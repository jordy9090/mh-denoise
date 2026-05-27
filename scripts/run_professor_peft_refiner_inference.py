import argparse, json, re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]

MASK_MARKERS = ["<MASK>", "[MASK]", "[needs revision]", "[NEEDS REVISION]"]

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def get_field(ex, *names, default=""):
    for name in names:
        value = ex.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default

def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

def sanitize_draft(text):
    text = str(text or "")
    for marker in MASK_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if not text:
        return "No usable partial draft is available. Rewrite from the question and unsafe response."
    return text

def format_aspect_signal(ex):
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        active = [f"{name}={float(score):.2f}" for name, score in pairs if float(score) >= 0.10]
        return ", ".join(active[:4]) if active else f"{pairs[0][0]}={float(pairs[0][1]):.2f}"
    for key in ["target_dimension", "condition_dim", "used_dimension", "predicted_dimension", "dimension"]:
        value = ex.get(key)
        if value:
            return str(value)
    return "unspecified"

def build_user_content(ex):
    q = clean_text(get_field(ex, "question", "query", "user_question"))
    u = clean_text(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"))
    z = sanitize_draft(get_field(ex, "z_t", "draft", "masked_unsafe_draft", default=""))
    aspect = format_aspect_signal(ex)
    t = ex.get("t", ex.get("corruption_level", ""))
    T = ex.get("T", "")
    t_str = f"{t}/{T}" if str(t) and str(T) else str(t)

    return (
        "Rewrite the unsafe counseling response into a safe final response.\n"
        "Use the question, the unsafe response, the aspect signal, and the corrupted draft.\n"
        "Return only the final response. Do not include section labels, analysis, metadata, or the draft.\n\n"
        f"Question:\n{q}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Aspect signal:\n{aspect}\n\n"
        f"Corruption strength:\n{t_str}\n\n"
        f"Risk-deleted draft z_t:\n{z}\n\n"
        "Safety requirements:\n"
        "- Validate the user's feelings without blaming or shaming them.\n"
        "- Do not diagnose the user or state that they have a disorder.\n"
        "- Do not give direct medication or treatment instructions.\n"
        "- Give concrete, bounded next steps when appropriate.\n"
        "- Encourage professional or trusted support for serious or persistent distress."
    )

def build_prompt(tokenizer, ex):
    system = (
        "You are a careful mental-health response refiner. "
        "Your job is to transform unsafe or low-quality counseling answers into safe, empathetic, specific, professionally bounded answers."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_content(ex)},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system}\n\nUser: {build_user_content(ex)}\n\nAssistant:"

def cleanup(text):
    text = text.strip()

    # Remove leaked section markers.
    bad_markers = [
        "Final response:",
        "Safe response:",
        "Response:",
        "Analysis:",
        "Notes:",
        "Draft:",
        "Aspect signal:",
        "Safety requirements:",
        "FINAL RESPONSE",
        "TOTAL RESPONSE",
    ]
    for m in bad_markers:
        if m in text:
            text = text.split(m)[-1].strip()

    # Cut common decoding artifacts observed in validation samples.
    cut_patterns = [
        r"\s*oi\b.*$",
        r"\s*\.oi\b.*$",
        r"\s*なければ.*$",
        r"\s*</tr>.*$",
        r"\s*/></tr>.*$",
        r"\s*この質問.*$",
    ]
    for pat in cut_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Remove stray HTML-like fragments.
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text

@torch.no_grad()
def generate_one(model, tok, prompt, max_source_len, max_new_tokens):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_source_len).to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.18,
        no_repeat_ngram_size=5,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    new_tokens = gen[0][enc["input_ids"].shape[-1]:]
    return cleanup(tok.decode(new_tokens, skip_special_tokens=True))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=896)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    rows = read_jsonl(args.input)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    outs = []
    for ex in tqdm(rows):
        prompt = build_prompt(tok, ex)
        pred = generate_one(model, tok, prompt, args.max_source_len, args.max_new_tokens)
        out = dict(ex)
        out["professor_peft_response"] = pred
        out["method"] = "professor_peft_safecorrupt"
        out["adapter_dir"] = args.adapter_dir
        outs.append(out)

    write_jsonl(outs, args.output)
    print("saved to", args.output)

if __name__ == "__main__":
    main()
