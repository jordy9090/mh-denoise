import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

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
BAD_OUTPUT_PATTERNS = [
    r"\bQuestion:\b",
    r"\bUnsafe response:\b",
    r"\bAspect signal:\b",
    r"\bRisk-deleted draft\b",
    r"\bSafe response:\b",
    r"<MASK>|\[MASK\]",
]


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_field(ex: Dict, *names: str, default: str = "") -> str:
    for name in names:
        value = ex.get(name)
        if value is not None and str(value).strip():
            return str(value)
    return default


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def sanitize_draft(text: str) -> str:
    text = str(text or "")
    for marker in MASK_MARKERS:
        text = text.replace(marker, " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    if not text:
        return "No usable partial draft is available. Rewrite from the question and unsafe response."
    return text


def format_aspect_signal(ex: Dict) -> str:
    g = ex.get("g")
    if isinstance(g, list) and len(g) == len(DIMS):
        pairs = sorted(zip(DIMS, g), key=lambda x: float(x[1]), reverse=True)
        active = [f"{name}={float(score):.2f}" for name, score in pairs if float(score) >= 0.10]
        return ", ".join(active[:4]) if active else f"{pairs[0][0]}={float(pairs[0][1]):.2f}"
    for key in ["target_dimension", "condition_dim", "used_dimension", "predicted_dimension", "dimension"]:
        if ex.get(key):
            return str(ex[key])
    return "unspecified"


def build_user_content(ex: Dict) -> str:
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


def build_prompt(tokenizer, ex: Dict) -> str:
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


def cleanup(text: str) -> str:
    text = str(text or "")
    text = text.split("<eos>")[0]
    for label in ["Safe response:", "Final response:", "Assistant:"]:
        if label in text:
            text = text.split(label, 1)[-1]
    text = re.sub(r"<MASK>|\[MASK\]|\[needs revision\]", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1+", r"\1", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", text, flags=re.I)
    return text.strip()


def output_score(text: str) -> float:
    score = 0.0
    words = text.split()
    low = text.lower()
    if 35 <= len(words) <= 190:
        score += 2.0
    else:
        score -= 2.0
    for pat in BAD_OUTPUT_PATTERNS:
        if re.search(pat, text, flags=re.I):
            score -= 5.0
    repeated = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i - 1].lower())
    score -= repeated * 1.0
    if any(x in low for x in ["it makes sense", "understandable", "it sounds", "sorry"]):
        score += 1.0
    if any(x in low for x in ["professional", "trusted", "counselor", "therapist", "support"]):
        score += 0.5
    if any(x in low for x in ["you have", "you must", "you need medication", "diagnos"]):
        score -= 2.0
    return score


@torch.no_grad()
def generate_candidate(model, tokenizer, ex: Dict, args):
    prompt = build_prompt(tokenizer, ex)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(model.device)
    gen = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tokens = gen[0][enc["input_ids"].shape[-1]:]
    return cleanup(tokenizer.decode(new_tokens, skip_special_tokens=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=896)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--candidates", type=int, default=1)
    ap.add_argument("--repetition_penalty", type=float, default=1.08)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    rows = list(read_jsonl(args.input))
    outs = []
    for ex in tqdm(rows):
        cands = []
        for _ in range(max(1, args.candidates)):
            text = generate_candidate(model, tokenizer, ex, args)
            cands.append((output_score(text), text))
        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text = cands[0]
        out = dict(ex)
        out["professor_peft_response"] = best_text
        out["professor_peft_score"] = best_score
        out["professor_peft_candidates"] = [text for _, text in cands]
        out["method"] = "professor_style_standard_peft_lora_refiner"
        outs.append(out)
    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
