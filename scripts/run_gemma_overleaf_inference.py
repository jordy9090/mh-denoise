import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


DIMS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]
DIM2ID = {d: i for i, d in enumerate(DIMS)}

DIM_KEYWORDS = {
    "toxicity": ["too sensitive", "overreact", "loosen up", "banter", "honestly", "probably just", "your fault", "dramatic", "big deal", "paranoid"],
    "medical_advice": ["diagnosable", "diagnosis", "disorder", "condition", "symptom", "treatment", "medication", "medicine", "cognitive restructuring", "social anxiety", "depression", "requires", "you should", "you need to", "you must"],
    "factual_consistency": ["clearly", "always", "often", "direct indicator", "underlying", "suggests", "means that", "drives", "manifest", "resilient", "proves", "obviously"],
    "specificity": ["general", "generally", "things", "stuff", "sometimes", "normal", "common", "focus on other things", "helpful approach", "wellness", "support", "take care"],
    "empathy": ["observe", "noticing", "common behavior", "pattern", "involved", "detached", "complicated", "it happens", "people do"],
    "overall_quality": ["just", "maybe", "things", "stuff", "handle it", "figure out", "focus", "general", "complicated", "sometimes", "resolve over time"],
}


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
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x):
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return x if x in DIM2ID else "overall_quality"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def format_probs(probs):
    return ", ".join([f"{k}={probs.get(k, 0.0):.2f}" for k in DIMS])


def risk_corrupt(text, dim, t=1.0):
    keys = list(dict.fromkeys(DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]))
    out = text
    records = []

    for key in sorted(keys, key=len, reverse=True):
        low = out.lower()
        start = low.find(key.lower())
        if start < 0:
            continue
        end = start + len(key)
        raw = out[start:end]
        out = out[:start] + f"<risk>{raw}</risk>" + out[end:]
        records.append({"text": raw, "source": "keyword", "dim": dim})
        if len(records) >= max(1, int(round(4 * t))):
            break

    if records:
        return out, records

    sents = split_sentences(text)
    if not sents:
        return text, []
    idx = 0
    sents[idx] = f"<risk>{sents[idx]}</risk>"
    return " ".join(sents), [{"text": sents[idx], "source": "fallback_sentence", "dim": dim}]


def build_router_text(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "Question:\n" + q.strip() +
        "\n\nUnsafe response:\n" + u.strip() +
        "\n\nPredict which CounselBench quality dimension is violated."
    )


@torch.no_grad()
def predict_dim(router, router_tok, ex, device):
    text = build_router_text(ex)
    enc = router_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = router(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    top = max(range(len(probs)), key=lambda i: probs[i])
    return DIMS[top], {DIMS[i]: float(probs[i]) for i in range(len(DIMS))}


def build_prompt(ex, dim, probs, draft, t):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "You are a safety-aware mental-health response denoiser.\n"
        "Your task is to rewrite an unsafe or low-quality counseling response into a safe, empathetic, specific, and professionally bounded response.\n"
        "Return only the final response shown to the user.\n"
        "Do not mention labels, risk tags, corruption, drafts, analysis, or hidden reasoning.\n"
        "Do not diagnose. Do not prescribe medication. Do not shame the user. Do not make unsupported clinical claims.\n\n"
        f"Question:\n{q.strip()}\n\n"
        f"Unsafe response:\n{u.strip()}\n\n"
        f"Active violation dimension:\n{dim}\n\n"
        f"Aspect probabilities:\n{format_probs(probs)}\n\n"
        "Corruption source:\nrisk\n\n"
        f"Corruption level t:\n{t:.2f}\n\n"
        f"Corrupted draft z_t:\n{draft}\n\n"
        "Safe response:\n"
    )


def cleanup(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    for marker in ["Safe response:", "Final response:", "Analysis:", "Thought:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
    text = text.replace("<risk>", "").replace("</risk>", "")
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir)
    router_device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(router_device)
    router.eval()

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
    device = next(model.parameters()).device

    outs = []
    rows = read_jsonl(args.input)

    for ex in tqdm(rows):
        dim, probs = predict_dim(router, router_tok, ex, router_device)
        draft, risk_spans = risk_corrupt(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"), dim, t=1.0)
        prompt = build_prompt(ex, dim, probs, draft, t=1.0)

        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(device)

        gen_kwargs = dict(
            **enc,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        if args.temperature and args.temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
        else:
            gen_kwargs.update(dict(do_sample=False))

        with torch.no_grad():
            gen = model.generate(**gen_kwargs)

        new = gen[0][enc["input_ids"].shape[-1]:]
        text = cleanup(tok.decode(new, skip_special_tokens=True))

        out = dict(ex)
        out["predicted_dimension"] = dim
        out["router_probs"] = probs
        out["risk_draft"] = draft
        out["risk_spans"] = risk_spans
        out["overleaf_denoised_response"] = text
        out["method"] = "overleaf_lite_router_risk_discrete_lora_denoiser"
        outs.append(out)

    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
