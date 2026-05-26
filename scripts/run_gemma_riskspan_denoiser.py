import argparse
import json
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

DIM_PATTERNS = {
    "medical_advice": [
        ("diagnosable", "overconfident diagnostic framing", "avoid diagnosis; suggest qualified support"),
        ("diagnosis", "diagnostic labeling", "avoid diagnosis; use cautious wording"),
        ("disorder", "diagnostic labeling", "avoid diagnostic labels"),
        ("borderline personality disorder", "specific diagnostic label", "avoid diagnosing"),
        ("adjustment disorder", "specific diagnostic label", "avoid diagnosing"),
        ("social anxiety", "possible diagnostic claim", "avoid diagnosing"),
        ("cognitive restructuring", "direct therapeutic technique prescription", "offer general coping options"),
        ("medication", "medical treatment advice", "refer to licensed professional"),
        ("you should", "directive advice", "soften into optional suggestion"),
        ("you need to", "directive advice", "soften into optional suggestion"),
        ("you must", "directive advice", "soften into optional suggestion"),
        ("formal diagnosis", "diagnostic escalation", "suggest qualified evaluation without diagnosing"),
    ],
    "toxicity": [
        ("too sensitive", "invalidating phrase", "validate the user's feeling"),
        ("overreact", "dismissive phrasing", "avoid judgment"),
        ("loosen up", "dismissive advice", "validate discomfort"),
        ("big deal", "minimizing phrase", "validate concern"),
        ("mountain out of a molehill", "dismissive idiom", "remove judgment"),
        ("paranoid", "stigmatizing phrase", "avoid stigmatizing labels"),
    ],
    "factual_consistency": [
        ("clearly", "overconfident claim", "use uncertainty-aware wording"),
        ("always", "absolute claim", "avoid absolutes"),
        ("direct indicator", "unsupported causal claim", "avoid causal overclaim"),
        ("underlying", "unsupported hidden-cause claim", "avoid speculation"),
        ("suggests", "possibly unsupported inference", "soften inference"),
        ("means that", "overconfident interpretation", "use cautious wording"),
        ("proves", "overclaim", "avoid overclaim"),
    ],
    "specificity": [
        ("things", "generic wording", "make response more concrete"),
        ("stuff", "generic wording", "make response more concrete"),
        ("generally", "generic advice", "add situation-specific guidance"),
        ("focus on other things", "vague coping advice", "offer concrete coping step"),
        ("take care", "generic closing", "add concrete next step"),
    ],
    "empathy": [
        ("observe", "detached tone", "acknowledge feeling"),
        ("noticing", "emotionally distant wording", "validate emotion"),
        ("common behavior", "overly analytical response", "center user's experience"),
        ("pattern", "overly analytical response", "center user's experience"),
        ("it happens", "minimizing phrase", "validate difficulty"),
    ],
    "overall_quality": [
        ("just", "minimizing or oversimplified wording", "use more supportive framing"),
        ("maybe", "vague response", "provide clearer support"),
        ("handle it", "unsupported directive", "offer supportive next step"),
        ("figure out", "burdens user", "provide guidance"),
        ("resolve over time", "passive reassurance", "offer active support"),
        ("next steps yourself", "burdens user", "provide concrete next step"),
    ],
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


def build_router_text(ex):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "Question:\n" + q.strip()
        + "\n\nUnsafe response:\n" + u.strip()
        + "\n\nTask: identify the primary violated response-quality dimension."
    )


@torch.no_grad()
def predict_dim(router, router_tok, ex, device):
    text = build_router_text(ex)
    enc = router_tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = router(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    top = max(range(len(probs)), key=lambda i: probs[i])
    return DIMS[top], {DIMS[i]: float(probs[i]) for i in range(len(DIMS))}


def find_risk_spans(text, dim):
    patterns = DIM_PATTERNS.get(dim, []) + DIM_PATTERNS["overall_quality"]
    out = []
    low = text.lower()
    occupied = []
    for phrase, reason, action in sorted(patterns, key=lambda x: len(x[0]), reverse=True):
        start = low.find(phrase.lower())
        if start < 0:
            continue
        end = start + len(phrase)
        overlap = any(not (end <= a or start >= b) for a, b in occupied)
        if overlap:
            continue
        occupied.append((start, end))
        out.append({
            "text": text[start:end],
            "dimension": dim,
            "risk_score": 1.0,
            "reason": reason,
            "suggested_action": action,
            "source": "phrase_rule",
        })
        if len(out) >= 5:
            break
    return out


def format_probs(probs):
    items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    return ", ".join([f"{k}={v:.2f}" for k, v in items])


def format_spans(spans):
    if not spans:
        return "None."
    lines = []
    for i, sp in enumerate(spans[:6], 1):
        lines.append(
            f'{i}. span: "{sp["text"]}" | dimension: {sp["dimension"]} | reason: {sp["reason"]} | action: {sp["suggested_action"]}'
        )
    return "\n".join(lines)


def build_prompt(ex, dim, probs, spans):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    return (
        "You are a safety-aware mental-health response refiner.\n"
        "Rewrite the unsafe response into a safe, empathetic, specific, and professionally bounded response.\n"
        "Return only the final response shown to the user.\n"
        "Do not mention labels, risk spans, probabilities, corruption, drafts, or analysis.\n"
        "Do not diagnose. Do not prescribe medication. Do not shame the user. Do not make unsupported clinical claims.\n\n"
        f"Question:\n{q.strip()}\n\n"
        f"Unsafe response:\n{u.strip()}\n\n"
        f"Active violation dimension:\n{dim}\n\n"
        f"Aspect probabilities:\n{format_probs(probs)}\n\n"
        f"Detected risk/edit spans:\n{format_spans(spans)}\n\n"
        "Safe response:\n"
    )


def cleanup(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    for marker in ["Safe response:", "Final response:", "Analysis:", "Thought:", "Risk spans:"]:
        if marker in text:
            text = text.split(marker)[-1].strip()
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
    ap.add_argument("--use_oracle_dim", action="store_true")
    args = ap.parse_args()

    router_tok = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir)
    r_device = "cuda" if torch.cuda.is_available() else "cpu"
    router.to(r_device)
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
    for ex in tqdm(read_jsonl(args.input)):
        pred_dim, probs = predict_dim(router, router_tok, ex, r_device)
        dim = get_field(ex, "target_dimension") if args.use_oracle_dim else pred_dim
        spans = find_risk_spans(get_field(ex, "unsafe_response", "corrupted_response", "bad_response"), dim)
        prompt = build_prompt(ex, dim, probs, spans)

        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=args.max_source_len).to(device)

        kwargs = dict(
            **enc,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if args.temperature and args.temperature > 0:
            kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
        else:
            kwargs.update(dict(do_sample=False))

        with torch.no_grad():
            gen = model.generate(**kwargs)

        new_tokens = gen[0][enc["input_ids"].shape[-1]:]
        text = cleanup(tok.decode(new_tokens, skip_special_tokens=True))

        out = dict(ex)
        out["predicted_dimension"] = pred_dim
        out["used_dimension"] = dim
        out["router_probs"] = probs
        out["risk_spans"] = spans
        out["riskspan_response"] = text
        out["method"] = "riskspan_router_lora_denoiser"
        outs.append(out)

    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
