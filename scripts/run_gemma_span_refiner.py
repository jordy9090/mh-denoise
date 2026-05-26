import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


DIM_KEYWORDS = {
    "toxicity": [
        "sensitive", "overreact", "overreacting", "loosen", "banter", "handle",
        "honestly", "probably", "fault", "blame", "dramatic", "big deal",
        "mountain", "molehill", "paranoid", "conspiracies",
    ],
    "medical_advice": [
        "diagnosable", "diagnosis", "disorder", "condition", "symptom", "treatment",
        "medication", "medicine", "cognitive restructuring", "anxiety", "depression",
        "requires", "should", "need", "must", "criteria", "intervention",
    ],
    "factual_consistency": [
        "clearly", "always", "often", "direct indicator", "underlying", "suggests",
        "means", "drives", "manifest", "resilient", "stability", "known psychological",
    ],
    "specificity": [
        "general", "generally", "things", "stuff", "sometimes", "normal", "common",
        "focus", "helpful approach", "wellness", "support", "take care",
    ],
    "empathy": [
        "observe", "noticing", "common behavior", "pattern", "involved",
        "detached", "complicated",
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle", "figure out", "focus",
        "general", "complicated", "sometimes", "resolve", "next steps yourself",
    ],
}


BAD_MARKERS = [
    "too sensitive",
    "overreacting",
    "paranoid",
    "diagnosable",
    "you have",
    "you need",
    "you must",
    "disorder",
    "cognitive restructuring",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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


def normalize_dim(dim):
    d = str(dim).strip().lower().replace("-", "_").replace(" ", "_")
    return d if d in DIM_KEYWORDS else "overall_quality"


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def mask_bad_spans_by_text(unsafe, dim, seed):
    rng = random.Random(seed)
    dim = normalize_dim(dim)
    text = unsafe
    mask = "[MASK]"

    keys = DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]
    replaced_any = False

    for key in sorted(keys, key=len, reverse=True):
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.9:
            text = pattern.sub(mask, text, count=1)
            replaced_any = True

    sents = split_sentences(text)
    if sents:
        scored = []
        for i, s in enumerate(sents):
            sl = s.lower()
            score = sum(1 for k in keys if k.lower() in sl)
            if len(s.split()) < 7:
                score += 0.3
            scored.append((score, i, s))

        scored.sort(reverse=True)
        n_mask = 1 if replaced_any else min(2, len(sents))

        for _, i, _ in scored[:n_mask]:
            if rng.random() < 0.85 or not replaced_any:
                sents[i] = mask
                replaced_any = True

        text = " ".join(sents)

    if mask not in text:
        text = mask + " " + text

    if rng.random() < 0.7:
        if not text.rstrip().endswith((".", "!", "?")):
            text = text.rstrip() + "."
        text += " [MASK]"

    return text


def build_prompt(ex, seed):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
    draft = mask_bad_spans_by_text(u, d, seed)

    prompt = (
        "You are refining a mental-health counseling response.\n"
        "Rewrite the unsafe response into a safe, empathetic, specific, and professionally bounded response.\n"
        "Do not diagnose. Do not give direct medical instructions. Do not shame the user.\n\n"
        f"Question:\n{q}\n\n"
        f"Violated quality dimension:\n{d}\n\n"
        f"Unsafe response:\n{u}\n\n"
        f"Masked unsafe draft:\n{draft}\n\n"
        "Safe response:\n"
    )
    return prompt, draft


def cleanup(text):
    text = text.strip()
    text = text.split("<eos>")[0].strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", text, flags=re.I)
    return text.strip()


def repetition_penalty_score(text):
    toks = text.lower().split()
    if len(toks) < 2:
        return 0.0
    rep = sum(1 for i in range(1, len(toks)) if toks[i] == toks[i - 1])
    return rep / len(toks)


def bad_marker_count(text):
    t = text.lower()
    return sum(1 for m in BAD_MARKERS if m in t)


def score_candidate(text):
    words = text.split()
    score = 0.0
    score -= repetition_penalty_score(text) * 10.0
    score -= bad_marker_count(text) * 2.0

    if len(words) < 35:
        score -= 3.0
    if len(words) > 180:
        score -= 1.0

    low = text.lower()
    if "it sounds" in low or "i'm sorry" in low or "understandable" in low:
        score += 1.0
    if "professional" in low or "qualified" in low or "trusted" in low:
        score += 0.5
    if "if you suspect" in low or "if there is immediate" in low:
        score += 0.5

    return score


@torch.no_grad()
def generate_one(model, tokenizer, ex, device, args, seed):
    prompt, draft = build_prompt(ex, seed)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_source_len,
    ).to(device)

    gen = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    new_tokens = gen[0][enc["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return cleanup(text), draft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=768)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.12)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    device = next(model.parameters()).device

    outs = []
    rows = list(read_jsonl(args.input))

    for idx, ex in enumerate(tqdm(rows)):
        cands = []
        for c in range(args.candidates):
            text, draft = generate_one(
                model,
                tokenizer,
                ex,
                device,
                args,
                args.seed + idx * 100 + c,
            )
            cands.append((score_candidate(text), text, draft))

        cands.sort(key=lambda x: x[0], reverse=True)
        best_score, best_text, best_draft = cands[0]

        out = dict(ex)
        out["gemma_span_response"] = best_text
        out["gemma_span_score"] = best_score
        out["gemma_masked_draft"] = best_draft
        out["gemma_span_candidates"] = [t for _, t, _ in cands]
        out["method"] = "gemma_masked_draft_span_refiner_v2"
        outs.append(out)

    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
