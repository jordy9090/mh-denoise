import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


DIM_KEYWORDS = {
    "toxicity": [
        "sensitive", "overreact", "loosen", "banter", "handle", "honestly", "probably",
        "fault", "blame", "dramatic", "big deal", "mountain", "molehill", "paranoid",
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
    "diagnosable",
    "you have",
    "you need",
    "you must",
    "too sensitive",
    "overreacting",
    "paranoid",
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


def mask_bad_spans_by_text(unsafe, dim, mask_token, seed):
    rng = random.Random(seed)
    dim = normalize_dim(dim)
    text = unsafe

    keys = DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]

    replaced_any = False

    for key in sorted(keys, key=len, reverse=True):
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.9:
            text = pattern.sub(mask_token, text, count=1)
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
                sents[i] = mask_token
                replaced_any = True

        text = " ".join(sents)

    if mask_token not in text:
        text = mask_token + " " + text

    if rng.random() < 0.7:
        text = text.rstrip()
        if not text.endswith((".", "!", "?")):
            text += "."
        text += f" {mask_token}"

    return text


def build_input(ex, tokenizer, seed):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    d = normalize_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))

    mask_token = tokenizer.mask_token if tokenizer.mask_token else "<mask>"
    draft = mask_bad_spans_by_text(u, d, mask_token, seed)

    return (
        "Refine the unsafe counseling response into a safe, empathetic, specific, and professionally bounded response.\n"
        f"Question: {q}\n"
        f"Violated quality dimension: {d}\n"
        f"Unsafe response: {u}\n"
        f"Masked unsafe draft: {draft}\n"
        "Safe response:"
    ), draft


def repetition_penalty(text):
    toks = text.lower().split()
    if len(toks) < 2:
        return 0
    rep = 0
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            rep += 1
    return rep / len(toks)


def bad_marker_count(text):
    t = text.lower()
    return sum(1 for m in BAD_MARKERS if m in t)


def score_candidate(text):
    words = text.split()
    score = 0.0
    score -= repetition_penalty(text) * 10.0
    score -= bad_marker_count(text) * 1.5
    if len(words) < 35:
        score -= 3.0
    if len(words) > 180:
        score -= 1.0
    if "it sounds" in text.lower() or "i'm sorry" in text.lower() or "understandable" in text.lower():
        score += 1.0
    if "professional" in text.lower() or "qualified" in text.lower() or "trusted" in text.lower():
        score += 0.5
    return score


def cleanup(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([,.;:])\1{1,}", r"\1", text)
    text = re.sub(r"\b(\w+)( \1\b){2,}", r"\1", text, flags=re.I)
    return text.strip()


@torch.no_grad()
def generate_candidates(model, tokenizer, ex, device, args, idx):
    candidates = []

    for c in range(args.candidates):
        inp, draft = build_input(ex, tokenizer, args.seed + idx * 100 + c)
        enc = tokenizer(
            inp,
            return_tensors="pt",
            max_length=args.max_source_len,
            truncation=True,
        ).to(device)

        gen = model.generate(
            **enc,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            num_return_sequences=1,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            length_penalty=args.length_penalty,
            early_stopping=True,
        )

        text = tokenizer.decode(gen[0], skip_special_tokens=True)
        text = cleanup(text)
        candidates.append((score_candidate(text), text, draft))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_source_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    ap.add_argument("--candidates", type=int, default=5)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--do_sample", action="store_true")
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--repetition_penalty", type=float, default=1.15)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.mask_token is None and "<mask>" in tokenizer.get_vocab():
        tokenizer.mask_token = "<mask>"

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device)
    model.eval()

    outs = []
    rows = list(read_jsonl(args.input))

    for idx, ex in enumerate(tqdm(rows)):
        cands = generate_candidates(model, tokenizer, ex, device, args, idx)
        best_score, best_text, best_draft = cands[0]

        out = dict(ex)
        out["seq2seq_span_response"] = best_text
        out["seq2seq_span_score"] = best_score
        out["seq2seq_masked_draft"] = best_draft
        out["seq2seq_candidates"] = [t for _, t, _ in cands]
        out["method"] = "seq2seq_span_infilling_refiner_v2"
        outs.append(out)

    write_jsonl(outs, args.output)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
