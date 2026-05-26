import argparse
import json
import math
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


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
    "toxicity": [
        "too sensitive", "overreact", "overreacting", "loosen up", "banter",
        "handle it", "honestly", "probably just", "your fault", "dramatic",
        "big deal", "mountain out of a molehill", "paranoid", "conspiracies",
        "stop whining", "weak", "crazy",
    ],
    "medical_advice": [
        "diagnosable", "diagnosis", "disorder", "condition", "symptom",
        "treatment", "medication", "medicine", "cognitive restructuring",
        "social anxiety", "depression", "anxiety disorder", "requires",
        "you should", "you need to", "you must", "criteria", "intervention",
        "emergency evaluation",
    ],
    "factual_consistency": [
        "clearly", "always", "often", "direct indicator", "underlying",
        "suggests", "means that", "drives", "manifest", "resilient",
        "emotional stability", "known psychological", "proves", "obviously",
    ],
    "specificity": [
        "general", "generally", "things", "stuff", "sometimes", "normal",
        "common", "focus on other things", "helpful approach", "wellness",
        "support", "take care", "try to handle",
    ],
    "empathy": [
        "observe", "noticing", "common behavior", "pattern", "involved",
        "detached", "complicated", "it happens", "people do",
    ],
    "overall_quality": [
        "just", "maybe", "things", "stuff", "handle it", "figure out",
        "focus", "general", "complicated", "sometimes", "resolve over time",
        "next steps yourself",
    ],
}


SAFE_ANCHOR_PATTERNS = [
    "it sounds",
    "it makes sense",
    "i'm sorry",
    "you may want",
    "it may help",
    "qualified professional",
    "trusted",
    "if you suspect",
    "immediate danger",
    "emergency",
]


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


def find_keyword_spans(text, dim):
    keys = list(dict.fromkeys(DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]))
    spans = []
    low = text.lower()
    for key in sorted(keys, key=len, reverse=True):
        start = low.find(key.lower())
        if start >= 0:
            spans.append({
                "start": start,
                "end": start + len(key),
                "text": text[start:start+len(key)],
                "score": 1.0 if key in DIM_KEYWORDS.get(dim, []) else 0.6,
                "source": "keyword",
            })
    return spans


def sentence_risk_scores(text, dim):
    sents = split_sentences(text)
    keys = list(dict.fromkeys(DIM_KEYWORDS.get(dim, []) + DIM_KEYWORDS["overall_quality"]))
    scored = []
    for i, s in enumerate(sents):
        sl = s.lower()
        score = 0.0
        for k in keys:
            if k.lower() in sl:
                score += 1.0 if k in DIM_KEYWORDS.get(dim, []) else 0.55
        if len(s.split()) < 7:
            score += 0.15
        if any(x in sl for x in ["should", "must", "need to", "diagnos", "too sensitive"]):
            score += 0.4
        scored.append({"idx": i, "sent": s, "score": score})
    return scored


def choose_num_corrupt(t, n_available):
    if n_available <= 0:
        return 0
    return max(1, min(n_available, int(math.ceil(t * min(4, n_available)))))


def apply_phrase_corruption(text, dim, t, rng, mode):
    spans = find_keyword_spans(text, dim)
    if not spans:
        return text, []

    spans = sorted(spans, key=lambda x: x["score"], reverse=True)
    n = choose_num_corrupt(t, len(spans))
    chosen = spans[:n]

    # Avoid index shift by replacing from right to left.
    out = text
    records = []
    for sp in sorted(chosen, key=lambda x: x["start"], reverse=True):
        raw = out[sp["start"]:sp["end"]]
        if mode == "mark":
            rep = f"<risk>{raw}</risk>"
        elif mode == "delete":
            rep = "[removed unsafe phrase]"
        else:
            rep = f"[revise unsafe phrase: {raw}]"
        out = out[:sp["start"]] + rep + out[sp["end"]:]
        records.append({**sp, "corruption": mode})
    return out, records


def random_corrupt(text, t, rng):
    sents = split_sentences(text)
    if not sents:
        return text, []
    n = choose_num_corrupt(t, len(sents))
    idxs = rng.sample(range(len(sents)), k=n)
    records = []
    for idx in idxs:
        old = sents[idx]
        mode = rng.choice(["mark", "delete", "hint"])
        if mode == "mark":
            sents[idx] = f"<risk>{old}</risk>"
        elif mode == "delete":
            sents[idx] = "[removed phrase]"
        else:
            sents[idx] = f"[revise phrase: {old}]"
        records.append({"idx": idx, "text": old, "score": 0.0, "source": "random", "corruption": mode})
    return " ".join(sents), records


def risk_corrupt(text, dim, t, rng):
    mode = rng.choices(["mark", "delete", "hint"], weights=[0.45, 0.35, 0.20], k=1)[0]
    out, spans = apply_phrase_corruption(text, dim, t, rng, mode)
    if spans:
        return out, spans

    scored = sentence_risk_scores(text, dim)
    if not scored:
        return text, []
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    n = choose_num_corrupt(t, len(scored))
    sents = split_sentences(text)
    records = []
    for item in scored[:n]:
        idx = item["idx"]
        old = sents[idx]
        mode = rng.choices(["mark", "delete", "hint"], weights=[0.45, 0.35, 0.20], k=1)[0]
        if mode == "mark":
            sents[idx] = f"<risk>{old}</risk>"
        elif mode == "delete":
            sents[idx] = "[removed unsafe sentence]"
        else:
            sents[idx] = f"[revise unsafe sentence: {old}]"
        records.append({**item, "source": "sentence_risk", "corruption": mode})
    return " ".join(sents), records


def safe_anchor_sentences(safe):
    sents = split_sentences(safe)
    anchors = []
    for s in sents:
        sl = s.lower()
        if any(p in sl for p in SAFE_ANCHOR_PATTERNS):
            anchors.append(s)
    return anchors or sents[:2]


def bridge_corrupt(unsafe, safe, dim, t, rng):
    unsafe_sents = split_sentences(unsafe)
    if not unsafe_sents:
        return unsafe, []
    anchors = safe_anchor_sentences(safe)
    scored = sorted(sentence_risk_scores(unsafe, dim), key=lambda x: x["score"], reverse=True)
    n = choose_num_corrupt(t, len(scored))
    records = []
    for j, item in enumerate(scored[:n]):
        idx = item["idx"]
        old = unsafe_sents[idx]
        if anchors and rng.random() < 0.65:
            new = anchors[j % len(anchors)]
            unsafe_sents[idx] = f"[safe anchor: {new}]"
            ctype = "safe_anchor"
        else:
            unsafe_sents[idx] = f"[revise unsafe sentence: {old}]"
            ctype = "rewrite_hint"
        records.append({**item, "old": old, "corruption": ctype, "source": "bridge"})
    return " ".join(unsafe_sents), records


def build_condition_vector(oracle_dim, router_probs, rng, teacher_force_prob=0.65):
    use_oracle = rng.random() < teacher_force_prob or router_probs is None
    if use_oracle:
        probs = [0.0] * len(DIMS)
        probs[DIM2ID[oracle_dim]] = 1.0
        source = "oracle"
    else:
        probs = router_probs
        source = "router_pred"
    top_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "condition_source": source,
        "condition_dim": DIMS[top_idx],
        "condition_probs": {DIMS[i]: float(probs[i]) for i in range(len(DIMS))},
    }


@torch.no_grad()
def predict_router(rows, model_dir, batch_size=16, max_len=512):
    if not model_dir:
        return [None] * len(rows)

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    probs_all = []
    for i in tqdm(range(0, len(rows), batch_size), desc="router predict"):
        batch_rows = rows[i:i+batch_size]
        texts = []
        for ex in batch_rows:
            q = get_field(ex, "question", "query", "user_question")
            u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
            texts.append(
                "Question:\n" + q.strip() +
                "\n\nUnsafe response:\n" + u.strip() +
                "\n\nPredict which CounselBench quality dimension is violated."
            )
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        probs_all.extend(probs)
    return probs_all


def make_rows(rows, router_probs, examples_per_row, seed, teacher_force_prob):
    rng = random.Random(seed)
    out = []
    corruption_types = ["empty", "random", "risk", "bridge"]
    t_values = [0.25, 0.50, 0.75, 1.00]

    for i, ex in enumerate(rows):
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        y = get_field(ex, "safe_response", "target_response", "response")
        oracle_dim = norm_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))
        rp = router_probs[i]

        for k in range(examples_per_row):
            ctype = corruption_types[k % len(corruption_types)]
            t = t_values[(i + k) % len(t_values)]

            cond = build_condition_vector(oracle_dim, rp, rng, teacher_force_prob)
            active_dim = cond["condition_dim"]

            if ctype == "empty":
                draft = ""
                risk_spans = []
            elif ctype == "random":
                draft, risk_spans = random_corrupt(u, t, rng)
            elif ctype == "risk":
                draft, risk_spans = risk_corrupt(u, active_dim, t, rng)
            elif ctype == "bridge":
                draft, risk_spans = bridge_corrupt(u, y, active_dim, t, rng)
            else:
                raise ValueError(ctype)

            # Example-level loss weight approximates risk-weighted objective.
            # Risk corruption and stronger corruption receive slightly larger weights.
            loss_weight = 1.0
            if ctype == "risk":
                loss_weight += 0.35
            if ctype == "bridge":
                loss_weight += 0.20
            loss_weight += 0.25 * float(t)

            out.append({
                "question": q,
                "unsafe_response": u,
                "safe_response": y,
                "target_dimension": oracle_dim,
                "condition_dim": cond["condition_dim"],
                "condition_source": cond["condition_source"],
                "condition_probs": cond["condition_probs"],
                "corruption_type": ctype,
                "corruption_level": t,
                "corrupted_draft": draft,
                "risk_spans": risk_spans,
                "loss_weight": loss_weight,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--router_dir", default="")
    ap.add_argument("--examples_per_row", type=int, default=8)
    ap.add_argument("--teacher_force_prob", type=float, default=0.65)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    router_probs = predict_router(rows, args.router_dir) if args.router_dir else [None] * len(rows)
    out = make_rows(rows, router_probs, args.examples_per_row, args.seed, args.teacher_force_prob)

    write_jsonl(out, args.output)

    manifest = {
        "input": args.input,
        "output": args.output,
        "router_dir": args.router_dir,
        "examples_per_row": args.examples_per_row,
        "teacher_force_prob": args.teacher_force_prob,
        "n_input": len(rows),
        "n_output": len(out),
        "dims": DIMS,
        "corruption_types": ["empty", "random", "risk", "bridge"],
    }
    with open(str(Path(args.output).with_suffix(".manifest.json")), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
