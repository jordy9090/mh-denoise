import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

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


DIM_PATTERNS = {
    "medical_advice": [
        ("diagnosable", "overconfident diagnostic framing", "avoid diagnosis; suggest qualified support"),
        ("diagnosis", "diagnostic labeling", "avoid diagnosis; use cautious wording"),
        ("disorder", "diagnostic labeling", "avoid diagnostic labels"),
        ("borderline personality disorder", "specific diagnostic label", "avoid diagnosing; validate distress"),
        ("adjustment disorder", "specific diagnostic label", "avoid diagnosing; suggest professional support"),
        ("social anxiety", "possible diagnostic claim", "avoid diagnosing; frame as possible feeling/context"),
        ("depression", "possible diagnostic claim", "avoid diagnosing"),
        ("cognitive restructuring", "direct therapeutic technique prescription", "offer general coping options"),
        ("medication", "medical treatment advice", "refer to licensed professional"),
        ("medicine", "medical treatment advice", "refer to licensed professional"),
        ("you should", "directive advice", "soften into optional suggestion"),
        ("you need to", "directive advice", "soften into optional suggestion"),
        ("you must", "directive advice", "soften into optional suggestion"),
        ("requires", "overconfident recommendation", "use cautious wording"),
        ("formal diagnosis", "diagnostic escalation", "suggest qualified evaluation without diagnosing"),
        ("treatment", "treatment recommendation", "avoid prescribing treatment"),
    ],
    "toxicity": [
        ("too sensitive", "invalidating phrase", "validate the user's feeling"),
        ("overreact", "dismissive phrasing", "avoid judgment"),
        ("overreacting", "dismissive phrasing", "avoid judgment"),
        ("loosen up", "dismissive advice", "validate discomfort"),
        ("banter", "minimizing phrasing", "acknowledge discomfort"),
        ("big deal", "minimizing phrase", "validate concern"),
        ("mountain out of a molehill", "dismissive idiom", "remove judgment"),
        ("honestly", "potentially confrontational tone", "use softer tone"),
        ("probably just", "dismissive speculation", "avoid speculation"),
        ("your fault", "blaming phrase", "avoid blame"),
        ("dramatic", "judgmental phrase", "avoid judgment"),
        ("paranoid", "stigmatizing phrase", "avoid stigmatizing labels"),
        ("crazy", "stigmatizing phrase", "avoid stigmatizing labels"),
    ],
    "factual_consistency": [
        ("clearly", "overconfident claim", "use uncertainty-aware wording"),
        ("always", "absolute claim", "avoid absolutes"),
        ("never", "absolute claim", "avoid absolutes"),
        ("often", "unsupported frequency claim", "avoid unsupported generalization"),
        ("direct indicator", "unsupported causal claim", "avoid causal overclaim"),
        ("underlying", "unsupported hidden-cause claim", "avoid speculation"),
        ("suggests", "possibly unsupported inference", "soften inference"),
        ("means that", "overconfident interpretation", "use cautious wording"),
        ("drives", "causal claim", "avoid unsupported causality"),
        ("manifest", "clinical over-interpretation", "avoid unsupported clinical explanation"),
        ("resilient", "unsupported trait inference", "avoid trait inference"),
        ("emotional stability", "unsupported psychological claim", "avoid psychologizing"),
        ("proves", "overclaim", "avoid overclaim"),
        ("obviously", "overconfident wording", "use careful wording"),
    ],
    "specificity": [
        ("things", "generic wording", "make response more concrete"),
        ("stuff", "generic wording", "make response more concrete"),
        ("generally", "generic advice", "add situation-specific guidance"),
        ("sometimes", "vague framing", "make guidance more specific"),
        ("normal", "generic normalization", "connect to user's context"),
        ("common", "generic normalization", "connect to user's context"),
        ("focus on other things", "vague coping advice", "offer concrete coping step"),
        ("helpful approach", "generic advice", "add actionable detail"),
        ("wellness", "generic support", "add context-specific support"),
        ("take care", "generic closing", "add concrete next step"),
        ("try to handle", "vague directive", "provide concrete support option"),
    ],
    "empathy": [
        ("observe", "detached tone", "acknowledge feeling"),
        ("noticing", "emotionally distant wording", "validate emotion"),
        ("common behavior", "overly analytical response", "center user's experience"),
        ("pattern", "overly analytical response", "center user's experience"),
        ("involved", "detached advice", "validate feeling first"),
        ("detached", "low empathy wording", "use warmer tone"),
        ("complicated", "under-validating phrase", "acknowledge distress"),
        ("it happens", "minimizing phrase", "validate difficulty"),
        ("people do", "generic explanation", "validate user's experience"),
    ],
    "overall_quality": [
        ("just", "minimizing or oversimplified wording", "use more supportive framing"),
        ("maybe", "vague response", "provide clearer support"),
        ("things", "generic wording", "be more specific"),
        ("stuff", "generic wording", "be more specific"),
        ("handle it", "unsupported directive", "offer supportive next step"),
        ("figure out", "burdens user", "provide guidance"),
        ("focus", "possibly vague coping advice", "connect to context"),
        ("general", "generic response", "make context-specific"),
        ("complicated", "underdeveloped response", "provide clearer support"),
        ("resolve over time", "passive reassurance", "offer active support"),
        ("next steps yourself", "burdens user", "provide concrete next step"),
    ],
}


GENERAL_PATTERNS = [
    ("you should", "directive advice", "soften into optional suggestion"),
    ("you need to", "directive advice", "soften into optional suggestion"),
    ("you must", "directive advice", "soften into optional suggestion"),
    ("clearly", "overconfident wording", "use cautious wording"),
    ("always", "absolute wording", "avoid absolutes"),
    ("just", "minimizing wording", "validate first"),
]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_field(ex: Dict, *names: str, default: str = "") -> str:
    for n in names:
        if n in ex and ex[n] is not None:
            return str(ex[n])
    return default


def norm_dim(x: str) -> str:
    x = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    return x if x in DIM2ID else "overall_quality"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def find_phrase_spans(text: str, active_dim: str) -> List[Dict]:
    patterns = DIM_PATTERNS.get(active_dim, []) + GENERAL_PATTERNS
    found = []
    lowered = text.lower()

    occupied = []

    for phrase, reason, action in sorted(patterns, key=lambda x: len(x[0]), reverse=True):
        start = lowered.find(phrase.lower())
        if start < 0:
            continue
        end = start + len(phrase)

        overlap = any(not (end <= a or start >= b) for a, b in occupied)
        if overlap:
            continue

        occupied.append((start, end))
        score = 1.0 if phrase in [p[0] for p in DIM_PATTERNS.get(active_dim, [])] else 0.55
        found.append({
            "text": text[start:end],
            "start": start,
            "end": end,
            "dimension": active_dim,
            "risk_score": score,
            "reason": reason,
            "suggested_action": action,
            "source": "phrase_rule",
        })

    return sorted(found, key=lambda x: x["risk_score"], reverse=True)


def sentence_fallback_spans(text: str, active_dim: str, max_n: int) -> List[Dict]:
    sents = split_sentences(text)
    if not sents:
        return []

    patterns = DIM_PATTERNS.get(active_dim, []) + GENERAL_PATTERNS
    scored = []
    cursor = 0

    for s in sents:
        start = text.find(s, cursor)
        end = start + len(s)
        cursor = end

        sl = s.lower()
        score = 0.0
        reasons = []
        actions = []

        for phrase, reason, action in patterns:
            if phrase.lower() in sl:
                score += 1.0
                reasons.append(reason)
                actions.append(action)

        if any(w in sl for w in ["should", "must", "need to", "diagnos", "too sensitive", "clearly"]):
            score += 0.5

        if score > 0:
            scored.append({
                "text": s,
                "start": start,
                "end": end,
                "dimension": active_dim,
                "risk_score": min(score, 3.0),
                "reason": "; ".join(sorted(set(reasons))) or "potential quality/safety risk",
                "suggested_action": "; ".join(sorted(set(actions))) or "revise into safer counseling wording",
                "source": "sentence_rule",
            })

    scored = sorted(scored, key=lambda x: x["risk_score"], reverse=True)
    return scored[:max_n]


def random_spans(text: str, active_dim: str, rng: random.Random, max_n: int) -> List[Dict]:
    sents = split_sentences(text)
    if not sents:
        return []
    idxs = list(range(len(sents)))
    rng.shuffle(idxs)
    selected = []
    cursor_map = []
    cursor = 0
    for s in sents:
        start = text.find(s, cursor)
        end = start + len(s)
        cursor = end
        cursor_map.append((start, end))

    for idx in idxs[:max_n]:
        start, end = cursor_map[idx]
        selected.append({
            "text": sents[idx],
            "start": start,
            "end": end,
            "dimension": active_dim,
            "risk_score": 0.25,
            "reason": "random corruption control span",
            "suggested_action": "rewrite if needed",
            "source": "random_sentence",
        })
    return selected


def safe_hints(safe: str, max_n: int = 2) -> List[str]:
    sents = split_sentences(safe)
    anchors = []
    prefs = [
        "it sounds",
        "i'm sorry",
        "understandable",
        "it may help",
        "consider",
        "trusted",
        "professional",
        "support",
        "if you suspect",
    ]
    for s in sents:
        sl = s.lower()
        if any(p in sl for p in prefs):
            anchors.append(s)
    if not anchors:
        anchors = sents[:max_n]
    return anchors[:max_n]


@torch.no_grad()
def predict_router(rows: List[Dict], router_dir: str, batch_size: int = 16, max_len: int = 512):
    if not router_dir:
        return [None] * len(rows)

    tok = AutoTokenizer.from_pretrained(router_dir)
    model = AutoModelForSequenceClassification.from_pretrained(router_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    out = []
    for i in tqdm(range(0, len(rows), batch_size), desc="router predict"):
        batch = rows[i:i+batch_size]
        texts = []
        for ex in batch:
            q = get_field(ex, "question", "query", "user_question")
            u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
            texts.append(
                "Question:\n" + q.strip()
                + "\n\nUnsafe response:\n" + u.strip()
                + "\n\nTask: identify the primary violated response-quality dimension."
            )
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        out.extend(probs)
    return out


def choose_condition(oracle_dim: str, router_probs, rng: random.Random, teacher_force_prob: float):
    if router_probs is None or rng.random() < teacher_force_prob:
        probs = [0.0] * len(DIMS)
        probs[DIM2ID[oracle_dim]] = 1.0
        source = "oracle"
    else:
        probs = router_probs
        source = "router_pred"

    top = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "condition_source": source,
        "condition_dim": DIMS[top],
        "condition_probs": {DIMS[i]: float(probs[i]) for i in range(len(DIMS))},
    }


def num_spans_from_t(t: float, available: int) -> int:
    if available <= 0:
        return 0
    return max(1, min(available, int(math.ceil(t * 4))))


def build_example(ex: Dict, router_probs, rng: random.Random, corruption_type: str, t: float, teacher_force_prob: float):
    q = get_field(ex, "question", "query", "user_question")
    u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
    y = get_field(ex, "safe_response", "target_response", "response")
    oracle_dim = norm_dim(get_field(ex, "target_dimension", "dimension", "violated_dimension"))

    cond = choose_condition(oracle_dim, router_probs, rng, teacher_force_prob)
    active_dim = cond["condition_dim"]

    risk_spans = []
    rewrite_hints = []

    max_n = int(math.ceil(t * 4))

    if corruption_type == "empty":
        risk_spans = []
    elif corruption_type == "random":
        risk_spans = random_spans(u, active_dim, rng, max_n=max_n)
    elif corruption_type == "risk":
        spans = find_phrase_spans(u, active_dim)
        if not spans:
            spans = sentence_fallback_spans(u, active_dim, max_n=max_n)
        risk_spans = spans[:num_spans_from_t(t, len(spans))]
    elif corruption_type == "bridge":
        spans = find_phrase_spans(u, active_dim)
        if not spans:
            spans = sentence_fallback_spans(u, active_dim, max_n=max_n)
        risk_spans = spans[:num_spans_from_t(t, len(spans))]
        rewrite_hints = safe_hints(y, max_n=2)
    else:
        raise ValueError(f"Unknown corruption_type: {corruption_type}")

    loss_weight = 1.0 + 0.15 * float(t)
    if corruption_type == "risk":
        loss_weight += 0.35
    elif corruption_type == "bridge":
        loss_weight += 0.20
    elif corruption_type == "random":
        loss_weight += 0.05

    return {
        "question": q,
        "unsafe_response": u,
        "safe_response": y,
        "target_dimension": oracle_dim,
        "condition_dim": cond["condition_dim"],
        "condition_source": cond["condition_source"],
        "condition_probs": cond["condition_probs"],
        "corruption_type": corruption_type,
        "corruption_level": float(t),
        "risk_spans": risk_spans,
        "rewrite_hints": rewrite_hints,
        "loss_weight": float(loss_weight),
    }


def make_dataset(rows, router_probs, examples_per_row: int, teacher_force_prob: float, seed: int):
    rng = random.Random(seed)
    corruption_cycle = ["empty", "random", "risk", "bridge"]
    t_cycle = [0.25, 0.50, 0.75, 1.00]
    out = []

    for i, ex in enumerate(rows):
        for k in range(examples_per_row):
            ctype = corruption_cycle[k % len(corruption_cycle)]
            t = t_cycle[(i + k) % len(t_cycle)]
            out.append(build_example(
                ex=ex,
                router_probs=router_probs[i],
                rng=rng,
                corruption_type=ctype,
                t=t,
                teacher_force_prob=teacher_force_prob,
            ))

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
    out = make_dataset(rows, router_probs, args.examples_per_row, args.teacher_force_prob, args.seed)
    write_jsonl(out, args.output)

    manifest = {
        "input": args.input,
        "output": args.output,
        "router_dir": args.router_dir,
        "n_input": len(rows),
        "n_output": len(out),
        "examples_per_row": args.examples_per_row,
        "teacher_force_prob": args.teacher_force_prob,
        "dims": DIMS,
        "corruption_types": ["empty", "random", "risk", "bridge"],
        "notes": "Risk spans are stored as structured fields; no <risk> tags or draft markers are inserted into the unsafe response.",
    }

    mpath = str(Path(args.output).with_suffix(".manifest.json"))
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
