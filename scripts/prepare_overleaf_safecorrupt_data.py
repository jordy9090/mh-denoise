import argparse
import json
import random
import re
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DIMS = ["overall_quality", "empathy", "specificity", "medical_advice", "factual_consistency", "toxicity"]


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


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def sim(a, b):
    al, bl = a.lower(), b.lower()
    wa = set(re.findall(r"[a-zA-Z']+", al))
    wb = set(re.findall(r"[a-zA-Z']+", bl))
    j = len(wa & wb) / max(1, len(wa | wb))
    seq = SequenceMatcher(None, al, bl).ratio()
    return 0.55 * seq + 0.45 * j


def monotonic_align(u, y):
    U = split_sentences(u) or [u]
    Y = split_sentences(y) or [y]
    n, m = len(U), len(Y)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]
    gap = -0.15
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
        back[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
        back[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim(U[i - 1], Y[j - 1])
            opts = [(dp[i - 1][j - 1] + s, "sub"), (dp[i - 1][j] + gap, "del"), (dp[i][j - 1] + gap, "ins")]
            best = max(opts, key=lambda x: x[0])
            dp[i][j] = best[0]
            back[i][j] = best[1]
    i, j = n, m
    pairs = []
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "sub":
            pairs.append({"a": U[i - 1], "b": Y[j - 1], "op": "substitute", "align_score": sim(U[i - 1], Y[j - 1])})
            i -= 1
            j -= 1
        elif op == "del":
            pairs.append({"a": U[i - 1], "b": "", "op": "delete", "align_score": 0.0})
            i -= 1
        elif op == "ins":
            pairs.append({"a": "", "b": Y[j - 1], "op": "insert", "align_score": 0.0})
            j -= 1
        else:
            break
    return list(reversed(pairs))


@torch.no_grad()
def predict_router(rows, router_dir):
    tok = AutoTokenizer.from_pretrained(router_dir)
    model = AutoModelForSequenceClassification.from_pretrained(router_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    outs = []
    for i in tqdm(range(0, len(rows), 16), desc="router"):
        batch = rows[i : i + 16]
        texts = []
        for ex in batch:
            q = get_field(ex, "question", "query", "user_question")
            u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
            texts.append("Question:\n" + q.strip() + "\n\nCandidate response:\n" + u.strip())
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        probs = torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()
        outs.extend(probs)
    return outs


@torch.no_grad()
def score_unsafe_spans(rows, risk_dir):
    tok = AutoTokenizer.from_pretrained(risk_dir)
    model = AutoModelForSequenceClassification.from_pretrained(risk_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    flat = []
    for ri, row in enumerate(rows):
        q = row["question"]
        for li, p in enumerate(row["alignments"]):
            span = p["a"]
            if not span:
                continue
            text = "Question:\n" + q.strip() + "\n\nCandidate span:\n" + span.strip()
            flat.append((ri, li, text))
    scores = [[[0.0] * len(DIMS) for _ in r["alignments"]] for r in rows]
    for i in tqdm(range(0, len(flat), 32), desc="risk"):
        batch = flat[i : i + 32]
        enc = tok([x[2] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=384).to(device)
        probs = torch.sigmoid(model(**enc).logits.float()).detach().cpu().tolist()
        for (ri, li, _), p in zip(batch, probs):
            scores[ri][li] = [float(x) for x in p]
    return scores


def active_risk(g, risk_vec):
    return max(float(g[k]) * float(risk_vec[k]) for k in range(len(DIMS)))


def build_safe_corrupted_zt(pairs, g, t, T, mask_token, rho, lambda_mask, random_mask_prob):
    """Corrupt the clean target side y to produce z_t.

    This is the training-time denoising view: y -> z_t, then model learns
    p_theta(y | q, u, g, z_t, t). Risk is estimated on the aligned unsafe span
    and transferred to the corresponding safe span b.
    """
    beta = t / float(T)
    parts = []
    states = []
    weighted = []
    for p in pairs:
        b = p.get("b", "")
        if not b:
            continue
        r = active_risk(g, p.get("risk_vec", [0.0] * len(DIMS)))
        p_mask = max(0.0, min(0.95, rho + lambda_mask * beta * r + random_mask_prob * beta))
        if random.random() < p_mask:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(b)
            state = "SAFE"
        states.append(state)
        weighted.append({
            "safe_span": b,
            "unsafe_span": p.get("a", ""),
            "risk": float(r),
            "state": state,
            "op": p.get("op", ""),
            "p_mask": float(p_mask),
        })
    return " ".join(parts).strip(), states, weighted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--lambda_mask", type=float, default=0.80)
    ap.add_argument("--random_mask_prob", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    base = read_jsonl(args.input)
    hats = predict_router(base, args.router_dir)

    staged = []
    for ex, g in zip(base, hats):
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        y = get_field(ex, "safe_response", "target_response", "response")
        staged.append({"question": q, "unsafe_response": u, "safe_response": y, "g": g, "alignments": monotonic_align(u, y)})

    risk_scores = score_unsafe_spans(staged, args.risk_scorer_dir)
    for row, rs in zip(staged, risk_scores):
        for p, risk_vec in zip(row["alignments"], rs):
            p["risk_vec"] = risk_vec

    out = []
    schedule = [("empty", 0), ("safe_corrupt", 1), ("safe_corrupt", 2), ("safe_corrupt", 3), ("safe_corrupt", 4)]
    for row in staged:
        for source, t in schedule:
            if source == "empty":
                z_t, states, weighted = "", [], []
            else:
                z_t, states, weighted = build_safe_corrupted_zt(
                    row["alignments"], row["g"], t, args.T, args.mask_token, args.rho, args.lambda_mask, args.random_mask_prob
                )
            out.append({
                "question": row["question"],
                "unsafe_response": row["unsafe_response"],
                "safe_response": row["safe_response"],
                "g": row["g"],
                "source": source,
                "t": t,
                "T": args.T,
                "z_t": z_t,
                "states": states,
                "alignments": row["alignments"],
                "target_weight_spans": weighted,
                "data_version": "overleaf_safecorrupt_v1",
            })

    write_jsonl(out, args.output)
    print("wrote", len(out), args.output)
    print("source:", Counter(r["source"] for r in out))
    print("t:", Counter(r["t"] for r in out))
    with open(str(Path(args.output).with_suffix(".manifest.json")), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"n_input": len(base), "n_output": len(out), "dims": DIMS}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
