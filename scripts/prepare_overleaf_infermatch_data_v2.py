import argparse
import json
import random
from collections import Counter
from pathlib import Path

from prepare_overleaf_infermatch_data import (
    DIMS,
    build_unsafe_zt,
    get_field,
    monotonic_align,
    predict_router,
    read_jsonl,
    rl,
    score_risks,
    split_sentences,
    write_jsonl,
)


DATA_VERSION = "infermatch_full_mixture_v2"
DIM2ID = {d: i for i, d in enumerate(DIMS)}
DIM_ALIASES = {
    "medical": "medical_advice",
    "medical_boundary": "medical_advice",
    "factual": "factual_consistency",
    "fact": "factual_consistency",
    "overall": "overall_quality",
}


def parse_timesteps(value):
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def preview(text, n=140):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 3] + "..."


def normalize_dim(x):
    x = str(x or "").strip().lower().replace("-", "_").replace(" ", "_")
    return DIM_ALIASES.get(x, x)


def gold_dimension_values(ex):
    if "target_dimensions" in ex and isinstance(ex["target_dimensions"], list):
        return ex["target_dimensions"]
    for field in ["target_dimension", "dimension", "violated_dimension", "used_dimension", "condition_dim"]:
        if field in ex and ex[field] is not None and str(ex[field]).strip():
            return [ex[field]]
    return []


def display_target_dimension(ex):
    if "target_dimensions" in ex and isinstance(ex["target_dimensions"], list):
        return ex["target_dimensions"]
    values = gold_dimension_values(ex)
    return values[0] if values else ""


def make_gold_aspect_vector(ex, warning_counter=None):
    vec = [0.0] * len(DIMS)
    raw_dims = gold_dimension_values(ex)

    if not raw_dims:
        vec[DIM2ID["overall_quality"]] = 1.0
        if warning_counter is not None:
            warning_counter["missing"] += 1
        return vec

    for raw_dim in raw_dims:
        dim = normalize_dim(raw_dim)
        if dim in DIM2ID:
            vec[DIM2ID[dim]] = 1.0
        else:
            vec[DIM2ID["overall_quality"]] = 1.0
            if warning_counter is not None:
                warning_counter["unknown"] += 1

    if not any(vec):
        vec[DIM2ID["overall_quality"]] = 1.0
        if warning_counter is not None:
            warning_counter["missing"] += 1
    return vec


def base_weight_record(pair, risk, state, p_mask=0.0):
    return {
        "safe_span": pair.get("b", ""),
        "unsafe_span": pair.get("a", ""),
        "risk": float(risk),
        "state": state,
        "op": pair.get("op", ""),
        "p_mask": float(p_mask),
        "align_score": float(pair.get("align_score", 0.0)),
    }


def fallback_safe_spans(safe_response):
    spans = split_sentences(safe_response)
    if spans:
        return [{"a": "", "b": span, "op": "insert", "align_score": 0.0, "risk_vec": [0.0] * len(DIMS)} for span in spans]
    safe_response = str(safe_response or "").strip()
    if safe_response:
        return [{"a": "", "b": safe_response, "op": "insert", "align_score": 0.0, "risk_vec": [0.0] * len(DIMS)}]
    return []


def fallback_unsafe_spans(unsafe_response):
    spans = split_sentences(unsafe_response)
    if spans:
        return [{"a": span, "b": "", "op": "delete", "align_score": 0.0, "risk_vec": [0.0] * len(DIMS)} for span in spans]
    unsafe_response = str(unsafe_response or "").strip()
    if unsafe_response:
        return [{"a": unsafe_response, "b": "", "op": "delete", "align_score": 0.0, "risk_vec": [0.0] * len(DIMS)}]
    return []


def build_safe_zt(row, t, T, mask_token, rho, lambda_mask, random_mask_prob, rng):
    beta = t / float(T)
    pairs = [p for p in row["alignments"] if p.get("b")]
    if not pairs:
        pairs = fallback_safe_spans(row["safe_response"])

    parts = []
    states = []
    weighted = []

    for pair in pairs:
        b = pair.get("b", "")
        if not b:
            continue

        risk = rl(row["g"], pair.get("risk_vec", [0.0] * len(DIMS)))
        p_mask = clamp01(rho + lambda_mask * beta * risk + random_mask_prob * beta)
        if rng.random() < p_mask:
            parts.append(mask_token)
            state = "MASK"
        else:
            parts.append(b)
            state = "SAFE"

        states.append(state)
        weighted.append(base_weight_record(pair, risk, state, p_mask))

    z_t = " ".join(x for x in parts if x).strip()
    if not z_t and row["safe_response"].strip():
        z_t = mask_token
        states = ["MASK"]
        weighted = [{
            "safe_span": row["safe_response"],
            "unsafe_span": "",
            "risk": 0.0,
            "state": "MASK",
            "op": "fallback_safe_mask",
            "p_mask": 1.0,
            "align_score": 0.0,
        }]
    return z_t, states, weighted


def bridge_state(pair, risk, beta, rho, lambda_mask, risk_threshold, weak_align_threshold, rng):
    a = pair.get("a", "")
    b = pair.get("b", "")
    align_score = float(pair.get("align_score", 0.0))
    p_mask = clamp01(rho + lambda_mask * beta * risk)

    if a and b and align_score < weak_align_threshold and risk >= risk_threshold:
        return "MASK", p_mask
    if not a and b:
        return ("MASK", p_mask) if rng.random() < beta * p_mask else ("SAFE", p_mask)
    if a and not b:
        return ("MASK", p_mask) if rng.random() < beta * p_mask else ("UNSAFE", p_mask)

    p_safe = 1.0 - beta
    p_mask_total = beta * p_mask
    draw = rng.random()
    if draw < p_safe:
        return "SAFE", p_mask
    if draw < p_safe + p_mask_total:
        return "MASK", p_mask
    return "UNSAFE", p_mask


def build_bridge_zt(row, t, T, mask_token, rho, lambda_mask, risk_threshold, weak_align_threshold, rng):
    beta = t / float(T)
    pairs = row["alignments"] or []
    if not pairs:
        pairs = fallback_unsafe_spans(row["unsafe_response"]) + fallback_safe_spans(row["safe_response"])

    parts = []
    states = []
    weighted = []

    for pair in pairs:
        a = pair.get("a", "")
        b = pair.get("b", "")
        if not a and not b:
            continue

        risk = rl(row["g"], pair.get("risk_vec", [0.0] * len(DIMS)))
        state, p_mask = bridge_state(pair, risk, beta, rho, lambda_mask, risk_threshold, weak_align_threshold, rng)
        if state == "SAFE":
            realized = b or mask_token
        elif state == "UNSAFE":
            realized = a or mask_token
        else:
            realized = mask_token

        if realized:
            parts.append(realized)
        states.append(state)
        if b:
            weighted.append(base_weight_record(pair, risk, state, p_mask))

    z_t = " ".join(parts).strip()
    if not z_t:
        z_t, states, weighted = build_unsafe_zt(
            row["alignments"],
            row["g"],
            t,
            T,
            mask_token,
            risk_threshold,
            0.35,
            rho,
            lambda_mask,
        )
    if not z_t and row["unsafe_response"].strip():
        z_t = row["unsafe_response"].strip()
        states = ["UNSAFE_FALLBACK"]
    return z_t, states, weighted


def make_row(row, source, t, T, z_t, states, weighted):
    return {
        "question": row["question"],
        "unsafe_response": row["unsafe_response"],
        "safe_response": row["safe_response"],
        "target_dimension": row.get("target_dimension", ""),
        "g": row["g"],
        "g_pred": row["g_pred"],
        "g_gold": row["g_gold"],
        "g_source": row["g_source"],
        "source": source,
        "t": t,
        "T": T,
        "z_t": z_t,
        "states": states,
        "alignments": row["alignments"],
        "target_weight_spans": weighted,
        "data_version": DATA_VERSION,
    }


def print_samples(rows, limit=3):
    print("\n[samples first_3]")
    for row in rows[:limit]:
        print(json.dumps({
            "target_dimension": row.get("target_dimension"),
            "g_gold": row.get("g_gold"),
            "g_pred": row.get("g_pred"),
            "g": row.get("g"),
            "g_source": row.get("g_source"),
            "source": row.get("source"),
            "t": row.get("t"),
            "z_t": preview(row.get("z_t")),
        }, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--router_dir", required=True)
    ap.add_argument("--risk_scorer_dir", required=True)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--mask_token", default="<MASK>")
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--lambda_mask", type=float, default=0.75)
    ap.add_argument("--risk_threshold", type=float, default=0.35)
    ap.add_argument("--mask_threshold", type=float, default=0.35)
    ap.add_argument("--weak_align_threshold", type=float, default=0.35)
    ap.add_argument("--safe_random_mask_prob", type=float, default=0.10)
    ap.add_argument("--unsafe_timesteps", default="2,2,3,4")
    ap.add_argument("--safe_timesteps", default="2")
    ap.add_argument("--bridge_timesteps", default="2")
    ap.add_argument("--use_gold_aspect_mixing", action="store_true")
    ap.add_argument("--aspect_tf_prob", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.aspect_tf_prob = clamp01(args.aspect_tf_prob)

    rng = random.Random(args.seed)

    base = read_jsonl(args.input)
    hats = predict_router(base, args.router_dir)

    staged = []
    gold_warnings = Counter()
    for ex, g_pred in zip(base, hats):
        q = get_field(ex, "question", "query", "user_question")
        u = get_field(ex, "unsafe_response", "corrupted_response", "bad_response")
        y = get_field(ex, "safe_response", "target_response", "response")
        g_pred = [float(x) for x in g_pred]
        g_gold = make_gold_aspect_vector(ex, gold_warnings)
        if args.use_gold_aspect_mixing and rng.random() < args.aspect_tf_prob:
            g = g_gold
            g_source = "gold"
        else:
            g = g_pred
            g_source = "pred"
        staged.append({
            "question": q,
            "unsafe_response": u,
            "safe_response": y,
            "target_dimension": display_target_dimension(ex),
            "g": g,
            "g_pred": g_pred,
            "g_gold": g_gold,
            "g_source": g_source,
            "alignments": monotonic_align(u, y),
        })

    risk_scores = score_risks(staged, args.risk_scorer_dir)
    for row, rs in zip(staged, risk_scores):
        for pair, risk_vec in zip(row["alignments"], rs):
            pair["risk_vec"] = risk_vec

    unsafe_timesteps = parse_timesteps(args.unsafe_timesteps)
    safe_timesteps = parse_timesteps(args.safe_timesteps)
    bridge_timesteps = parse_timesteps(args.bridge_timesteps)

    out = []
    for row in staged:
        out.append(make_row(row, "empty", 0, args.T, "", [], []))

        for t in unsafe_timesteps:
            z_t, states, weighted = build_unsafe_zt(
                row["alignments"],
                row["g"],
                t,
                args.T,
                args.mask_token,
                args.risk_threshold,
                args.mask_threshold,
                args.rho,
                args.lambda_mask,
            )
            out.append(make_row(row, "unsafe", t, args.T, z_t, states, weighted))

        for t in safe_timesteps:
            z_t, states, weighted = build_safe_zt(
                row,
                t,
                args.T,
                args.mask_token,
                args.rho,
                args.lambda_mask,
                args.safe_random_mask_prob,
                rng,
            )
            out.append(make_row(row, "safe", t, args.T, z_t, states, weighted))

        for t in bridge_timesteps:
            z_t, states, weighted = build_bridge_zt(
                row,
                t,
                args.T,
                args.mask_token,
                args.rho,
                args.lambda_mask,
                args.risk_threshold,
                args.weak_align_threshold,
                rng,
            )
            out.append(make_row(row, "bridge", t, args.T, z_t, states, weighted))

    write_jsonl(out, args.output)
    print("wrote", len(out), args.output)
    print("source:", Counter(r["source"] for r in out))
    print("t:", Counter(r["t"] for r in out))
    print("g_source:", Counter(r["g_source"] for r in out))
    print("gold_dimension_warnings:", gold_warnings)
    print_samples(out)

    g_source_distribution = Counter(r["g_source"] for r in out)
    manifest = vars(args) | {
        "n_input": len(base),
        "n_output": len(out),
        "dims": DIMS,
        "data_version": DATA_VERSION,
        "use_gold_aspect_mixing": args.use_gold_aspect_mixing,
        "aspect_tf_prob": args.aspect_tf_prob,
        "source_distribution": dict(Counter(r["source"] for r in out)),
        "timestep_distribution": dict(Counter(str(r["t"]) for r in out)),
        "g_source_distribution": dict(g_source_distribution),
        "unknown_gold_dimension_count": int(gold_warnings["unknown"]),
        "missing_gold_dimension_count": int(gold_warnings["missing"]),
        "gold_dimension_warning_distribution": dict(gold_warnings),
    }
    with open(str(Path(args.output).with_suffix(".manifest.json")), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
