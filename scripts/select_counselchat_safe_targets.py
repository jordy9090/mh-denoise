import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


FILTER_LEVELS = {
    "strict": {
        "overall_score": 4,
        "empathy_score": 4,
        "specificity_score": 4,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
    "relaxed_1": {
        "overall_score": 4,
        "empathy_score": 4,
        "specificity_score": 3,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
    "relaxed_2": {
        "overall_score": 4,
        "empathy_score": 3,
        "specificity_score": 3,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
}


def clean_text(value):
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def normalize_question(text):
    return clean_text(text).lower()


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_float(row, key):
    try:
        return float(row.get(key))
    except Exception:
        return None


def no_medical_advice(value):
    return str(value).strip().lower() in {"no", "0", "false", "none", "n"}


def passes_filter(row, filter_level):
    cfg = FILTER_LEVELS[filter_level]
    for key in ["overall_score", "empathy_score", "specificity_score", "factual_consistency_score"]:
        value = as_float(row, key)
        if value is None or value < cfg[key]:
            return False
    toxicity = as_float(row, "toxicity_score")
    if toxicity is None or toxicity > cfg["toxicity_score"]:
        return False
    if not no_medical_advice(row.get("medical_advice_score")):
        return False
    return True


def dedupe_by_question(rows):
    best = {}
    for row in rows:
        key = normalize_question(row.get("question", ""))
        if not key:
            continue
        score_sum = (
            float(row.get("overall_score", 0))
            + float(row.get("empathy_score", 0))
            + float(row.get("specificity_score", 0))
            + float(row.get("factual_consistency_score", 0))
            + (5 - float(row.get("toxicity_score", 5)))
            + 0.01 * float(row.get("upvotes", 0) or 0)
        )
        if key not in best or score_sum > best[key][0]:
            best[key] = (score_sum, row)
    return [item[1] for item in best.values()]


def build_output(row, out_i):
    return {
        "id": f"counselchat_{out_i:04d}",
        "question_id": str(row.get("question_id", row.get("questionID", ""))),
        "question": row.get("question", ""),
        "safe_response": row.get("safe_response", ""),
        "topic": row.get("topic", ""),
        "source": "CounselChat",
        "safe_target_source": "nbertagnolli/counsel-chat",
        "overall_score": row.get("overall_score"),
        "empathy_score": row.get("empathy_score"),
        "specificity_score": row.get("specificity_score"),
        "medical_advice_score": row.get("medical_advice_score"),
        "factual_consistency_score": row.get("factual_consistency_score"),
        "toxicity_score": row.get("toxicity_score"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/raw/counselchat_safe_201.jsonl")
    parser.add_argument("--n_questions", type=int, default=201)
    parser.add_argument("--filter_level", choices=sorted(FILTER_LEVELS), default="strict")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    passing = [row for row in rows if passes_filter(row, args.filter_level)]
    unique = dedupe_by_question(passing)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(unique)
    else:
        unique = sorted(
            unique,
            key=lambda r: (
                float(r.get("overall_score", 0)),
                float(r.get("empathy_score", 0)),
                float(r.get("specificity_score", 0)),
                float(r.get("factual_consistency_score", 0)),
                5 - float(r.get("toxicity_score", 5)),
                float(r.get("upvotes", 0) or 0),
            ),
            reverse=True,
        )

    selected = unique[: args.n_questions]
    out_rows = [build_output(row, i) for i, row in enumerate(selected)]
    write_jsonl(out_rows, args.output)

    print("input count:", len(rows))
    print("filter level:", args.filter_level)
    print("passing rows:", len(passing))
    print("unique passing questions:", len(unique))
    print("selected rows:", len(out_rows))
    print("topic distribution:", dict(Counter(row.get("topic", "") for row in out_rows)))
    if len(out_rows) < args.n_questions:
        print(f"WARNING: requested {args.n_questions} but selected only {len(out_rows)}. Try relaxed_1, then relaxed_2.")
    print(f"saved to {args.output}")
    print("toxicity convention: CounselBench lower-is-better; strict uses toxicity_score <= 1")


if __name__ == "__main__":
    main()
