import argparse
import json
import re
from collections import Counter
from pathlib import Path


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


def base_output(row, out_i, mix_source):
    out = dict(row)
    out["id"] = f"exp300_safe_{out_i:04d}"
    out["question_id"] = str(row.get("question_id", row.get("questionID", "")))
    out["question"] = row.get("question", "")
    out["safe_response"] = row.get("safe_response", "")
    out["topic"] = row.get("topic", "")
    out["source"] = row.get("source", "")
    out["safe_target_source"] = row.get("safe_target_source", "")
    out["safe_target_mix_source"] = mix_source
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--counselbench_eval", required=True)
    parser.add_argument("--counselchat", required=True)
    parser.add_argument("--output", default="data/raw/exp300_safe_targets.jsonl")
    parser.add_argument("--expected_total", type=int, default=300)
    args = parser.parse_args()

    sources = [
        ("counselbench_eval_strict", read_jsonl(args.counselbench_eval)),
        ("counselchat_judged", read_jsonl(args.counselchat)),
    ]

    merged = []
    seen = set()
    duplicate_count = 0
    for mix_source, rows in sources:
        for row in rows:
            key = normalize_question(row.get("question", ""))
            if not key:
                raise SystemExit(f"Missing question in {mix_source}: {row.get('id')}")
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            merged.append(base_output(row, len(merged), mix_source))

    if len(merged) != args.expected_total:
        print("source distribution:", dict(Counter(row["safe_target_mix_source"] for row in merged)))
        print("duplicates removed:", duplicate_count)
        raise SystemExit(f"Expected {args.expected_total} merged safe targets, got {len(merged)}.")

    write_jsonl(merged, args.output)
    print("source distribution:", dict(Counter(row["safe_target_mix_source"] for row in merged)))
    print("duplicates removed:", duplicate_count)
    print("total:", len(merged))
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
