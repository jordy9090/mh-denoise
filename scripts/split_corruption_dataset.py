import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


DIMENSIONS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_question_key(ex):
    if ex.get("question_id"):
        return ex["question_id"]

    ex_id = ex["id"]
    dim = ex["target_dimension"]

    suffix = "_" + dim
    if ex_id.endswith(suffix):
        return ex_id[: -len(suffix)]

    return ex_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="data/splits")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    by_qid = defaultdict(list)
    for ex in rows:
        by_qid[get_question_key(ex)].append(ex)

    qids = list(by_qid.keys())
    random.seed(args.seed)
    random.shuffle(qids)

    n_test = max(1, int(len(qids) * args.test_ratio))
    test_qids = set(qids[:n_test])
    train_qids = set(qids[n_test:])

    train_rows = [ex for qid in train_qids for ex in by_qid[qid]]
    test_rows = [ex for qid in test_qids for ex in by_qid[qid]]

    write_jsonl(train_rows, f"{args.out_dir}/train.jsonl")
    write_jsonl(test_rows, f"{args.out_dir}/test.jsonl")

    print("questions:", len(qids))
    print("train questions:", len(train_qids), "rows:", len(train_rows))
    print("test questions:", len(test_qids), "rows:", len(test_rows))


if __name__ == "__main__":
    main()
