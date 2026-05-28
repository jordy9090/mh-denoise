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
    parser.add_argument("--valid_ratio", type=float, default=0.0)
    parser.add_argument("--train_name", default="train.jsonl")
    parser.add_argument("--valid_name", default="valid.jsonl")
    parser.add_argument("--test_name", default="test.jsonl")
    parser.add_argument("--group_by_question", action="store_true", default=True)
    parser.add_argument("--row_level_split", action="store_false", dest="group_by_question")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.input)

    random.seed(args.seed)

    if args.group_by_question:
        by_qid = defaultdict(list)
        for ex in rows:
            by_qid[get_question_key(ex)].append(ex)

        qids = list(by_qid.keys())
        random.shuffle(qids)

        n_test = max(1, int(len(qids) * args.test_ratio))
        remaining = len(qids) - n_test
        n_valid = int(len(qids) * args.valid_ratio) if args.valid_ratio > 0 else 0
        n_valid = min(n_valid, max(0, remaining - 1)) if remaining else 0

        test_qids = set(qids[:n_test])
        valid_qids = set(qids[n_test:n_test + n_valid])
        train_qids = set(qids[n_test + n_valid:])

        train_rows = [ex for qid in train_qids for ex in by_qid[qid]]
        valid_rows = [ex for qid in valid_qids for ex in by_qid[qid]]
        test_rows = [ex for qid in test_qids for ex in by_qid[qid]]

        print("questions:", len(qids))
        print("train questions:", len(train_qids), "rows:", len(train_rows))
        print("valid questions:", len(valid_qids), "rows:", len(valid_rows))
        print("test questions:", len(test_qids), "rows:", len(test_rows))
    else:
        random.shuffle(rows)
        n_test = max(1, int(len(rows) * args.test_ratio))
        remaining = len(rows) - n_test
        n_valid = int(len(rows) * args.valid_ratio) if args.valid_ratio > 0 else 0
        n_valid = min(n_valid, max(0, remaining - 1)) if remaining else 0
        test_rows = rows[:n_test]
        valid_rows = rows[n_test:n_test + n_valid]
        train_rows = rows[n_test + n_valid:]

        print("rows:", len(rows))
        print("train rows:", len(train_rows))
        print("valid rows:", len(valid_rows))
        print("test rows:", len(test_rows))

    write_jsonl(train_rows, f"{args.out_dir}/{args.train_name}")
    if args.valid_ratio > 0:
        write_jsonl(valid_rows, f"{args.out_dir}/{args.valid_name}")
    write_jsonl(test_rows, f"{args.out_dir}/{args.test_name}")


if __name__ == "__main__":
    main()
