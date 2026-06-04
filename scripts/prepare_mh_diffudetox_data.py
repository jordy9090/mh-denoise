import argparse
import json
import random
from pathlib import Path


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


def clean_text(x):
    return " ".join((x or "").replace("\n", " ").split()).strip()


def make_src(ex):
    q = clean_text(ex["question"])
    u = clean_text(ex["unsafe_response"])
    k = clean_text(ex.get("target_dimension", "unknown"))

    return (
        f"violation dimension: {k}. "
        f"user question: {q} "
        f"unsafe response: {u}"
    )


def convert_pair(ex):
    return {
        "src": make_src(ex),
        "trg": clean_text(ex["safe_response"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/splits/train.jsonl")
    parser.add_argument("--test_file", default="data/splits/test.jsonl")
    parser.add_argument("--out_dir", default="external/diffu-detox-main/datasets/mh_refine")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_file)
    test_rows = load_jsonl(args.test_file)

    random.seed(args.seed)
    random.shuffle(train_rows)

    n_valid = max(1, int(len(train_rows) * args.valid_ratio))
    valid_rows = train_rows[:n_valid]
    train_rows2 = train_rows[n_valid:]

    train_pairs = [convert_pair(x) for x in train_rows2]
    valid_pairs = [convert_pair(x) for x in valid_rows]
    test_pairs = [convert_pair(x) for x in test_rows]

    # Unconditional fluent/supportive response corpus.
    # Use only train split safe responses to prevent leakage.
    seen = set()
    flu_rows = []
    for ex in train_rows2:
        y = clean_text(ex["safe_response"])
        if y and y not in seen:
            flu_rows.append({"src": "", "trg": y})
            seen.add(y)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_pairs, out / "mhrefine_train.jsonl")
    write_jsonl(valid_pairs, out / "mhrefine_valid.jsonl")
    write_jsonl(test_pairs, out / "mhrefine_test.jsonl")

    write_jsonl(flu_rows, out / "mhflu_train.jsonl")

    manifest = {
        "conditional_train": len(train_pairs),
        "conditional_valid": len(valid_pairs),
        "conditional_test": len(test_pairs),
        "unconditional_train_unique_safe": len(flu_rows),
        "conditional_format": {
            "src": "question + unsafe_response + target_dimension",
            "trg": "safe_response"
        },
        "unconditional_format": {
            "src": "empty",
            "trg": "safe_response"
        }
    }

    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Saved to", out)


if __name__ == "__main__":
    main()
