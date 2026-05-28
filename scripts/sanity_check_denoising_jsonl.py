import argparse
import json
from collections import Counter


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def response_len(row):
    text = row.get("safe_response") or row.get("unsafe_response") or row.get("z_t") or ""
    return len(str(text).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--short_words", type=int, default=10)
    args = parser.parse_args()

    paths = args.paths + args.input
    if not paths:
        parser.error("provide at least one JSONL path or --input PATH")

    for path in paths:
        rows = read_jsonl(path)
        print(f"\n== {path} ==")
        print("rows:", len(rows))
        print("source:", dict(Counter(row.get("source", "<missing>") for row in rows)))
        print("t:", dict(Counter(row.get("t", "<missing>") for row in rows)))
        print("empty output count:", sum(1 for row in rows if not str(row.get("safe_response", "")).strip()))
        print("short response count:", sum(1 for row in rows if response_len(row) < args.short_words))
        print("first row keys:", sorted(rows[0].keys()) if rows else [])


if __name__ == "__main__":
    main()
