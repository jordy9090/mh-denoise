import argparse
import json
import re
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value):
    text = str(value or "")
    text = text.replace("```json", "").replace("```", "")
    text = text.replace("\\n", " ").replace('\\"', '"').replace("\\", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


def row_key(row):
    parts = [
        str(row.get("example_id") or row.get("id") or row.get("question_id") or ""),
        str(row.get("target_dimension") or row.get("dimension") or ""),
        str(row.get("unsafe_response") or "")[:300],
    ]
    return "||".join(parts)


def response_from_row(row):
    for key in [
        "cleaned_response",
        "prompt_cleaning_raw_output",
        "cleaned_response_raw",
        "raw_output",
    ]:
        value = row.get(key)
        if value is not None and str(value).strip():
            return clean_text(value)
    return ""


def merge_regen(base_rows, regen_rows):
    regen_by_key = {row_key(row): row for row in regen_rows}
    out = []
    replaced = 0
    still_empty = 0

    for row in base_rows:
        current = clean_text(row.get("cleaned_response", ""))
        if current:
            new = dict(row)
            new["cleaned_response"] = current
            out.append(new)
            continue

        regen = regen_by_key.get(row_key(row))
        replacement = response_from_row(regen or {})
        new = dict(row)
        if replacement:
            new["cleaned_response_before_rescue"] = row.get("cleaned_response", "")
            new["cleaned_response"] = replacement
            new["prompt_cleaning_parse_error"] = bool((regen or {}).get("prompt_cleaning_parse_error", False))
            if regen and regen.get("prompt_cleaning_raw_output"):
                new["prompt_cleaning_raw_output"] = regen.get("prompt_cleaning_raw_output")
            new["prompt_cleaning_rescued"] = True
            replaced += 1
        else:
            still_empty += 1
        out.append(new)

    return out, replaced, still_empty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--empty_output", default="")
    parser.add_argument("--merge_regen", default="")
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    if args.merge_regen:
        regen_rows = read_jsonl(args.merge_regen)
        rows, replaced, still_empty = merge_regen(rows, regen_rows)
        print("regen rows:", len(regen_rows))
        print("replaced empty rows:", replaced)
        print("still empty after merge:", still_empty)
    else:
        for row in rows:
            row["cleaned_response"] = response_from_row(row)

    write_jsonl(rows, args.output)

    empty_rows = [row for row in rows if not clean_text(row.get("cleaned_response", ""))]
    if args.empty_output:
        write_jsonl(empty_rows, args.empty_output)

    print("input rows:", len(rows))
    print("empty rows:", len(empty_rows))
    print("saved:", args.output)
    if args.empty_output:
        print("empty subset saved:", args.empty_output)


if __name__ == "__main__":
    main()
