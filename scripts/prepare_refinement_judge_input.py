import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value):
    return " ".join(str(value or "").replace("\n", " ").split()).strip()


def base_id(row, input_index):
    for key in ("example_id", "id", "question_id"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row_{input_index:06d}"


def build_record(row, input_path, input_index, response, system, id_prefix, input_slot="", warning=""):
    bid = base_id(row, input_index)
    prefix_parts = [x for x in (id_prefix, input_slot) if x]
    prefix = "_".join(prefix_parts) + "_" if prefix_parts else ""
    mode = row.get("mode", "")
    mode_part = f"_{mode}" if mode not in ("", None) else ""

    record = {
        "example_id": f"{prefix}{bid}{mode_part}__{system}",
        "base_example_id": bid,
        "question": clean_text(row.get("question")),
        "response": clean_text(response),
        "system": system,
        "mode": row.get("mode", ""),
        "t": row.get("t", None),
        "reference_safe_response": clean_text(row.get("safe_response")),
        "unsafe_response": clean_text(row.get("unsafe_response")),
        "input_path": str(input_path),
    }

    for key in ("source", "target_dimension", "target_dimensions", "g"):
        if key in row:
            record[key] = row[key]

    if warning:
        record["prepare_warning"] = warning

    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="Input refinement JSONL. May be repeated.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--response_field", default="peft_response")
    parser.add_argument("--system_name", required=True)
    parser.add_argument("--id_prefix", default="")
    parser.add_argument("--mode_filter", default="")
    parser.add_argument("--include_unsafe_baseline", action="store_true")
    parser.add_argument("--include_safe_reference", action="store_true")
    parser.add_argument("--unsafe_system_name", default="unsafe_baseline")
    parser.add_argument("--safe_reference_system_name", default="safe_reference")
    args = parser.parse_args()

    out_rows = []
    counts = Counter()

    include_input_slot = len(args.input) > 1
    for file_i, path in enumerate(args.input):
        input_slot = f"input{file_i}" if include_input_slot else ""
        for i, row in enumerate(read_jsonl(path)):
            counts["input_rows"] += 1

            if args.mode_filter and str(row.get("mode", "")) != args.mode_filter:
                counts["filtered_by_mode"] += 1
                continue

            response = row.get(args.response_field, "")
            warning = "" if clean_text(response) else f"empty_or_missing_response_field:{args.response_field}"
            if warning:
                counts["empty_main_response"] += 1
            out_rows.append(build_record(row, path, i, response, args.system_name, args.id_prefix, input_slot, warning))
            counts[f"system:{args.system_name}"] += 1

            if args.include_unsafe_baseline:
                unsafe = row.get("unsafe_response", "")
                warning = "" if clean_text(unsafe) else "empty_or_missing_response_field:unsafe_response"
                if warning:
                    counts["empty_unsafe_response"] += 1
                out_rows.append(build_record(row, path, i, unsafe, args.unsafe_system_name, args.id_prefix, input_slot, warning))
                counts[f"system:{args.unsafe_system_name}"] += 1

            if args.include_safe_reference:
                safe = row.get("safe_response", "")
                warning = "" if clean_text(safe) else "empty_or_missing_response_field:safe_response"
                if warning:
                    counts["empty_safe_response"] += 1
                out_rows.append(build_record(row, path, i, safe, args.safe_reference_system_name, args.id_prefix, input_slot, warning))
                counts[f"system:{args.safe_reference_system_name}"] += 1

    write_jsonl(out_rows, args.output)

    counts["output_rows"] = len(out_rows)
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
