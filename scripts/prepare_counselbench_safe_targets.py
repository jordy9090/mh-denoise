import argparse
import json
import re
from pathlib import Path

from prepare_counselbench_eval_100 import (
    DEFAULT_DATASET,
    load_selected_splits,
    select_safe_targets,
    unique_question_count,
    write_safe_targets,
)


FILTER_ORDER = ["strict", "relaxed_1", "relaxed_2"]


def discover_referenced_counselbench_datasets(repo_root):
    pattern = re.compile(r"""["']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]*CounselBench[A-Za-z0-9_.-]*)["']""")
    found = []
    for path in Path(repo_root).rglob("*"):
        if path.is_dir() or path.suffix not in {".py", ".md", ".txt", ".sh"}:
            continue
        if any(part.startswith(".git") for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in pattern.findall(text):
            if match not in found:
                found.append(match)
    return found


def try_dataset(dataset_name, n_questions, seed, shuffle):
    reports = []
    df = load_selected_splits(dataset_name, "all")
    for filter_level in FILTER_ORDER:
        filtered, selected = select_safe_targets(
            df,
            n_questions=n_questions,
            filter_level=filter_level,
            seed=seed,
            shuffle=shuffle,
        )
        reports.append({
            "dataset_name": dataset_name,
            "filter_level": filter_level,
            "total_candidate_rows": len(df),
            "filtered_rows": len(filtered),
            "filtered_unique_questions": unique_question_count(filtered),
            "selected_unique_questions": len(selected),
        })
        if len(selected) >= n_questions:
            return filter_level, selected, reports
    return None, None, reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_questions", type=int, default=300)
    parser.add_argument("--output", default="data/raw/counselbench_eval_300.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--primary_dataset", default=DEFAULT_DATASET)
    args = parser.parse_args()

    all_reports = []

    filter_level, selected, reports = try_dataset(args.primary_dataset, args.n_questions, args.seed, args.shuffle)
    all_reports.extend(reports)
    if selected is not None:
        write_safe_targets(selected, args.output, args.primary_dataset, filter_level)
        print(json.dumps({"status": "ok", "selected_dataset": args.primary_dataset, "filter_level": filter_level, "reports": all_reports}, ensure_ascii=False, indent=2))
        print(f"saved to {args.output}")
        return

    referenced = discover_referenced_counselbench_datasets(args.repo_root)
    broader = [name for name in referenced if name != args.primary_dataset]
    print("Primary CounselBench-Eval all-splits did not reach target. Referenced broader CounselBench-family datasets:", broader)

    for dataset_name in broader:
        filter_level, selected, reports = try_dataset(dataset_name, args.n_questions, args.seed, args.shuffle)
        all_reports.extend(reports)
        if selected is not None:
            write_safe_targets(selected, args.output, dataset_name, filter_level)
            print(json.dumps({"status": "ok", "selected_dataset": dataset_name, "filter_level": filter_level, "reports": all_reports}, ensure_ascii=False, indent=2))
            print(f"saved to {args.output}")
            return

    print(json.dumps({"status": "insufficient_safe_targets", "reports": all_reports}, ensure_ascii=False, indent=2))
    raise SystemExit(
        f"Could not obtain {args.n_questions} safe targets from CounselBench-family sources referenced in this repo. "
        "Do not add CounselChat or MentalChat unless explicitly requested."
    )


if __name__ == "__main__":
    main()
