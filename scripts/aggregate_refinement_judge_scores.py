import argparse
import csv
import json
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path


METRICS = [
    "overall_quality",
    "empathy",
    "specificity",
    "medical_advice",
    "factual_consistency",
    "toxicity",
]

PAPER_DERIVED_METRICS = [
    "medical_advice_violation",
]

POSITIVE_METRICS = [
    "overall_quality",
    "empathy",
    "specificity",
    "factual_consistency",
]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def as_number(value):
    try:
        return float(value)
    except Exception:
        return None


def medical_advice_violation_score(scores):
    value = as_number(scores.get("medical_advice_violation"))
    if value is not None:
        return value

    raw = scores.get("medical_advice")
    if isinstance(raw, str):
        norm = raw.strip().lower().rstrip(".")
        if norm == "yes":
            return 1.0
        if norm == "no":
            return 0.0
        return None

    value = as_number(raw)
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value
    return 6.0 - value


def medical_advice_safety_score(scores):
    if as_number(scores.get("medical_advice_violation")) is not None:
        violation = as_number(scores.get("medical_advice_violation"))
        return 5.0 - 4.0 * violation

    raw = scores.get("medical_advice")
    if isinstance(raw, str):
        norm = raw.strip().lower().rstrip(".")
        if norm == "yes":
            return 1.0
        if norm == "no":
            return 5.0
        return None

    value = as_number(raw)
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return 5.0 - 4.0 * value
    return value


def normalized_scores(row):
    scores = row.get("judge_scores") or {}
    out = {}
    for key in METRICS:
        value = as_number(scores.get(key))
        if value is None:
            return None
        out[key] = value
    return out


def parse_group_by(value):
    cols = [x.strip() for x in value.split(",") if x.strip()]
    return cols or ["system"]


def group_key(row, cols):
    return tuple("" if row.get(col) is None else str(row.get(col)) for col in cols)


def metric_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), stdev(values)


def aggregate_rows(rows, group_cols):
    grouped = defaultdict(list)
    for row in rows:
        grouped[group_key(row, group_cols)].append(row)

    out = []
    for key, items in sorted(grouped.items()):
        score_dicts = []
        for row in items:
            scores = row.get("judge_scores") or {}
            if isinstance(scores, dict) and scores:
                score_dicts.append(scores)

        record = {col: value for col, value in zip(group_cols, key)}
        record["n"] = len(score_dicts)
        record["n_total"] = len(items)

        for metric in METRICS:
            vals = []
            for scores in score_dicts:
                value = as_number(scores.get(metric))
                if value is not None:
                    vals.append(value)
            avg, sd = metric_stats(vals)
            record[f"{metric}_mean"] = avg
            record[f"{metric}_std"] = sd

        medical_violation_vals = []
        for scores in score_dicts:
            value = medical_advice_violation_score(scores)
            if value is not None:
                medical_violation_vals.append(value)
        avg, sd = metric_stats(medical_violation_vals)
        record["medical_advice_violation_mean"] = avg
        record["medical_advice_violation_std"] = sd

        quality_vals = []
        for scores in score_dicts:
            adjusted = []
            for metric in POSITIVE_METRICS:
                value = as_number(scores.get(metric))
                if value is not None:
                    adjusted.append(value)
            med_safety = medical_advice_safety_score(scores)
            if med_safety is not None:
                adjusted.append(med_safety)
            toxicity = as_number(scores.get("toxicity"))
            if toxicity is not None:
                adjusted.append(6.0 - toxicity)
            if adjusted:
                quality_vals.append(mean(adjusted))
        avg, sd = metric_stats(quality_vals)
        record["quality_safety_average_mean"] = avg
        record["quality_safety_average_std"] = sd
        out.append(record)

    return out


def write_csv(rows, path, group_cols):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    metric_cols = []
    for metric in METRICS:
        metric_cols.extend([f"{metric}_mean", f"{metric}_std"])
    for metric in PAPER_DERIVED_METRICS:
        metric_cols.extend([f"{metric}_mean", f"{metric}_std"])
    fieldnames = group_cols + ["n", "n_total"] + metric_cols + [
        "quality_safety_average_mean",
        "quality_safety_average_std",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_file(input_path, output_csv, group_cols):
    rows = list(read_jsonl(input_path))
    summary = aggregate_rows(rows, group_cols)
    write_csv(summary, output_csv, group_cols)
    return len(rows), len(summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--group_by", default="system")
    args = parser.parse_args()

    group_cols = parse_group_by(args.group_by)
    n_rows, n_groups = aggregate_file(args.input, args.output_csv, group_cols)
    print(f"input rows: {n_rows}")
    print(f"groups: {n_groups}")
    print(f"saved: {args.output_csv}")
    print("toxicity convention: lower is better; quality_safety_average uses 6 - toxicity")
    print("paper metric: medical_advice_violation = 6 - medical_advice; lower is better")


if __name__ == "__main__":
    main()
