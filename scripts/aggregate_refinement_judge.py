import argparse
import json
from collections import defaultdict
from statistics import mean
from pathlib import Path


SCORE_KEYS = [
    "overall_quality",
    "empathy",
    "specificity",
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


def fmt(x):
    return f"{x:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", default="outputs/analysis")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    by_method = defaultdict(list)
    by_method_dim = defaultdict(list)

    for r in rows:
        by_method[r["method"]].append(r)
        by_method_dim[(r["method"], r.get("target_dimension", "unknown"))].append(r)

    summary = []

    for method, items in sorted(by_method.items()):
        rec = {
            "method": method,
            "n": len(items),
        }
        for k in SCORE_KEYS:
            rec[k] = mean([x["judgment"][k] for x in items])

        med = [x["judgment"]["medical_advice"] for x in items]
        rec["medical_advice_yes_rate"] = sum(m == "yes" for m in med) / len(med)
        rec["medical_advice_yes_or_unsure_rate"] = sum(m in {"yes", "unsure"} for m in med) / len(med)
        rec["toxicity_3plus_rate"] = sum(x["judgment"]["toxicity"] >= 3 for x in items) / len(items)

        summary.append(rec)

    # JSON summary
    json_path = f"{args.out_dir}/refinement_judge_summary.json"
    with open(json_path, "w", encoding="utf-8") as out:
        json.dump(summary, out, ensure_ascii=False, indent=2)

    # Markdown table
    md_path = f"{args.out_dir}/refinement_judge_summary.md"
    with open(md_path, "w", encoding="utf-8") as out:
        out.write("# Refinement Judge Summary\n\n")
        out.write("| method | n | overall ↑ | empathy ↑ | specificity ↑ | factual ↑ | toxicity ↓ | med yes ↓ | med yes/unsure ↓ | tox>=3 ↓ |\n")
        out.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in summary:
            out.write(
                f"| {r['method']} | {r['n']} | "
                f"{fmt(r['overall_quality'])} | {fmt(r['empathy'])} | {fmt(r['specificity'])} | "
                f"{fmt(r['factual_consistency'])} | {fmt(r['toxicity'])} | "
                f"{fmt(r['medical_advice_yes_rate'])} | {fmt(r['medical_advice_yes_or_unsure_rate'])} | "
                f"{fmt(r['toxicity_3plus_rate'])} |\n"
            )

        out.write("\n\n## By target dimension\n\n")
        out.write("| method | target_dimension | n | overall ↑ | empathy ↑ | specificity ↑ | factual ↑ | toxicity ↓ | med yes ↓ |\n")
        out.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for (method, dim), items in sorted(by_method_dim.items()):
            med = [x["judgment"]["medical_advice"] for x in items]
            out.write(
                f"| {method} | {dim} | {len(items)} | "
                f"{fmt(mean([x['judgment']['overall_quality'] for x in items]))} | "
                f"{fmt(mean([x['judgment']['empathy'] for x in items]))} | "
                f"{fmt(mean([x['judgment']['specificity'] for x in items]))} | "
                f"{fmt(mean([x['judgment']['factual_consistency'] for x in items]))} | "
                f"{fmt(mean([x['judgment']['toxicity'] for x in items]))} | "
                f"{fmt(sum(m == 'yes' for m in med) / len(med))} |\n"
            )

    print("Saved:", json_path)
    print("Saved:", md_path)


if __name__ == "__main__":
    main()
