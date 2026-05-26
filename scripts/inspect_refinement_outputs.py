import json
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def short(text, n=1200):
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " ..."


def main():
    test_path = "data/splits/test.jsonl"
    prompt_path = "outputs/refinement/prompt_cleaning_gemma4_test.jsonl"
    refiner_path = "outputs/refinement/lora_refiner_gemma4_test.jsonl"
    out_path = "outputs/analysis/refinement_preview.md"

    test = load_jsonl(test_path)
    prompt = load_jsonl(prompt_path)
    refiner = load_jsonl(refiner_path)

    assert len(test) == len(prompt) == len(refiner), (len(test), len(prompt), len(refiner))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    by_dim = defaultdict(list)
    for i, ex in enumerate(test):
        dim = ex.get("target_dimension", "unknown")
        by_dim[dim].append(i)

    selected = []
    for dim, idxs in sorted(by_dim.items()):
        selected.extend(idxs[:2])  # dimension별 2개씩

    with open(out_path, "w", encoding="utf-8") as out:
        out.write("# Refinement Output Preview\n\n")
        out.write(f"- test rows: {len(test)}\n")
        out.write(f"- prompt rows: {len(prompt)}\n")
        out.write(f"- refiner rows: {len(refiner)}\n\n")

        for i in selected:
            ex = test[i]
            p = prompt[i]
            r = refiner[i]

            out.write("---\n\n")
            out.write(f"## Row {i} / Dimension: `{ex.get('target_dimension')}`\n\n")
            out.write("### Question\n")
            out.write(short(ex.get("question")) + "\n\n")
            out.write("### Unsafe response\n")
            out.write(short(ex.get("unsafe_response")) + "\n\n")
            out.write("### Prompt-only cleaned response\n")
            out.write(short(p.get("cleaned_response")) + "\n\n")
            out.write("### LoRA refiner response\n")
            out.write(short(r.get("refiner_response")) + "\n\n")
            out.write("### Original safe target\n")
            out.write(short(ex.get("safe_response")) + "\n\n")

    print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    main()
