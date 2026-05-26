import json
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


def make_record(ex, response, method, i):
    return {
        "id": ex.get("id", f"test_{i:04d}") + f"__{method}",
        "base_id": ex.get("id", f"test_{i:04d}"),
        "question_id": ex.get("question_id"),
        "question": ex["question"],
        "target_dimension": ex.get("target_dimension"),
        "response": response,
        "method": method,
        "safe_reference": ex.get("safe_response"),
        "unsafe_response": ex.get("unsafe_response"),
    }


def main():
    test = load_jsonl("data/splits/test.jsonl")
    prompt = load_jsonl("outputs/refinement/prompt_cleaning_gemma4_test.jsonl")
    refiner = load_jsonl("outputs/refinement/lora_refiner_gemma4_test.jsonl")

    assert len(test) == len(prompt) == len(refiner), (len(test), len(prompt), len(refiner))

    raw_rows = []
    prompt_rows = []
    refiner_rows = []
    all_rows = []

    for i, ex in enumerate(test):
        raw = make_record(ex, ex.get("unsafe_response", ""), "unsafe_raw", i)
        pr = make_record(ex, prompt[i].get("cleaned_response", ""), "prompt_only", i)
        rf = make_record(ex, refiner[i].get("refiner_response", ""), "lora_refiner", i)

        raw_rows.append(raw)
        prompt_rows.append(pr)
        refiner_rows.append(rf)
        all_rows.extend([raw, pr, rf])

    write_jsonl(raw_rows, "outputs/eval_inputs/unsafe_raw_test.jsonl")
    write_jsonl(prompt_rows, "outputs/eval_inputs/prompt_only_test.jsonl")
    write_jsonl(refiner_rows, "outputs/eval_inputs/lora_refiner_test.jsonl")
    write_jsonl(all_rows, "outputs/eval_inputs/refinement_all_methods_test.jsonl")

    print("Saved:")
    print("  outputs/eval_inputs/unsafe_raw_test.jsonl", len(raw_rows))
    print("  outputs/eval_inputs/prompt_only_test.jsonl", len(prompt_rows))
    print("  outputs/eval_inputs/lora_refiner_test.jsonl", len(refiner_rows))
    print("  outputs/eval_inputs/refinement_all_methods_test.jsonl", len(all_rows))


if __name__ == "__main__":
    main()
