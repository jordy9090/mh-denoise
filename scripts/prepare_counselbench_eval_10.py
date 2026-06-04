import json
import re
from pathlib import Path

import pandas as pd


INPUT_PATH = "/home/user/.cache/huggingface/hub/datasets--izi-ano--CounselBench-Eval/snapshots/8d56a96ea1de3f3f190f77f4ca9bc3503d731af7/counselbench_eval.csv"
OUTPUT_PATH = "data/raw/counselbench_eval_10.jsonl"


def clean_text(x):
    x = str(x)
    x = re.sub(r"<br\s*/?>", "\n", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def main():
    df = pd.read_csv(INPUT_PATH)

    # Use only high-scoring responses as safe target candidates.
    for col in ["overall_score", "empathy_score", "specificity_score", "factual_consistency_score", "toxicity_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    safe_df = df[
        (df["overall_score"] >= 4)
        & (df["empathy_score"] >= 4)
        & (df["specificity_score"] >= 4)
        & (df["factual_consistency_score"] >= 4)
        & (df["toxicity_score"] <= 1)
        & (df["medical_advice_score"].astype(str).str.lower() == "no")
    ].copy()

    safe_df = safe_df.drop_duplicates(subset=["questionID", "questionText", "response"])
    safe_df = safe_df.head(10)

    print("selected rows:", len(safe_df))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, row in safe_df.iterrows():
            question = clean_text(row["questionText"])
            response = clean_text(row["response"])

            record = {
                "id": f"cb_eval_{i:04d}",
                "question_id": str(row["questionID"]),
                "question": question,
                "safe_response": response,
                "topic": str(row["topic"]),
                "source": "CounselBench-Eval",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
