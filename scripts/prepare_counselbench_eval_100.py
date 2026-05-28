import json
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


OUTPUT_PATH = "data/raw/counselbench_eval_100.jsonl"


def clean_text(x):
    x = str(x)
    x = re.sub(r"<br\s*/?>", "\n", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def load_counselbench_eval():
    # H100에 HF cache가 없을 수 있어서 datasets로 직접 로드
    ds = load_dataset("izi-ano/CounselBench-Eval", split="test")
    return ds.to_pandas()


def main():
    df = load_counselbench_eval()

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

    safe_df["score_sum"] = (
        safe_df["overall_score"]
        + safe_df["empathy_score"]
        + safe_df["specificity_score"]
        + safe_df["factual_consistency_score"]
    )

    safe_df = safe_df.sort_values("score_sum", ascending=False)
    safe_df = safe_df.drop_duplicates(subset=["questionID"])
    safe_df = safe_df.head(100)

    print("selected unique questions:", len(safe_df))

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for new_i, (_, row) in enumerate(safe_df.iterrows()):
            record = {
                "id": f"cb_eval_{new_i:04d}",
                "question_id": str(row["questionID"]),
                "question": clean_text(row["questionText"]),
                "safe_response": clean_text(row["response"]),
                "topic": str(row["topic"]),
                "source": "CounselBench-Eval",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
