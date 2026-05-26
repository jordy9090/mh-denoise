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


def main():
    ds = load_dataset("izi-ano/CounselBench-Eval", split="test")
    df = ds.to_pandas()

    for col in ["overall_score", "empathy_score", "specificity_score", "factual_consistency_score", "toxicity_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1순위: 아주 안전한 high-quality response
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
        - safe_df["toxicity_score"]
    )

    safe_df = safe_df.sort_values("score_sum", ascending=False)
    safe_df = safe_df.drop_duplicates(subset=["questionID"])

    # 99개만 나오면 1개 부족하니까 relaxed 후보로 채움
    if len(safe_df) < 100:
        relaxed_df = df[
            (df["overall_score"] >= 4)
            & (df["empathy_score"] >= 3)
            & (df["specificity_score"] >= 3)
            & (df["factual_consistency_score"] >= 4)
            & (df["toxicity_score"] <= 1)
            & (df["medical_advice_score"].astype(str).str.lower() == "no")
        ].copy()

        relaxed_df["score_sum"] = (
            relaxed_df["overall_score"]
            + relaxed_df["empathy_score"]
            + relaxed_df["specificity_score"]
            + relaxed_df["factual_consistency_score"]
            - relaxed_df["toxicity_score"]
        )

        relaxed_df = relaxed_df.sort_values("score_sum", ascending=False)
        relaxed_df = relaxed_df.drop_duplicates(subset=["questionID"])

        safe_ids = set(safe_df["questionID"].astype(str))
        relaxed_df = relaxed_df[~relaxed_df["questionID"].astype(str).isin(safe_ids)]

        safe_df = pd.concat([safe_df, relaxed_df], ignore_index=True)

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
