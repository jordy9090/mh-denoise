import argparse
import json
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


OUTPUT_PATH = "data/raw/counselbench_eval_100.jsonl"
DEFAULT_DATASET = "izi-ano/CounselBench-Eval"

FILTER_LEVELS = {
    "strict": {
        "overall_score": 4,
        "empathy_score": 4,
        "specificity_score": 4,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
    "relaxed_1": {
        "overall_score": 4,
        "empathy_score": 4,
        "specificity_score": 3,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
    "relaxed_2": {
        "overall_score": 4,
        "empathy_score": 3,
        "specificity_score": 3,
        "factual_consistency_score": 4,
        "toxicity_score": 1,
    },
}

QUESTION_ID_COLS = ["questionID", "question_id", "qid"]
QUESTION_TEXT_COLS = ["questionText", "question", "query", "user_question", "post"]
RESPONSE_COLS = ["response", "safe_response", "answer", "assistant_response"]
TOPIC_COLS = ["topic", "category", "domain"]


def clean_text(x):
    x = str(x if x is not None else "")
    x = re.sub(r"<br\s*/?>", "\n", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def first_existing_col(df, candidates, required=False):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def normalize_question_key(text):
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def row_question_key(row, qid_col=None, q_col=None):
    if qid_col and pd.notna(row.get(qid_col)) and str(row.get(qid_col)).strip():
        return "id:" + str(row.get(qid_col)).strip()
    if q_col:
        return "text:" + normalize_question_key(row.get(q_col))
    return ""


def is_no_medical_advice(value):
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value) == 0.0
    text = str(value).strip().lower()
    return text in {"no", "false", "0", "0.0", "none", "n"}


def coerce_score_columns(df):
    out = df.copy()
    for col in ["overall_score", "empathy_score", "specificity_score", "factual_consistency_score", "toxicity_score"]:
        if col not in out.columns:
            raise KeyError(f"Missing score column: {col}. Available columns: {list(out.columns)}")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "medical_advice_score" not in out.columns:
        raise KeyError(f"Missing score column: medical_advice_score. Available columns: {list(out.columns)}")
    return out


def apply_safe_filter(df, filter_level):
    if filter_level not in FILTER_LEVELS:
        raise ValueError(f"Unknown filter_level={filter_level}. Expected one of {sorted(FILTER_LEVELS)}")

    out = coerce_score_columns(df)
    cfg = FILTER_LEVELS[filter_level]
    mask = (
        (out["overall_score"] >= cfg["overall_score"])
        & (out["empathy_score"] >= cfg["empathy_score"])
        & (out["specificity_score"] >= cfg["specificity_score"])
        & (out["factual_consistency_score"] >= cfg["factual_consistency_score"])
        & (out["toxicity_score"] <= cfg["toxicity_score"])
        & out["medical_advice_score"].map(is_no_medical_advice)
    )
    return out[mask].copy()


def dataframe_from_dataset(ds, split_name):
    df = ds.to_pandas()
    df["_split"] = split_name
    return df


def load_selected_splits(dataset_name, splits):
    split_names = [x.strip() for x in splits.split(",") if x.strip()]
    if not split_names:
        raise ValueError("--splits must not be empty")

    frames = []
    if len(split_names) == 1 and split_names[0].lower() == "all":
        loaded = load_dataset(dataset_name)
        if isinstance(loaded, DatasetDict):
            for split_name, ds in loaded.items():
                frames.append(dataframe_from_dataset(ds, split_name))
        elif isinstance(loaded, Dataset):
            frames.append(dataframe_from_dataset(loaded, "default"))
        else:
            raise TypeError(f"Unsupported dataset object from load_dataset({dataset_name!r}): {type(loaded)}")
    else:
        for split_name in split_names:
            ds = load_dataset(dataset_name, split=split_name)
            frames.append(dataframe_from_dataset(ds, split_name))

    if not frames:
        raise RuntimeError(f"No splits loaded for dataset={dataset_name} splits={splits}")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_question_key(df):
    qid_col = first_existing_col(df, QUESTION_ID_COLS)
    q_col = first_existing_col(df, QUESTION_TEXT_COLS, required=True)
    out = df.copy()
    out["_question_key"] = [row_question_key(row, qid_col, q_col) for _, row in out.iterrows()]
    return out


def unique_question_count(df):
    if df.empty:
        return 0
    keyed = add_question_key(df)
    return keyed["_question_key"].nunique()


def select_safe_targets(df, n_questions, filter_level, seed=42, shuffle=False):
    qid_col = first_existing_col(df, QUESTION_ID_COLS)
    q_col = first_existing_col(df, QUESTION_TEXT_COLS, required=True)
    response_col = first_existing_col(df, RESPONSE_COLS, required=True)

    safe_df = apply_safe_filter(df, filter_level)
    if safe_df.empty:
        return safe_df, safe_df

    safe_df["_question_key"] = [row_question_key(row, qid_col, q_col) for _, row in safe_df.iterrows()]
    safe_df["score_sum"] = (
        safe_df["overall_score"]
        + safe_df["empathy_score"]
        + safe_df["specificity_score"]
        + safe_df["factual_consistency_score"]
        + (5 - safe_df["toxicity_score"])
    )
    safe_df["_response_len"] = safe_df[response_col].map(lambda x: len(clean_text(x).split()))
    safe_df = safe_df.sort_values(["score_sum", "_response_len"], ascending=[False, False])
    selected = safe_df.drop_duplicates(subset=["_question_key"], keep="first").copy()

    if shuffle:
        selected = selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        selected = selected.reset_index(drop=True)

    selected = selected.head(n_questions).copy()
    return safe_df, selected


def build_record(row, new_i, dataset_name, filter_level):
    qid_col = first_existing_col(pd.DataFrame([row]), QUESTION_ID_COLS)
    q_col = first_existing_col(pd.DataFrame([row]), QUESTION_TEXT_COLS, required=True)
    response_col = first_existing_col(pd.DataFrame([row]), RESPONSE_COLS, required=True)
    topic_col = first_existing_col(pd.DataFrame([row]), TOPIC_COLS)

    source_name = dataset_name.split("/")[-1]
    record = {
        "id": f"cb_eval_{new_i:04d}",
        "question_id": str(row.get(qid_col, row.get("_question_key", ""))) if qid_col else str(row.get("_question_key", "")),
        "question": clean_text(row.get(q_col)),
        "safe_response": clean_text(row.get(response_col)),
        "topic": str(row.get(topic_col, "")) if topic_col else "",
        "source": "CounselBench-Eval" if dataset_name == DEFAULT_DATASET else source_name,
        "safe_target_source": dataset_name,
        "split": str(row.get("_split", "")),
        "filter_level": filter_level,
    }
    return record


def write_safe_targets(rows, output_path, dataset_name, filter_level):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for new_i, (_, row) in enumerate(rows.iterrows()):
            out.write(json.dumps(build_record(row, new_i, dataset_name, filter_level), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--filter_level", choices=sorted(FILTER_LEVELS), default="strict")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--n_questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    df = load_selected_splits(args.dataset_name, args.splits)
    filtered, selected = select_safe_targets(
        df,
        n_questions=args.n_questions,
        filter_level=args.filter_level,
        seed=args.seed,
        shuffle=args.shuffle,
    )

    print("dataset:", args.dataset_name)
    print("splits:", args.splits)
    print("filter_level:", args.filter_level)
    print("total candidate rows before filtering:", len(df))
    print("filtered rows:", len(filtered))
    print("filtered unique questions:", unique_question_count(filtered))
    print("selected unique questions:", len(selected))
    if len(selected) < args.n_questions:
        print(
            f"WARNING: requested {args.n_questions} questions but only selected "
            f"{len(selected)} unique safe questions. Run scripts/audit_counselbench_safe_pool.py "
            "and consider --splits all with relaxed_1 or relaxed_2."
        )

    write_safe_targets(selected, args.output, args.dataset_name, args.filter_level)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
