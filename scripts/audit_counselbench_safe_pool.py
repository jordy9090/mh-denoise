import argparse
import json

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from prepare_counselbench_eval_100 import (
    DEFAULT_DATASET,
    FILTER_LEVELS,
    apply_safe_filter,
    unique_question_count,
)


def dataframe_from_dataset(ds, split_name):
    df = ds.to_pandas()
    df["_split"] = split_name
    return df


def load_all_splits(dataset_name):
    loaded = load_dataset(dataset_name)
    if isinstance(loaded, DatasetDict):
        return {split_name: dataframe_from_dataset(ds, split_name) for split_name, ds in loaded.items()}
    if isinstance(loaded, Dataset):
        return {"default": dataframe_from_dataset(loaded, "default")}
    raise TypeError(f"Unsupported dataset object from load_dataset({dataset_name!r}): {type(loaded)}")


def summarize_df(name, df):
    summary = {
        "split": name,
        "row_count": len(df),
        "column_names": list(df.columns),
    }

    try:
        summary["unique_question_count"] = unique_question_count(df)
    except Exception as exc:
        summary["unique_question_count"] = None
        summary["unique_question_count_error"] = str(exc)

    for level in FILTER_LEVELS:
        try:
            filtered = apply_safe_filter(df, level)
            summary[f"{level}_pass_row_count"] = len(filtered)
            summary[f"{level}_pass_unique_question_count"] = unique_question_count(filtered)
        except Exception as exc:
            summary[f"{level}_pass_row_count"] = None
            summary[f"{level}_pass_unique_question_count"] = None
            summary[f"{level}_error"] = str(exc)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DEFAULT_DATASET)
    args = parser.parse_args()

    split_frames = load_all_splits(args.dataset_name)
    summaries = []

    for split_name, df in split_frames.items():
        summaries.append(summarize_df(split_name, df))

    combined = pd.concat(split_frames.values(), ignore_index=True, sort=False)
    summaries.append(summarize_df("ALL", combined))

    print(json.dumps({"dataset_name": args.dataset_name, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
