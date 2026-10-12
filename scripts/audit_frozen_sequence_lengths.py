#!/usr/bin/env python3
"""Audit exact SFT/DPO sequence budgets for a frozen first-run export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from selective_risk_refinement_utils import build_sft_prompt, clean_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def round_up(value: int, multiple: int = 64) -> int:
    return int(math.ceil(value / multiple) * multiple)


def summarize(values: list[int], limit: int) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "rows": len(values),
        "min": min(values),
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": max(values),
        "configured_limit": limit,
        "truncated_rows": sum(value > limit for value in values),
        "smallest_multiple_of_64_preserving_all": round_up(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sft-source-limit", type=int, default=512)
    parser.add_argument("--target-limit", type=int, default=160)
    parser.add_argument("--dpo-prompt-limit", type=int, default=768)
    parser.add_argument("--dpo-completion-limit", type=int, default=512)
    args = parser.parse_args()

    data = Path(args.data_dir).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.eos_token is None:
        raise RuntimeError("Tokenizer has no EOS token")

    details = []
    summary = {}
    for split in ("train", "valid"):
        source_path = data / f"sft_{split}.jsonl"
        dpo_path = data / f"dpo_{split}.jsonl"
        rows = read_jsonl(source_path)
        dpo_rows = read_jsonl(dpo_path)
        if len(rows) != len(dpo_rows):
            raise RuntimeError(f"{split}: frozen SFT/DPO row count mismatch")
        metrics = {
            "sft_prompt": [], "sft_target_with_eos": [],
            "dpo_prompt": [], "dpo_chosen_with_eos": [], "dpo_rejected_with_eos": [],
        }
        for row, dpo_row in zip(rows, dpo_rows, strict=True):
            dpo_id = str(dpo_row.get("id") or (dpo_row.get("metadata") or {}).get("canonical_id") or "")
            dpo_input = dpo_row.get("input") or {}
            if dpo_id != row["id"]:
                raise RuntimeError(f"{split}: frozen SFT/DPO ID mismatch: {row['id']}")
            if (
                dpo_input.get("question") != row["question"]
                or dpo_row["chosen"] != row["safe_response"]
                or dpo_row["rejected"] != row["unsafe_response"]
                or dpo_input.get("corrupted_response") != row["unsafe_response"]
            ):
                raise RuntimeError(f"{split}: frozen SFT/DPO text mismatch: {row['id']}")
            prompt = build_sft_prompt(tokenizer, row)
            sft_target = clean_text(row["safe_response"])
            dpo_chosen = str(dpo_row["chosen"])
            rejected = str(dpo_row["rejected"])
            if not sft_target.endswith(tokenizer.eos_token):
                sft_target += tokenizer.eos_token
            if not dpo_chosen.endswith(tokenizer.eos_token):
                dpo_chosen += tokenizer.eos_token
            if not rejected.endswith(tokenizer.eos_token):
                rejected += tokenizer.eos_token

            # Match ProfessorRefinerDataset exactly.
            sft_prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            sft_target_ids = tokenizer(sft_target, add_special_tokens=False, truncation=False)["input_ids"]

            # Match TRL 1.4 DPOTrainer's non-conversational tokenize_fn exactly:
            # tokenize prompt and prompt+completion with tokenizer defaults, then slice.
            dpo_prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
            dpo_chosen_full = tokenizer(prompt + dpo_chosen, truncation=False)["input_ids"]
            dpo_rejected_full = tokenizer(prompt + rejected, truncation=False)["input_ids"]
            if dpo_chosen_full[: len(dpo_prompt_ids)] != dpo_prompt_ids:
                raise RuntimeError(f"DPO chosen prompt prefix mismatch: {row['canonical_id']}")
            if dpo_rejected_full[: len(dpo_prompt_ids)] != dpo_prompt_ids:
                raise RuntimeError(f"DPO rejected prompt prefix mismatch: {row['canonical_id']}")
            dpo_chosen_ids = dpo_chosen_full[len(dpo_prompt_ids) :]
            dpo_rejected_ids = dpo_rejected_full[len(dpo_prompt_ids) :]

            lengths = {
                "sft_prompt": len(sft_prompt_ids),
                "sft_target_with_eos": len(sft_target_ids),
                "dpo_prompt": len(dpo_prompt_ids),
                "dpo_chosen_with_eos": len(dpo_chosen_ids),
                "dpo_rejected_with_eos": len(dpo_rejected_ids),
            }
            for name, value in lengths.items():
                metrics[name].append(value)
            details.append({"canonical_id": row["canonical_id"], "split": split, **lengths})

        summary[split] = {
            "sft_prompt": summarize(metrics["sft_prompt"], args.sft_source_limit),
            "sft_target_with_eos": summarize(metrics["sft_target_with_eos"], args.target_limit),
            "denoiser_target_with_eos": summarize(metrics["sft_target_with_eos"], args.target_limit),
            "dpo_prompt": summarize(metrics["dpo_prompt"], args.dpo_prompt_limit),
            "dpo_chosen_with_eos": summarize(metrics["dpo_chosen_with_eos"], args.dpo_completion_limit),
            "dpo_rejected_with_eos": summarize(metrics["dpo_rejected_with_eos"], args.dpo_completion_limit),
        }

    all_rows = details
    selected = {
        "sft_source_len": round_up(max(row["sft_prompt"] for row in all_rows)),
        "sft_and_denoiser_target_len": round_up(max(row["sft_target_with_eos"] for row in all_rows)),
        "dpo_prompt_len": round_up(max(row["dpo_prompt"] for row in all_rows)),
        "dpo_completion_len": round_up(max(
            max(row["dpo_chosen_with_eos"], row["dpo_rejected_with_eos"])
            for row in all_rows
        )),
        "generation_max_new_tokens": 512,
    }
    payload = {
        "status": "complete_cpu_tokenizer_only",
        "data_dir": str(data),
        "tokenizer": {
            "path": str(tokenizer_path), "class": tokenizer.__class__.__name__,
            "truncation_side": tokenizer.truncation_side, "eos_token_id": tokenizer.eos_token_id,
            "local_files_only": True,
        },
        "inputs": {
            name: {"sha256": sha256(data / name)}
            for split in ("train", "valid")
            for name in (f"sft_{split}.jsonl", f"dpo_{split}.jsonl")
        },
        "configured_limits": {
            "sft_source": args.sft_source_limit, "sft_and_denoiser_target": args.target_limit,
            "dpo_prompt": args.dpo_prompt_limit, "dpo_completion": args.dpo_completion_limit,
        },
        "summary": summary,
        "selected_complete_lengths_before_context_check": selected,
        "model_context_check": "pending_exact_model_config_recovery",
        "details": details,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
