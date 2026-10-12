#!/usr/bin/env python3
"""CPU-only exact input-length audit for the frozen Revision 2 TEST inference path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from selective_risk_refinement_utils import (
    build_risk_tune_prompt,
    build_sft_prompt,
    canonical_example,
)


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


def summarize(lengths: list[tuple[str, int]], frozen_limit: int) -> dict[str, Any]:
    ordered = sorted(value for _, value in lengths)
    return {
        "rows": len(lengths),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
        "frozen_train_valid_limit": frozen_limit,
        "rows_exceeding_frozen_limit": sum(value > frozen_limit for _, value in lengths),
        "smallest_multiple_of_64_preserving_all": round_up(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("sft", "proposed"), required=True)
    parser.add_argument("--frozen-limit", type=int, required=True)
    parser.add_argument("--generation-budget", type=int, default=1536)
    parser.add_argument("--model-context-limit", type=int, default=131072)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = load_fullpaper_tokenizer(tokenizer_path)
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError("Input is empty")

    lengths: list[tuple[str, int]] = []
    for raw in rows:
        row = canonical_example(raw)
        canonical_id = str(row.get("canonical_id") or row.get("id") or "")
        if not canonical_id:
            raise RuntimeError("Input row is missing canonical_id/id")
        if args.stage == "sft":
            prompt = build_sft_prompt(tokenizer, row)
        else:
            sft_response = str(row.get("sft_response") or "")
            if not sft_response:
                raise RuntimeError(f"Proposed input lacks sft_response: {canonical_id}")
            prompt = build_risk_tune_prompt(
                tokenizer,
                {
                    **row,
                    "z_t_from_sft": sft_response,
                    "g_sft": [0.0] * 6,
                    "aspect_conditioning_status": "available",
                },
            )
        token_count = len(
            tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        )
        lengths.append((canonical_id, token_count))

    stats = summarize(lengths, args.frozen_limit)
    selected_limit = max(args.frozen_limit, stats["smallest_multiple_of_64_preserving_all"])
    if selected_limit + args.generation_budget > args.model_context_limit:
        raise RuntimeError("Complete input plus generation budget exceeds model context")
    payload = {
        "status": "complete_cpu_tokenizer_only",
        "stage": args.stage,
        "input": {"path": str(input_path), "sha256": sha256(input_path), "rows": len(rows)},
        "tokenizer": {
            "path": str(tokenizer_path),
            "class": tokenizer.__class__.__name__,
            "local_files_only": True,
        },
        "generation_budget": args.generation_budget,
        "model_context_limit": args.model_context_limit,
        "summary": stats,
        "selected_complete_max_source_len": selected_limit,
        "selection_rule": (
            "retain the frozen TRAIN/VALID limit when sufficient; otherwise use the smallest "
            "multiple of 64 that preserves every exact TEST prompt"
        ),
        "details": [
            {"canonical_id": canonical_id, "input_tokens": token_count}
            for canonical_id, token_count in lengths
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "details"}, indent=2))


if __name__ == "__main__":
    main()
