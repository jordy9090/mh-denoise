#!/usr/bin/env python3
"""Reconstruct and audit saved selective-refinement prompts without model loading."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from selective_risk_refinement_utils import (
    build_risk_tune_prompt,
    build_risk_tune_user_content,
)


SECTION_HEADERS = (
    "Question:",
    "Original unsafe response:",
    "SFT refined response:",
    "Risk-aware corrupted SFT response:",
    "Aspect scores of SFT response:",
    "Safety requirements:",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def section_bodies(content: str) -> dict[str, str]:
    positions = []
    for header in SECTION_HEADERS:
        start = content.find(header)
        if start < 0:
            raise RuntimeError(f"Prompt section is missing: {header}")
        positions.append((start, header))
    positions.sort()
    bodies = {}
    for index, (start, header) in enumerate(positions):
        body_start = start + len(header)
        body_end = positions[index + 1][0] if index + 1 < len(positions) else len(content)
        bodies[header.removesuffix(":")] = content[body_start:body_end].strip()
    return bodies


def retained_body(prompt: str, body: str, offsets: list[tuple[int, int]]) -> dict[str, Any]:
    start = prompt.find(body)
    if start < 0:
        raise RuntimeError("Section body was not preserved verbatim by the current chat template")
    end = start + len(body)
    covered = [
        (max(left, start), min(right, end))
        for left, right in offsets
        if right > start and left < end and right > left
    ]
    if not covered:
        return {
            "body_chars": len(body), "retained_chars": 0, "retained_fraction": 0.0,
            "fully_retained": False, "retained_first": "", "retained_last": "",
        }
    retained_start = min(left for left, _ in covered)
    retained_end = max(right for _, right in covered)
    retained = prompt[retained_start:retained_end]
    retained_chars = sum(max(0, right - left) for left, right in covered)
    return {
        "body_chars": len(body),
        "retained_chars": retained_chars,
        "retained_fraction": retained_chars / max(1, len(body)),
        "fully_retained": retained_start <= start and retained_end >= end,
        "retained_first": retained[:160],
        "retained_last": retained[-160:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--limits", type=int, nargs="+", default=(512, 1280))
    parser.add_argument("--used-denoiser-only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokenizer_path = Path(args.tokenizer).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    rows = read_jsonl(input_path)
    if args.used_denoiser_only:
        rows = [row for row in rows if row.get("used_denoiser")]

    audits = []
    for row in rows:
        content = build_risk_tune_user_content(row)
        prompt = build_risk_tune_prompt(tokenizer, row)
        bodies = section_bodies(content)
        raw = tokenizer(prompt, add_special_tokens=True, truncation=False)
        by_limit = {}
        for limit in args.limits:
            encoded = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=limit,
                return_offsets_mapping=True,
            )
            offsets = [tuple(value) for value in encoded.pop("offset_mapping")]
            retained_prompt = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
            by_limit[str(limit)] = {
                "tokens": len(encoded["input_ids"]),
                "was_truncated": len(raw["input_ids"]) > limit,
                "retained_prompt": retained_prompt,
                "retained_prompt_first": retained_prompt[:300],
                "retained_prompt_last": retained_prompt[-300:],
                "sections": {
                    name: retained_body(prompt, body, offsets)
                    for name, body in bodies.items()
                },
            }
        audits.append({
            "canonical_id": row.get("canonical_id") or row.get("id"),
            "raw_prompt_tokens": len(raw["input_ids"]),
            "raw_prompt": prompt,
            "limits": by_limit,
        })

    with (output / "prompt_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audits:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    token_lengths = [row["raw_prompt_tokens"] for row in audits]
    summary = {
        "status": "complete_cpu_tokenizer_only_prompt_reconstruction",
        "provenance": "reconstructed_with_current_code; original run did not save prompt strings or input token IDs",
        "input": {"path": str(input_path), "sha256": sha256(input_path), "selected_rows": len(audits)},
        "tokenizer": {
            "path": str(tokenizer_path),
            "class": tokenizer.__class__.__name__,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
            "local_files_only": True,
        },
        "prompt_builder": {
            "path": str((Path(__file__).resolve().parent / "selective_risk_refinement_utils.py")),
            "sha256": sha256(Path(__file__).resolve().parent / "selective_risk_refinement_utils.py"),
        },
        "raw_prompt_tokens": {
            "min": min(token_lengths) if token_lengths else None,
            "max": max(token_lengths) if token_lengths else None,
            "mean": sum(token_lengths) / len(token_lengths) if token_lengths else None,
        },
        "limits": {
            str(limit): {
                "truncated_rows": sum(row["limits"][str(limit)]["was_truncated"] for row in audits),
                "rows_with_any_incomplete_section": sum(
                    not all(section["fully_retained"] for section in row["limits"][str(limit)]["sections"].values())
                    for row in audits
                ),
            }
            for limit in args.limits
        },
        "output": "prompt_audit.jsonl",
    }
    (output / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
