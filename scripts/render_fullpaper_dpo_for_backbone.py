#!/usr/bin/env python3
"""Render frozen DPO semantic inputs with one backbone's official chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from selective_risk_refinement_utils import build_sft_prompt
from fullpaper_backbone_utils import load_fullpaper_tokenizer


VERSION = "fullpaper-backbone-dpo-render-v1"
FROZEN_DATASET = "fullpaper_production_accepted652_reuse_reviewed_v2_20261004_frozen319"
FROZEN_PROVENANCE = "reviewed-reuse-first-experiment-freeze-v1-20261004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_dpo_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or (row.get("metadata") or {}).get("canonical_id") or "")


def render_pair(
    source_sft: dict[str, Any],
    source_dpo: dict[str, Any],
    tokenizer: Any,
    repo: str,
    revision: str,
) -> dict[str, Any]:
    pair_id = str(source_sft["id"])
    if source_dpo_id(source_dpo) != pair_id:
        raise RuntimeError(f"Frozen SFT/DPO ID mismatch: {pair_id}")
    nested_input = source_dpo.get("input") or {}
    dpo_question = nested_input.get("question", source_dpo.get("question"))
    dpo_rejected = source_dpo["rejected"]
    if nested_input.get("corrupted_response", dpo_rejected) != dpo_rejected:
        raise RuntimeError(f"Frozen DPO nested rejected text mismatch: {pair_id}")
    if (
        source_sft["question"] != dpo_question
        or source_sft["safe_response"] != source_dpo["chosen"]
        or source_sft["unsafe_response"] != dpo_rejected
    ):
        raise RuntimeError(f"Frozen SFT/DPO question or response text mismatch: {pair_id}")

    split = str(source_sft["split"]).lower()
    if split not in {"train", "valid"}:
        raise RuntimeError(f"Unsupported frozen split for {pair_id}: {split}")
    question_hash = str(source_sft["question_normalized_sha256"])
    original_metadata = source_dpo.get("audit_metadata") or source_dpo.get("metadata") or {}
    audit_metadata = {
        **original_metadata,
        "contract_version": "fullpaper-dpo-v1",
        "dataset": FROZEN_DATASET,
        "provenance": FROZEN_PROVENANCE,
        "canonical_id": pair_id,
        "question_normalized_sha256": question_hash,
        "original_split": split,
        "development_role": split,
        "chosen_sha256": hashlib.sha256(source_dpo["chosen"].encode()).hexdigest(),
        "rejected_sha256": hashlib.sha256(dpo_rejected.encode()).hexdigest(),
        "prompt_render_contract": VERSION,
        "prompt_renderer_repo": repo,
        "prompt_renderer_revision": revision,
        "enable_thinking": False,
    }
    return {
        "id": pair_id,
        "question_group_id": question_hash,
        "prompt": build_sft_prompt(tokenizer, source_sft),
        "chosen": source_dpo["chosen"],
        "rejected": dpo_rejected,
        "audit_metadata": audit_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-file", required=True)
    parser.add_argument("--dpo-file", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sft_path, dpo_path = Path(args.sft_file).resolve(), Path(args.dpo_file).resolve()
    sft, dpo = read(sft_path), read(dpo_path)
    if len(sft) != len(dpo) or [row["id"] for row in sft] != [source_dpo_id(row) for row in dpo]:
        raise RuntimeError("Frozen SFT/DPO membership or ordering differs")
    tokenizer = load_fullpaper_tokenizer(args.tokenizer)
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned backbone tokenizer has no chat template")
    rendered = []
    for source_sft, source_dpo in zip(sft, dpo, strict=True):
        rendered.append(render_pair(source_sft, source_dpo, tokenizer, args.repo, args.revision))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, rendered)
    manifest = {
        "status": "complete",
        "version": VERSION,
        "repo": args.repo,
        "revision": args.revision,
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "enable_thinking": False,
        "rows": len(rendered),
        "source": {"sft_sha256": sha256(sft_path), "dpo_sha256": sha256(dpo_path)},
        "output": {"path": str(output), "sha256": sha256(output)},
        "chosen_rejected_text_unchanged": True,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
