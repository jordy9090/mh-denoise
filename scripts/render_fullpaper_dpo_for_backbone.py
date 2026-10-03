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
    if len(sft) != len(dpo) or [row["id"] for row in sft] != [row["id"] for row in dpo]:
        raise RuntimeError("Frozen SFT/DPO membership or ordering differs")
    tokenizer = load_fullpaper_tokenizer(args.tokenizer)
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned backbone tokenizer has no chat template")
    rendered = []
    for source_sft, source_dpo in zip(sft, dpo, strict=True):
        if source_sft["safe_response"] != source_dpo["chosen"] or source_sft["unsafe_response"] != source_dpo["rejected"]:
            raise RuntimeError(f"Frozen pair text mismatch: {source_sft['id']}")
        row = dict(source_dpo)
        row["prompt"] = build_sft_prompt(tokenizer, source_sft)
        row["audit_metadata"] = {
            **source_dpo["audit_metadata"],
            "prompt_render_contract": VERSION,
            "prompt_renderer_repo": args.repo,
            "prompt_renderer_revision": args.revision,
            "enable_thinking": False,
        }
        rendered.append(row)
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
