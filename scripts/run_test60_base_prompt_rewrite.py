#!/usr/bin/env python3
"""Generate the TEST60 base-model control with the frozen SFT prompt/decoding."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, "/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008/scripts")

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from run_gemma_selective_risk_refinement import load_base_model
from selective_risk_refinement_utils import build_sft_prompt, canonical_example, generate_response


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
INPUT = ROOT / "inputs/test60_sft_frozen.jsonl"
OUTPUT = ROOT / "generation/base_prompt_rewrite_test60.jsonl"
MANIFEST = OUTPUT.with_suffix(".manifest.json")
BASE_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
TOKENIZER_SOURCE = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/gemma/sft/final"
)
MAX_SOURCE_LEN = 1024
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_ids_hash(token_ids: list[int]) -> str:
    return sha256_text(json.dumps(token_ids, separators=(",", ":")))


def main() -> None:
    started = time.monotonic()
    rows = read_jsonl(INPUT)
    if len(rows) != 60:
        raise RuntimeError(f"Expected 60 frozen inputs, got {len(rows)}")
    if [int(row["source_order"]) for row in rows] != list(range(60)):
        raise RuntimeError("Frozen input order is not 0..59")
    existing_history = read_jsonl(OUTPUT)
    existing: dict[str, dict[str, Any]] = {}
    for row in existing_history:
        existing[row["canonical_id"]] = row
    unknown = set(existing) - {row["canonical_id"] for row in rows}
    if unknown:
        raise RuntimeError(f"Output contains unknown IDs: {sorted(unknown)}")
    pending = [row for row in rows if row["canonical_id"] not in existing]
    if not pending and MANIFEST.exists():
        print(MANIFEST.read_text(encoding="utf-8"), flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_fullpaper_tokenizer(str(TOKENIZER_SOURCE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_base_model(str(BASE_MODEL), use_4bit=torch.cuda.is_available())
    model.eval()

    for input_row in pending:
        row = canonical_example(input_row)
        prompt = build_sft_prompt(tokenizer, row)
        untruncated = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        used = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )["input_ids"]
        raw, response, generation = generate_response(
            model,
            tokenizer,
            prompt,
            max_source_len=MAX_SOURCE_LEN,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            repetition_penalty=REPETITION_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        )
        result = {
            "version": VERSION,
            "split": "TEST",
            "source_order": int(input_row["source_order"]),
            "case_id": input_row["case_id"],
            "canonical_id": input_row["canonical_id"],
            "question": input_row["question"],
            "unsafe_response": input_row["unsafe_response"],
            "safe_response": input_row.get("safe_response", ""),
            "arm": "base_prompt_rewrite",
            "adapter": None,
            "base_prompt_response_raw": raw,
            "base_prompt_response": response,
            "generation": generation,
            "technical_valid": (
                bool(response.strip())
                and not generation["length_limit_reached"]
                and not generation["source_input_truncated"]
            ),
            "prompt_provenance": {
                "renderer": "selective_risk_refinement_utils.build_sft_prompt",
                "rendered_prompt": prompt,
                "rendered_prompt_sha256": sha256_text(prompt),
                "input_token_ids_hash_scheme": "sha256(compact-json-array-of-decimal-token-ids)",
                "input_token_ids_untruncated_sha256": token_ids_hash([int(value) for value in untruncated]),
                "input_token_ids_used_sha256": token_ids_hash([int(value) for value in used]),
                "source_input_tokens_untruncated": len(untruncated),
                "source_input_tokens_used": len(used),
                "source_input_truncated": len(untruncated) > len(used),
            },
        }
        append_jsonl(OUTPUT, result)
        existing[input_row["canonical_id"]] = result
        print(
            json.dumps(
                {
                    "arm": "base_prompt_rewrite",
                    "case_id": input_row["case_id"],
                    "completed": len(existing),
                    "total": 60,
                    "technical_valid": result["technical_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ordered = [existing[row["canonical_id"]] for row in rows]
    write_jsonl(OUTPUT, ordered)
    manifest = {
        "version": VERSION,
        "status": "complete" if len(ordered) == 60 else "partial",
        "arm": "BasePromptRewrite",
        "purpose": "No-adapter control using the exact SFT prompt renderer and first-stage decoding",
        "rows": len(ordered),
        "technical_valid": sum(row["technical_valid"] for row in ordered),
        "base_model": str(BASE_MODEL.resolve()),
        "tokenizer_source": str(TOKENIZER_SOURCE.resolve()),
        "adapter": None,
        "settings": {
            "max_source_len": MAX_SOURCE_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "load_in_4bit": torch.cuda.is_available(),
            "prompt_renderer": "selective_risk_refinement_utils.build_sft_prompt",
        },
        "input": {"path": str(INPUT.resolve()), "sha256": sha256_file(INPUT), "rows": len(rows)},
        "output": {"path": str(OUTPUT.resolve()), "sha256": sha256_file(OUTPUT), "rows": len(ordered)},
        "generation_completion": {
            "eos_reached": sum(bool(row["generation"]["eos_reached"]) for row in ordered),
            "length_limit_reached": sum(bool(row["generation"]["length_limit_reached"]) for row in ordered),
            "source_input_truncated": sum(bool(row["generation"]["source_input_truncated"]) for row in ordered),
        },
        "runtime_seconds_this_invocation": round(time.monotonic() - started, 6),
        "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "software": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "bitsandbytes": package_version("bitsandbytes"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(MANIFEST, manifest)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
