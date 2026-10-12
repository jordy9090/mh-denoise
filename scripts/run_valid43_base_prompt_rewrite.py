#!/usr/bin/env python3
"""Generate the frozen VALID43 no-adapter rewrite control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

FROZEN_CODE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008/scripts")
sys.path.insert(0, str(FROZEN_CODE))

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from run_gemma_selective_risk_refinement import load_base_model
from selective_risk_refinement_utils import build_sft_prompt, canonical_example, generate_response


VERSION = "base-first-residual-valid-v1-20261011"
DEFAULT_ROOT = Path("/mnt/ssd00/hsoh/base_first_residual_valid_20261011_v1")
INPUT = Path(
    "/home/user/hsoh/mh-denoise/data/fullpaper_acl_pipeline/"
    "production_accepted652_reuse_export_reviewed_v2_20261004/sft_valid.jsonl"
)
EXPECTED_INPUT_SHA256 = "e8b155cd7e420bb80df64e0fff76d0ca330fa736a39dbce7493cd3babedd6256"
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_ids_hash(token_ids: list[int]) -> str:
    return sha256_text(json.dumps(token_ids, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / "generation/base_prompt_rewrite_valid43.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    contract_path = root / "run_contract.json"
    log_started = datetime.now(timezone.utc).isoformat()

    input_sha256 = sha256_file(INPUT)
    if input_sha256 != EXPECTED_INPUT_SHA256:
        raise RuntimeError(f"VALID43 input hash mismatch: {input_sha256}")
    rows = read_jsonl(INPUT)
    if len(rows) != 43:
        raise RuntimeError(f"Expected 43 VALID rows, got {len(rows)}")

    ids = [str(row.get("canonical_id") or row.get("id") or "") for row in rows]
    if any(not value for value in ids) or len(set(ids)) != 43:
        raise RuntimeError("VALID43 canonical IDs are missing or duplicated")

    settings = {
        "base_model": str(BASE_MODEL.resolve()),
        "tokenizer_source": str(TOKENIZER_SOURCE.resolve()),
        "adapter": None,
        "max_source_len": MAX_SOURCE_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "load_in_4bit": bool(torch.cuda.is_available()),
        "prompt_renderer": "selective_risk_refinement_utils.build_sft_prompt",
    }
    contract = {
        "version": VERSION,
        "status": "frozen_before_generation",
        "created_at_utc": log_started,
        "purpose": "VALID-only diagnostic of no-adapter prompting before any residual-refiner design",
        "prohibitions": [
            "no TEST response regeneration",
            "no TEST-informed prompt tuning",
            "no existing artifact overwrite",
        ],
        "input": {"path": str(INPUT.resolve()), "sha256": input_sha256, "rows": len(rows)},
        "canonical_ids": ids,
        "canonical_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":"))),
        "settings": settings,
        "frozen_code": {
            "renderer_module": str((FROZEN_CODE / "selective_risk_refinement_utils.py").resolve()),
            "renderer_module_sha256": sha256_file(FROZEN_CODE / "selective_risk_refinement_utils.py"),
            "inference_module": str((FROZEN_CODE / "run_gemma_selective_risk_refinement.py").resolve()),
            "inference_module_sha256": sha256_file(FROZEN_CODE / "run_gemma_selective_risk_refinement.py"),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        comparable_keys = ("version", "purpose", "input", "canonical_ids", "canonical_ids_sha256", "settings", "frozen_code")
        if any(existing_contract.get(key) != contract.get(key) for key in comparable_keys):
            raise RuntimeError("Existing run contract differs from the frozen contract")
    else:
        atomic_json(contract_path, contract)

    existing_history = read_jsonl(output)
    existing: dict[str, dict[str, Any]] = {}
    for row in existing_history:
        canonical_id = str(row["canonical_id"])
        if canonical_id in existing and existing[canonical_id] != row:
            raise RuntimeError(f"Conflicting checkpoint rows for {canonical_id}")
        existing[canonical_id] = row
    unknown = set(existing) - set(ids)
    if unknown:
        raise RuntimeError(f"Output contains unknown IDs: {sorted(unknown)}")

    pending = [row for row in rows if str(row.get("canonical_id") or row.get("id")) not in existing]
    if not pending and manifest_path.exists():
        print(manifest_path.read_text(encoding="utf-8"), flush=True)
        return

    started = time.monotonic()
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
        canonical_id = str(row.get("canonical_id") or row.get("id"))
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
            "split": "VALID",
            "source_order": ids.index(canonical_id),
            "id": row.get("id"),
            "canonical_id": canonical_id,
            "question": row["question"],
            "unsafe_response": row["unsafe_response"],
            "safe_response": row.get("safe_response", ""),
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
        append_jsonl(output, result)
        existing[canonical_id] = result
        print(
            json.dumps(
                {
                    "arm": "base_prompt_rewrite",
                    "canonical_id": canonical_id,
                    "completed": len(existing),
                    "total": 43,
                    "technical_valid": result["technical_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ordered = [existing[canonical_id] for canonical_id in ids]
    write_jsonl(output, ordered)
    manifest = {
        "version": VERSION,
        "status": "complete" if len(ordered) == 43 else "partial",
        "arm": "BasePromptRewrite",
        "purpose": "No-adapter VALID43 control using exact SFT prompt renderer and first-stage decoding",
        "rows": len(ordered),
        "technical_valid": sum(bool(row["technical_valid"]) for row in ordered),
        "settings": settings,
        "input": {"path": str(INPUT.resolve()), "sha256": input_sha256, "rows": len(rows)},
        "output": {"path": str(output.resolve()), "sha256": sha256_file(output), "rows": len(ordered)},
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
    atomic_json(manifest_path, manifest)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
