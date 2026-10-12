#!/usr/bin/env python3
"""Generate VALID43 second-pass controls from frozen BasePromptRewrite drafts."""

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
from peft import PeftModel

FROZEN_CODE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008/scripts")
sys.path.insert(0, str(FROZEN_CODE))

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from run_gemma_selective_risk_refinement import load_base_model
from selective_risk_refinement_utils import build_risk_tune_prompt, configure_risk_contract, generate_response


VERSION = "base-first-residual-valid-v1-20261011"
DEFAULT_ROOT = Path("/mnt/ssd00/hsoh/base_first_residual_valid_20261011_v1")
BASE_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
CHECKPOINT_ROOT = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/gemma"
)
TOKENIZER_SOURCE = CHECKPOINT_ROOT / "sft/final"
DENOISER_ADAPTER = CHECKPOINT_ROOT / "proposed_mask_off/final"
MAX_SOURCE_LEN = 4096
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4
ARMS = ("base_self_refine", "trained_denoiser_on_base")


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


def token_ids_hash(values: list[int]) -> str:
    return sha256_text(json.dumps(values, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_path = root / "generation/base_prompt_rewrite_valid43_extraction_recovered.jsonl"
    risk_path = root / "risk/unique_pair_full_scorer_risk.jsonl"
    assignment_path = root / "inputs/pointwise_assignments.jsonl"
    output = root / "generation/base_draft_second_pass_controls_valid43.jsonl"
    manifest_path = output.with_suffix(".manifest.json")
    contract_path = root / "base_draft_second_pass_contract.json"

    base_rows = read_jsonl(input_path)
    assignments = read_jsonl(assignment_path)
    risk_rows = read_jsonl(risk_path)
    if len(base_rows) != 43:
        raise RuntimeError(f"Expected 43 recovered base rows, got {len(base_rows)}")
    base_assignments = [row for row in assignments if row["method"] == "BasePromptRewrite"]
    if len(base_assignments) != 43:
        raise RuntimeError(f"Expected 43 base assignments, got {len(base_assignments)}")
    risk_by_hash = {row["question_response_sha256"]: row for row in risk_rows}
    assignment_by_id = {row["canonical_id"]: row for row in base_assignments}
    ids = [row["canonical_id"] for row in base_rows]
    if len(set(ids)) != 43 or set(ids) != set(assignment_by_id):
        raise RuntimeError("Base control ID set mismatch")
    for row in base_rows:
        assignment = assignment_by_id[row["canonical_id"]]
        if assignment["response"] != row["base_prompt_response_evaluation"]:
            raise RuntimeError(f"Recovered base response mismatch for {row['canonical_id']}")
        if assignment["question_response_sha256"] not in risk_by_hash:
            raise RuntimeError(f"Missing frozen risk score for {row['canonical_id']}")

    settings = {
        "base_model": str(BASE_MODEL.resolve()),
        "tokenizer_source": str(TOKENIZER_SOURCE.resolve()),
        "trained_denoiser_adapter": str(DENOISER_ADAPTER.resolve()),
        "arms": {
            "base_self_refine": "no adapter",
            "trained_denoiser_on_base": "proposed_mask_off/final adapter",
        },
        "draft": "frozen BasePromptRewrite VALID43 response after documented extraction recovery",
        "prompt_renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
        "zt_strategy": "no_mask",
        "risk_contract": "fullpaper_v1",
        "max_source_len": MAX_SOURCE_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "repetition_penalty": REPETITION_PENALTY,
        "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
        "load_in_4bit": bool(torch.cuda.is_available()),
        "generation_scope": "always-generate candidate diagnostic; not a deployment gate",
    }
    contract = {
        "version": VERSION,
        "status": "frozen_before_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "test_used_for_tuning": False,
        "purpose": "Separate second-pass prompting from trained-denoiser adapter effects on identical BasePromptRewrite drafts",
        "input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": len(base_rows)},
        "risk": {"path": str(risk_path), "sha256": sha256_file(risk_path), "rows": len(risk_rows)},
        "canonical_ids": ids,
        "canonical_ids_sha256": sha256_text(json.dumps(ids, separators=(",", ":"))),
        "settings": settings,
        "artifacts": {
            "denoiser_adapter_config_sha256": sha256_file(DENOISER_ADAPTER / "adapter_config.json"),
            "denoiser_adapter_weights_sha256": sha256_file(DENOISER_ADAPTER / "adapter_model.safetensors"),
            "renderer_module_sha256": sha256_file(FROZEN_CODE / "selective_risk_refinement_utils.py"),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        for key in ("version", "purpose", "input", "risk", "canonical_ids", "canonical_ids_sha256", "settings", "artifacts"):
            if existing_contract.get(key) != contract.get(key):
                raise RuntimeError(f"Existing frozen contract differs at {key}")
    else:
        atomic_json(contract_path, contract)

    history = read_jsonl(output)
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    for row in history:
        key = (row["canonical_id"], row["arm"])
        if key in existing and existing[key] != row:
            raise RuntimeError(f"Conflicting checkpoint rows for {key}")
        existing[key] = row
    expected_keys = {(canonical_id, arm) for canonical_id in ids for arm in ARMS}
    if set(existing) - expected_keys:
        raise RuntimeError("Control output contains unknown ID/arm keys")
    pending_keys = expected_keys - set(existing)
    if not pending_keys and manifest_path.exists():
        print(manifest_path.read_text(encoding="utf-8"), flush=True)
        return

    configure_risk_contract("fullpaper_v1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tokenizer = load_fullpaper_tokenizer(str(TOKENIZER_SOURCE))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = load_base_model(str(BASE_MODEL), use_4bit=torch.cuda.is_available())
    model = PeftModel.from_pretrained(base_model, str(DENOISER_ADAPTER), adapter_name="trained_denoiser")
    model.eval()
    started = time.monotonic()

    for base_row in base_rows:
        canonical_id = base_row["canonical_id"]
        assignment = assignment_by_id[canonical_id]
        risk = risk_by_hash[assignment["question_response_sha256"]]
        draft = base_row["base_prompt_response_evaluation"]
        prompt_row = {
            **base_row,
            "sft_response": draft,
            "z_t_from_sft": draft,
            "g_sft": risk["g"],
            "aspect_conditioning_status": "available",
            "risk_component_mode": "full",
        }
        prompt = build_risk_tune_prompt(tokenizer, prompt_row)
        untruncated = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
        used = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SOURCE_LEN,
        )["input_ids"]
        provenance = {
            "renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
            "rendered_prompt": prompt,
            "rendered_prompt_sha256": sha256_text(prompt),
            "input_token_ids_hash_scheme": "sha256(compact-json-array-of-decimal-token-ids)",
            "input_token_ids_untruncated_sha256": token_ids_hash([int(value) for value in untruncated]),
            "input_token_ids_used_sha256": token_ids_hash([int(value) for value in used]),
            "source_input_tokens_untruncated": len(untruncated),
            "source_input_tokens_used": len(used),
            "source_input_truncated": len(untruncated) > len(used),
        }
        if provenance["source_input_truncated"]:
            raise RuntimeError(f"Unexpected second-pass prompt truncation for {canonical_id}")

        for arm in ARMS:
            key = (canonical_id, arm)
            if key in existing:
                continue
            if arm == "base_self_refine":
                with model.disable_adapter():
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
                adapter = None
            else:
                model.set_adapter("trained_denoiser")
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
                adapter = str(DENOISER_ADAPTER.resolve())
            raw_words = len(raw.split())
            response_words = len(response.split())
            severe_cleanup_loss = bool(raw_words >= 20 and response_words < 0.8 * raw_words)
            evaluation_response = raw.strip() if severe_cleanup_loss else response.strip()
            result = {
                "version": VERSION,
                "split": "VALID",
                "source_order": int(base_row["source_order"]),
                "canonical_id": canonical_id,
                "question": base_row["question"],
                "unsafe_response": base_row["unsafe_response"],
                "safe_response": base_row["safe_response"],
                "base_draft": draft,
                "base_draft_sha256": sha256_text(draft),
                "base_draft_risk_score": risk["risk_score"],
                "base_draft_focus_risk_score": risk["focus_risk_score"],
                "base_draft_g": risk["g"],
                "arm": arm,
                "adapter": adapter,
                "candidate_raw_decoded_output": raw,
                "candidate_legacy_postprocessed": response,
                "candidate_response": evaluation_response,
                "cleanup_recovery_applied": severe_cleanup_loss,
                "generation": generation,
                "technical_valid": (
                    bool(evaluation_response)
                    and generation["eos_reached"]
                    and not generation["length_limit_reached"]
                    and not generation["source_input_truncated"]
                ),
                "prompt_provenance": provenance,
            }
            append_jsonl(output, result)
            existing[key] = result
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "canonical_id": canonical_id,
                        "completed": len(existing),
                        "total": len(expected_keys),
                        "technical_valid": result["technical_valid"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    ordered = [existing[(canonical_id, arm)] for canonical_id in ids for arm in ARMS]
    write_jsonl(output, ordered)
    manifest = {
        "version": VERSION,
        "status": "complete" if len(ordered) == len(expected_keys) else "partial",
        "rows": len(ordered),
        "cases": len(ids),
        "arms": list(ARMS),
        "technical_valid": sum(bool(row["technical_valid"]) for row in ordered),
        "cleanup_recovery_applied": sum(bool(row["cleanup_recovery_applied"]) for row in ordered),
        "settings": settings,
        "input": {"path": str(input_path), "sha256": sha256_file(input_path), "rows": len(base_rows)},
        "risk": {"path": str(risk_path), "sha256": sha256_file(risk_path), "rows": len(risk_rows)},
        "output": {"path": str(output), "sha256": sha256_file(output), "rows": len(ordered)},
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
            "peft": package_version("peft"),
            "bitsandbytes": package_version("bitsandbytes"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(manifest_path, manifest)
    del model, base_model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
