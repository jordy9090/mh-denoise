#!/usr/bin/env python3
"""Freeze the exact VALID9/TEST8 inputs for the original-prompt adapter control."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from selective_risk_refinement_utils import word_count


VERSION = "original-prompt-sft-control-v1-20261009"
WORKTREE = Path(__file__).resolve().parents[1]
SOURCE_REPO = Path("/home/user/hsoh/mh-denoise")
OUTPUT_ROOT = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
CHECKPOINT_ROOT = SOURCE_REPO / (
    "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma"
)
VALID_OUTPUT = CHECKPOINT_ROOT / "generation_1536/proposed_mask_off_valid_outputs.jsonl"
VALID_MANIFEST = VALID_OUTPUT.with_suffix(".manifest.json")
TEST_ROOT = CHECKPOINT_ROOT / "generation_1536_test_surface_relaxed_v1_20261008"
TEST_OUTPUT = TEST_ROOT / "part24/proposed_mask_off_test_outputs.jsonl"
TEST_MANIFEST = TEST_OUTPUT.with_suffix(".manifest.json")
TEST_GENERATION_MANIFESTS = (
    TEST_ROOT / "part15/proposed_missing_outputs.manifest.json",
    TEST_ROOT / "part24/proposed_missing_outputs.manifest.json",
)
GENERATION_KEYS = (
    "base_model",
    "sft_adapter_dir",
    "risk_adapter_dir",
    "router_dir",
    "risk_scorer_dir",
    "risk_contract",
    "component_mode",
    "reuse_sft_response",
    "sft_response_field",
    "sft_prompt_style",
    "zt_strategy",
    "T",
    "timestep",
    "risk_threshold",
    "gate_risk_threshold",
    "gate_strategy",
    "gate_focus_aspect",
    "gate_focus_threshold",
    "mask_threshold",
    "mask_token",
    "t2_frac",
    "t3_frac",
    "risk_tag_format",
    "min_risk_delta",
    "min_focus_risk_delta",
    "specificity_min_ratio",
    "keyword_overlap_slack",
    "min_word_count",
    "router_max_len",
    "risk_max_len",
    "max_source_len",
    "max_new_tokens",
    "temperature",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "no_4bit",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise RuntimeError(f"Refusing to overwrite different frozen file: {path}")
    path.write_text(data, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != data:
        raise RuntimeError(f"Refusing to overwrite different frozen file: {path}")
    path.write_text(data, encoding="utf-8")


def adapter_files(path: Path) -> dict[str, str]:
    names = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    )
    return {name: sha256_file(path / name) for name in names if (path / name).exists()}


def frozen_case(split: str, source_order: int, called_order: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "split": split,
        "source_order": source_order,
        "called_order": called_order,
        "case_id": f"TEST-{source_order + 1:02d}" if split == "TEST" else None,
        "canonical_id": row["canonical_id"],
        "question": row["question"],
        "unsafe_response": row.get("unsafe_response", ""),
        "safe_response": row.get("safe_response", ""),
        "sft_response": row["sft_response"],
        "sft_generation": row.get("sft_generation"),
        "z_t_from_sft": row["z_t_from_sft"],
        "z_t_from_sft_infos": row.get("z_t_from_sft_infos") or [],
        "g_sft": row["g_sft"],
        "aspect_conditioning_status": row.get("aspect_conditioning_status", "available"),
        "risk_component_mode": row.get("risk_component_mode", "full"),
        "call_reason": row["denoiser_call_reason"],
        "original_candidate_raw": row["denoiser_response_raw"],
        "original_candidate": row["denoiser_response"],
        "original_generation": row["denoiser_generation"],
        "original_stored_accepted": bool(row["accepted_denoiser"]),
        "original_stored_reject_reason": row.get("reject_reason"),
        "original_stored_final_response": row["final_response"],
        "original_acceptance_policy": row.get("acceptance_policy") or "legacy",
        "sft_metrics_stored": {
            "risk_score": row["sft_risk_score"],
            "focus_risk_score": row.get("sft_focus_risk_score"),
            "word_count": row.get("sft_word_count") or word_count(row["sft_response"]),
            "bad_safety_count": row["sft_bad_safety_count"],
            "generic_count": row["sft_generic_count"],
            "keyword_overlap": row["sft_keyword_overlap"],
        },
        "original_metrics_stored": {
            "risk_score": row["denoiser_risk_score"],
            "focus_risk_score": row.get("denoiser_focus_risk_score"),
            "word_count": row.get("denoiser_word_count") or word_count(row["denoiser_response"]),
            "specificity_ratio": row["specificity_ratio"],
            "bad_safety_count": row["denoiser_bad_safety_count"],
            "generic_count": row["denoiser_generic_count"],
            "keyword_overlap": row["denoiser_keyword_overlap"],
        },
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    valid_manifest = json.loads(VALID_MANIFEST.read_text(encoding="utf-8"))
    test_manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(VALID_OUTPUT) != valid_manifest["output"]["sha256"]:
        raise RuntimeError("VALID source hash differs from its manifest")
    if sha256_file(TEST_OUTPUT) != test_manifest["output"]["sha256"]:
        raise RuntimeError("TEST source hash differs from its manifest")

    manifests = [valid_manifest] + [json.loads(path.read_text(encoding="utf-8")) for path in TEST_GENERATION_MANIFESTS]
    settings = [{key: manifest["settings"].get(key) for key in GENERATION_KEYS} for manifest in manifests]
    if any(item != settings[0] for item in settings[1:]):
        raise RuntimeError("VALID and TEST generation settings differ")
    fixed = settings[0]
    if fixed["zt_strategy"] != "no_mask" or fixed["temperature"] != 0.0:
        raise RuntimeError("Unexpected frozen denoiser generation contract")

    cases = []
    for split, path, expected in (("VALID", VALID_OUTPUT, 9), ("TEST", TEST_OUTPUT, 8)):
        rows = read_jsonl(path)
        called_order = 0
        for source_order, row in enumerate(rows):
            if not row.get("used_denoiser"):
                continue
            if row["z_t_from_sft"] != row["sft_response"]:
                raise RuntimeError(f"{split} {row['canonical_id']} violates stored no-mask identity")
            cases.append(frozen_case(split, source_order, called_order, row))
            called_order += 1
        if called_order != expected:
            raise RuntimeError(f"Expected {expected} called {split} rows, got {called_order}")
    if len({row["canonical_id"] for row in cases}) != 17:
        raise RuntimeError("Called canonical IDs are not unique across splits")

    sft_adapter = Path(fixed["sft_adapter_dir"])
    risk_adapter = Path(fixed["risk_adapter_dir"])
    contract = {
        "version": VERSION,
        "status": "prepared",
        "scope": {"VALID_called": 9, "TEST_called": 8, "new_generations": 17},
        "post_hoc_test_comparison": True,
        "source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True
        ).strip(),
        "sources": {
            "VALID": {"path": str(VALID_OUTPUT), "sha256": sha256_file(VALID_OUTPUT), "manifest": str(VALID_MANIFEST), "manifest_sha256": sha256_file(VALID_MANIFEST)},
            "TEST": {"path": str(TEST_OUTPUT), "sha256": sha256_file(TEST_OUTPUT), "manifest": str(TEST_MANIFEST), "manifest_sha256": sha256_file(TEST_MANIFEST)},
            "TEST_generation_manifests": [
                {"path": str(path), "sha256": sha256_file(path)} for path in TEST_GENERATION_MANIFESTS
            ],
        },
        "fixed_generation_settings": fixed,
        "adapter_only_change": {
            "original_denoiser": str(risk_adapter),
            "original_prompt_sft_control": str(sft_adapter),
            "same_base_model": fixed["base_model"],
            "same_tokenizer_source": str(sft_adapter),
            "same_renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
            "feedback_added": False,
            "adapter_stacking": False,
        },
        "artifact_hashes": {
            "original_adapter": adapter_files(risk_adapter),
            "control_adapter": adapter_files(sft_adapter),
            "renderer_code": sha256_file(WORKTREE / "scripts/selective_risk_refinement_utils.py"),
            "generation_runner_code": sha256_file(WORKTREE / "scripts/run_gemma_selective_risk_refinement.py"),
        },
        "representative_replay_required_before_generation": {"VALID": cases[0]["canonical_id"], "TEST": next(row["canonical_id"] for row in cases if row["split"] == "TEST")},
        "evaluation_authorization": {
            "historical_no_paid_api_contracts_are_preserved": True,
            "paid_gpt41_authorized_for_this_separate_experiment": True,
            "model": "gpt-4.1-2025-04-14",
            "api_key_recorded": False,
        },
        "forbidden": ["retraining", "response regeneration outside the 17 control candidates", "feedback prompts", "TEST-driven prompt or policy adjustment", "overwriting source outputs"],
    }
    write_jsonl(OUTPUT_ROOT / "frozen_called_cases.jsonl", cases)
    write_json(OUTPUT_ROOT / "experiment_contract.json", contract)
    print(json.dumps({"status": "prepared", "cases": len(cases), "output_root": str(OUTPUT_ROOT)}, sort_keys=True))


if __name__ == "__main__":
    main()
