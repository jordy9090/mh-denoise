#!/usr/bin/env python3
"""Freeze the corrected TEST60 inputs and the reviewer-facing ablation matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
REPO = Path("/home/user/hsoh/mh-denoise")
WORKTREE = Path("/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008")
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
HOME_LINK = Path("/home/user/hsoh/acl_component_ablation_matrix_20261009_v1")
CORRECTED = Path("/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix")
FROZEN_JUDGE = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
CONTROL = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
NO_SELECTOR = Path("/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1")
PUBLICATION = Path("/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1")
CKPT = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/gemma"

SFT_SOURCE = CORRECTED / "generation/sft_test_outputs.jsonl"
PROPOSED_SOURCE = CORRECTED / "generation/proposed_mask_off_test_outputs.jsonl"
FINAL_SOURCE = CORRECTED / "test60_final_outputs.jsonl"
CONTROL_SOURCE = CONTROL / "results/control_generations.jsonl"
SFT_VALID_MANIFEST = CKPT / "generation_1536/sft_valid_outputs.manifest.json"

BASE_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
ROUTER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/router/final"
SCORER = REPO / "outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910/scorer/final"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite a different frozen artifact: {path}")
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise RuntimeError(f"Refusing to overwrite a different frozen artifact: {path}")
    path.write_text(text, encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_meta(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def adapter_meta(path: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        candidate = path / name
        if candidate.exists():
            files[name] = file_meta(candidate)
    if "adapter_config.json" not in files or "adapter_model.safetensors" not in files:
        raise RuntimeError(f"Incomplete adapter: {path}")
    return {"path": str(path.resolve()), "files": files}


def add_order(row: dict[str, Any], source_order: int, case_id: str) -> dict[str, Any]:
    return {**row, "source_order": source_order, "case_id": case_id}


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("inputs", "generation", "requests", "results", "logs", "publication", "code", "risk"):
        (ROOT / name).mkdir(exist_ok=True)
    if HOME_LINK.exists() or HOME_LINK.is_symlink():
        if HOME_LINK.resolve() != ROOT.resolve():
            raise RuntimeError(f"Existing home path points elsewhere: {HOME_LINK}")
    else:
        HOME_LINK.symlink_to(ROOT, target_is_directory=True)

    sft_rows = read_jsonl(SFT_SOURCE)
    proposed_rows = read_jsonl(PROPOSED_SOURCE)
    final_rows = sorted(read_jsonl(FINAL_SOURCE), key=lambda row: int(row["source_order"]))
    if not (len(sft_rows) == len(proposed_rows) == len(final_rows) == 60):
        raise RuntimeError("Corrected sources must each contain exactly 60 rows")
    final_by_id = {row["canonical_id"]: row for row in final_rows}
    if len(final_by_id) != 60:
        raise RuntimeError("Corrected TEST60 contains duplicate canonical IDs")
    ordered_ids = [row["canonical_id"] for row in final_rows]
    if [int(row["source_order"]) for row in final_rows] != list(range(60)):
        raise RuntimeError("Corrected TEST60 source order must be 0..59")

    def normalize_source(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
        by_id = {row["canonical_id"]: row for row in rows}
        if set(by_id) != set(ordered_ids):
            raise RuntimeError(f"{label} canonical IDs differ from corrected TEST60")
        normalized = []
        for final in final_rows:
            row = by_id[final["canonical_id"]]
            if row["question"] != final["question"]:
                raise RuntimeError(f"{label} question mismatch for {final['canonical_id']}")
            if row["sft_response"] != final["sft_response"]:
                raise RuntimeError(f"{label} SFT mismatch for {final['canonical_id']}")
            normalized.append(add_order(row, int(final["source_order"]), final["case_id"]))
        return normalized

    frozen_sft = normalize_source(sft_rows, "SFT")
    frozen_proposed = normalize_source(proposed_rows, "Proposed")
    called = [row for row in frozen_proposed if row.get("used_denoiser") is True]
    gate_off = [row for row in frozen_proposed if row.get("used_denoiser") is not True]
    if len(called) != 18 or len(gate_off) != 42:
        raise RuntimeError(f"Expected 18 called and 42 gate-off, got {len(called)} and {len(gate_off)}")
    final_called_ids = [row["canonical_id"] for row in final_rows if row.get("denoiser_called")]
    if final_called_ids != [row["canonical_id"] for row in called]:
        raise RuntimeError("Corrected final and generation called-ID order differs")

    control_rows = [row for row in read_jsonl(CONTROL_SOURCE) if row.get("split") == "TEST"]
    control_by_id = {row["canonical_id"]: row for row in control_rows}
    reusable_control_ids = [row["canonical_id"] for row in called if row["canonical_id"] in control_by_id]
    missing_control_ids = [row["canonical_id"] for row in called if row["canonical_id"] not in control_by_id]
    if len(reusable_control_ids) != 8 or len(missing_control_ids) != 10:
        raise RuntimeError(
            f"Expected 8 reusable and 10 missing TEST controls, got {len(reusable_control_ids)} and {len(missing_control_ids)}"
        )

    inputs = ROOT / "inputs"
    write_jsonl(inputs / "test60_sft_frozen.jsonl", frozen_sft)
    write_jsonl(inputs / "test60_main_proposed_frozen.jsonl", frozen_proposed)
    write_jsonl(inputs / "test60_main_called18.jsonl", called)
    write_jsonl(inputs / "test60_main_gateoff42.jsonl", gate_off)
    write_jsonl(inputs / "test60_control_called18.jsonl", called)
    write_jsonl(
        ROOT / "generation/original_prompt_sft_control_reused8.jsonl",
        [control_by_id[row_id] for row_id in reusable_control_ids],
    )
    (inputs / "test60_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")
    response_hashes = [
        {
            "source_order": row["source_order"],
            "case_id": row["case_id"],
            "canonical_id": row["canonical_id"],
            "question_sha256": sha256_text(row["question"]),
            "sft_response_sha256": sha256_text(row["sft_response"]),
            "dpo_response_sha256": sha256_text(row["dpo_response"]),
            "proposed_response_sha256": sha256_text(row["final_response"]),
        }
        for row in final_rows
    ]
    write_jsonl(inputs / "corrected_test60_response_hashes.jsonl", response_hashes)

    valid_manifests = {}
    for arm in ("proposed_mask_off", "proposed_mask_on", "proposed_without_router", "proposed_without_scorer"):
        path = CKPT / f"generation_1536/{arm}_valid_outputs.manifest.json"
        valid_manifests[arm] = {
            "artifact": file_meta(path),
            "settings": read_json(path)["settings"],
        }

    common_generation = {
        key: valid_manifests["proposed_mask_off"]["settings"][key]
        for key in (
            "base_model",
            "sft_adapter_dir",
            "router_dir",
            "risk_scorer_dir",
            "risk_contract",
            "T",
            "timestep",
            "gate_strategy",
            "gate_focus_aspect",
            "gate_focus_threshold",
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
    }
    if common_generation["max_source_len"] != 4096 or common_generation["max_new_tokens"] != 1536:
        raise RuntimeError("Unexpected frozen generation length contract")

    judge_contract = read_json(FROZEN_JUDGE / "run_contract.json")
    if judge_contract["model_settings"] != {
        "model": "gpt-4.1-2025-04-14",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "pointwise_max_tokens": 384,
        "selector_max_tokens": 256,
    }:
        raise RuntimeError("Frozen GPT-4.1 contract changed")
    sft_generation_settings = read_json(SFT_VALID_MANIFEST)["settings"]
    expected_sft_settings = {
        "max_source_len": 1024,
        "max_new_tokens": 1536,
        "temperature": 0,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
        "sft_prompt_style": "sft_plain",
        "no_4bit": False,
    }
    for key, expected in expected_sft_settings.items():
        if sft_generation_settings.get(key) != expected:
            raise RuntimeError(f"Unexpected frozen SFT generation setting {key}")

    matrix = [
        {"name": "BasePromptRewrite", "role": "main_baseline", "status": "new_generation_60", "isolates": "fine-tuning versus the same SFT rewrite prompt"},
        {"name": "SFT", "role": "main_baseline", "status": "reuse_60", "isolates": "supervised fine-tuning"},
        {"name": "DPO", "role": "main_baseline", "status": "reuse_60", "isolates": "preference optimization"},
        {"name": "Proposed", "role": "main_method", "status": "reuse_60", "isolates": "full trained denoiser, gate, and frozen GPT selector"},
        {"name": "GatedNoSelector", "role": "ablation", "status": "reuse_60", "isolates": "selector contribution"},
        {"name": "AlwaysOnNoSelector", "role": "ablation", "status": "reuse_18_generate_42", "isolates": "gate contribution without selector"},
        {"name": "AlwaysOnGPTSelector", "role": "ablation", "status": "reuse_18_generate_42_then_select", "isolates": "gate contribution with the same selector"},
        {"name": "OriginalPromptSFTControl", "role": "ablation", "status": "reuse_8_generate_10_then_select", "isolates": "denoiser additional training at fixed denoiser prompt/input"},
        {"name": "MaskON", "role": "ablation", "status": "new_test60_generation_then_select", "isolates": "risk-span masking"},
        {"name": "WithoutRouter", "role": "ablation", "status": "new_test60_generation_then_select", "isolates": "router"},
        {"name": "WithoutScorer", "role": "ablation", "status": "new_test60_generation_then_select", "isolates": "scorer"},
        {"name": "CurrentRules", "role": "policy_sensitivity", "status": "replay_18", "isolates": "surface_relaxed_v1 versus GPT selector"},
        {"name": "RiskOnly", "role": "policy_sensitivity", "status": "replay_18", "isolates": "risk-only acceptance versus GPT selector"},
        {"name": "UnsafeResponse", "role": "diagnostic_lower_bound", "status": "reuse_60", "isolates": "evaluator sanity only"},
        {"name": "ReferenceResponse", "role": "diagnostic_oracle", "status": "reuse_60", "isolates": "upper-bound diagnostic only"},
    ]

    script_paths = [
        REPO / "scripts/prepare_acl_component_ablation_matrix.py",
        REPO / "scripts/run_test60_base_prompt_rewrite.py",
        REPO / "scripts/run_test60_original_prompt_sft_control.py",
        REPO / "scripts/run_acl_component_ablation_inference_tmux.sh",
        WORKTREE / "scripts/run_gemma_selective_risk_refinement.py",
        WORKTREE / "scripts/selective_risk_refinement_utils.py",
    ]
    contract = {
        "version": VERSION,
        "status": "frozen_before_new_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Reviewer-resistant TEST60 baseline and component ablation matrix",
        "post_hoc_exploratory_test_comparison": True,
        "scope": {
            "questions": 60,
            "main_denoiser_called": 18,
            "main_gate_off": 42,
            "new_base_generations": 60,
            "new_always_on_generations": 42,
            "reused_sft_adapter_controls": 8,
            "new_sft_adapter_controls": 10,
            "component_generation_arms": 3,
            "no_training": True,
            "no_existing_output_overwrite": True,
            "no_test_tuning": True,
        },
        "paper_organization": {
            "main_baselines": ["BasePromptRewrite", "SFT", "DPO"],
            "main_method": "Proposed",
            "component_ablations": [
                "GatedNoSelector",
                "AlwaysOnNoSelector",
                "AlwaysOnGPTSelector",
                "OriginalPromptSFTControl",
                "MaskON",
                "WithoutRouter",
                "WithoutScorer",
            ],
            "policy_sensitivity": ["CurrentRules", "RiskOnly", "GPTSelector"],
            "diagnostics_not_deployable_baselines": ["UnsafeResponse", "ReferenceResponse"],
        },
        "matrix": matrix,
        "frozen_sources": {
            "corrected_sft60": file_meta(SFT_SOURCE, 60),
            "corrected_proposed_generation60": file_meta(PROPOSED_SOURCE, 60),
            "corrected_final60": file_meta(FINAL_SOURCE, 60),
            "corrected_final_contract": file_meta(CORRECTED / "run_contract.json"),
            "frozen_gpt_contract": file_meta(FROZEN_JUDGE / "run_contract.json"),
            "existing_no_selector_bundle": file_meta(NO_SELECTOR / "TEST60_four_method_selector_ablation_research_bundle.zip"),
            "existing_publication_bundle": file_meta(PUBLICATION / "corrected_TEST60_GPT41_pointwise_publication_research_bundle.zip"),
            "existing_test_control_generations": file_meta(CONTROL_SOURCE, len(read_jsonl(CONTROL_SOURCE))),
            "valid_component_manifests": valid_manifests,
            "sft_generation_manifest": file_meta(SFT_VALID_MANIFEST),
        },
        "frozen_inputs": {
            "ordered_ids_newline_sha256": sha256_text("\n".join(ordered_ids) + "\n"),
            "test60_sft": file_meta(inputs / "test60_sft_frozen.jsonl", 60),
            "main_called18": file_meta(inputs / "test60_main_called18.jsonl", 18),
            "main_gateoff42": file_meta(inputs / "test60_main_gateoff42.jsonl", 42),
            "response_hashes": file_meta(inputs / "corrected_test60_response_hashes.jsonl", 60),
        },
        "generation": {
            "common": common_generation,
            "first_stage_sft_decoding": expected_sft_settings,
            "acceptance_policy_required": "surface_relaxed_v1",
            "base_prompt_renderer": "selective_risk_refinement_utils.build_sft_prompt",
            "denoiser_prompt_renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
            "always_on_only_change": {"gate_risk_threshold": -1.0},
            "arms_from_valid_frozen_manifests": {
                arm: {
                    "risk_adapter_dir": data["settings"]["risk_adapter_dir"],
                    "component_mode": data["settings"]["component_mode"],
                    "zt_strategy": data["settings"]["zt_strategy"],
                    "risk_threshold": data["settings"]["risk_threshold"],
                    "gate_risk_threshold": data["settings"]["gate_risk_threshold"],
                    "mask_threshold": data["settings"]["mask_threshold"],
                }
                for arm, data in valid_manifests.items()
            },
        },
        "models": {
            "base_model": {"path": str(BASE_MODEL.resolve())},
            "sft_adapter": adapter_meta(CKPT / "sft/final"),
            "dpo_adapter": adapter_meta(CKPT / "dpo/final"),
            "mask_off_adapter": adapter_meta(CKPT / "proposed_mask_off/final"),
            "mask_on_adapter": adapter_meta(CKPT / "proposed_mask_on/final"),
            "without_router_adapter": adapter_meta(CKPT / "proposed_without_router/final"),
            "without_scorer_adapter": adapter_meta(CKPT / "proposed_without_scorer/final"),
            "router": {"path": str(ROUTER.resolve()), "model": file_meta(ROUTER / "model.safetensors")},
            "scorer": {"path": str(SCORER.resolve()), "model": file_meta(SCORER / "model.safetensors")},
        },
        "judge_and_selector": {
            "contract": judge_contract,
            "same_gpt_model_for_selector_and_pointwise_disclosure_required": True,
            "local_auxiliary_judge": "meta-llama/Llama-3.1-8B-Instruct",
        },
        "code": {
            "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "inference_worktree_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True).strip(),
            "artifacts": {str(path): file_meta(path) for path in script_paths},
        },
    }
    write_json(ROOT / "run_contract.json", contract)
    (ROOT / "run_contract.sha256").write_text(
        f"{sha256_file(ROOT / 'run_contract.json')}  run_contract.json\n", encoding="utf-8"
    )
    shutil.copy2(REPO / "scripts/prepare_acl_component_ablation_matrix.py", ROOT / "code")
    print(
        json.dumps(
            {
                "status": "prepared",
                "root": str(ROOT),
                "called": len(called),
                "gate_off": len(gate_off),
                "control_reused": len(reusable_control_ids),
                "control_new": len(missing_control_ids),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
