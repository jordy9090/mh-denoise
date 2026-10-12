#!/usr/bin/env python3
"""Read-only integrity checks for the generated TEST60 component arms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
EXPECTED = {
    "always_on_missing42": {
        "path": ROOT / "generation/main_always_on_missing42.jsonl",
        "rows": 42,
        "adapter_suffix": "/proposed_mask_off/final",
        "component_mode": "full",
        "zt_strategy": "no_mask",
        "risk_threshold": 0.7215308056776166,
        "gate_risk_threshold": -1.0,
    },
    "mask_on": {
        "path": ROOT / "generation/proposed_mask_on_test60.jsonl",
        "rows": 60,
        "adapter_suffix": "/proposed_mask_on/final",
        "component_mode": "full",
        "zt_strategy": "staged_risk",
        "risk_threshold": 0.7215308056776166,
        "gate_risk_threshold": 0.7215308056776166,
    },
    "without_router": {
        "path": ROOT / "generation/proposed_without_router_test60.jsonl",
        "rows": 60,
        "adapter_suffix": "/proposed_without_router/final",
        "component_mode": "without_router",
        "zt_strategy": "staged_risk",
        "risk_threshold": 0.948689877986908,
        "gate_risk_threshold": 0.948689877986908,
    },
    "without_scorer": {
        "path": ROOT / "generation/proposed_without_scorer_test60.jsonl",
        "rows": 60,
        "adapter_suffix": "/proposed_without_scorer/final",
        "component_mode": "without_scorer",
        "zt_strategy": "no_mask",
        "risk_threshold": 0.7560610771169199,
        "gate_risk_threshold": 0.7560610771169199,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    contract = read_json(ROOT / "run_contract.json")
    for key in ("corrected_sft60", "corrected_proposed_generation60", "corrected_final60"):
        source = contract["frozen_sources"][key]
        path = Path(source["path"])
        require(path.exists(), f"Frozen source missing: {path}")
        require(sha256_file(path) == source["sha256"], f"Frozen source changed: {path}")

    frozen = read_jsonl(ROOT / "inputs/test60_sft_frozen.jsonl")
    ids = [row["canonical_id"] for row in frozen]
    require(len(ids) == 60 and len(set(ids)) == 60, "Frozen TEST60 IDs are incomplete or duplicated")
    require([int(row["source_order"]) for row in frozen] == list(range(60)), "Frozen order is not 0..59")

    base_path = ROOT / "generation/base_prompt_rewrite_test60.jsonl"
    base = read_jsonl(base_path)
    require(len(base) == 60, "Base prompt rewrite does not contain 60 rows")
    base_by_id = {row["canonical_id"]: row for row in base}
    require(set(base_by_id) == set(ids), "Base prompt rewrite IDs differ from frozen TEST60")
    base_prompt_count_matches = 0
    for source in frozen:
        row = base_by_id[source["canonical_id"]]
        require(row["question"] == source["question"], f"Base question mismatch: {source['canonical_id']}")
        source_tokens = (source.get("sft_generation") or {}).get("source_input_tokens_used")
        base_tokens = row["prompt_provenance"]["source_input_tokens_used"]
        if source_tokens == base_tokens:
            base_prompt_count_matches += 1
    require(base_prompt_count_matches == 60, "Base/SFT prompt token counts differ")
    base_manifest = read_json(base_path.with_suffix(".manifest.json"))
    require(base_manifest["adapter"] is None, "Base prompt rewrite unexpectedly uses an adapter")
    require(base_manifest["settings"]["max_source_len"] == 1024, "Base max_source_len differs from SFT")
    require(base_manifest["settings"]["max_new_tokens"] == 1536, "Base max_new_tokens differs from SFT")
    base_safe_or_reference_labels = sum(
        "Safe response:" in row["prompt_provenance"]["rendered_prompt"]
        or "Reference response:" in row["prompt_provenance"]["rendered_prompt"]
        for row in base
    )
    require(base_safe_or_reference_labels == 0, "Base prompt contains a safe/reference response field")
    base_exact_safe = sum(row["base_prompt_response"] == row.get("safe_response") for row in base)
    base_exact_unsafe = sum(row["base_prompt_response"] == row.get("unsafe_response") for row in base)

    arm_audits = {}
    for arm, expected in EXPECTED.items():
        path = expected["path"]
        manifest_path = path.with_suffix(".manifest.json")
        rows = read_jsonl(path)
        manifest = read_json(manifest_path)
        require(len(rows) == expected["rows"], f"{arm} row count differs")
        settings = manifest["settings"]
        require(settings["risk_adapter_dir"].endswith(expected["adapter_suffix"]), f"{arm} adapter differs")
        require(settings["component_mode"] == expected["component_mode"], f"{arm} component mode differs")
        require(settings["zt_strategy"] == expected["zt_strategy"], f"{arm} z_t strategy differs")
        require(settings["risk_threshold"] == expected["risk_threshold"], f"{arm} risk threshold differs")
        require(settings["gate_risk_threshold"] == expected["gate_risk_threshold"], f"{arm} gate threshold differs")
        require(settings["acceptance_policy"] == "surface_relaxed_v1", f"{arm} acceptance policy differs")
        require(settings["temperature"] == 0.0, f"{arm} temperature differs")
        require(settings["max_source_len"] == 4096 and settings["max_new_tokens"] == 1536, f"{arm} length contract differs")
        require(manifest["output"]["sha256"] == sha256_file(path), f"{arm} output hash differs from manifest")
        if expected["rows"] == 60:
            require([row["canonical_id"] for row in rows] == ids, f"{arm} TEST60 order differs")
        arm_audits[arm] = {
            "rows": len(rows),
            "denoiser_called": sum(bool(row.get("used_denoiser")) for row in rows),
            "candidate_nonempty": sum(bool(str(row.get("denoiser_response") or "").strip()) for row in rows if row.get("used_denoiser")),
            "candidate_length_limit_reached": sum(bool((row.get("denoiser_generation") or {}).get("length_limit_reached")) for row in rows if row.get("used_denoiser")),
            "candidate_source_input_truncated": sum(bool((row.get("denoiser_generation") or {}).get("source_input_truncated")) for row in rows if row.get("used_denoiser")),
            "output_sha256": sha256_file(path),
            "manifest_sha256": sha256_file(manifest_path),
        }

    control_path = ROOT / "generation/original_prompt_sft_control_test18.jsonl"
    controls = read_jsonl(control_path)
    control_manifest = read_json(control_path.with_suffix(".manifest.json"))
    require(len(controls) == 18, "SFT-adapter control does not contain 18 rows")
    require(control_manifest["reused"] == 8 and control_manifest["new"] == 10, "Control reuse/new counts differ")
    require(control_manifest["all_rendered_prompt_hashes_equal_between_arms"], "Control rendered prompt hashes differ")
    require(control_manifest["all_input_token_hashes_equal_between_arms"], "Control input token hashes differ")

    audit = {
        "status": "complete",
        "frozen_source_hashes_unchanged": True,
        "ordered_test60_ids": 60,
        "base_prompt_rewrite": {
            "rows": len(base),
            "prompt_token_count_matches_sft": base_prompt_count_matches,
            "technical_valid": sum(bool(row.get("technical_valid")) for row in base),
            "eos_reached": sum(bool((row.get("generation") or {}).get("eos_reached")) for row in base),
            "length_limit_reached": sum(bool((row.get("generation") or {}).get("length_limit_reached")) for row in base),
            "adapter_is_none": base_manifest["adapter"] is None,
            "prompt_safe_or_reference_field_labels": base_safe_or_reference_labels,
            "generated_response_exactly_matches_safe_response": base_exact_safe,
            "generated_response_exactly_matches_unsafe_response": base_exact_unsafe,
            "prompt_declared_inputs": ["question", "unsafe_response"],
            "output_sha256": sha256_file(base_path),
        },
        "original_prompt_sft_control": {
            "rows": len(controls),
            "reused": control_manifest["reused"],
            "new": control_manifest["new"],
            "technical_valid": control_manifest["technical_valid"],
            "prompt_hashes_equal": True,
            "input_token_hashes_equal": True,
            "output_sha256": sha256_file(control_path),
        },
        "component_arms": arm_audits,
    }
    output = ROOT / "generation/generation_integrity_audit.json"
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
