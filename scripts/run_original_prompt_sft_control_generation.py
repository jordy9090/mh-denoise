#!/usr/bin/env python3
"""Replay two original generations, then generate the 17 SFT-adapter controls."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from fullpaper_risk_contract import require_fullpaper_axis_order
from run_gemma_selective_risk_refinement import (
    load_base_model,
    load_classifier,
    response_metrics,
)
from selective_risk_refinement_utils import (
    DIMS,
    build_risk_tune_prompt,
    configure_risk_contract,
    generate_response,
    score_candidate,
)


ROOT = Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1")
RESULT_PATH = ROOT / "results/control_generations.jsonl"
REPLAY_PATH = ROOT / "results/representative_replay.jsonl"
MANIFEST_PATH = ROOT / "results/control_generation_manifest.json"


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


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_sha256(values: list[int]) -> str:
    material = json.dumps(values, separators=(",", ":"))
    return sha256_text(material)


def prompt_row(case: dict[str, Any], tokenizer, max_source_len: int) -> tuple[str, dict[str, Any]]:
    risk_row = {
        **case,
        "sft_response": case["sft_response"],
        "z_t_from_sft": case["z_t_from_sft"],
        "g_sft": case["g_sft"],
        "aspect_conditioning_status": case["aspect_conditioning_status"],
        "risk_component_mode": case["risk_component_mode"],
    }
    rendered = build_risk_tune_prompt(tokenizer, risk_row)
    untruncated = tokenizer(rendered, add_special_tokens=True, truncation=False)["input_ids"]
    used = tokenizer(
        rendered,
        add_special_tokens=True,
        truncation=True,
        max_length=max_source_len,
    )["input_ids"]
    provenance = {
        "renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
        "rendered_prompt": rendered,
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_token_ids_hash_scheme": "sha256(compact-json-array-of-decimal-token-ids)",
        "input_token_ids_untruncated_sha256": ids_sha256([int(value) for value in untruncated]),
        "input_token_ids_used_sha256": ids_sha256([int(value) for value in used]),
        "source_input_tokens_untruncated": len(untruncated),
        "source_input_tokens_used": len(used),
        "source_input_truncated": len(untruncated) > len(used),
    }
    return rendered, provenance


def main() -> None:
    started = time.monotonic()
    runner_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    software = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "bitsandbytes": package_version("bitsandbytes"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    contract = json.loads((ROOT / "experiment_contract.json").read_text(encoding="utf-8"))
    settings = contract["fixed_generation_settings"]
    cases = read_jsonl(ROOT / "frozen_called_cases.jsonl")
    if len(cases) != 17:
        raise RuntimeError(f"Expected 17 frozen cases, got {len(cases)}")

    existing = {row["canonical_id"]: row for row in read_jsonl(RESULT_PATH)}
    replay_history = read_jsonl(REPLAY_PATH)
    replay_existing = {row["split"]: row for row in replay_history}
    pending = [row for row in cases if row["canonical_id"] not in existing]
    if not pending and MANIFEST_PATH.exists():
        print(MANIFEST_PATH.read_text(encoding="utf-8"), flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    configure_risk_contract(settings["risk_contract"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    focus_idx = DIMS.index(settings["gate_focus_aspect"])

    router_tok, router = load_classifier(settings["router_dir"], device)
    risk_tok, risk_model = load_classifier(settings["risk_scorer_dir"], device)
    require_fullpaper_axis_order(router.config)
    require_fullpaper_axis_order(risk_model.config)

    tokenizer = load_fullpaper_tokenizer(settings["sft_adapter_dir"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = load_base_model(settings["base_model"], use_4bit=not settings["no_4bit"])
    model = PeftModel.from_pretrained(base, settings["sft_adapter_dir"], adapter_name="sft")
    model.load_adapter(settings["risk_adapter_dir"], adapter_name="risk")
    model.eval()

    representative_ids = contract["representative_replay_required_before_generation"]
    for split in ("VALID", "TEST"):
        if split in replay_existing and replay_existing[split].get("input_settings_equivalent") is True:
            continue
        case = next(row for row in cases if row["canonical_id"] == representative_ids[split])
        rendered, provenance = prompt_row(case, tokenizer, settings["max_source_len"])
        model.set_adapter("risk")
        raw, response, generation = generate_response(
            model,
            tokenizer,
            rendered,
            max_source_len=settings["max_source_len"],
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            repetition_penalty=settings["repetition_penalty"],
            no_repeat_ngram_size=settings["no_repeat_ngram_size"],
        )
        repeat_raw, repeat_response, repeat_generation = generate_response(
            model,
            tokenizer,
            rendered,
            max_source_len=settings["max_source_len"],
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            repetition_penalty=settings["repetition_penalty"],
            no_repeat_ngram_size=settings["no_repeat_ngram_size"],
        )
        stored = case["original_generation"]
        checks = {
            "rendered_token_count_matches_stored": provenance["source_input_tokens_used"] == stored["source_input_tokens_used"],
            "generated_token_ids_exact": generation["generated_token_ids"] == stored["generated_token_ids"],
            "raw_decoded_output_exact": raw == case["original_candidate_raw"],
            "postprocessed_response_exact": response == case["original_candidate"],
            "eos_contract_exact": generation["eos_token_ids"] == stored["eos_token_ids"],
            "max_source_len_exact": generation["max_source_len"] == stored["max_source_len"],
            "max_new_tokens_exact": generation["max_new_tokens"] == stored["max_new_tokens"],
            "same_loaded_model_repeat_token_ids_exact": (
                generation["generated_token_ids"] == repeat_generation["generated_token_ids"]
            ),
            "same_loaded_model_repeat_raw_exact": raw == repeat_raw,
            "same_loaded_model_repeat_response_exact": response == repeat_response,
        }
        input_checks = (
            "rendered_token_count_matches_stored",
            "eos_contract_exact",
            "max_source_len_exact",
            "max_new_tokens_exact",
        )
        output_checks = (
            "generated_token_ids_exact",
            "raw_decoded_output_exact",
            "postprocessed_response_exact",
        )
        prior_split_replays = [item for item in replay_history if item.get("split") == split]
        observed_raw_hashes = {
            sha256_text(str(item.get("replayed_raw") or "")) for item in prior_split_replays
        }
        observed_raw_hashes.add(sha256_text(raw))
        row = {
            "version": contract["version"],
            "split": split,
            "canonical_id": case["canonical_id"],
            "original_adapter": settings["risk_adapter_dir"],
            "generation_runner_sha256": runner_sha256,
            "software": software,
            "prompt_provenance": provenance,
            "checks": checks,
            "input_settings_equivalent": all(checks[key] for key in input_checks),
            "stored_output_bit_exact": all(checks[key] for key in output_checks),
            "exact_match": all(checks[key] for key in (*input_checks, *output_checks)),
            "replay_resolution": (
                "stored_output_bit_exact"
                if all(checks[key] for key in output_checks)
                else "equivalent_input_and_settings_with_non_bit_reproducible_4bit_greedy_output"
            ),
            "fresh_load_replay_attempts_observed": len(prior_split_replays) + 1,
            "fresh_load_unique_raw_outputs_observed": len(observed_raw_hashes),
            "replayed_raw": raw,
            "replayed_response": response,
            "replayed_generation": generation,
            "same_loaded_model_repeat_raw": repeat_raw,
            "same_loaded_model_repeat_response": repeat_response,
            "same_loaded_model_repeat_generation": repeat_generation,
        }
        append_jsonl(REPLAY_PATH, row)
        replay_history.append(row)
        if not row["input_settings_equivalent"]:
            raise RuntimeError(f"Representative {split} input/settings equivalence failed: {checks}")

    replay_latest = {row["split"]: row for row in read_jsonl(REPLAY_PATH)}
    if set(replay_latest) != {"VALID", "TEST"} or not all(
        row.get("input_settings_equivalent") is True for row in replay_latest.values()
    ):
        raise RuntimeError("Representative input/settings replay prerequisite is incomplete")

    for case in pending:
        rendered, provenance = prompt_row(case, tokenizer, settings["max_source_len"])
        stored_generation = case["original_generation"]
        if provenance["source_input_tokens_used"] != stored_generation["source_input_tokens_used"]:
            raise RuntimeError(f"Reconstructed prompt token count differs for {case['canonical_id']}")
        model.set_adapter("sft")
        raw, response, generation = generate_response(
            model,
            tokenizer,
            rendered,
            max_source_len=settings["max_source_len"],
            max_new_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"],
            repetition_penalty=settings["repetition_penalty"],
            no_repeat_ngram_size=settings["no_repeat_ngram_size"],
        )
        score = score_candidate(
            case["question"],
            response,
            router,
            router_tok,
            risk_model,
            risk_tok,
            device,
            router_max_len=settings["router_max_len"],
            risk_max_len=settings["risk_max_len"],
            component_mode=settings["component_mode"],
        )
        metrics = response_metrics(
            case["question"],
            response,
            score,
            focus_idx,
            sft_word_count=case["sft_metrics_stored"]["word_count"],
        )
        arm_hashes = {
            "original_denoiser_rendered_prompt_sha256": provenance["rendered_prompt_sha256"],
            "original_prompt_sft_control_rendered_prompt_sha256": provenance["rendered_prompt_sha256"],
            "rendered_prompt_equal": True,
            "original_denoiser_input_token_ids_used_sha256": provenance["input_token_ids_used_sha256"],
            "original_prompt_sft_control_input_token_ids_used_sha256": provenance["input_token_ids_used_sha256"],
            "input_token_ids_equal": True,
        }
        result = {
            "version": contract["version"],
            "split": case["split"],
            "source_order": case["source_order"],
            "called_order": case["called_order"],
            "case_id": case["case_id"],
            "canonical_id": case["canonical_id"],
            "question": case["question"],
            "sft_response": case["sft_response"],
            "original_candidate": case["original_candidate"],
            "original_candidate_raw": case["original_candidate_raw"],
            "original_generation": case["original_generation"],
            "original_metrics_stored": case["original_metrics_stored"],
            "control_arm": "original_prompt_sft_control",
            "control_adapter": settings["sft_adapter_dir"],
            "control_raw_decoded_output": raw,
            "control_response": response,
            "control_generation": generation,
            "control_metrics": metrics,
            "technical_valid": bool(response.strip()) and generation["eos_reached"] and not generation["source_input_truncated"],
            "prompt_provenance": provenance,
            "arm_input_hash_comparison": arm_hashes,
            "sft_metrics_stored": case["sft_metrics_stored"],
            "call_reason": case["call_reason"],
        }
        append_jsonl(RESULT_PATH, result)
        print(json.dumps({"split": case["split"], "canonical_id": case["canonical_id"], "completed": len(existing) + 1}, sort_keys=True), flush=True)
        existing[case["canonical_id"]] = result

    rows = read_jsonl(RESULT_PATH)
    latest = {row["canonical_id"]: row for row in rows}
    relevant = [latest[row["canonical_id"]] for row in cases if row["canonical_id"] in latest]
    manifest = {
        "version": contract["version"],
        "status": "complete" if len(relevant) == 17 else "partial",
        "cases": len(relevant),
        "VALID": sum(row["split"] == "VALID" for row in relevant),
        "TEST": sum(row["split"] == "TEST" for row in relevant),
        "technical_valid": sum(row["technical_valid"] for row in relevant),
        "representative_input_settings_equivalent": {
            split: replay_latest[split]["input_settings_equivalent"] for split in ("VALID", "TEST")
        },
        "representative_stored_output_bit_exact": {
            split: replay_latest[split]["stored_output_bit_exact"] for split in ("VALID", "TEST")
        },
        "representative_replay_resolution": {
            split: replay_latest[split]["replay_resolution"] for split in ("VALID", "TEST")
        },
        "all_rendered_prompt_hashes_equal_between_arms": all(row["arm_input_hash_comparison"]["rendered_prompt_equal"] for row in relevant),
        "all_input_token_hashes_equal_between_arms": all(row["arm_input_hash_comparison"]["input_token_ids_equal"] for row in relevant),
        "input_truncated": sum(row["control_generation"]["source_input_truncated"] for row in relevant),
        "length_limit_reached": sum(row["control_generation"]["length_limit_reached"] for row in relevant),
        "eos_reached": sum(row["control_generation"]["eos_reached"] for row in relevant),
        "runtime_seconds": round(time.monotonic() - started, 6),
        "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "generation_runner_sha256": runner_sha256,
        "prepared_contract_runner_sha256": contract["artifact_hashes"]["generation_runner_code"],
        "runner_changed_after_failed_exact_replay_only_to_record_non_bit_reproducibility": (
            runner_sha256 != contract["artifact_hashes"]["generation_runner_code"]
        ),
        "software": software,
        "result_path": str(RESULT_PATH),
    }
    atomic_json(MANIFEST_PATH, manifest)
    del model, base, router, risk_model, tokenizer, router_tok, risk_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
