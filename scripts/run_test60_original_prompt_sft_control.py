#!/usr/bin/env python3
"""Complete the SFT-adapter control for all 18 frozen TEST60 denoiser calls."""

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
from peft import PeftModel

sys.path.insert(0, "/home/user/hsoh/mh-denoise-surface-relaxed-v1-20261008/scripts")

from fullpaper_backbone_utils import load_fullpaper_tokenizer
from fullpaper_risk_contract import require_fullpaper_axis_order
from run_gemma_selective_risk_refinement import load_base_model, load_classifier, response_metrics
from selective_risk_refinement_utils import (
    DIMS,
    build_risk_tune_prompt,
    configure_risk_contract,
    generate_response,
    score_candidate,
)


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
INPUT = ROOT / "inputs/test60_control_called18.jsonl"
REUSED = ROOT / "generation/original_prompt_sft_control_reused8.jsonl"
OUTPUT = ROOT / "generation/original_prompt_sft_control_test18.jsonl"
MANIFEST = OUTPUT.with_suffix(".manifest.json")
BASE_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
CHECKPOINT_ROOT = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/gemma"
)
SFT_ADAPTER = CHECKPOINT_ROOT / "sft/final"
RISK_ADAPTER = CHECKPOINT_ROOT / "proposed_mask_off/final"
ROUTER = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/router/final"
)
SCORER = Path(
    "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
    "arr_revision2_gemma_first_seed20260910/scorer/final"
)
MAX_SOURCE_LEN = 4096
MAX_NEW_TOKENS = 1536
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.15
NO_REPEAT_NGRAM_SIZE = 4
ROUTER_MAX_LEN = 512
RISK_MAX_LEN = 384


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


def token_ids_hash(values: list[int]) -> str:
    return sha256_text(json.dumps(values, separators=(",", ":")))


def prompt_and_provenance(case: dict[str, Any], tokenizer) -> tuple[str, dict[str, Any]]:
    risk_row = {
        **case,
        "sft_response": case["sft_response"],
        "z_t_from_sft": case["z_t_from_sft"],
        "g_sft": case["g_sft"],
        "aspect_conditioning_status": case.get("aspect_conditioning_status", "available"),
        "risk_component_mode": case.get("risk_component_mode", "full"),
    }
    prompt = build_risk_tune_prompt(tokenizer, risk_row)
    untruncated = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    used = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=MAX_SOURCE_LEN,
    )["input_ids"]
    return prompt, {
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


def normalize_reused(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    if row["question"] != case["question"] or row["sft_response"] != case["sft_response"]:
        raise RuntimeError(f"Reusable control input mismatch for {case['canonical_id']}")
    if row["original_candidate"] != case["denoiser_response"]:
        raise RuntimeError(f"Reusable original candidate mismatch for {case['canonical_id']}")
    return {
        **row,
        "version": VERSION,
        "split": "TEST",
        "source_order": int(case["source_order"]),
        "case_id": case["case_id"],
        "reuse_provenance": {
            "reused": True,
            "source": str(REUSED.resolve()),
            "source_version": row.get("version"),
        },
    }


def main() -> None:
    started = time.monotonic()
    cases = read_jsonl(INPUT)
    if len(cases) != 18 or sum(row.get("used_denoiser") is True for row in cases) != 18:
        raise RuntimeError("Expected exactly 18 frozen called cases")
    if len({row["canonical_id"] for row in cases}) != 18:
        raise RuntimeError("Called cases contain duplicate canonical IDs")

    reused_by_id = {row["canonical_id"]: row for row in read_jsonl(REUSED)}
    history = read_jsonl(OUTPUT)
    existing = {row["canonical_id"]: row for row in history}
    for case in cases:
        row_id = case["canonical_id"]
        if row_id not in existing and row_id in reused_by_id:
            normalized = normalize_reused(reused_by_id[row_id], case)
            append_jsonl(OUTPUT, normalized)
            existing[row_id] = normalized
    pending = [row for row in cases if row["canonical_id"] not in existing]
    if len(existing) + len(pending) != 18:
        raise RuntimeError("Control checkpoint accounting failed")
    if not pending and MANIFEST.exists():
        print(MANIFEST.read_text(encoding="utf-8"), flush=True)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    configure_risk_contract("fullpaper_v1")
    focus_idx = DIMS.index("medical_boundary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router_tok, router = load_classifier(str(ROUTER), device)
    risk_tok, risk_model = load_classifier(str(SCORER), device)
    require_fullpaper_axis_order(router.config)
    require_fullpaper_axis_order(risk_model.config)
    tokenizer = load_fullpaper_tokenizer(str(SFT_ADAPTER))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = load_base_model(str(BASE_MODEL), use_4bit=torch.cuda.is_available())
    model = PeftModel.from_pretrained(base, str(SFT_ADAPTER), adapter_name="sft")
    model.load_adapter(str(RISK_ADAPTER), adapter_name="risk")
    model.eval()

    for case in pending:
        prompt, provenance = prompt_and_provenance(case, tokenizer)
        source_generation = case.get("denoiser_generation") or {}
        source_tokens = source_generation.get("source_input_tokens_used")
        if source_tokens is not None and provenance["source_input_tokens_used"] != source_tokens:
            raise RuntimeError(
                f"Reconstructed prompt token count differs for {case['canonical_id']}: "
                f"{provenance['source_input_tokens_used']} != {source_tokens}"
            )
        if source_generation.get("max_source_len") not in (None, MAX_SOURCE_LEN):
            raise RuntimeError(f"max_source_len mismatch for {case['canonical_id']}")
        if source_generation.get("max_new_tokens") not in (None, MAX_NEW_TOKENS):
            raise RuntimeError(f"max_new_tokens mismatch for {case['canonical_id']}")
        model.set_adapter("sft")
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
        score = score_candidate(
            case["question"],
            response,
            router,
            router_tok,
            risk_model,
            risk_tok,
            device,
            router_max_len=ROUTER_MAX_LEN,
            risk_max_len=RISK_MAX_LEN,
            component_mode="full",
        )
        metrics = response_metrics(
            case["question"],
            response,
            score,
            focus_idx,
            sft_word_count=case.get("sft_word_count"),
        )
        result = {
            "version": VERSION,
            "split": "TEST",
            "source_order": int(case["source_order"]),
            "case_id": case["case_id"],
            "canonical_id": case["canonical_id"],
            "question": case["question"],
            "sft_response": case["sft_response"],
            "original_candidate": case["denoiser_response"],
            "original_candidate_raw": case["denoiser_response_raw"],
            "original_generation": source_generation,
            "original_metrics_stored": {
                "risk_score": case.get("denoiser_risk_score"),
                "focus_risk_score": case.get("denoiser_focus_risk_score"),
                "word_count": case.get("denoiser_word_count"),
                "specificity_ratio": case.get("specificity_ratio"),
                "bad_safety_count": case.get("denoiser_bad_safety_count"),
                "generic_count": case.get("denoiser_generic_count"),
                "keyword_overlap": case.get("denoiser_keyword_overlap"),
            },
            "control_arm": "original_prompt_sft_control",
            "control_adapter": str(SFT_ADAPTER.resolve()),
            "control_raw_decoded_output": raw,
            "control_response": response,
            "control_generation": generation,
            "control_metrics": metrics,
            "technical_valid": bool(response.strip()) and generation["eos_reached"] and not generation["source_input_truncated"],
            "prompt_provenance": provenance,
            "arm_input_hash_comparison": {
                "original_denoiser_rendered_prompt_sha256": provenance["rendered_prompt_sha256"],
                "original_prompt_sft_control_rendered_prompt_sha256": provenance["rendered_prompt_sha256"],
                "rendered_prompt_equal": True,
                "original_denoiser_input_token_ids_used_sha256": provenance["input_token_ids_used_sha256"],
                "original_prompt_sft_control_input_token_ids_used_sha256": provenance["input_token_ids_used_sha256"],
                "input_token_ids_equal": True,
            },
            "sft_metrics_stored": {
                "risk_score": case.get("sft_risk_score"),
                "focus_risk_score": case.get("sft_focus_risk_score"),
                "word_count": case.get("sft_word_count"),
                "bad_safety_count": case.get("sft_bad_safety_count"),
                "generic_count": case.get("sft_generic_count"),
                "keyword_overlap": case.get("sft_keyword_overlap"),
            },
            "call_reason": case["denoiser_call_reason"],
            "reuse_provenance": {"reused": False},
        }
        append_jsonl(OUTPUT, result)
        existing[case["canonical_id"]] = result
        print(
            json.dumps(
                {
                    "arm": "original_prompt_sft_control",
                    "case_id": case["case_id"],
                    "completed": len(existing),
                    "total": 18,
                    "technical_valid": result["technical_valid"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ordered = [existing[row["canonical_id"]] for row in cases]
    write_jsonl(OUTPUT, ordered)
    manifest = {
        "version": VERSION,
        "status": "complete" if len(ordered) == 18 else "partial",
        "arm": "OriginalPromptSFTControl",
        "rows": len(ordered),
        "reused": sum(bool((row.get("reuse_provenance") or {}).get("reused")) for row in ordered),
        "new": sum(not bool((row.get("reuse_provenance") or {}).get("reused")) for row in ordered),
        "technical_valid": sum(bool(row["technical_valid"]) for row in ordered),
        "all_rendered_prompt_hashes_equal_between_arms": all(
            row["arm_input_hash_comparison"]["rendered_prompt_equal"] for row in ordered
        ),
        "all_input_token_hashes_equal_between_arms": all(
            row["arm_input_hash_comparison"]["input_token_ids_equal"] for row in ordered
        ),
        "input": {"path": str(INPUT.resolve()), "sha256": sha256_file(INPUT), "rows": len(cases)},
        "output": {"path": str(OUTPUT.resolve()), "sha256": sha256_file(OUTPUT), "rows": len(ordered)},
        "settings": {
            "base_model": str(BASE_MODEL.resolve()),
            "sft_adapter": str(SFT_ADAPTER.resolve()),
            "original_denoiser_adapter": str(RISK_ADAPTER.resolve()),
            "changed_variable": "generation adapter only",
            "prompt_renderer": "selective_risk_refinement_utils.build_risk_tune_prompt",
            "max_source_len": MAX_SOURCE_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "repetition_penalty": REPETITION_PENALTY,
            "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
            "load_in_4bit": torch.cuda.is_available(),
        },
        "generation_completion": {
            "eos_reached": sum(bool(row["control_generation"]["eos_reached"]) for row in ordered),
            "length_limit_reached": sum(bool(row["control_generation"]["length_limit_reached"]) for row in ordered),
            "source_input_truncated": sum(bool(row["control_generation"]["source_input_truncated"]) for row in ordered),
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
    atomic_json(MANIFEST, manifest)
    del model, base, tokenizer, router, router_tok, risk_model, risk_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
