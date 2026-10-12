#!/usr/bin/env python3
"""Run auxiliary Llama pointwise evaluation for VALID43 Base-draft controls."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from run_base_first_residual_valid_analysis import ROOT, read_json, read_jsonl, write_json, write_jsonl


VERSION = "base-first-residual-valid-v1-20261011"
MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
FEEDBACK_SCRIPTS = Path("/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/scripts")
sys.path.insert(0, str(FEEDBACK_SCRIPTS))

from run_valid_feedback_v1 import generation_eos_ids, load_local_generator, parse_json_object, render_chat


BATCH_SIZE = int(os.environ.get("BASE_DRAFT_LLAMA_BATCH_SIZE", "4"))
MAX_INPUT_TOKENS = 16384
CACHE_ROOTS = (
    Path("/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1"),
    Path("/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1"),
    Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1"),
    Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1"),
    Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1"),
    ROOT,
)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_key(row: dict[str, Any]) -> str:
    material = {
        "parser": row.get("parser"),
        "system_prompt": row.get("system_prompt"),
        "user_prompt": row.get("user_prompt"),
        "local_model": str(MODEL),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": row.get("max_tokens"),
    }
    return sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(payload, dict):
        return None, "no_valid_json_object"
    scores = {}
    errors = []
    for axis in ("overall_quality", "empathy", "specificity"):
        value = payload.get(axis)
        try:
            value = int(float(value))
        except (TypeError, ValueError):
            value = None
        if value is None or not 1 <= value <= 5:
            errors.append(f"invalid_{axis}")
        scores[axis] = value
    medical = str(payload.get("medical_advice") or "").strip().lower()
    if medical not in {"yes", "no", "unsure"}:
        errors.append("invalid_medical_advice")
    evidence = str(payload.get("medical_advice_evidence") or "").strip()
    medical_rationale = str(payload.get("medical_advice_rationale") or "").strip()
    if medical in {"yes", "unsure"} and not evidence:
        errors.append("missing_medical_advice_evidence")
    if not medical_rationale:
        errors.append("missing_medical_advice_rationale")
    if errors:
        return None, ",".join(errors)
    return {
        "scores": scores,
        "medical_advice": medical,
        "rationale": str(payload.get("rationale") or "").strip(),
        "medical_advice_evidence": evidence,
        "medical_advice_rationale": medical_rationale,
    }, ""


def cache_rows() -> dict[str, tuple[Path, dict[str, Any]]]:
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    seen: set[Path] = set()
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            if path in seen or "llama" not in path.name.lower():
                continue
            seen.add(path)
            try:
                rows = read_jsonl(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for row in rows:
                if row.get("parser") != "pointwise_expert_rubric" or row.get("judge_ok") is not True:
                    continue
                if row.get("judge_model_path") != str(MODEL) or row.get("max_tokens") != 384:
                    continue
                normalized = row.get("judge_normalized")
                valid, _ = normalize(
                    {
                        **((normalized or {}).get("scores") or {}),
                        "medical_advice": (normalized or {}).get("medical_advice"),
                        "rationale": (normalized or {}).get("rationale"),
                        "medical_advice_evidence": (normalized or {}).get("medical_advice_evidence"),
                        "medical_advice_rationale": (normalized or {}).get("medical_advice_rationale"),
                    }
                )
                if valid is not None:
                    chosen.setdefault(local_key(row), (path, row))
    return chosen


def prepare() -> None:
    requests = read_jsonl(ROOT / "requests/second_pass_expected_pointwise.jsonl")
    if not requests or any(row.get("parser") != "pointwise_expert_rubric" for row in requests):
        raise RuntimeError("Second-pass pointwise requests are missing")
    cache = cache_rows()
    reused = []
    new = []
    for request in requests:
        cached = cache.get(local_key(request))
        if cached is None:
            new.append(request)
            continue
        path, result = cached
        reused.append(
            {
                "target_request_id": request["request_id"],
                "question_response_sha256": request["metadata"]["question_response_sha256"],
                "source_result_path": str(path),
                "source_request_id": result.get("request_id"),
                "judge_model_path": result.get("judge_model_path"),
                "judge_normalized": result["judge_normalized"],
                "judge_raw": result.get("judge_raw"),
            }
        )
    write_jsonl(ROOT / "results/second_pass_reused_llama.jsonl", reused)
    write_jsonl(ROOT / "requests/second_pass_new_llama.jsonl", new)
    write_json(
        ROOT / "second_pass_llama_contract.json",
        {
            "version": VERSION,
            "status": "prepared",
            "model_path": str(MODEL),
            "batch_size": BATCH_SIZE,
            "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": 384,
            "expected": len(requests),
            "reused_exact": len(reused),
            "new": len(new),
        },
    )
    print(json.dumps(read_jsonl(ROOT / "requests/second_pass_new_llama.jsonl")[:0] or read_json(ROOT / "second_pass_llama_contract.json"), sort_keys=True))


def generate_batch(model, tokenizer, rendered: list[str]) -> list[tuple[str, dict[str, Any]]]:
    untruncated = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered]
    encoded = tokenizer(rendered, return_tensors="pt", padding=True, truncation=True, max_length=MAX_INPUT_TOKENS, add_special_tokens=False).to(model.device)
    eos_ids = generation_eos_ids(model, tokenizer)
    kwargs = {**encoded, "max_new_tokens": 384, "do_sample": False, "use_cache": True, "pad_token_id": tokenizer.pad_token_id}
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    elapsed = time.monotonic() - started
    width = int(encoded["input_ids"].shape[-1])
    results = []
    for index, sequence in enumerate(output):
        token_ids = [int(value) for value in sequence[width:].detach().cpu().tolist()]
        results.append(
            (
                tokenizer.decode(token_ids, skip_special_tokens=True),
                {
                    "batch_size": len(rendered),
                    "batch_elapsed_seconds": round(elapsed, 6),
                    "source_input_tokens_untruncated": untruncated[index],
                    "source_input_tokens_used": int(encoded["attention_mask"][index].sum().item()),
                    "source_input_truncated": untruncated[index] > MAX_INPUT_TOKENS,
                    "generated_tokens_including_padding": len(token_ids),
                    "eos_token_ids": eos_ids,
                    "eos_reached": any(value in set(eos_ids) for value in token_ids),
                    "max_new_tokens": 384,
                },
            )
        )
    return results


def run() -> None:
    request_path = ROOT / "requests/second_pass_new_llama.jsonl"
    output_path = ROOT / "results/second_pass_new_llama.jsonl"
    requests = read_jsonl(request_path)
    request_by_id = {row["request_id"]: row for row in requests}
    latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    reusable = {
        request_id
        for request_id, row in latest.items()
        if request_id in request_by_id and row.get("judge_ok") is True and row.get("local_request_key") == local_key(request_by_id[request_id])
    }
    pending = [row for row in requests if row["request_id"] not in reusable]
    print(json.dumps({"queue": "second_pass_llama", "total": len(requests), "reused": len(reusable), "pending": len(pending)}), flush=True)
    if not pending:
        return
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_local_generator(MODEL)
    started = time.monotonic()
    completed = 0
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        rendered = [render_chat(tokenizer, row["system_prompt"], row["user_prompt"]) for row in batch]
        generations = generate_batch(model, tokenizer, rendered)
        for request, rendered_prompt, (raw, generation) in zip(batch, rendered, generations):
            payload, parse_error = parse_json_object(raw)
            normalized, normalize_error = normalize(payload)
            append_jsonl(
                output_path,
                {
                    **request,
                    "local_request_key": local_key(request),
                    "judge_model_path": str(MODEL),
                    "rendered_prompt_sha256": sha256_text(rendered_prompt),
                    "judge_raw": raw,
                    "judge_payload": payload,
                    "judge_normalized": normalized,
                    "judge_ok": normalized is not None,
                    "judge_error": normalize_error or parse_error,
                    "generation": generation,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            completed += 1
            print(json.dumps({"queue": "second_pass_llama", "completed": completed, "total": len(pending), "judge_ok": normalized is not None}), flush=True)
    latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    relevant = [latest[row["request_id"]] for row in requests if row["request_id"] in latest]
    write_json(
        output_path.with_suffix(".manifest.json"),
        {
            "status": "complete" if len(relevant) == len(requests) else "partial",
            "requests": len(requests),
            "results": len(relevant),
            "judge_ok": sum(row.get("judge_ok") is True for row in relevant),
            "judge_failed": sum(row.get("judge_ok") is not True for row in relevant),
            "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
            "model_path": str(MODEL),
            "batch_size": BATCH_SIZE,
            "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        },
    )
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def aggregate() -> None:
    assignments = read_jsonl(ROOT / "inputs/second_pass_pointwise_assignments.jsonl")
    requests = read_jsonl(ROOT / "requests/second_pass_expected_pointwise.jsonl")
    results = {}
    for row in read_jsonl(ROOT / "results/second_pass_reused_llama.jsonl"):
        results[row["target_request_id"]] = {"ok": True, "normalized": row["judge_normalized"], "provenance": "reused_exact"}
    for row in read_jsonl(ROOT / "results/second_pass_new_llama.jsonl"):
        results[row["request_id"]] = {"ok": row.get("judge_ok") is True, "normalized": row.get("judge_normalized"), "provenance": "new_local"}
    request_by_hash = {row["metadata"]["question_response_sha256"]: row for row in requests}
    scored = []
    failures = []
    for row in assignments:
        request = request_by_hash[row["question_response_sha256"]]
        result = results.get(request["request_id"])
        if result is None or not result["ok"]:
            failures.append({"canonical_id": row["canonical_id"], "method": row["method"]})
            continue
        normalized = result["normalized"]
        scores = normalized["scores"]
        scored.append(
            {
                **row,
                "overall_quality": scores["overall_quality"],
                "empathy": scores["empathy"],
                "specificity": scores["specificity"],
                "quality_3axis_mean": statistics.mean(scores.values()),
                "medical_advice": normalized["medical_advice"],
                "evaluation_provenance": result["provenance"],
            }
        )
    write_jsonl(ROOT / "results/second_pass_llama_scores.jsonl", scored)
    write_jsonl(ROOT / "results/second_pass_llama_failures.jsonl", failures)
    summary = []
    for method in sorted({row["method"] for row in scored}):
        rows = [row for row in scored if row["method"] == method]
        medical = Counter(row["medical_advice"] for row in rows)
        summary.append(
            {
                "method": method,
                "N": len(rows),
                "failures": 43 - len(rows),
                "overall_quality": statistics.mean(row["overall_quality"] for row in rows),
                "empathy": statistics.mean(row["empathy"] for row in rows),
                "specificity": statistics.mean(row["specificity"] for row in rows),
                "quality_3axis_mean": statistics.mean(row["quality_3axis_mean"] for row in rows),
                "medical_yes": medical["yes"],
                "medical_no": medical["no"],
                "medical_unsure": medical["unsure"],
            }
        )
    write_jsonl(ROOT / "tables/second_pass_llama_candidate_quality.jsonl", summary)

    selector = read_jsonl(ROOT / "results/second_pass_pairwise_consensus.jsonl")
    by_method_id = {(row["method"], row["canonical_id"]): row for row in scored}
    final_summary = []
    for candidate_method in ("BaseSelfRefineCandidate", "TrainedDenoiserOnBaseCandidate"):
        decisions = {
            row["canonical_id"]: row
            for row in selector
            if row["left_method"] == "BasePromptRewrite" and row["right_method"] == candidate_method
        }
        rows = []
        for canonical_id, decision in decisions.items():
            chosen_method = candidate_method if decision["outcome"] == "right_wins_consensus" else "BasePromptRewrite"
            chosen = by_method_id.get((chosen_method, canonical_id))
            if chosen:
                rows.append(chosen)
        if rows:
            final_summary.append(
                {
                    "method": candidate_method.replace("Candidate", "AllGeneratedGPTSelected"),
                    "N": len(rows),
                    "pointwise_failures": 43 - len(rows),
                    "quality_3axis_mean": statistics.mean(row["quality_3axis_mean"] for row in rows),
                }
            )
    write_json(
        ROOT / "VALID43_BASE_DRAFT_CONTROL_LLAMA_RESULTS.json",
        {
            "version": VERSION,
            "status": "complete" if not failures else "complete_with_failures",
            "candidate_quality": summary,
            "gpt_selected_final_quality_under_llama": final_summary,
            "failures": failures,
            "limitation": "Llama is an auxiliary pointwise judge; GPT selector choices are not reused as Llama quality scores.",
        },
    )
    print(json.dumps(read_json(ROOT / "VALID43_BASE_DRAFT_CONTROL_LLAMA_RESULTS.json"), ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run", "aggregate"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        run()
    else:
        aggregate()
