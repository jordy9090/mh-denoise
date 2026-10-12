#!/usr/bin/env python3
"""Cache and run the auxiliary Llama pointwise evaluation for the TEST60 matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
FEEDBACK_SCRIPTS = Path("/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/scripts")
sys.path.insert(0, str(FEEDBACK_SCRIPTS))

from run_valid_feedback_v1 import (  # noqa: E402
    generation_eos_ids,
    load_local_generator,
    parse_json_object,
    render_chat,
)


BATCH_SIZE = int(os.environ.get("ACL_LLAMA_BATCH_SIZE", "4"))
MAX_INPUT_TOKENS = 16384
CACHE_ROOTS = (
    Path("/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1"),
    Path("/home/user/hsoh/test60_gated_denoiser_no_selector_ablation_20261009_v1"),
    Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1"),
    Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(value: str) -> str:
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


def cache_rows() -> tuple[dict[str, tuple[Path, dict[str, Any]]], list[dict[str, Any]]]:
    chosen: dict[str, tuple[Path, dict[str, Any]]] = {}
    conflicts = []
    for root in CACHE_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.jsonl")):
            in_results_dir = any(part.lower() == "results" for part in path.parts)
            if "llama" not in path.name.lower() or ("result" not in path.name.lower() and not in_results_dir):
                continue
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
                if valid is None:
                    continue
                key = local_key(row)
                prior = chosen.get(key)
                if prior is not None and prior[1].get("judge_normalized") != normalized:
                    conflicts.append(
                        {
                            "request_key": key,
                            "kept_path": str(prior[0]),
                            "other_path": str(path),
                            "resolution": "kept_first_by_cache_priority",
                        }
                    )
                    continue
                if prior is None:
                    chosen[key] = (path, row)
    return chosen, conflicts


def prepare() -> None:
    requests = read_jsonl(ROOT / "requests/expected_pointwise_requests.jsonl")
    if not requests or any(row.get("parser") != "pointwise_expert_rubric" for row in requests):
        raise RuntimeError("Pointwise request contract is missing or invalid")
    cache, conflicts = cache_rows()
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
                "target_request_key": local_key(request),
                "question_response_sha256": request["metadata"]["question_response_sha256"],
                "source_result_path": str(path),
                "source_request_id": result.get("request_id"),
                "judge_model_path": result.get("judge_model_path"),
                "judge_normalized": result["judge_normalized"],
                "judge_raw": result.get("judge_raw"),
            }
        )
    write_jsonl(ROOT / "results/reused_llama_pointwise_cache.jsonl", reused)
    write_jsonl(ROOT / "results/llama_pointwise_cache_conflicts.jsonl", conflicts)
    write_jsonl(ROOT / "requests/new_llama_pointwise_requests.jsonl", new)
    manifest = {
        "version": VERSION,
        "status": "prepared",
        "model_path": str(MODEL),
        "batch_size": BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": 384,
        "expected": len(requests),
        "reused_exact": len(reused),
        "new": len(new),
        "cache_conflicts_logged": len(conflicts),
    }
    write_json(ROOT / "llama_pointwise_contract.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def generate_batch(model, tokenizer, rendered: list[str], max_new_tokens: int) -> list[tuple[str, dict[str, Any]]]:
    import torch

    untruncated = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        add_special_tokens=False,
    ).to(model.device)
    eos_ids = generation_eos_ids(model, tokenizer)
    kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**kwargs)
    elapsed = time.monotonic() - started
    input_width = int(encoded["input_ids"].shape[-1])
    results = []
    for index, sequence in enumerate(output):
        token_ids = [int(value) for value in sequence[input_width:].detach().cpu().tolist()]
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
                    "max_new_tokens": max_new_tokens,
                },
            )
        )
    return results


def run() -> None:
    import torch

    request_path = ROOT / "requests/new_llama_pointwise_requests.jsonl"
    output_path = ROOT / "results/new_llama_pointwise_results.jsonl"
    requests = read_jsonl(request_path)
    request_by_id = {row["request_id"]: row for row in requests}
    latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    checkpoint_rows_seen = len(latest)
    reusable = {
        request_id
        for request_id, row in latest.items()
        if request_id in request_by_id
        and row.get("judge_ok") is True
        and row.get("local_request_key") == local_key(request_by_id[request_id])
    }
    pending = [row for row in requests if row["request_id"] not in reusable]
    print(json.dumps({"queue": "llama_pointwise", "total": len(requests), "reused": len(reusable), "pending": len(pending)}), flush=True)
    if not pending:
        write_json(
            output_path.with_suffix(".manifest.json"),
            {
                "status": "complete",
                "requests": len(requests),
                "results": len(reusable),
                "judge_ok": len(reusable),
                "judge_failed": 0,
                "same_run_new": 0,
                "reused": len(reusable),
                "model_path": str(MODEL),
                "batch_size_this_invocation": BATCH_SIZE,
                "checkpoint_unique_rows_seen": checkpoint_rows_seen,
                "observed_result_batch_sizes": sorted(
                    {
                        int((row.get("generation") or {}).get("batch_size"))
                        for row in latest.values()
                        if (row.get("generation") or {}).get("batch_size") is not None
                    }
                ),
            },
        )
        return
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_local_generator(MODEL)
    started = time.monotonic()
    completed = 0
    first10_written = False
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        rendered = [render_chat(tokenizer, row["system_prompt"], row["user_prompt"]) for row in batch]
        generations = generate_batch(model, tokenizer, rendered, 384)
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
            print(json.dumps({"queue": "llama_pointwise", "completed_this_run": completed, "pending_total": len(pending), "judge_ok": normalized is not None}), flush=True)
            if completed >= 10 and not first10_written:
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = completed / elapsed * 60
                remaining = len(pending) - completed
                write_json(
                    output_path.with_suffix(".first10.json"),
                    {
                        "milestone_completed": completed,
                        "measured_throughput_requests_per_minute": round(throughput, 3),
                        "remaining_unique_requests": remaining,
                        "eta_minutes": round(remaining / throughput, 3) if throughput else None,
                        "batch_size": BATCH_SIZE,
                    },
                )
                first10_written = True
    latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    relevant = [latest[row["request_id"]] for row in requests if row["request_id"] in latest]
    manifest = {
        "status": "complete" if len(relevant) == len(requests) else "partial",
        "requests": len(requests),
        "results": len(relevant),
        "judge_ok": sum(row.get("judge_ok") is True for row in relevant),
        "judge_failed": sum(row.get("judge_ok") is not True for row in relevant),
        "same_run_new": completed,
        "reused": len(reusable),
        "checkpoint_unique_rows_seen": checkpoint_rows_seen,
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        "model_path": str(MODEL),
        "batch_size_this_invocation": BATCH_SIZE,
        "observed_result_batch_sizes": sorted(
            {
                int((row.get("generation") or {}).get("batch_size"))
                for row in relevant
                if (row.get("generation") or {}).get("batch_size") is not None
            }
        ),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": 384,
        "peak_allocated_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output_path.with_suffix(".manifest.json"), manifest)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
