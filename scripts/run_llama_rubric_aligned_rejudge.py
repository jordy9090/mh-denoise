#!/usr/bin/env python3
"""Run the aligned expert-rubric pointwise queues with one local Llama instance."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


FEEDBACK_SCRIPTS = Path("/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/scripts")
sys.path.insert(0, str(FEEDBACK_SCRIPTS))

from run_valid_feedback_v1 import (  # noqa: E402
    generation_eos_ids,
    load_local_generator,
    parse_json_object,
    render_chat,
)


ROOT = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
BATCH_SIZE = 4
MAX_INPUT_TOKENS = 16384


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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def request_hash(row: dict[str, Any]) -> str:
    material = {
        "request_id": row["request_id"],
        "parser": row["parser"],
        "system_prompt": row["system_prompt"],
        "user_prompt": row["user_prompt"],
        "metadata": row["metadata"],
        "local_model": str(MODEL),
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": row["max_tokens"],
    }
    return sha256_text(json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def normalize_expert(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
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


def run_queue(model, tokenizer, request_path: Path, output_path: Path, queue_name: str) -> dict[str, Any]:
    requests = read_jsonl(request_path)
    if any(row["parser"] != "pointwise_expert_rubric" for row in requests):
        raise RuntimeError(f"Unexpected parser in {queue_name}")
    if len({row["request_id"] for row in requests}) != len(requests):
        raise RuntimeError(f"Duplicate request IDs in {queue_name}")
    request_by_id = {row["request_id"]: row for row in requests}
    latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    reusable = {
        request_id
        for request_id, row in latest.items()
        if request_id in request_by_id
        and row.get("judge_ok") is True
        and row.get("request_sha256") == request_hash(request_by_id[request_id])
    }
    pending = [row for row in requests if row["request_id"] not in reusable]
    print(json.dumps({"queue": queue_name, "total": len(requests), "reused": len(reusable), "pending": len(pending)}), flush=True)
    started = time.monotonic()
    completed = 0
    first10_written = False
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        max_tokens = {row["max_tokens"] for row in batch}
        if len(max_tokens) != 1:
            raise RuntimeError("Mixed max token contracts in one local batch")
        rendered = [render_chat(tokenizer, row["system_prompt"], row["user_prompt"]) for row in batch]
        generations = generate_batch(model, tokenizer, rendered, max_tokens.pop())
        for request, rendered_prompt, (raw, generation) in zip(batch, rendered, generations):
            payload, parse_error = parse_json_object(raw)
            normalized, normalize_error = normalize_expert(payload)
            append_jsonl(
                output_path,
                {
                    **request,
                    "request_sha256": request_hash(request),
                    "judge_model_path": str(MODEL),
                    "rendered_prompt_sha256": sha256_text(rendered_prompt),
                    "judge_raw": raw,
                    "judge_payload": payload,
                    "judge_normalized": normalized,
                    "judge_ok": normalized is not None,
                    "judge_error": normalize_error or parse_error,
                    "generation": generation,
                },
            )
            completed += 1
            print(json.dumps({"queue": queue_name, "completed_this_run": completed, "pending_total": len(pending), "judge_ok": normalized is not None}), flush=True)
            if completed >= 10 and not first10_written:
                elapsed = max(time.monotonic() - started, 1e-9)
                throughput = completed / elapsed * 60
                remaining = len(pending) - completed
                atomic_json(
                    output_path.with_suffix(".first10.json"),
                    {
                        "queue": queue_name,
                        "milestone_completed": completed,
                        "measured_throughput_requests_per_minute": round(throughput, 3),
                        "remaining_unique_requests": remaining,
                        "eta_minutes": round(remaining / throughput, 3) if throughput else None,
                        "batch_size": BATCH_SIZE,
                    },
                )
                first10_written = True
    final_latest = {row["request_id"]: row for row in read_jsonl(output_path)}
    relevant = [final_latest[row["request_id"]] for row in requests if row["request_id"] in final_latest]
    manifest = {
        "queue": queue_name,
        "status": "complete" if len(relevant) == len(requests) else "partial",
        "requests": len(requests),
        "results": len(relevant),
        "judge_ok": sum(row.get("judge_ok") is True for row in relevant),
        "judge_failed": sum(row.get("judge_ok") is not True for row in relevant),
        "same_run_new": completed,
        "reused": len(reusable),
        "elapsed_seconds_this_run": round(time.monotonic() - started, 3),
        "model_path": str(MODEL),
        "batch_size": BATCH_SIZE,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": 384,
        "request_path": str(request_path),
        "output_path": str(output_path),
    }
    atomic_json(output_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> None:
    import torch

    queues = (
        (
            ROOT / "requests/expert15_pointwise.jsonl",
            ROOT / "results/expert15_pointwise_llama.jsonl",
            "expert15_pointwise_llama",
        ),
        (
            ROOT / "requests/remaining_pointwise.jsonl",
            ROOT / "results/remaining_pointwise_llama.jsonl",
            "remaining_pointwise_llama",
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model = load_local_generator(MODEL)
    manifests = [run_queue(model, tokenizer, *queue) for queue in queues]
    peak = int(torch.cuda.max_memory_allocated())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    final = {
        "status": "complete" if all(row["status"] == "complete" for row in manifests) else "partial",
        "model_path": str(MODEL),
        "one_model_instance": True,
        "batch_size": BATCH_SIZE,
        "peak_allocated_cuda_bytes": peak,
        "queues": manifests,
    }
    atomic_json(ROOT / "results/llama_rubric_aligned_manifest.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
