#!/usr/bin/env python3
"""Blind two-order original-denoiser vs SFT-adapter-control comparison on TEST18."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "acl-component-ablation-matrix-v1-20261009"
ROOT = Path("/mnt/ssd00/hsoh/acl_component_ablation_matrix_20261009_v1")
FROZEN = Path("/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1")
MODEL = "gpt-4.1-2025-04-14"
LOCAL_MODEL = Path(
    "/home/user/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
FEEDBACK_SCRIPTS = Path("/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/scripts")
sys.path.insert(0, str(FEEDBACK_SCRIPTS))
sys.path.insert(0, "/home/user/hsoh/mh-denoise/scripts")

from run_acl_component_llama_pointwise import generate_batch  # noqa: E402
from run_acl_component_selector import cache_results, request_key  # noqa: E402
from run_valid_feedback_v1 import load_local_generator, parse_json_object, render_chat  # noqa: E402


BATCH_SIZE = int(os.environ.get("ACL_LLAMA_BATCH_SIZE", "4"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "local_model": str(LOCAL_MODEL),
        "max_input_tokens": 16384,
        "max_new_tokens": row.get("max_tokens"),
    }
    return sha256_text(json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def normalize_choice(payload: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(payload, dict):
        return None, "no_valid_json_object"
    choice = str(payload.get("choice") or "").strip().lower()
    choice = {"response a": "a", "response b": "b"}.get(choice, choice)
    if choice not in {"a", "b", "tie"}:
        return None, "invalid_choice"
    return {
        "choice": choice.upper() if choice in {"a", "b"} else "tie",
        "rationale": str(payload.get("rationale") or "").strip(),
    }, ""


def make_request(
    original: dict[str, Any],
    control: dict[str, Any],
    order: int,
    system_prompt: str,
    user_template: str,
) -> dict[str, Any]:
    if order == 0:
        response_a, response_b = original["candidate_response"], control["candidate_response"]
        variant_a, variant_b = "original_denoiser", "sft_adapter_control"
    else:
        response_a, response_b = control["candidate_response"], original["candidate_response"]
        variant_a, variant_b = "sft_adapter_control", "original_denoiser"
    user_prompt = user_template.format(
        question=original["question"], response_a=response_a, response_b=response_b
    )
    metadata = {
        "split": "TEST",
        "comparison_type": "original_denoiser_vs_sft_adapter_same_prompt",
        "canonical_id": original["canonical_id"],
        "case_id": original["case_id"],
        "source_order": original["source_order"],
        "order_index": order,
        "variant_for_A": variant_a,
        "variant_for_B": variant_b,
        "question_sha256": original["question_sha256"],
        "response_A_sha256": sha256_text(response_a),
        "response_B_sha256": sha256_text(response_b),
    }
    request_id = sha256_text(
        json.dumps(
            {"version": VERSION, "queue": "adapter_pairwise", "metadata": metadata, "user_prompt": user_prompt},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )[:32]
    return {
        "version": VERSION,
        "queue": "adapter_pairwise",
        "request_id": request_id,
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        "parser": "blind_selector",
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "system_prompt_sha256": sha256_text(system_prompt),
        "user_prompt_sha256": sha256_text(user_prompt),
        "metadata": metadata,
    }


def prepare() -> None:
    candidates = read_jsonl(ROOT / "inputs/standardized_candidates.jsonl")
    original = {row["canonical_id"]: row for row in candidates if row["arm"] == "OriginalDenoiser"}
    control = {row["canonical_id"]: row for row in candidates if row["arm"] == "OriginalPromptSFTControl"}
    if len(original) != 18 or set(original) != set(control):
        raise RuntimeError("Original/control TEST18 coverage differs")
    contract = read_json(FROZEN / "run_contract.json")
    system_prompt = contract["prompts"]["selector_system"]
    user_template = contract["prompts"]["selector_user_template"]
    requests = []
    identical = []
    for row in sorted(original.values(), key=lambda item: int(item["source_order"])):
        if not row["technical_valid"] or not control[row["canonical_id"]]["technical_valid"]:
            continue
        if row["candidate_response"] == control[row["canonical_id"]]["candidate_response"]:
            identical.append(
                {
                    "canonical_id": row["canonical_id"],
                    "case_id": row["case_id"],
                    "source_order": row["source_order"],
                    "reason": "candidate_texts_identical",
                }
            )
            continue
        requests.extend(
            make_request(row, control[row["canonical_id"]], order, system_prompt, user_template)
            for order in (0, 1)
        )
    cache, conflicts = cache_results()
    gpt_reused = []
    gpt_new = []
    for request in requests:
        cached = cache.get(request_key(request))
        if cached is None:
            gpt_new.append(request)
        else:
            path, result = cached
            gpt_reused.append(
                {
                    "target_request_id": request["request_id"],
                    "target_metadata": request["metadata"],
                    "source_result_path": str(path),
                    "source_request_id": result.get("request_id"),
                    "normalized": result["normalized"],
                }
            )
    write_jsonl(ROOT / "requests/adapter_pairwise_expected.jsonl", requests)
    write_jsonl(ROOT / "results/adapter_pairwise_identical_text.jsonl", identical)
    write_jsonl(ROOT / "requests/adapter_pairwise_gpt41_new.jsonl", gpt_new)
    write_jsonl(ROOT / "results/adapter_pairwise_gpt41_reused.jsonl", gpt_reused)
    write_jsonl(ROOT / "results/adapter_pairwise_gpt41_cache_conflicts.jsonl", conflicts)
    # No prior local result is reused unless its exact prompt and frozen local settings match.
    local_cache = {}
    for cache_root in (
        Path("/mnt/ssd00/hsoh/original_prompt_sft_control_20261009_v1"),
        FROZEN,
    ):
        for path in sorted(cache_root.rglob("*.jsonl")):
            if "llama" not in path.name.lower():
                continue
            try:
                rows = read_jsonl(path)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for result in rows:
                if result.get("parser") != "blind_selector" or result.get("judge_ok") is not True:
                    continue
                if result.get("judge_model_path") != str(LOCAL_MODEL):
                    continue
                normalized, _ = normalize_choice(result.get("judge_payload"))
                if normalized is not None:
                    local_cache[local_key(result)] = (path, result)
    llama_reused = []
    llama_new = []
    for request in requests:
        cached = local_cache.get(local_key(request))
        if cached is None:
            llama_new.append(request)
        else:
            path, result = cached
            llama_reused.append(
                {
                    "target_request_id": request["request_id"],
                    "target_metadata": request["metadata"],
                    "source_result_path": str(path),
                    "source_request_id": result.get("request_id"),
                    "judge_normalized": result["judge_normalized"],
                }
            )
    write_jsonl(ROOT / "requests/adapter_pairwise_llama_new.jsonl", llama_new)
    write_jsonl(ROOT / "results/adapter_pairwise_llama_reused.jsonl", llama_reused)
    manifest = {
        "version": VERSION,
        "status": "prepared",
        "cases": len(requests) // 2 + len(identical),
        "identical_text_shared_ties": len(identical),
        "requests": len(requests),
        "gpt41_reused": len(gpt_reused),
        "gpt41_new": len(gpt_new),
        "llama_reused": len(llama_reused),
        "llama_new": len(llama_new),
        "model": MODEL,
        "local_model": str(LOCAL_MODEL),
        "local_batch_size": BATCH_SIZE,
        "prompt": "frozen expert-aligned selector prompt",
        "two_reversed_orders": True,
    }
    write_json(ROOT / "adapter_pairwise_contract.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


def run_llama() -> None:
    import torch

    requests = read_jsonl(ROOT / "requests/adapter_pairwise_llama_new.jsonl")
    output = ROOT / "results/adapter_pairwise_llama_new.jsonl"
    latest = {row["request_id"]: row for row in read_jsonl(output)}
    pending = [
        row
        for row in requests
        if row["request_id"] not in latest
        or latest[row["request_id"]].get("judge_ok") is not True
        or latest[row["request_id"]].get("local_request_key") != local_key(row)
    ]
    print(json.dumps({"queue": "adapter_pairwise_llama", "total": len(requests), "pending": len(pending)}), flush=True)
    if not pending:
        return
    tokenizer, model = load_local_generator(LOCAL_MODEL)
    started = time.monotonic()
    completed = 0
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset : offset + BATCH_SIZE]
        rendered = [render_chat(tokenizer, row["system_prompt"], row["user_prompt"]) for row in batch]
        generations = generate_batch(model, tokenizer, rendered, 256)
        for request, rendered_prompt, (raw, generation) in zip(batch, rendered, generations):
            payload, parse_error = parse_json_object(raw)
            normalized, normalize_error = normalize_choice(payload)
            append_jsonl(
                output,
                {
                    **request,
                    "local_request_key": local_key(request),
                    "judge_model_path": str(LOCAL_MODEL),
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
            print(json.dumps({"queue": "adapter_pairwise_llama", "completed": completed, "pending_total": len(pending), "judge_ok": normalized is not None}), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "run_complete", "elapsed_seconds": round(time.monotonic() - started, 3)}), flush=True)


def result_maps() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    gpt = {}
    for row in read_jsonl(ROOT / "results/adapter_pairwise_gpt41_reused.jsonl"):
        gpt[row["target_request_id"]] = {"ok": True, "normalized": row["normalized"], "provenance": "exact_cache_reuse"}
    for path in sorted((ROOT / "results").glob("adapter_pairwise_gpt41_new*_results.jsonl")):
        for row in read_jsonl(path):
            gpt[row["request_id"]] = row
    llama = {}
    for row in read_jsonl(ROOT / "results/adapter_pairwise_llama_reused.jsonl"):
        llama[row["target_request_id"]] = {"judge_ok": True, "judge_normalized": row["judge_normalized"], "provenance": "exact_cache_reuse"}
    for row in read_jsonl(ROOT / "results/adapter_pairwise_llama_new.jsonl"):
        llama[row["request_id"]] = row
    return gpt, llama


def aggregate() -> None:
    requests = read_jsonl(ROOT / "requests/adapter_pairwise_expected.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[request["metadata"]["canonical_id"]].append(request)
    gpt, llama = result_maps()
    identical = read_jsonl(ROOT / "results/adapter_pairwise_identical_text.jsonl")
    all_rows = []
    summaries = []
    for judge, results, ok_field, normalized_field in (
        ("GPT-4.1", gpt, "ok", "normalized"),
        ("Llama", llama, "judge_ok", "judge_normalized"),
    ):
        judged = []
        for item in identical:
            result_row = {
                "judge": judge,
                "canonical_id": item["canonical_id"],
                "case_id": item["case_id"],
                "source_order": item["source_order"],
                "consensus": "tie",
                "orders": [],
                "shared_without_judge_call": True,
            }
            judged.append(result_row)
            all_rows.append(result_row)
        for row_id, pair in grouped.items():
            pair = sorted(pair, key=lambda row: int(row["metadata"]["order_index"]))
            orders = []
            winners = []
            failed = False
            for request in pair:
                result = results.get(request["request_id"])
                if result is None or result.get(ok_field) is not True:
                    failed = True
                    orders.append({"request_id": request["request_id"], "ok": False})
                    continue
                choice = str((result.get(normalized_field) or {}).get("choice") or "")
                selected = request["metadata"][f"variant_for_{choice}"] if choice in {"A", "B"} else "tie"
                winners.append(selected)
                orders.append({"request_id": request["request_id"], "ok": True, "choice": choice, "selected_variant": selected})
            if failed:
                consensus = "technical_failure"
            elif "tie" in winners:
                consensus = "tie"
            elif len(set(winners)) > 1:
                consensus = "order_disagreement"
            else:
                consensus = winners[0]
            result_row = {
                "judge": judge,
                "canonical_id": row_id,
                "case_id": pair[0]["metadata"]["case_id"],
                "source_order": pair[0]["metadata"]["source_order"],
                "consensus": consensus,
                "orders": orders,
            }
            judged.append(result_row)
            all_rows.append(result_row)
        counts = Counter(row["consensus"] for row in judged)
        summaries.append(
            {
                "judge": judge,
                "N": len(judged),
                "original_denoiser_wins": counts["original_denoiser"],
                "sft_adapter_control_wins": counts["sft_adapter_control"],
                "ties": counts["tie"],
                "order_disagreements": counts["order_disagreement"],
                "technical_failures": counts["technical_failure"],
            }
        )
    write_jsonl(ROOT / "results/adapter_pairwise_consensus.jsonl", all_rows)
    write_jsonl(ROOT / "tables/adapter_pairwise_summary.jsonl", summaries)
    write_json(
        ROOT / "results/adapter_pairwise_completion.json",
        {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "summaries": summaries},
    )
    print(json.dumps({"status": "complete", "summaries": summaries}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run-llama", "aggregate"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run-llama":
        run_llama()
    else:
        aggregate()


if __name__ == "__main__":
    main()
