#!/usr/bin/env python3
"""Add the frozen VALID9 arm C and evaluate only the new B-C comparison."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from run_valid_feedback_v1 import (
    AXES,
    DEFECTS,
    EXPECTED_COMMIT,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    load_local_generator,
    local_generate,
    normalize_judge_payload,
    parse_json_object,
    read_jsonl,
    render_chat,
    rounded_mean,
    sha256_file,
    sha256_text,
    stable_unique,
)


EXPERIMENT_VERSION = "feedback_v1_valid9_bc_adapter_ablation_20261009"
SOURCE_DIR = Path(
    "/home/user/hsoh/mh-denoise-feedback-v1-valid-20261008/"
    "outputs/valid_feedback_v1_20261008"
)
DEFAULT_OUTPUT = Path("/mnt/ssd00/hsoh/feedback_v1_bc_20261009")
SOURCE_FILES = (
    "experiment_contract.json",
    "frozen_valid9_inputs.jsonl",
    "feedback.jsonl",
    "candidate_generations.jsonl",
    "candidate_results.jsonl",
    "blind_judge_results.jsonl",
    "blind_judge_manifest.json",
    "summary.json",
    "REPORT.md",
    "final_manifest.json",
)
B_VARIANT = "feedback_B_sft"
C_VARIANT = "feedback_C_sft_adapter"


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    ).strip()


def update_state(output_dir: Path, stage: str, status: str, **extra: Any) -> None:
    path = output_dir / "run_state.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if status != "failed":
        state.pop("error", None)
    state.update(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "stage": stage,
            "status": status,
            "updated_unix": time.time(),
            **extra,
        }
    )
    atomic_write_json(path, state)


def source_hashes(source_dir: Path) -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (source_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing preserved feedback_v1 artifacts: {missing}")
    return {name: sha256_file(source_dir / name) for name in SOURCE_FILES}


def load_sources(source_dir: Path) -> dict[str, Any]:
    contract = json.loads((source_dir / "experiment_contract.json").read_text(encoding="utf-8"))
    frozen_rows = read_jsonl(source_dir / "frozen_valid9_inputs.jsonl")
    feedback_rows = read_jsonl(source_dir / "feedback.jsonl")
    generation_rows = read_jsonl(source_dir / "candidate_generations.jsonl")
    result_rows = read_jsonl(source_dir / "candidate_results.jsonl")
    frozen = stable_unique(frozen_rows, lambda row: row["canonical_id"])
    feedback = stable_unique(feedback_rows, lambda row: row["canonical_id"])
    b_generated = stable_unique(
        [row for row in generation_rows if row["variant"] == B_VARIANT],
        lambda row: row["canonical_id"],
    )
    b_results = stable_unique(
        [row for row in result_rows if row["variant"] == B_VARIANT],
        lambda row: row["canonical_id"],
    )
    ids = [row["canonical_id"] for row in sorted(frozen_rows, key=lambda row: row["source_order"])]
    if len(ids) != 9 or len(set(ids)) != 9:
        raise RuntimeError(f"Expected nine frozen VALID IDs, got {len(ids)}")
    if set(feedback) != set(ids) or set(b_generated) != set(ids) or set(b_results) != set(ids):
        raise RuntimeError("Frozen, feedback, B generation, and B result ID sets differ")

    fixed = contract["fixed_backbone"]
    settings = contract["fixed_inference"]
    for canonical_id in ids:
        base = frozen[canonical_id]
        fb = feedback[canonical_id]
        b_row = b_generated[canonical_id]
        checks = {
            "B draft source": b_row.get("draft_source") == "sft_response",
            "B draft text": b_row.get("draft_text") == base["sft_response"],
            "B feedback": b_row.get("feedback_text") == fb["feedback_for_both_arms"],
            "B feedback hash": b_row.get("feedback_sha256") == fb["feedback_sha256"],
            "B base model": b_row.get("model_base_path") == fixed["base_model"],
            "B adapter": b_row.get("adapter_path") == fixed["risk_adapter_dir"],
            "B rendered prompt hash": b_row.get("rendered_prompt_sha256")
            == sha256_text(b_row.get("rendered_prompt") or ""),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise RuntimeError(f"Frozen B contract mismatch for {canonical_id}: {failed}")
    return {
        "contract": contract,
        "frozen": frozen,
        "feedback": feedback,
        "b_generated": b_generated,
        "b_results": b_results,
        "ids": ids,
        "fixed": fixed,
        "settings": settings,
    }


def prepare(args) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    update_state(output_dir, "prepare", "running")
    if current_commit() != EXPECTED_COMMIT:
        raise RuntimeError(f"Expected commit {EXPECTED_COMMIT}, found {current_commit()}")
    source = load_sources(args.source_dir)
    hashes = source_hashes(args.source_dir)
    fixed = source["fixed"]
    settings = source["settings"]
    expected_sft = Path(
        "/home/user/hsoh/mh-denoise/outputs/fullpaper_acl/"
        "arr_revision2_gemma_first_seed20260910/gemma/sft/final"
    ).resolve()
    if Path(fixed["sft_adapter_dir"]).resolve() != expected_sft:
        raise RuntimeError(f"Unexpected SFT adapter: {fixed['sft_adapter_dir']}")
    if not expected_sft.exists():
        raise FileNotFoundError(expected_sft)

    extension = {
        "experiment_version": EXPERIMENT_VERSION,
        "git_commit": EXPECTED_COMMIT,
        "scope": "VALID-only; add C for the same frozen nine feedback_v1 IDs",
        "source_feedback_v1_dir": str(args.source_dir.resolve()),
        "preserved_source_sha256": hashes,
        "canonical_ids_in_source_order": source["ids"],
        "generation_arms": {
            "A_preserved": {
                "draft_input": "existing denoiser candidate",
                "adapter_path": fixed["risk_adapter_dir"],
                "regenerated": False,
            },
            "B_preserved": {
                "draft_input": "SFT original response",
                "adapter_path": fixed["risk_adapter_dir"],
                "regenerated": False,
            },
            "C_new": {
                "draft_input": "the same SFT original response used by B",
                "adapter_path": fixed["sft_adapter_dir"],
                "generations_per_id": 1,
            },
        },
        "controlled_B_C_fields": {
            "same_ids": True,
            "same_sft_draft": True,
            "same_stored_feedback": True,
            "same_rendered_prompt": True,
            "same_base_model": fixed["base_model"],
            "same_tokenizer_source": fixed["sft_adapter_dir"],
            "same_decoding": {
                key: settings[key]
                for key in (
                    "max_source_len",
                    "max_new_tokens",
                    "temperature",
                    "repetition_penalty",
                    "no_repeat_ngram_size",
                )
            },
            "only_generation_adapter_changed": True,
        },
        "selection": {
            "policy": "surface_relaxed_v1",
            "settings": {
                key: settings[key]
                for key in (
                    "min_word_count",
                    "min_risk_delta",
                    "gate_strategy",
                    "min_focus_risk_delta",
                    "specificity_min_ratio",
                    "keyword_overlap_slack",
                    "gate_risk_threshold",
                    "gate_focus_threshold",
                )
            },
            "policy_changed": False,
        },
        "judge": {
            "model_path": source["contract"]["judge"]["model_path"],
            "prompt_template_sha256": sha256_text(JUDGE_USER_TEMPLATE),
            "new_pairs_only": "B-C",
            "new_calls": 9,
            "position_scheme": "source-order parity; B is Response A for even rows, C is Response A for odd rows",
            "existing_54_rerun": False,
        },
        "no_feedback_regeneration": True,
        "no_A_or_B_regeneration": True,
        "no_test_inference": True,
        "no_policy_change": True,
    }
    atomic_write_json(output_dir / "extension_contract.json", extension)
    update_state(output_dir, "prepare", "complete", valid_ids=9)


def generate_c(args) -> None:
    import torch
    from peft import PeftModel

    from fullpaper_backbone_utils import load_fullpaper_tokenizer
    from run_gemma_selective_risk_refinement import load_base_model
    from selective_risk_refinement_utils import generate_response

    output_dir = args.output_dir
    update_state(output_dir, "generate_c", "running")
    source = load_sources(args.source_dir)
    fixed = source["fixed"]
    settings = source["settings"]
    output_path = output_dir / "c_candidate_generations.jsonl"
    existing = stable_unique(read_jsonl(output_path), lambda row: row["canonical_id"])
    unknown = set(existing) - set(source["ids"])
    if unknown:
        raise RuntimeError(f"Unexpected C IDs already stored: {sorted(unknown)}")
    requests = [canonical_id for canonical_id in source["ids"] if canonical_id not in existing]
    peak = 0
    if requests:
        torch.cuda.reset_peak_memory_stats()
        tokenizer = load_fullpaper_tokenizer(fixed["sft_adapter_dir"])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base_model = load_base_model(fixed["base_model"], use_4bit=True)
        model = PeftModel.from_pretrained(
            base_model,
            fixed["sft_adapter_dir"],
            adapter_name="sft",
            is_trainable=False,
        )
        model.set_adapter("sft")
        model.eval()
        for index, canonical_id in enumerate(requests, 1):
            frozen = source["frozen"][canonical_id]
            feedback = source["feedback"][canonical_id]
            b_row = source["b_generated"][canonical_id]
            rendered = b_row["rendered_prompt"]
            raw, candidate, generation = generate_response(
                model,
                tokenizer,
                rendered,
                max_source_len=settings["max_source_len"],
                max_new_tokens=settings["max_new_tokens"],
                temperature=settings["temperature"],
                repetition_penalty=settings["repetition_penalty"],
                no_repeat_ngram_size=settings["no_repeat_ngram_size"],
            )
            append_jsonl(
                output_path,
                {
                    "experiment_version": EXPERIMENT_VERSION,
                    "canonical_id": canonical_id,
                    "source_order": frozen["source_order"],
                    "variant": C_VARIANT,
                    "draft_source": "sft_response",
                    "draft_text": frozen["sft_response"],
                    "draft_sha256": sha256_text(frozen["sft_response"]),
                    "feedback_source": str((args.source_dir / "feedback.jsonl").resolve()),
                    "feedback_text": feedback["feedback_for_both_arms"],
                    "feedback_sha256": feedback["feedback_sha256"],
                    "rendered_prompt_source": "exact stored feedback_B_sft rendered_prompt",
                    "rendered_prompt": rendered,
                    "rendered_prompt_sha256": sha256_text(rendered),
                    "matches_B_rendered_prompt": sha256_text(rendered)
                    == b_row["rendered_prompt_sha256"],
                    "model_base_path": fixed["base_model"],
                    "adapter_path": fixed["sft_adapter_dir"],
                    "B_adapter_path": b_row["adapter_path"],
                    "raw_decoded_output": raw,
                    "candidate_text": candidate,
                    "candidate_sha256": sha256_text(candidate),
                    "generation": generation,
                },
            )
            print(f"C generation {index}/{len(requests)} {canonical_id}", flush=True)
        peak = int(torch.cuda.max_memory_allocated())
        del model, base_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    rows = read_jsonl(output_path)
    unique = stable_unique(rows, lambda row: row["canonical_id"])
    if len(unique) != 9:
        raise RuntimeError(f"C generation incomplete: {len(unique)}/9")
    for canonical_id, row in unique.items():
        b_row = source["b_generated"][canonical_id]
        if row["draft_text"] != source["frozen"][canonical_id]["sft_response"]:
            raise RuntimeError(f"C draft differs from frozen SFT for {canonical_id}")
        if row["feedback_sha256"] != source["feedback"][canonical_id]["feedback_sha256"]:
            raise RuntimeError(f"C feedback differs from frozen feedback for {canonical_id}")
        if row["rendered_prompt"] != b_row["rendered_prompt"]:
            raise RuntimeError(f"C rendered prompt differs from B for {canonical_id}")
        if row["model_base_path"] != b_row["model_base_path"]:
            raise RuntimeError(f"C base model differs from B for {canonical_id}")
        if row["adapter_path"] == b_row["adapter_path"]:
            raise RuntimeError(f"C adapter did not change for {canonical_id}")
    atomic_write_json(
        output_dir / "c_candidate_generation_manifest.json",
        {
            "status": "complete",
            "rows": len(rows),
            "new_generations_this_invocation": len(requests),
            "adapter_path": fixed["sft_adapter_dir"],
            "B_adapter_path": fixed["risk_adapter_dir"],
            "same_B_rendered_prompt_all_rows": all(row["matches_B_rendered_prompt"] for row in rows),
            "output_path": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "peak_allocated_cuda_bytes_this_invocation": peak,
        },
    )
    update_state(output_dir, "generate_c", "complete", c_rows=9)


def score_c(args) -> None:
    import torch

    from run_gemma_selective_risk_refinement import (
        acceptance_policy_settings,
        acceptance_reasons,
        load_classifier,
        response_metrics,
        should_call_denoiser,
    )
    from selective_risk_refinement_utils import DIMS, configure_risk_contract, score_candidate

    output_dir = args.output_dir
    update_state(output_dir, "score_c", "running")
    source = load_sources(args.source_dir)
    fixed = source["fixed"]
    settings = source["settings"]
    generated = stable_unique(
        read_jsonl(output_dir / "c_candidate_generations.jsonl"),
        lambda row: row["canonical_id"],
    )
    output_path = output_dir / "c_candidate_results.jsonl"
    if output_path.exists():
        rows = read_jsonl(output_path)
        if len(stable_unique(rows, lambda row: row["canonical_id"])) != 9:
            raise RuntimeError("Partial C risk results are not resumable")
        update_state(output_dir, "score_c", "complete", c_result_rows=9)
        return

    configure_risk_contract("fullpaper_v1")
    focus_idx = DIMS.index(settings["gate_focus_aspect"])
    policy_args = SimpleNamespace(
        acceptance_policy="surface_relaxed_v1",
        min_word_count=settings["min_word_count"],
        min_risk_delta=settings["min_risk_delta"],
        gate_strategy=settings["gate_strategy"],
        min_focus_risk_delta=settings["min_focus_risk_delta"],
        specificity_min_ratio=settings["specificity_min_ratio"],
        keyword_overlap_slack=settings["keyword_overlap_slack"],
        gate_risk_threshold=settings["gate_risk_threshold"],
        gate_focus_threshold=settings["gate_focus_threshold"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.reset_peak_memory_stats()
    router_tok, router = load_classifier(fixed["router_dir"], device)
    risk_tok, risk_model = load_classifier(fixed["risk_scorer_dir"], device)

    def metrics(question: str, response: str, sft_words: int | None = None):
        scores = score_candidate(
            question,
            response,
            router,
            router_tok,
            risk_model,
            risk_tok,
            device,
            router_max_len=settings["router_max_len"],
            risk_max_len=settings["risk_max_len"],
            component_mode="full",
        )
        return response_metrics(
            question,
            response,
            scores,
            focus_idx,
            sft_word_count=sft_words,
        )

    results = []
    for canonical_id in source["ids"]:
        frozen = source["frozen"][canonical_id]
        c_row = generated[canonical_id]
        sft_metrics = metrics(frozen["question"], frozen["sft_response"])
        candidate_metrics = metrics(
            frozen["question"],
            c_row["candidate_text"],
            sft_words=sft_metrics["word_count"],
        )
        gate_called, gate_reason = should_call_denoiser(sft_metrics, policy_args)
        reasons = acceptance_reasons(sft_metrics, candidate_metrics, policy_args)
        accepted = not reasons
        surface_flags = []
        if candidate_metrics["specificity_ratio"] < settings["specificity_min_ratio"]:
            surface_flags.append("specificity_ratio_low")
        if candidate_metrics["generic_count"] > sft_metrics["generic_count"]:
            surface_flags.append("genericity_increased")
        results.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "canonical_id": canonical_id,
                "source_order": frozen["source_order"],
                "variant": C_VARIANT,
                "question": frozen["question"],
                "sft_response": frozen["sft_response"],
                "candidate_raw_decoded_output": c_row["raw_decoded_output"],
                "candidate_text": c_row["candidate_text"],
                "candidate_generation": c_row["generation"],
                "draft_source": "sft_response",
                "generation_adapter_path": c_row["adapter_path"],
                "fixed_gate_membership": True,
                "gate_replay_called": gate_called,
                "gate_replay_reason": gate_reason,
                "acceptance_policy": "surface_relaxed_v1",
                "acceptance_settings": acceptance_policy_settings(policy_args),
                "accepted_candidate": accepted,
                "reject_reasons": reasons,
                "reject_reason": ",".join(reasons) if reasons else None,
                "surface_heuristic_flags": surface_flags,
                "sft_metrics": sft_metrics,
                "candidate_metrics": candidate_metrics,
                "policy_final_source": C_VARIANT if accepted else "sft",
                "policy_final_response": c_row["candidate_text"] if accepted else frozen["sft_response"],
                "candidate_changed_from_sft": c_row["candidate_text"] != frozen["sft_response"],
            }
        )
    peak = int(torch.cuda.max_memory_allocated())
    del router, router_tok, risk_model, risk_tok
    gc.collect()
    torch.cuda.empty_cache()
    atomic_write_jsonl(output_path, results)
    reasons = Counter(reason for row in results for reason in row["reject_reasons"])
    atomic_write_json(
        output_dir / "c_candidate_results_manifest.json",
        {
            "status": "complete",
            "rows": len(results),
            "acceptance_policy": "surface_relaxed_v1",
            "accepted": sum(row["accepted_candidate"] for row in results),
            "rejected": sum(not row["accepted_candidate"] for row in results),
            "reject_reason_counts": dict(sorted(reasons.items())),
            "output_path": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "peak_allocated_cuda_bytes": peak,
        },
    )
    update_state(output_dir, "score_c", "complete", c_result_rows=9)


def build_bc_requests(args) -> list[dict[str, Any]]:
    source = load_sources(args.source_dir)
    c_generated = stable_unique(
        read_jsonl(args.output_dir / "c_candidate_generations.jsonl"),
        lambda row: row["canonical_id"],
    )
    judge_model = source["contract"]["judge"]["model_path"]
    requests = []
    for canonical_id in source["ids"]:
        frozen = source["frozen"][canonical_id]
        b_text = source["b_generated"][canonical_id]["candidate_text"]
        c_text = c_generated[canonical_id]["candidate_text"]
        b_is_a = frozen["source_order"] % 2 == 0
        variant_a, variant_b = (B_VARIANT, C_VARIANT) if b_is_a else (C_VARIANT, B_VARIANT)
        response_a, response_b = (b_text, c_text) if b_is_a else (c_text, b_text)
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=frozen["question"],
            response_a=response_a,
            response_b=response_b,
        )
        requests.append(
            {
                "experiment_version": EXPERIMENT_VERSION,
                "comparison_id": sha256_text(f"{EXPERIMENT_VERSION}|{canonical_id}|B|C")[:24],
                "canonical_id": canonical_id,
                "source_order": frozen["source_order"],
                "pair": [B_VARIANT, C_VARIANT],
                "variant_for_A": variant_a,
                "variant_for_B": variant_b,
                "question": frozen["question"],
                "response_A": response_a,
                "response_B": response_b,
                "response_A_sha256": sha256_text(response_a),
                "response_B_sha256": sha256_text(response_b),
                "judge_model_path": judge_model,
                "system_prompt": JUDGE_SYSTEM_PROMPT,
                "user_prompt": user_prompt,
                "user_prompt_sha256": sha256_text(user_prompt),
                "position_scheme": "alternating_source_order_parity",
            }
        )
    path = args.output_dir / "bc_blind_judge_requests.jsonl"
    if path.exists() and read_jsonl(path) != requests:
        raise RuntimeError("Stored B-C blind requests changed")
    atomic_write_jsonl(path, requests)
    return requests


def judge_bc(args) -> None:
    import torch

    output_dir = args.output_dir
    update_state(output_dir, "judge_bc", "running")
    requests = build_bc_requests(args)
    if len(requests) != 9:
        raise RuntimeError(f"Expected nine B-C requests, got {len(requests)}")
    by_id = {row["comparison_id"]: row for row in requests}
    output_path = output_dir / "bc_blind_judge_results.jsonl"
    prior = read_jsonl(output_path)
    reusable = {}
    for row in prior:
        request = by_id.get(row.get("comparison_id"))
        exact = bool(
            request
            and row.get("judge_model_path") == request["judge_model_path"]
            and row.get("user_prompt_sha256") == request["user_prompt_sha256"]
            and row.get("response_A_sha256") == request["response_A_sha256"]
            and row.get("response_B_sha256") == request["response_B_sha256"]
            and row.get("variant_for_A") == request["variant_for_A"]
            and row.get("variant_for_B") == request["variant_for_B"]
        )
        if not exact:
            raise RuntimeError(f"Stored B-C judge row is not reusable: {row.get('comparison_id')}")
        if row["comparison_id"] in reusable:
            raise RuntimeError(f"Duplicate B-C judge result: {row['comparison_id']}")
        reusable[row["comparison_id"]] = row
    missing = [row for row in requests if row["comparison_id"] not in reusable]
    peak = 0
    if missing:
        torch.cuda.reset_peak_memory_stats()
        tokenizer, model = load_local_generator(Path(requests[0]["judge_model_path"]))
        for index, request in enumerate(missing, 1):
            rendered = render_chat(tokenizer, request["system_prompt"], request["user_prompt"])
            raw, generation = local_generate(
                model,
                tokenizer,
                rendered,
                args.judge_max_input_tokens,
                args.judge_max_new_tokens,
            )
            payload, parse_error = parse_json_object(raw)
            normalized, normalize_error = (
                normalize_judge_payload(payload) if payload is not None else (None, parse_error)
            )
            append_jsonl(
                output_path,
                {
                    **request,
                    "rendered_prompt": rendered,
                    "rendered_prompt_sha256": sha256_text(rendered),
                    "judge_raw": raw,
                    "judge_payload": payload,
                    "judge_normalized": normalized,
                    "judge_ok": normalized is not None,
                    "judge_error": normalize_error or parse_error,
                    "generation": generation,
                },
            )
            print(
                f"B-C judge {index}/{len(missing)} {request['canonical_id']} "
                f"{request['variant_for_A']} as A",
                flush=True,
            )
        peak = int(torch.cuda.max_memory_allocated())
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    rows = read_jsonl(output_path)
    unique = stable_unique(rows, lambda row: row["comparison_id"])
    if len(unique) != 9:
        raise RuntimeError(f"B-C blind judge incomplete: {len(unique)}/9")
    position_counts = Counter(row["variant_for_A"] for row in requests)
    atomic_write_json(
        output_dir / "bc_blind_judge_manifest.json",
        {
            "status": "complete",
            "requests": 9,
            "results": len(rows),
            "judge_ok": sum(bool(row.get("judge_ok")) for row in rows),
            "judge_failed": sum(not bool(row.get("judge_ok")) for row in rows),
            "new_local_judge_calls_this_invocation": len(missing),
            "same_extension_results_reused_this_invocation": len(reusable),
            "existing_feedback_v1_54_rerun": False,
            "position_A_counts": dict(position_counts),
            "method_names_exposed_in_prompt": False,
            "feedback_exposed_in_prompt": False,
            "risk_scores_exposed_in_prompt": False,
            "acceptance_exposed_in_prompt": False,
            "judge_model_path": requests[0]["judge_model_path"],
            "prompt_template_sha256": sha256_text(JUDGE_USER_TEMPLATE),
            "output_path": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "peak_allocated_cuda_bytes_this_invocation": peak,
        },
    )
    update_state(
        output_dir,
        "judge_bc",
        "complete",
        bc_judge_rows=9,
        bc_judge_ok=sum(bool(row.get("judge_ok")) for row in rows),
    )


def collect_existing_sft_scores(source_dir: Path) -> dict[tuple[str, str], float | None]:
    votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in read_jsonl(source_dir / "blind_judge_results.jsonl"):
        if not row.get("judge_ok"):
            continue
        payload = row["judge_normalized"]
        for position in ("A", "B"):
            if row[f"variant_for_{position}"] != "sft":
                continue
            for axis in AXES:
                votes[(row["canonical_id"], axis)].append(payload[f"response_{position}"][axis])
    return {key: rounded_mean(values) for key, values in votes.items()}


def summarize(args) -> None:
    output_dir = args.output_dir
    update_state(output_dir, "summarize", "running")
    source = load_sources(args.source_dir)
    c_generated = stable_unique(
        read_jsonl(output_dir / "c_candidate_generations.jsonl"),
        lambda row: row["canonical_id"],
    )
    c_results = stable_unique(
        read_jsonl(output_dir / "c_candidate_results.jsonl"),
        lambda row: row["canonical_id"],
    )
    judged = read_jsonl(output_dir / "bc_blind_judge_results.jsonl")
    score_by_case: dict[tuple[str, str, str], int] = {}
    defects: dict[tuple[str, str, str], bool] = {}
    preferences = Counter()
    for row in judged:
        if not row.get("judge_ok"):
            continue
        payload = row["judge_normalized"]
        for position in ("A", "B"):
            variant = row[f"variant_for_{position}"]
            item = payload[f"response_{position}"]
            for axis in AXES:
                score_by_case[(row["canonical_id"], variant, axis)] = item[axis]
            for defect in DEFECTS:
                defects[(row["canonical_id"], variant, defect)] = item[defect]
        preference = payload.get("preference")
        if preference == "tie":
            preferences["tie"] += 1
        elif preference in {"A", "B"}:
            preferences[row[f"variant_for_{preference}"]] += 1
        else:
            preferences["unparsed"] += 1

    candidate_quality = {
        variant: {
            axis: rounded_mean(
                score_by_case[(canonical_id, variant, axis)]
                for canonical_id in source["ids"]
                if (canonical_id, variant, axis) in score_by_case
            )
            for axis in AXES
        }
        for variant in (B_VARIANT, C_VARIANT)
    }
    candidate_delta = {
        axis: round(candidate_quality[C_VARIANT][axis] - candidate_quality[B_VARIANT][axis], 4)
        if candidate_quality[C_VARIANT][axis] is not None
        and candidate_quality[B_VARIANT][axis] is not None
        else None
        for axis in AXES
    }
    defect_summary = {
        variant: {
            defect: {
                "count": sum(
                    bool(defects.get((canonical_id, variant, defect), False))
                    for canonical_id in source["ids"]
                ),
                "case_ids": [
                    canonical_id
                    for canonical_id in source["ids"]
                    if defects.get((canonical_id, variant, defect), False)
                ],
            }
            for defect in DEFECTS
        }
        for variant in (B_VARIANT, C_VARIANT)
    }

    selection_rows = {
        B_VARIANT: [source["b_results"][canonical_id] for canonical_id in source["ids"]],
        C_VARIANT: [c_results[canonical_id] for canonical_id in source["ids"]],
    }
    selection = {}
    for variant, rows in selection_rows.items():
        rejection_reasons = Counter(reason for row in rows for reason in row["reject_reasons"])
        selection[variant] = {
            "accepted": sum(row["accepted_candidate"] for row in rows),
            "rejected_to_sft": sum(not row["accepted_candidate"] for row in rows),
            "reject_reason_counts": dict(sorted(rejection_reasons.items())),
            "final_source_by_id": {
                row["canonical_id"]: row["policy_final_source"] for row in rows
            },
        }

    sft_scores = collect_existing_sft_scores(args.source_dir)
    final_quality = {}
    for variant, rows in selection_rows.items():
        values = {axis: [] for axis in AXES}
        for row in rows:
            canonical_id = row["canonical_id"]
            for axis in AXES:
                if row["accepted_candidate"]:
                    value = score_by_case.get((canonical_id, variant, axis))
                else:
                    value = sft_scores.get((canonical_id, axis))
                if value is not None:
                    values[axis].append(value)
        final_quality[variant] = {axis: rounded_mean(values[axis]) for axis in AXES}
    final_delta = {
        axis: round(final_quality[C_VARIANT][axis] - final_quality[B_VARIANT][axis], 4)
        if final_quality[C_VARIANT][axis] is not None
        and final_quality[B_VARIANT][axis] is not None
        else None
        for axis in AXES
    }

    judge_by_id = {row["canonical_id"]: row for row in judged}
    cases = []
    for canonical_id in source["ids"]:
        frozen = source["frozen"][canonical_id]
        feedback = source["feedback"][canonical_id]
        b_generation = source["b_generated"][canonical_id]
        b_result = source["b_results"][canonical_id]
        c_generation = c_generated[canonical_id]
        c_result = c_results[canonical_id]
        cases.append(
            {
                "canonical_id": canonical_id,
                "source_order": frozen["source_order"],
                "question": frozen["question"],
                "sft_original_response": frozen["sft_response"],
                "stored_feedback": feedback["feedback_for_both_arms"],
                "rendered_prompt_shared_by_B_and_C": b_generation["rendered_prompt"],
                "B": {
                    "draft_input_source": "SFT original response",
                    "generation_adapter_path": b_generation["adapter_path"],
                    "candidate_text": b_generation["candidate_text"],
                    "accepted_candidate": b_result["accepted_candidate"],
                    "reject_reasons": b_result["reject_reasons"],
                    "surface_heuristic_flags": b_result["surface_heuristic_flags"],
                    "policy_final_source": b_result["policy_final_source"],
                    "policy_final_response": b_result["policy_final_response"],
                    "blind_candidate_scores": {
                        axis: score_by_case.get((canonical_id, B_VARIANT, axis)) for axis in AXES
                    },
                },
                "C": {
                    "draft_input_source": "SFT original response",
                    "generation_adapter_path": c_generation["adapter_path"],
                    "candidate_text": c_generation["candidate_text"],
                    "accepted_candidate": c_result["accepted_candidate"],
                    "reject_reasons": c_result["reject_reasons"],
                    "surface_heuristic_flags": c_result["surface_heuristic_flags"],
                    "policy_final_source": c_result["policy_final_source"],
                    "policy_final_response": c_result["policy_final_response"],
                    "blind_candidate_scores": {
                        axis: score_by_case.get((canonical_id, C_VARIANT, axis)) for axis in AXES
                    },
                },
                "blind_order": {
                    "variant_for_A": judge_by_id[canonical_id]["variant_for_A"],
                    "variant_for_B": judge_by_id[canonical_id]["variant_for_B"],
                },
                "judge_normalized": judge_by_id[canonical_id].get("judge_normalized"),
            }
        )
    atomic_write_jsonl(output_dir / "bc_case_results.jsonl", cases)

    known_failures = [
        {
            "canonical_id": "qa_24f1b77a40c9b22203b8f337",
            "arm": "B",
            "failure": "specificity-loss candidate accepted by surface_relaxed_v1",
            "accepted_candidate": source["b_results"]["qa_24f1b77a40c9b22203b8f337"]["accepted_candidate"],
            "surface_heuristic_flags": source["b_results"]["qa_24f1b77a40c9b22203b8f337"]["surface_heuristic_flags"],
        },
        {
            "canonical_id": "qa_b243b4c887feb8e3594dd74c",
            "arm": "B",
            "failure": "candidate rejected and policy fell back to the unchanged SFT original",
            "accepted_candidate": source["b_results"]["qa_b243b4c887feb8e3594dd74c"]["accepted_candidate"],
            "reject_reasons": source["b_results"]["qa_b243b4c887feb8e3594dd74c"]["reject_reasons"],
            "policy_final_source": source["b_results"]["qa_b243b4c887feb8e3594dd74c"]["policy_final_source"],
        },
    ]
    summary = {
        "experiment_version": EXPERIMENT_VERSION,
        "status": "complete",
        "valid_ids": 9,
        "new_C_generations": 9,
        "new_B_C_judge_requests": 9,
        "existing_54_judge_requests_rerun": 0,
        "candidate_quality_before_policy": candidate_quality,
        "candidate_score_delta_C_minus_B": candidate_delta,
        "candidate_pair_preferences": {
            "B": preferences[B_VARIANT],
            "C": preferences[C_VARIANT],
            "tie": preferences["tie"],
            "unparsed": preferences["unparsed"],
        },
        "candidate_defects": defect_summary,
        "surface_relaxed_v1_selection": selection,
        "derived_policy_final_quality": final_quality,
        "derived_policy_final_score_delta_C_minus_B": final_delta,
        "derived_policy_final_quality_method": (
            "accepted candidates use the new direct B-C blind score; SFT fallbacks use the existing "
            "feedback_v1 per-case SFT score. No additional final-output judge call was made."
        ),
        "known_failures": known_failures,
        "limits": [
            "This is a nine-case VALID development ablation, not a TEST result.",
            "The new score delta comes only from nine B-C blind comparisons.",
            "A single local automated judge is not expert validation and may be noisy.",
            "Candidate quality and fixed-policy final output are reported separately.",
            "No TEST inference, feedback regeneration, A/B regeneration, retraining, or policy change occurred.",
        ],
    }
    atomic_write_json(output_dir / "summary_bc.json", summary)

    labels = {B_VARIANT: "B: SFT input + proposed Mask-OFF adapter", C_VARIANT: "C: SFT input + SFT adapter"}
    lines = [
        "# VALID feedback_v1: B-C adapter ablation",
        "",
        "- Status: complete for the same 9 frozen VALID IDs",
        "- New work only: 9 arm-C generations and 9 blind B-C judge calls",
        "- Preserved without regeneration: stored feedback, arm A, arm B, and all 54 prior judge results",
        "- Selection policy remains `surface_relaxed_v1`; no TEST inference or policy change was performed",
        "",
        "## Generation conditions",
        "",
        "The draft input and the generation adapter are separate experimental factors:",
        "",
        "| Arm | Draft input text | Generation adapter | This extension |",
        "|---|---|---|---|",
        f"| A | Existing denoiser candidate | `{source['fixed']['risk_adapter_dir']}` | Preserved; not regenerated or newly judged |",
        f"| B | SFT original response | `{source['fixed']['risk_adapter_dir']}` | Preserved; compared with C |",
        f"| C | The same SFT original response as B | `{source['fixed']['sft_adapter_dir']}` | Generated once per ID |",
        "",
        "B and C used the exact same stored feedback, rendered prompt, base model, tokenizer source, and decoding settings. Only the generation adapter changed. Anonymous Response A/B placement alternated by fixed source order (B in A: 5; C in A: 4).",
        "",
        "## Candidate quality before policy",
        "",
        "| Candidate | Overall Quality | Empathy | Specificity | Medical Advice |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in (B_VARIANT, C_VARIANT):
        scores = candidate_quality[variant]
        lines.append(
            f"| {labels[variant]} | {scores['overall_quality']} | {scores['empathy']} | "
            f"{scores['specificity']} | {scores['medical_advice']} |"
        )
    lines.append(
        f"| C - B | {candidate_delta['overall_quality']:+.4f} | {candidate_delta['empathy']:+.4f} | "
        f"{candidate_delta['specificity']:+.4f} | {candidate_delta['medical_advice']:+.4f} |"
    )
    lines.extend(
        [
            "",
            f"Pair preference: B {preferences[B_VARIANT]}, C {preferences[C_VARIANT]}, tie {preferences['tie']}, unparsed {preferences['unparsed']}.",
            "",
            "## Fixed-policy final output",
            "",
            "Rejected candidates fall back to the unchanged SFT original. The quality rows below are derived without new final-output judge calls: accepted candidates use their new B-C score, while fallbacks use the preserved per-case SFT score from feedback_v1.",
            "",
            "| Route | Accepted candidate | SFT fallback | Overall Quality | Empathy | Specificity | Medical Advice |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in (B_VARIANT, C_VARIANT):
        scores = final_quality[variant]
        lines.append(
            f"| {labels[variant]} | {selection[variant]['accepted']} | {selection[variant]['rejected_to_sft']} | "
            f"{scores['overall_quality']} | {scores['empathy']} | {scores['specificity']} | {scores['medical_advice']} |"
        )
    lines.append(
        f"| C - B |  |  | {final_delta['overall_quality']:+.4f} | {final_delta['empathy']:+.4f} | "
        f"{final_delta['specificity']:+.4f} | {final_delta['medical_advice']:+.4f} |"
    )
    lines.extend(
        [
            "",
            "## Known failures",
            "",
            "- `qa_24f1b77a40c9b22203b8f337`: arm B's materially shortened, specificity-losing candidate carried `specificity_ratio_low` but was accepted by `surface_relaxed_v1`; this remains a known false acceptance.",
            "- `qa_b243b4c887feb8e3594dd74c`: arm B was rejected for `risk_not_improved`, so the final output fell back to the unchanged SFT original; this remains a known fallback failure case.",
            "",
            "## Limits",
            "",
            "This is a nine-case VALID development ablation. Score changes do not establish TEST or expert-rated improvement. The judge is one local automated model and can be noisy. Candidate quality and post-policy final output must not be conflated.",
        ]
    )
    (output_dir / "REPORT_BC.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if source_hashes(args.source_dir) != json.loads(
        (output_dir / "extension_contract.json").read_text(encoding="utf-8")
    )["preserved_source_sha256"]:
        raise RuntimeError("A preserved feedback_v1 source artifact changed during the C extension")

    bundle = output_dir / "valid_feedback_v1_BC_adapter_ablation_research_bundle.zip"
    bundle_names = [
        "REPORT_BC.md",
        "summary_bc.json",
        "extension_contract.json",
        "c_candidate_generations.jsonl",
        "c_candidate_generation_manifest.json",
        "c_candidate_results.jsonl",
        "c_candidate_results_manifest.json",
        "bc_blind_judge_requests.jsonl",
        "bc_blind_judge_results.jsonl",
        "bc_blind_judge_manifest.json",
        "bc_case_results.jsonl",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in bundle_names:
            archive.write(output_dir / name, arcname=name)
    atomic_write_json(
        output_dir / "final_manifest.json",
        {
            "status": "complete",
            "experiment_version": EXPERIMENT_VERSION,
            "valid_ids": 9,
            "new_C_generations": 9,
            "new_B_C_judge_calls": 9,
            "existing_54_judge_calls_rerun": 0,
            "source_feedback_or_A_B_regenerated": False,
            "test_inference_run": False,
            "policy_changed": False,
            "report_path": str((output_dir / "REPORT_BC.md").resolve()),
            "bundle_path": str(bundle.resolve()),
            "bundle_sha256": sha256_file(bundle),
        },
    )
    update_state(
        output_dir,
        "complete",
        "complete",
        report=str((output_dir / "REPORT_BC.md").resolve()),
        bundle=str(bundle.resolve()),
    )
    print(f"complete: {output_dir / 'REPORT_BC.md'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--judge-max-input-tokens", type=int, default=16384)
    parser.add_argument("--judge-max-new-tokens", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    stages = (prepare, generate_c, score_c, judge_bc, summarize)
    current = "prepare"
    try:
        for stage in stages:
            current = stage.__name__
            stage(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        update_state(args.output_dir, current, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
