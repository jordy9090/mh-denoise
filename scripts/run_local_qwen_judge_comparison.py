#!/usr/bin/env python3
"""Frozen local-Qwen comparison against stored dev120 paired-QC decisions.

This program has no OpenAI client and no network fallback.  Model loading is
``local_files_only`` and the caller must provide the exact repository and
revision recorded for the local snapshot.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_development_corruption_shard import (
    ACCEPTANCE_RULES,
    PAIRED_PROMPT,
    content_failure,
    validate_paired,
)
from corruption_contract_v2 import AXES, paired_graded_realized_axes
from run_paired_generator_diagnostic import ELIGIBILITY_PROMPT, SYSTEM_PROMPT, normalize_text, surface_flags


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120"
REPAIR = ROOT / "data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2"
DEFAULT_OUTPUT = ROOT / "data/fullpaper_acl_pipeline/local_qwen_judge_dev120_36"
SEED = 20260909
QUOTAS = {
    "retained_pass": 12,
    "held_primary": 10,
    "held_secondary": 3,
    "rejected": 6,
    "qc_conflict": 5,
}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_key(seed: int, text: str) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def last_graded_stage(row: dict[str, Any]) -> dict[str, Any] | None:
    graded = [stage for stage in row.get("stage_history", []) if stage.get("grade")]
    return graded[-1] if graded else None


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_development_rows() -> dict[str, dict[str, Any]]:
    checkpoint_paths = sorted(DEV.glob("results_checkpoint.part*.jsonl"))
    rows = [row for path in checkpoint_paths for row in read_jsonl(path)]
    return {row["canonical_id"]: row for row in rows}


def enrich_frozen_selection(selection: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach stage provenance without changing the frozen IDs or their order."""
    by_id = load_development_rows()
    enriched: list[dict[str, Any]] = []
    for position, selected in enumerate(selection):
        row = by_id.get(selected["canonical_id"])
        if row is None:
            raise RuntimeError(f"Frozen ID is absent from dev120 checkpoints: {selected['canonical_id']}")
        matches = [
            (history_index, stage)
            for history_index, stage in enumerate(row.get("stage_history", []))
            if stage.get("grade")
            and stage.get("candidate_response") == selected["candidate_response"]
        ]
        if not matches:
            raise RuntimeError(f"Frozen candidate text no longer matches a graded stage: {selected['canonical_id']}")
        history_index, stage = matches[-1]
        stage_index = int(stage["stage_index"])
        cumulative = list(row["intended_axes"][:stage_index])
        item = copy.deepcopy(selected)
        item.update({
            "selection_index": position,
            "selected_stage_index": stage_index,
            "selected_history_index": history_index,
            "selected_semantic_attempt": int(stage.get("semantic_attempt", 0)),
            "clean_sha256": text_hash(row["clean_response"]),
            "candidate_sha256": text_hash(stage["candidate_response"]),
            "cumulative_intended_axes": cumulative,
            "old_stage_accept": bool(stage.get("accepted")),
            "old_stage_reason": stage.get("reason"),
            "old_gpt_grade": copy.deepcopy(stage["grade"]),
            "old_clean_recheck": copy.deepcopy(stage.get("bounded_clean_recheck")),
            "old_terminal_status": row["status"],
            "old_terminal_failure_reason": row.get("failure_reason"),
            "terminal_comparable": (
                history_index == len(row.get("stage_history", [])) - 1
                and row["status"] in {"accepted", "rejected", "qc_conflict"}
            ),
        })
        if item["clean_response"] != row["clean_response"]:
            raise RuntimeError(f"Frozen clean text changed: {selected['canonical_id']}")
        enriched.append(item)
    if [row["canonical_id"] for row in enriched] != [row["canonical_id"] for row in selection]:
        raise AssertionError("Selection ID/order changed while adding provenance")
    return enriched


def select_balanced(rows: list[dict[str, Any]], quota: int, seed: int) -> list[dict[str, Any]]:
    """Greedily cover underrepresented intended axes with a stable tie-break."""
    chosen: list[dict[str, Any]] = []
    counts = Counter()
    pool = list(rows)
    while pool and len(chosen) < quota:
        pool.sort(
            key=lambda row: (
                sum(counts[axis] for axis in row["intended_axes"]),
                max((counts[axis] for axis in row["intended_axes"]), default=0),
                stable_key(seed, row["canonical_id"]),
            )
        )
        item = pool.pop(0)
        chosen.append(item)
        counts.update(item["intended_axes"])
    if len(chosen) != quota:
        raise RuntimeError(f"Selection pool short: requested {quota}, found {len(chosen)}")
    return chosen


def build_selection(output: Path) -> list[dict[str, Any]]:
    checkpoint_paths = sorted(DEV.glob("results_checkpoint.part*.jsonl"))
    source_rows = [row for path in checkpoint_paths for row in read_jsonl(path)]
    by_id = {row["canonical_id"]: row for row in source_rows}
    holds = {row["canonical_id"]: row for row in read_jsonl(REPAIR / "held_out.jsonl")}
    retained = {
        row["metadata"]["canonical_id"]
        for row in read_jsonl(REPAIR / "train_sft.jsonl")
    }

    pools: dict[str, list[dict[str, Any]]] = {key: [] for key in QUOTAS}
    for cid, row in by_id.items():
        graded = last_graded_stage(row)
        if not graded:
            continue
        if cid in holds:
            category = "held_" + holds[cid]["audit_category"]
        elif cid in retained:
            category = "retained_pass"
        elif row["status"] == "qc_conflict":
            category = "qc_conflict"
        elif row["status"] == "rejected":
            category = "rejected"
        else:
            continue
        reference = row.get("final_grade") or graded["grade"]
        pools[category].append(
            {
                "canonical_id": cid,
                "category": category,
                "source": row["source"],
                "source_component": row["source_component"],
                "split": row["split"],
                "question": row["question"],
                "clean_response": row["clean_response"],
                "candidate_response": graded["candidate_response"],
                "intended_axes": row["intended_axes"],
                "axis_count": row["axis_count"],
                "old_status": row["status"],
                "old_failure_reason": row.get("failure_reason"),
                "old_gpt_grade": reference,
                "audit_category": holds.get(cid, {}).get("audit_category"),
                "audit_hold_reason": holds.get(cid, {}).get("hold_reason"),
                "audit_evidence_quote": holds.get(cid, {}).get("evidence_quote"),
                "audit_surface_flags": holds.get(cid, {}).get("surface_flags", {}),
            }
        )

    selected: list[dict[str, Any]] = []
    for offset, (category, quota) in enumerate(QUOTAS.items()):
        selected.extend(select_balanced(pools[category], quota, SEED + offset))
    selected.sort(key=lambda row: stable_key(SEED + 99, row["canonical_id"]))
    for index, row in enumerate(selected):
        row["selection_index"] = index
        row["ab_candidate_label"] = "A" if int(stable_key(SEED, row["canonical_id"]), 16) % 2 == 0 else "B"

    if len(selected) != 36 or any(row["split"] != "train" for row in selected):
        raise AssertionError("Frozen comparison must contain exactly 36 TRAIN rows")
    if set(axis for row in selected for axis in row["intended_axes"]) != set(AXES):
        raise AssertionError("Frozen comparison does not cover all six axes")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "selection.jsonl", selected)
    manifest = {
        "version": "local-qwen-judge-comparison-v1",
        "status": "selection_frozen",
        "seed": SEED,
        "max_examples": 36,
        "quotas": QUOTAS,
        "category_counts": dict(Counter(row["category"] for row in selected)),
        "axis_counts": dict(Counter(axis for row in selected for axis in row["intended_axes"])),
        "source_counts": dict(Counter(row["source"] for row in selected)),
        "input_hashes": {
            **{path.name: file_hash(path) for path in checkpoint_paths},
            "held_out.jsonl": file_hash(REPAIR / "held_out.jsonl"),
            "corrected_train_sft.jsonl": file_hash(REPAIR / "train_sft.jsonl"),
        },
        "selection_sha256": file_hash(output / "selection.jsonl"),
        "paired_prompt_sha256": hashlib.sha256(PAIRED_PROMPT.encode()).hexdigest(),
        "eligibility_prompt_sha256": hashlib.sha256(ELIGIBILITY_PROMPT.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "acceptance_rules": ACCEPTANCE_RULES,
        "openai_fallback": False,
        "canonical_valid_test_used": False,
    }
    (output / "selection_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return selected


def parse_json_object(text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("judge output is not a JSON object")
    return value


class LocalQwenJudge:
    def __init__(self, model_dir: Path, repo: str, revision: str, max_new_tokens: int,
                 call_timeout_seconds: float, system_prompt: str = SYSTEM_PROMPT) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(model_dir)
        self.model_dir = model_dir.resolve()
        self.repo = repo
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.call_timeout_seconds = call_timeout_seconds
        self.system_prompt = system_prompt
        self.optional_kernel_imports = {}
        # Qwen3.5's decorators resolve optional FLA functions while the model
        # module is imported. Importing FLA first avoids its circular import
        # falling back silently to the token-by-token PyTorch implementation.
        if repo == "Qwen/Qwen3.5-27B":
            try:
                module = importlib.import_module("fla")
                importlib.import_module("fla.ops.gated_delta_rule")
                self.optional_kernel_imports["fla"] = {
                    "available": True, "path": module.__file__, "gated_delta_rule_preloaded": True,
                }
            except Exception as error:
                self.optional_kernel_imports["fla"] = {"available": False, "error": repr(error)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise RuntimeError("Local judge tokenizer has no official chat template")
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
        self.parameter_devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        self.cpu_parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters() if parameter.device.type == "cpu"
        )
        self.delta_kernel_status = {}
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed = 0.0
        if repo == "Qwen/Qwen3.5-27B":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn"):
                function = getattr(modeling, name)
                cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
                implementation = cells.get("implementation")
                self.delta_kernel_status[name] = {
                    "accelerated": bool(
                        cells.get("is_new_implementation")
                        and cells["is_new_implementation"].cell_contents
                    ),
                    "implementation_module": (
                        getattr(implementation.cell_contents, "__module__", None)
                        if implementation is not None else None
                    ),
                }

    @torch.no_grad()
    def call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        return self.call_many([prompt])[0]

    @torch.no_grad()
    def call_many(self, prompts: list[str]) -> list[tuple[str, dict[str, Any]]]:
        rendered = [self.tokenizer.apply_chat_template(
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts]
        encoded = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.model.device)
        started = time.monotonic()
        output = self.model.generate(
            **encoded,
            do_sample=False,
            use_cache=True,
            max_new_tokens=self.max_new_tokens,
            max_time=self.call_timeout_seconds,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated_batch = output[:, encoded["input_ids"].shape[-1] :]
        elapsed = time.monotonic() - started
        self.calls += len(prompts)
        self.prompt_tokens += int(encoded["attention_mask"].sum().item())
        self.elapsed += elapsed
        results = []
        for index, generated in enumerate(generated_batch):
            positions = ((generated == self.tokenizer.eos_token_id).nonzero(as_tuple=False).flatten()
                         if self.tokenizer.eos_token_id is not None else [])
            eos = len(positions) > 0
            count = int(positions[0].item() + 1) if eos else int(generated.numel())
            self.completion_tokens += count
            results.append((self.tokenizer.decode(generated[:count], skip_special_tokens=True), {
                "input_tokens": int(encoded["attention_mask"][index].sum().item()),
                "output_tokens": count,
                "elapsed_seconds": elapsed / len(prompts),
                "batch_elapsed_seconds": elapsed,
                "batch_size": len(prompts),
                "enable_thinking": False,
                "use_cache": True,
                "eos_reached": eos,
                "call_timeout_seconds": self.call_timeout_seconds,
                "stop_reason": (
                    "eos" if eos else
                    "max_new_tokens" if count >= self.max_new_tokens else
                    "max_time_or_generation_stop"
                ),
            }))
        return results


def map_paired(payload: dict[str, Any], clean_label: str, candidate_label: str) -> dict[str, Any]:
    clean_values, candidate_values = {}, {}
    clean_scores, candidate_scores = {}, {}
    for axis in AXES:
        key = "violation" if axis == "medical_boundary" else "score"
        clean_scores[axis] = payload["scores"][axis][clean_label]
        candidate_scores[axis] = payload["scores"][axis][candidate_label]
        clean_values[axis] = clean_scores[axis][key]
        candidate_values[axis] = candidate_scores[axis][key]
    realized = list(paired_graded_realized_axes(clean_values, candidate_values))
    return {
        "clean_label": clean_label,
        "candidate_label": candidate_label,
        "clean_scores": clean_scores,
        "candidate_scores": candidate_scores,
        "deltas": {
            axis: (
                {"clean_violation": clean_values[axis], "candidate_violation": candidate_values[axis]}
                if axis == "medical_boundary"
                else candidate_values[axis] - clean_values[axis]
            )
            for axis in AXES
        },
        "realized_axes": realized,
        "content_checks": payload["content_checks"],
        "text_reason_contradiction": payload["text_reason_contradiction"],
        "overall_reason": payload.get("overall_reason", ""),
    }


def parse_eligibility(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    if set(payload.get("axes", {})) != set(AXES):
        raise ValueError("eligibility output must contain all six axes")
    degraded = []
    qnorm, rnorm = normalize_text(row["question"]), normalize_text(row["clean_response"])
    for axis in AXES:
        item = payload["axes"][axis]
        if not isinstance(item.get("material_degradation"), bool):
            raise ValueError(f"invalid eligibility boolean for {axis}")
        if item["material_degradation"]:
            source, span = item.get("evidence_source"), str(item.get("evidence_span") or "")
            if source == "response" and normalize_text(span) not in rnorm:
                raise ValueError(f"non-verbatim eligibility response evidence for {axis}")
            if source == "question" and normalize_text(span) not in qnorm:
                raise ValueError(f"non-verbatim eligibility question evidence for {axis}")
            degraded.append(axis)
    integrity_flags = surface_flags(row["question"], row["clean_response"], "clean")
    axis_for = {"non_response_meta": "overall_quality", "speaker_switch_question_copy": "overall_quality",
                "false_identity": "factual_consistency", "unsupported_specific_history": "factual_consistency"}
    for flag in integrity_flags:
        if axis_for[flag["kind"]] not in degraded:
            degraded.append(axis_for[flag["kind"]])
    return {**payload, "baseline_degraded_axes": degraded, "eligible": not degraded,
            "source_integrity_flags": integrity_flags}


def run_comparison(args: argparse.Namespace, selection: list[dict[str, Any]]) -> None:
    if len(selection) > 36:
        raise ValueError("Hard cap is 36 examples")
    snapshot = Path(args.model_dir).resolve()
    if snapshot.parent.name == "snapshots" and snapshot.name != args.revision:
        raise ValueError(f"Snapshot directory/revision mismatch: {snapshot.name} != {args.revision}")
    run_identity = {
        "repo": args.repo,
        "revision": args.revision,
        "local_path": str(snapshot),
        "selection_sha256": file_hash(args.selection_file),
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": False,
        "batch_size": 1,
        "use_cache": True,
    }
    completed_path = args.output_dir / "comparison_manifest.json"
    if completed_path.exists():
        completed = json.loads(completed_path.read_text())
        if completed.get("status") == "complete" and completed.get("run_identity") == run_identity:
            print(f"reuse complete comparison: {completed_path}")
            return
        raise RuntimeError("Existing comparison manifest has a different or incomplete run identity")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded local-judge comparison")
    free_before, total_vram = torch.cuda.mem_get_info(0)
    torch.cuda.reset_peak_memory_stats(0)
    model_load_started = time.monotonic()
    judge = LocalQwenJudge(
        Path(args.model_dir), args.repo, args.revision, args.max_new_tokens, args.call_timeout_seconds
    )
    model_load_seconds = time.monotonic() - model_load_started
    identity_path = args.output_dir / "run_identity.json"
    checkpoint_path = args.output_dir / "results_checkpoint.jsonl"
    if identity_path.exists() and json.loads(identity_path.read_text()) != run_identity:
        raise RuntimeError("Existing checkpoint run identity differs from requested run")
    existing = read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    if [row["canonical_id"] for row in existing] != [row["canonical_id"] for row in selection[:len(existing)]]:
        raise RuntimeError("Existing checkpoint is not an exact prefix of the frozen selection")
    for result, selected in zip(existing, selection, strict=False):
        if result.get("clean_sha256") != selected["clean_sha256"] or result.get("candidate_sha256") != selected["candidate_sha256"]:
            raise RuntimeError("Existing checkpoint text provenance differs from frozen selection")
    identity_path.write_text(json.dumps(run_identity, indent=2, ensure_ascii=False) + "\n")
    results = list(existing)
    peak_before = torch.cuda.max_memory_allocated(0)
    pending = selection[len(results):]
    call_ledger_path = args.output_dir / "local_call_ledger.jsonl"
    if existing and not call_ledger_path.exists():
        write_jsonl(call_ledger_path, [{
            "canonical_id": result["canonical_id"],
            "call_kind": "paired_grade",
            "usage": result["usage"],
            "reconstructed_from_completed_checkpoint": True,
        } for result in existing])
    for row in pending:
        candidate_label = row["ab_candidate_label"]
        clean_label = "B" if candidate_label == "A" else "A"
        a = row["candidate_response"] if candidate_label == "A" else row["clean_response"]
        b = row["clean_response"] if candidate_label == "A" else row["candidate_response"]
        started = time.monotonic()
        (args.output_dir / "runtime_state.json").write_text(json.dumps({
            "status": "running",
            "completed_rows": len(results),
            "current_selection_index": row["selection_index"],
            "current_canonical_id": row["canonical_id"],
            "current_call_kind": "paired_grade",
            "execution_batch_size": 1,
            "call_timeout_seconds": args.call_timeout_seconds,
            "call_started_unix": time.time(),
        }, indent=2) + "\n")
        raw, usage = judge.call(PAIRED_PROMPT.format(question=row["question"], a=a, b=b))
        with call_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "canonical_id": row["canonical_id"], "call_kind": "paired_grade", "usage": usage,
            }, ensure_ascii=False, sort_keys=True) + "\n")
        result: dict[str, Any] = {
            "canonical_id": row["canonical_id"],
            "selection_index": row["selection_index"],
            "category": row["category"],
            "selected_stage_index": row["selected_stage_index"],
            "selected_semantic_attempt": row["selected_semantic_attempt"],
            "clean_sha256": row["clean_sha256"],
            "candidate_sha256": row["candidate_sha256"],
            "cumulative_intended_axes": row["cumulative_intended_axes"],
            "usage": usage,
            "raw_output": raw,
        }
        try:
            payload = parse_json_object(raw)
            result["json_valid"] = True
        except Exception as exc:
            result.update({
                "json_valid": False,
                "schema_valid": False,
                "parse_or_validation_error": f"{type(exc).__name__}: {exc}",
            })
            payload = None
        try:
            if payload is None:
                raise RuntimeError("JSON parsing failed")
            raw_nonempty = sum(
                bool(str(payload["scores"][axis][label].get("evidence_span") or "").strip())
                for axis in AXES for label in ("A", "B")
            ) + sum(
                bool(str(item.get("evidence_span") or "").strip())
                for item in payload.get("content_checks", {}).values()
            )
            invalid = validate_paired(payload, a, b)
            grade = map_paired(payload, clean_label, candidate_label)
            grade["invalid_evidence"] = invalid
            failure = content_failure(row["question"], row["candidate_response"], grade)
            missing = [axis for axis in row["cumulative_intended_axes"] if axis not in grade["realized_axes"]]
            local_stage_accept = not failure and not missing
            conflict_flags = []
            if grade["clean_scores"]["specificity"]["score"] == 1:
                conflict_flags.append("clean_specificity_score_1")
            if grade["clean_scores"]["medical_boundary"]["violation"]:
                conflict_flags.append("clean_medical_boundary_violation")
            if grade["text_reason_contradiction"]["detected"]:
                conflict_flags.append("judge_text_reason_contradiction")
            recheck = None
            recheck_usage = None
            if conflict_flags:
                (args.output_dir / "runtime_state.json").write_text(json.dumps({
                    "status": "running",
                    "completed_rows": len(results),
                    "current_selection_index": row["selection_index"],
                    "current_canonical_id": row["canonical_id"],
                    "current_call_kind": "clean_conflict_recheck",
                }, indent=2) + "\n")
                eraw, recheck_usage = judge.call(ELIGIBILITY_PROMPT.format(
                    question=row["question"], response=row["clean_response"]
                ))
                with call_ledger_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "canonical_id": row["canonical_id"],
                        "call_kind": "clean_conflict_recheck",
                        "usage": recheck_usage,
                    }, ensure_ascii=False, sort_keys=True) + "\n")
                recheck = parse_eligibility(parse_json_object(eraw), row)
            local_terminal_projection = "qc_conflict" if conflict_flags else (
                "accepted" if local_stage_accept else "rejected"
            )
            result.update({
                "schema_valid": True,
                "local_grade": grade,
                "local_stage_accept": local_stage_accept,
                "local_failure_reason": failure or ("missing_intended_axes:" + ",".join(missing) if missing else None),
                "conflict_flags": conflict_flags,
                "local_clean_recheck": recheck,
                "local_clean_recheck_usage": recheck_usage,
                "local_terminal_projection": local_terminal_projection,
                "evidence_reported_nonempty": raw_nonempty,
                "evidence_source_verified": max(0, raw_nonempty - len(invalid)),
                "old_gpt_grade": row["old_gpt_grade"],
                "old_gpt_realized_axes": row["old_gpt_grade"]["realized_axes"],
                "old_stage_accept": row["old_stage_accept"],
                "old_stage_reason": row["old_stage_reason"],
                "old_clean_recheck": row.get("old_clean_recheck"),
                "old_terminal_status": row["old_terminal_status"],
                "old_terminal_failure_reason": row["old_terminal_failure_reason"],
                "terminal_comparable": row["terminal_comparable"],
                "audit_hold_reason": row.get("audit_hold_reason"),
            })
        except Exception as exc:
            if result.get("json_valid"):
                result.update({
                    "schema_valid": False,
                    "parse_or_validation_error": f"{type(exc).__name__}: {exc}",
                })
        # `started` precedes the local model call, so adding usage time here would double-count it.
        result["wall_seconds"] = time.monotonic() - started
        results.append(result)
        write_jsonl(checkpoint_path, results)
        (args.output_dir / "runtime_state.json").write_text(json.dumps({
            "status": "running", "completed_rows": len(results), "current_call_kind": None,
        }, indent=2) + "\n")
        print(json.dumps({
            "progress": f"{len(results)}/{len(selection)}",
            "canonical_id": row["canonical_id"],
            "paired_seconds": usage["elapsed_seconds"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "eos_reached": usage["eos_reached"],
            "stop_reason": usage["stop_reason"],
        }, ensure_ascii=False), flush=True)

    json_valid = [row for row in results if row.get("json_valid")]
    valid = [row for row in results if row.get("schema_valid")]
    reported = sum(row.get("evidence_reported_nonempty", 0) for row in valid)
    verified = sum(row.get("evidence_source_verified", 0) for row in valid)
    manifest = json.loads(args.selection_manifest.read_text())
    per_axis = {}
    for axis in AXES:
        comparable = [row for row in valid if row.get("old_gpt_realized_axes") is not None]
        value_key = "violation" if axis == "medical_boundary" else "score"
        per_axis[axis] = {
            "old_gpt_realized": sum(axis in row["old_gpt_realized_axes"] for row in comparable),
            "local_realized": sum(axis in row["local_grade"]["realized_axes"] for row in comparable),
            "realized_agreement": sum(
                (axis in row["old_gpt_realized_axes"]) == (axis in row["local_grade"]["realized_axes"])
                for row in comparable
            ),
            "clean_score_agreement": sum(
                row["old_gpt_grade"]["clean_scores"][axis][value_key]
                == row["local_grade"]["clean_scores"][axis][value_key]
                for row in comparable
            ),
            "candidate_score_agreement": sum(
                row["old_gpt_grade"]["candidate_scores"][axis][value_key]
                == row["local_grade"]["candidate_scores"][axis][value_key]
                for row in comparable
            ),
            "score_pair_agreement": sum(
                row["old_gpt_grade"]["clean_scores"][axis][value_key]
                == row["local_grade"]["clean_scores"][axis][value_key]
                and row["old_gpt_grade"]["candidate_scores"][axis][value_key]
                == row["local_grade"]["candidate_scores"][axis][value_key]
                for row in comparable
            ),
            "denominator": len(comparable),
        }
    manifest.update({
        "status": "complete",
        "run_identity": run_identity,
        "model": {"repo": args.repo, "revision": args.revision, "local_path": str(snapshot)},
        "local_judge_config": {
            "official_chat_template_sha256": hashlib.sha256(judge.tokenizer.chat_template.encode()).hexdigest(),
            "enable_thinking": False,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "call_timeout_seconds": args.call_timeout_seconds,
            "torch_dtype": "bfloat16",
            "trust_remote_code": False,
            "openai_fallback": False,
            "optional_kernel_imports": judge.optional_kernel_imports,
            "delta_kernel_status": judge.delta_kernel_status,
            "parameter_devices": judge.parameter_devices,
            "cpu_parameter_count": judge.cpu_parameter_count,
        },
        "processed": len(results),
        "json_valid": len(json_valid),
        "schema_valid": len(valid),
        "local_stage_accept": sum(bool(row.get("local_stage_accept")) for row in valid),
        "old_gpt_stage_accept": sum(bool(row.get("old_stage_accept")) for row in valid),
        "stage_acceptance_agreement": sum(
            bool(row.get("local_stage_accept")) == bool(row.get("old_stage_accept")) for row in valid
        ),
        "stage_acceptance_denominator": len(valid),
        "terminal_comparable_count": sum(bool(row.get("terminal_comparable")) for row in valid),
        "terminal_status_agreement": sum(
            row.get("local_terminal_projection") == row.get("old_terminal_status")
            for row in valid if row.get("terminal_comparable")
        ),
        "per_axis_realization": per_axis,
        "evidence_reported_nonempty": reported,
        "evidence_source_verified": verified,
        "evidence_source_verified_rate": verified / reported if reported else None,
        "medical_realized_disagreements": sum(
            (("medical_boundary" in row["local_grade"]["realized_axes"]) != ("medical_boundary" in row["old_gpt_realized_axes"]))
            for row in valid
        ),
        "harm_realized_disagreements": sum(
            (("toxicity_or_harm" in row["local_grade"]["realized_axes"]) != ("toxicity_or_harm" in row["old_gpt_realized_axes"]))
            for row in valid
        ),
        "wall_seconds": sum(row["wall_seconds"] for row in results),
        "model_load_seconds": model_load_seconds,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_total_bytes": int(total_vram),
        "cuda_free_bytes_before_model_load": int(free_before),
        "peak_cuda_bytes": max(peak_before, torch.cuda.max_memory_allocated(0)),
    })
    by_id = {row["canonical_id"]: row for row in selection}
    disagreements = []
    for result in valid:
        old_realized = set(result["old_gpt_realized_axes"])
        local_realized = set(result["local_grade"]["realized_axes"])
        important_axis_difference = bool(
            (old_realized ^ local_realized) & {"medical_boundary", "toxicity_or_harm"}
        )
        stage_difference = bool(result["local_stage_accept"] != result["old_stage_accept"])
        if important_axis_difference or stage_difference:
            source = by_id[result["canonical_id"]]
            disagreements.append({
                "canonical_id": result["canonical_id"],
                "selection_index": result["selection_index"],
                "category": result["category"],
                "question": source["question"],
                "clean_response": source["clean_response"],
                "candidate_response": source["candidate_response"],
                "cumulative_intended_axes": result["cumulative_intended_axes"],
                "old_gpt_stage_accept": result["old_stage_accept"],
                "local_stage_accept": result["local_stage_accept"],
                "old_gpt_grade": source["old_gpt_grade"],
                "local_grade": result["local_grade"],
                "audit_hold_reason": source.get("audit_hold_reason"),
            })
    write_jsonl(args.output_dir / "important_disagreements.jsonl", disagreements)
    manifest["important_disagreement_count"] = len(disagreements)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (args.output_dir / "runtime_state.json").write_text(json.dumps({
        "status": "complete", "completed_rows": len(results), "current_call_kind": None,
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_OUTPUT / "selection.jsonl")
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_OUTPUT / "selection_manifest.json")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--repo")
    parser.add_argument("--revision")
    parser.add_argument("--max-new-tokens", type=int, default=2400)
    parser.add_argument("--call-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if args.selection_file.exists():
        original = read_jsonl(args.selection_file)
        selection = enrich_frozen_selection(original)
        if selection != original:
            previous_hash = file_hash(args.selection_file)
            write_jsonl(args.selection_file, selection)
            manifest = json.loads(args.selection_manifest.read_text())
            manifest["selection_provenance_upgrade"] = {
                "previous_sha256": previous_hash,
                "ids_and_order_preserved": True,
                "stage_specific_gpt_grade": True,
            }
            manifest["selection_sha256"] = file_hash(args.selection_file)
            args.selection_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    else:
        selection = build_selection(args.selection_file.parent)
        selection = enrich_frozen_selection(selection)
        write_jsonl(args.selection_file, selection)
    if args.select_only:
        return
    if not all((args.model_dir, args.repo, args.revision)):
        raise ValueError("--model-dir, --repo, and --revision are required; no model name is inferred")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_comparison(args, selection)


if __name__ == "__main__":
    main()
