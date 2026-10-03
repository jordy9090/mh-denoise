#!/usr/bin/env python3
"""Bounded, resumable full-paper corruption production from an explicit selection.

This generalizes the immutable 120-row development builder without altering it.
The judge rubric, Qwen generator, axis operators, acceptance rule, retry limits,
and deterministic generation seeds are imported unchanged from that builder.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import fcntl
import hashlib
import json
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from build_development_corruption_shard import (
    ACCEPTANCE_RULES,
    BUILD_JUDGE_MAX_TOKENS,
    BUILD_SEED,
    base_record,
    canonical_json_hash,
    clean_qc_hash,
    generation_prompt_hash,
    paired_prompt_hash,
    process_one,
    candidate_a,
    editing_meta,
)
from corruption_contract_v2 import AXES, StageRequest, parse_json_object
from fullpaper_acl_pipeline import DEFAULT_OUTPUT_DIR, read_jsonl, sha256_file, write_json, write_jsonl
from run_paired_generator_diagnostic import (
    DEFAULT_REQUIRED_FREE_VRAM_MIB,
    ELIGIBILITY_PROMPT,
    ExternalJudge,
    GENERATORS,
    JUDGE_MODEL,
    LocalGenerator,
    gpu_status,
    load_repo_env,
)
from shared_api_budget import BudgetConfig, BudgetExceeded, SharedApiBudget
from source_integrity_contract import VERSION as SOURCE_INTEGRITY_VERSION
from source_integrity_contract import contract_hash, exact_offset


LOCAL_JUDGE_VERSION = "local-qwen35-27b-paired-qc-v6-20260926"

CLEAN_PREFLIGHT_RETRY_INSTRUCTION = """

Your previous clean-target JSON failed validation.
Validation error: {validation_error}

Return the complete JSON object again. Evidence from the question or response
must be one contiguous exact verbatim substring; do not paraphrase, normalize
whitespace, join passages, or use ellipses. Use whole_response with an empty
span only for a genuinely holistic judgment. Keep every required key and
return JSON only.

Previous output for correction:
<<<PREVIOUS_JSON>>>{previous_output}<<<END_PREVIOUS_JSON>>>
"""


VERSION = "fullpaper-corruption-production-v1"
DEFAULT_HOLDS = DEFAULT_OUTPUT_DIR / "development_training_shard_120_corrected_v2/held_out.jsonl"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".tmp.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".tmp.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def initialize_run_contract(output: Path, contract: dict[str, Any]) -> str:
    fingerprint = canonical_json_hash(contract)
    contract_payload = {**contract, "run_fingerprint": fingerprint}
    contract_path = output / "run_contract.json"
    with exclusive_lock(output / ".run_contract.lock"):
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            existing_fingerprint = existing.pop("run_fingerprint", None)
            if (
                existing_fingerprint != fingerprint
                or canonical_json_hash(existing) != fingerprint
            ):
                raise RuntimeError("Existing run contract fingerprint/configuration differs; refusing checkpoint reuse")
        else:
            stale = [
                path.name for path in output.iterdir()
                if path.name.startswith("results_checkpoint.") or path.name in {"accepted.jsonl", "train_sft.jsonl", "valid_sft.jsonl"}
            ]
            if stale:
                raise RuntimeError(f"Checkpoint artifacts exist without a run contract; refusing reuse: {stale}")
            atomic_write_json(contract_path, contract_payload)
    return fingerprint


def source_verified_spans(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return only explicit, typed local labels with exact source offsets.

    Response-level scores, holistic judgments, and omissions remain in paired
    QC and are deliberately not promoted to local scorer labels.
    """
    verified: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    grade = row["final_grade"]
    texts = {"clean": row["clean_response"], "candidate": row["corrupted_response"]}
    if "local_supervision" not in grade:
        for side in ("clean", "candidate"):
            for axis, score in grade.get(f"{side}_scores", {}).items():
                evidence = str(score.get("evidence_span") or "")
                if evidence:
                    excluded.append({
                        "canonical_id": row["canonical_id"], "axis": axis, "side": side,
                        "reported_evidence": evidence,
                        "reason": "legacy_untyped_response_evidence_not_local_supervision",
                    })
    for annotation in grade.get("local_supervision", []):
        side = str(annotation.get("side") or "")
        source_text = texts.get(side)
        record = {"canonical_id": row["canonical_id"], **annotation}
        if source_text is None:
            excluded.append({**record, "reason": "invalid_local_supervision_side"})
            continue
        evidence = str(annotation.get("text") or "")
        offset = exact_offset(source_text, evidence)
        expected_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        valid_label_scope = (
            (annotation.get("scope") == "local_defect" and annotation.get("label") == 1)
            or (annotation.get("scope") == "local_support" and annotation.get("label") == 0)
        )
        if (
            offset is None
            or int(annotation.get("start", -1)) != offset["start"]
            or int(annotation.get("end", -1)) != offset["end"]
            or annotation.get("source_sha256") != expected_hash
            or not valid_label_scope
        ):
            excluded.append({**record, "reason": "invalid_typed_local_supervision"})
        else:
            verified.append(record)
    for item in grade.get("invalid_evidence", []):
        excluded.append(
            {
                "canonical_id": row["canonical_id"],
                "axis": item.get("axis"),
                "side": "raw_label_" + str(item.get("response_label", "unknown")),
                "reported_evidence": item.get("invalid_span", ""),
                "reported_validation": item.get("reason", "invalid_evidence"),
                "reason": "judge_evidence_validator_rejected_non_verbatim_evidence",
            }
        )
    return verified, excluded


def selection_hash(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def text_edit_trace(before: str, after: str) -> dict[str, Any]:
    """Record exact changed locations as provenance without inferring labels."""
    hunks = []
    for tag, before_start, before_end, after_start, after_end in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        hunks.append({
            "operation": tag,
            "before_start": before_start,
            "before_end": before_end,
            "before_text": before[before_start:before_end],
            "after_start": after_start,
            "after_end": after_end,
            "after_text": after[after_start:after_end],
        })
    return {
        "before_sha256": hashlib.sha256(before.encode("utf-8")).hexdigest(),
        "after_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
        "changed_hunks": hunks,
        "changed_hunk_count": len(hunks),
        "label_status": "provenance_only_not_ground_truth",
    }


def load_selection(
    path: Path,
    max_inputs: int,
    canonical: dict[str, dict[str, Any]],
    clean_qc: dict[str, dict[str, Any]],
    held_ids: set[str],
    allowed_splits: set[str],
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Selection must be a nonempty list or an object containing rows")
    selected = []
    seen = set()
    seen_questions: set[str] = set()
    for slot, spec in enumerate(rows[:max_inputs]):
        canonical_id = str(spec.get("canonical_id", ""))
        axes = list(spec.get("intended_axes") or [])
        if canonical_id in seen or canonical_id not in canonical:
            raise ValueError(f"Duplicate or unknown canonical_id at selection row {slot}: {canonical_id}")
        row = canonical[canonical_id]
        qc = clean_qc.get(canonical_id)
        split = row.get("split")
        if canonical_id in held_ids:
            raise ValueError(f"Selection contains a known held canonical_id: {canonical_id}")
        if split not in allowed_splits:
            raise ValueError(f"Selection row split {split!r} is not enabled: {canonical_id}")
        if spec.get("split") not in (None, split):
            raise ValueError(f"Selection/canonical split mismatch: {canonical_id}")
        if (
            not qc
            or qc.get("qc_ok") is not True
            or qc.get("eligible") is not True
            or qc.get("baseline_degraded_axes") != []
        ):
            raise ValueError(f"Selection row is not a clean-QC-passing target: {canonical_id}")
        question_group = row.get("question_normalized_sha256")
        if not question_group:
            raise ValueError(f"Missing normalized question identifier: {canonical_id}")
        if question_group in seen_questions:
            raise ValueError(f"More than one response selected for normalized question {question_group}")
        if not 1 <= len(axes) <= 3:
            raise ValueError(f"Selection row has invalid axis count: {canonical_id}")
        allowed = {
            "overall_quality", "empathy", "specificity", "factual_consistency",
            "medical_boundary", "toxicity_or_harm",
        }
        if len(set(axes)) != len(axes) or not set(axes) <= allowed:
            raise ValueError(f"Selection row has invalid axes: {canonical_id}: {axes}")
        generation_objective = str(spec.get("generation_objective") or "response_pair")
        if generation_objective not in {"response_pair", "local_scorer"}:
            raise ValueError(
                f"Selection row has invalid generation objective: {canonical_id}: {generation_objective}"
            )
        if generation_objective == "local_scorer" and set(axes) - {"empathy", "specificity"}:
            raise ValueError(
                f"local_scorer selection supports only empathy/specificity: {canonical_id}: {axes}"
            )
        seen.add(canonical_id)
        seen_questions.add(question_group)
        selected.append(
            {
                "slot": slot,
                "clean": row,
                "eligibility": qc,
                "intended_axes": axes,
                "axis_count": len(axes),
                "generation_seeds": spec.get("generation_seeds", []),
                "generation_objective": generation_objective,
                "split": split,
            }
        )
    if not selected:
        raise ValueError("Processing cap selected zero inputs")
    for field in ("question_normalized_sha256", "duplicate_cluster_id", "source_group_id"):
        train_groups = {item["clean"].get(field) for item in selected if item["split"] == "train"}
        valid_groups = {item["clean"].get(field) for item in selected if item["split"] == "valid"}
        overlap = (train_groups - {None}) & (valid_groups - {None})
        if overlap:
            raise ValueError(f"TRAIN/VALID {field} overlap in production selection: {sorted(overlap)[:5]}")
    return selected


def export(output: Path, results: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    from fullpaper_scorer_export_contract import (
        VERSION as SCORER_EXPORT_CONTRACT_VERSION,
        assert_no_non_null_collisions,
        find_non_null_collisions,
        mask_collisions,
    )

    accepted = [row for row in results if row["status"] == "accepted"]
    conflicts = [row for row in results if row["status"] == "qc_conflict"]
    holds = [row for row in results if row["status"] == "qc_hold"]
    rejected = [row for row in results if row["status"] not in ("accepted", "qc_conflict", "qc_hold")]
    sft_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    dpo_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "valid": []}
    all_spans: list[dict[str, Any]] = []
    excluded_spans: list[dict[str, Any]] = []
    response_level_evidence: list[dict[str, Any]] = []
    for row in accepted:
        model_input = {"question": row["question"], "corrupted_response": row["corrupted_response"]}
        metadata = {
            key: row[key]
            for key in (
                "canonical_id", "source", "source_component", "intended_axes", "realized_axes",
                "unintended_axes", "axis_count", "judge_model", "generator_repo", "generator_revision",
                "generation_objective",
            )
            if key in row
        }
        metadata.update(
            {
                "question_normalized_sha256": row["question_normalized_sha256"],
                "duplicate_cluster_id": row["duplicate_cluster_id"],
                "source_group_id": row["source_group_id"],
            }
        )
        metadata["paired_qc"] = {
            key: row["final_grade"][key]
            for key in (
                "clean_scores", "candidate_scores", "deltas", "content_checks", "invalid_evidence",
                "clean_target_eligibility", "candidate_target_eligibility",
                "response_eligibility", "response_content_checks",
            )
            if key in row["final_grade"]
        }
        verified, excluded = source_verified_spans(row)
        all_spans.extend(verified)
        excluded_spans.extend(excluded)
        row_response_evidence = [
            {"canonical_id": row["canonical_id"], **item}
            for item in row["final_grade"].get("response_level_evidence", [])
        ]
        response_level_evidence.extend(row_response_evidence)
        metadata.update(
            {
                "span_supervision": verified,
                "response_level_qc_evidence": row_response_evidence,
                "source_verified_span_count": len(verified),
                "excluded_unverified_evidence_count": len(excluded),
                "source_integrity_contract": SOURCE_INTEGRITY_VERSION,
                "source_integrity_contract_sha256": contract_hash(),
                "split": row["split"],
                "generator_prompt_version": row.get("generator_prompt_version"),
                "judge_prompt_version": row.get("judge_prompt_version"),
                "judge_repo": row.get("judge_repo", row.get("judge_model")),
                "judge_revision": row.get("judge_revision"),
            }
        )
        sft_by_split[row["split"]].append({"input": model_input, "target": row["clean_response"], "metadata": metadata})
        dpo_by_split[row["split"]].append({"input": model_input, "chosen": row["clean_response"], "rejected": row["corrupted_response"], "metadata": metadata})

    question_by_id = {row["canonical_id"]: row["question"] for row in accepted}
    scorer_rows = []
    for span in all_spans:
        axis = span["axis"]
        scorer_rows.append({
            "canonical_id": span["canonical_id"],
            "question": question_by_id[span["canonical_id"]],
            "span": span["text"],
            "span_side": span.get("side"),
            "span_start": span.get("start"),
            "span_end": span.get("end"),
            "split": next(row["split"] for row in accepted if row["canonical_id"] == span["canonical_id"]),
            "labels": {name: (span.get("label") if name == axis else None) for name in AXES},
            "label_mask": {name: span.get("label") in (0, 1) and name == axis for name in AXES},
            "label_basis": {name: (f"explicit_{span.get('scope')}_exact_span" if name == axis else "unknown") for name in AXES},
            "source_records": [span],
        })
    original_collisions = find_non_null_collisions(scorer_rows)
    scorer_conflict_ledger = mask_collisions(
        scorer_rows, original_collisions, stage="production_string_input"
    )
    for scorer_row, span in zip(scorer_rows, all_spans, strict=True):
        axis = span["axis"]
        if scorer_row["labels"][axis] is None and span.get("label") in (0, 1):
            span["original_label"] = span["label"]
            span["label"] = None
            span["label_mask"] = False
            span["mask_reason"] = "same_scorer_input_axis_opposite_labels"
    assert_no_non_null_collisions(scorer_rows)
    local_label_counts = Counter(
        span["label"] for span in all_spans if span.get("label") in (0, 1)
    )
    semantic_training_approved = bool(
        manifest.get("semantic_review", {}).get("approved_for_training")
    )
    local_classes_present = local_label_counts[1] > 0 and local_label_counts[0] > 0
    clean_preflight_results = [
        row.get("automatic_clean_preflight") for row in results
        if row.get("automatic_clean_preflight") is not None
    ]
    clean_preflight_technical = sum(
        row.get("failure_reason") == "automatic_clean_preflight_infrastructure_exhausted"
        for row in results
    )
    typed_local_contract = manifest.get("version") == LOCAL_JUDGE_VERSION
    checks = {
        "terminal_partition": len(accepted) + len(rejected) + len(conflicts) + len(holds) == len(results),
        "accepted_intended_subset_realized": all(set(row["intended_axes"]) <= set(row["realized_axes"]) for row in accepted),
        "accepted_clean_canonical_split_only": all(row["split"] in {"train", "valid"} and row["baseline_degraded_axes"] == [] for row in accepted),
        "sft_dpo_count_matches": sum(map(len, sft_by_split.values())) == sum(map(len, dpo_by_split.values())) == len(accepted),
        "pair_contract": all(
            s["target"] == d["chosen"] and s["input"]["corrupted_response"] == d["rejected"]
            for split in ("train", "valid")
            for s, d in zip(sft_by_split[split], dpo_by_split[split], strict=True)
        ),
        "span_offsets_exact": all(
            (
                row["clean_response"] if span["side"] == "clean" else row["corrupted_response"]
            )[span["start"] : span["end"]]
            == span["text"]
            for row in accepted
            for span in next(
                item["metadata"]["span_supervision"]
                for item in sft_by_split[row["split"]]
                if item["metadata"]["canonical_id"] == row["canonical_id"]
            )
        ),
        "clean_gate_passed_for_every_accepted": all(
            row["final_grade"].get("clean_target_eligibility", {}).get("disposition") == "pass"
            for row in accepted
        ) if typed_local_contract else True,
        "no_clean_local_defect_in_accepted": all(
            not any(
                item.get("side") == "clean"
                and item.get("scope") == "local_defect"
                and item.get("label") == 1
                for item in row["final_grade"].get("local_supervision", [])
            )
            for row in accepted
        ) if typed_local_contract else True,
        "no_response_level_scope_promoted_to_local": all(
            span.get("scope") in {"local_defect", "local_support"}
            for span in all_spans
        ),
        "no_non_null_scorer_input_axis_conflicts": not find_non_null_collisions(scorer_rows),
    }
    if not all(checks.values()):
        raise AssertionError(checks)

    artifact_rows = {
        "accepted.jsonl": accepted,
        "rejected.jsonl": rejected,
        "qc_conflicts.jsonl": conflicts,
        "qc_holds.jsonl": holds,
        "train_sft.jsonl": sft_by_split["train"],
        "train_dpo.jsonl": dpo_by_split["train"],
        "valid_sft.jsonl": sft_by_split["valid"],
        "valid_dpo.jsonl": dpo_by_split["valid"],
        "source_verified_spans.jsonl": all_spans,
        "excluded_unverified_evidence.jsonl": excluded_spans,
        "response_level_qc_evidence.jsonl": response_level_evidence,
        "scorer_label_conflicts.jsonl": scorer_conflict_ledger,
    }
    manifest.update(
        {
            "status": "complete",
            "processed_inputs": len(results),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "qc_conflicts": len(conflicts),
            "qc_holds": len(holds),
            "accepted_by_split": dict(Counter(row["split"] for row in accepted)),
            "source_verified_spans": len(all_spans),
            "local_scorer_label_counts": {
                "positive": local_label_counts[1], "negative": local_label_counts[0],
                "unknown_masked_by_conflict": sum(span.get("label") is None for span in all_spans),
            },
            "local_scorer_supervision_status": {
                "has_positive_and_negative": local_classes_present,
                "training_ready": local_classes_present and semantic_training_approved,
                "note": (
                    "binary class presence is descriptive; semantic approval is also required"
                ),
            },
            "automatic_clean_preflight_counts": {
                "enabled": bool(manifest.get("automatic_clean_preflight")),
                "denominator": len(results),
                "pass": sum(result.get("eligible") is True for result in clean_preflight_results),
                "hold": sum(result.get("eligible") is False for result in clean_preflight_results),
                "technical_failure": clean_preflight_technical,
            },
            "structural_validation": {"passed": True, "checks": checks},
            "training_readiness": {
                "ready": local_classes_present and semantic_training_approved,
                "blocking_reasons": [
                    reason for reason, blocked in (
                        ("local_scorer_missing_positive_or_negative", not local_classes_present),
                        ("semantic_review_not_approved", not semantic_training_approved),
                    ) if blocked
                ],
                "not_inferred_from_binary_class_presence": True,
            },
            "scorer_export_contract_version": SCORER_EXPORT_CONTRACT_VERSION,
            "scorer_input_conflicts": {
                "original_groups": len(original_collisions),
                "masked_groups": len(scorer_conflict_ledger),
                "remaining_groups": len(find_non_null_collisions(scorer_rows)),
            },
            "response_level_qc_evidence": len(response_level_evidence),
            "excluded_unverified_evidence": len(excluded_spans),
            "data_contract_checks": checks,
        }
    )
    publish_names = list(artifact_rows)
    existing = [name for name in publish_names if (output / name).exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing published production artifacts: {existing}")
    with tempfile.TemporaryDirectory(dir=output, prefix=".export_staging.") as stage_name:
        stage = Path(stage_name)
        for name, rows in artifact_rows.items():
            write_jsonl(stage / name, rows)
        manifest["artifact_sha256"] = {
            name: sha256_file(stage / name) for name in publish_names
        }
        write_json(stage / "production_manifest.json", manifest)
        for name in publish_names:
            os.replace(stage / name, output / name)
        os.replace(stage / "production_manifest.json", output / "production_manifest.json")


class LocalProductionJudge:
    """Persistent Qwen3.5-27B backend with the frozen local production rubric."""

    def __init__(self, model_dir: Path, max_tokens: int, timeout_seconds: float, calls_dir: Path) -> None:
        from local_qwen_production_qc_v4 import (
            JUDGE_REPO, JUDGE_REVISION, SYSTEM_PROMPT, parse_eligibility,
        )
        from run_local_qwen_judge_comparison import LocalQwenJudge

        self.repo = JUDGE_REPO
        self.revision = JUDGE_REVISION
        self._parse_eligibility = parse_eligibility
        self.calls_dir = calls_dir
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self.call_file_index = len(list(self.calls_dir.glob("call_*.json")))
        self.context: dict[str, Any] = {}
        self.backend = LocalQwenJudge(
            model_dir, self.repo, self.revision, max_tokens, timeout_seconds,
            system_prompt=SYSTEM_PROMPT,
        )

    @property
    def calls(self) -> int:
        return self.backend.calls

    @property
    def prompt_tokens(self) -> int:
        return self.backend.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.backend.completion_tokens

    @property
    def elapsed(self) -> float:
        return self.backend.elapsed

    def set_context(self, **context: Any) -> None:
        self.context = context

    def _record_call(self, payload: dict[str, Any]) -> None:
        self.call_file_index += 1
        atomic_write_json(self.calls_dir / f"call_{self.call_file_index:06d}.json", payload)

    def call(self, prompt: str) -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        try:
            raw, usage = self.backend.call(prompt)
        except Exception as exc:
            self._record_call({
                "call_kind": "judge", "context": self.context,
                "status": "model_exception", "error": f"{type(exc).__name__}: {exc}",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
            })
            raise
        self._record_call({
            "call_kind": "judge", "context": self.context, "status": "returned",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_output": raw, "usage": usage,
        })
        return raw, usage

    def record_generation(self, context: dict[str, Any], candidate: Any, elapsed_seconds: float) -> None:
        self._record_call({
            "call_kind": "generator", "context": context, "status": "returned",
            "raw_output_sha256": hashlib.sha256(candidate.raw_output.encode()).hexdigest(),
            "raw_output": candidate.raw_output,
            "usage": {
                "current_response_tokens": candidate.current_response_tokens,
                "max_new_tokens": candidate.max_new_tokens,
                "output_tokens": candidate.generated_tokens,
                "eos_reached": candidate.eos_reached,
                "truncation_reason": candidate.truncation_reason,
                "elapsed_seconds": elapsed_seconds,
                "enable_thinking": False,
                "use_cache": True,
            },
        })

    def eligibility(self, row: dict[str, Any]) -> dict[str, Any]:
        from run_paired_generator_diagnostic import ELIGIBILITY_PROMPT

        self.set_context(canonical_id=row["canonical_id"], call_purpose="clean_conflict_recheck")
        raw, usage = self.call(ELIGIBILITY_PROMPT.format(
            question=row["question"], response=row["clean_response"]
        ))
        payload = self._parse_eligibility(parse_json_object(raw), row)
        payload["local_judge_usage"] = usage
        payload["raw_output"] = raw
        return payload

    def metadata(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "local_path": str(self.backend.model_dir),
            "max_new_tokens": self.backend.max_new_tokens,
            "call_timeout_seconds": self.backend.call_timeout_seconds,
            "enable_thinking": False,
            "do_sample": False,
            "use_cache": True,
            "batch_size": 1,
            "parameter_devices": self.backend.parameter_devices,
            "cpu_parameter_count": self.backend.cpu_parameter_count,
            "optional_kernel_imports": self.backend.optional_kernel_imports,
            "delta_kernel_status": self.backend.delta_kernel_status,
        }

    def close(self) -> None:
        import gc
        import torch

        del self.backend.model
        gc.collect()
        torch.cuda.empty_cache()


def _selection_generation_seed(item: dict[str, Any], stage: int, attempt: int) -> int:
    for record in item.get("generation_seeds", []):
        if int(record.get("stage_index", -1)) == stage:
            value = record.get(f"attempt_{attempt}")
            if value is not None:
                return int(value)
    from run_paired_generator_diagnostic import deterministic_generation_seed

    return deterministic_generation_seed(BUILD_SEED, item["clean"]["canonical_id"], stage, attempt)


def paired_grade_local(
    judge: LocalProductionJudge,
    row: dict[str, Any],
    candidate: str,
    stage: int,
    attempt: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    from local_qwen_production_qc_v4 import paired_prompt, validate_and_map

    candidate_label = "A" if candidate_a(row["canonical_id"], stage, attempt) else "B"
    clean_label = "B" if candidate_label == "A" else "A"
    a = candidate if candidate_label == "A" else row["clean_response"]
    b = row["clean_response"] if candidate_label == "A" else candidate
    infrastructure = []
    validation_error: str | None = None
    previous_output: str | None = None
    best_quarantined: tuple[int, dict[str, Any], str, dict[str, Any]] | None = None

    def complete_grade(
        grade: dict[str, Any], raw_output: str, call_usage: dict[str, Any]
    ) -> dict[str, Any]:
        grade.update({
            "raw_output": raw_output,
            "local_judge_usage": call_usage,
            "response_a_sha256": hashlib.sha256(a.encode()).hexdigest(),
            "response_b_sha256": hashlib.sha256(b.encode()).hexdigest(),
            "response_a_chars": len(a),
            "response_b_chars": len(b),
        })
        grade["unintended_axes"] = [
            axis for axis in grade["realized_axes"] if axis not in row["intended_axes"]
        ]
        return grade

    for infrastructure_attempt in range(1, ACCEPTANCE_RULES["max_infrastructure_attempts"] + 1):
        raw = None
        try:
            judge.set_context(
                canonical_id=row["canonical_id"], call_purpose="paired_stage_qc",
                stage_index=stage, semantic_attempt=attempt,
                infrastructure_attempt=infrastructure_attempt,
                validation_feedback=validation_error,
            )
            raw, usage = judge.call(paired_prompt(
                question=row["question"], a=a, b=b,
                validation_error=validation_error,
                previous_output=previous_output,
            ))
            payload = parse_json_object(raw)
            grade, _ = validate_and_map(
                payload, a, b, clean_label, candidate_label, question=row["question"]
            )
            return complete_grade(grade, raw, usage), infrastructure
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            if raw is not None:
                previous_output = raw
                try:
                    fallback_payload = parse_json_object(raw)
                    fallback_grade, fallback_invalid = validate_and_map(
                        fallback_payload, a, b, clean_label, candidate_label,
                        question=row["question"], quarantine_invalid_evidence=True,
                    )
                    candidate_fallback = (
                        len(fallback_invalid), fallback_grade, raw, usage
                    )
                    if best_quarantined is None or candidate_fallback[0] < best_quarantined[0]:
                        best_quarantined = candidate_fallback
                except Exception:
                    pass
            infrastructure.append({
                "kind": "local_paired_judge_parse_validation_or_timeout",
                "attempt": infrastructure_attempt,
                "reason": validation_error,
                "feedback_added_to_next_attempt": (
                    infrastructure_attempt < ACCEPTANCE_RULES["max_infrastructure_attempts"]
                ),
            })
    if best_quarantined is not None:
        invalid_count, grade, raw, usage = best_quarantined
        grade = complete_grade(grade, raw, usage)
        grade["strict_validation_attempts_exhausted"] = True
        grade["quarantined_invalid_evidence_count"] = invalid_count
        infrastructure.append({
            "kind": "local_paired_judge_non_verbatim_evidence_quarantined",
            "attempt": ACCEPTANCE_RULES["max_infrastructure_attempts"],
            "reason": (
                f"strict validation exhausted; removed {invalid_count} non-verbatim "
                "evidence item(s) without creating local labels"
            ),
        })
        return grade, infrastructure
    return None, infrastructure


def automatic_clean_preflight_local(
    judge: LocalProductionJudge, row: dict[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Run clean-only local QC before generation with bounded schema retries."""
    failures: list[dict[str, Any]] = []
    validation_error: str | None = None
    previous_output: str | None = None
    for attempt in range(1, ACCEPTANCE_RULES["max_infrastructure_attempts"] + 1):
        prompt = ELIGIBILITY_PROMPT.format(
            question=row["question"], response=row["clean_response"]
        )
        if validation_error is not None:
            prompt += CLEAN_PREFLIGHT_RETRY_INSTRUCTION.format(
                validation_error=validation_error,
                previous_output=previous_output or "",
            )
        raw = None
        try:
            judge.set_context(
                canonical_id=row["canonical_id"],
                call_purpose="automatic_clean_preflight",
                infrastructure_attempt=attempt,
                validation_feedback=validation_error,
            )
            raw, usage = judge.call(prompt)
            payload = parse_json_object(raw)
            for axis, item in payload.get("axes", {}).items():
                if not item.get("material_degradation"):
                    continue
                source = item.get("evidence_source")
                span = str(item.get("evidence_span") or "")
                if source == "response":
                    if exact_offset(row["clean_response"], span) is None:
                        raise ValueError(
                            f"non-verbatim clean-preflight response evidence for {axis}"
                        )
                elif source == "question":
                    if exact_offset(row["question"], span) is None:
                        raise ValueError(
                            f"non-verbatim clean-preflight question evidence for {axis}"
                        )
                elif source == "whole_response":
                    if span:
                        raise ValueError(
                            f"whole-response clean-preflight evidence must have empty span for {axis}"
                        )
                else:
                    raise ValueError(
                        f"material clean-preflight evidence has invalid source for {axis}: {source!r}"
                    )
            result = judge._parse_eligibility(payload, row)
            result.update({
                "attempt": attempt,
                "local_judge_usage": usage,
                "raw_output": raw,
            })
            return result, failures
        except Exception as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            if raw is not None:
                previous_output = raw
            failures.append({
                "kind": "automatic_clean_preflight_parse_validation_or_timeout",
                "attempt": attempt,
                "reason": validation_error,
                "feedback_added_to_next_attempt": (
                    attempt < ACCEPTANCE_RULES["max_infrastructure_attempts"]
                ),
            })
    return None, failures


def process_one_local(
    item: dict[str, Any], backend: LocalGenerator, judge: LocalProductionJudge
) -> dict[str, Any]:
    from local_qwen_production_qc_v4 import content_disposition
    from fullpaper_acl_pipeline import normalize_text

    row = item["clean"]
    infrastructure: list[dict[str, Any]] = []
    automatic_clean_preflight = None
    contextual_clean_review = row.get("contextual_clean_review")
    if contextual_clean_review is not None:
        from source_integrity_contract import validate_contextual_clean_review

        reviewed = validate_contextual_clean_review(
            row["question"], row["clean_response"], contextual_clean_review
        )
        if reviewed["disposition"] != "pass":
            return {
                "status": "qc_hold",
                "failure_reason": "contextual_clean_review_required",
                "contextual_clean_review": reviewed,
                "stage_history": [],
                "infrastructure_failures": infrastructure,
                "automatic_clean_preflight": None,
            }
    elif item.get("automatic_clean_preflight"):
        automatic_clean_preflight, failures = automatic_clean_preflight_local(judge, row)
        infrastructure.extend(failures)
        if automatic_clean_preflight is None:
            return {
                "status": "technical_failure",
                "failure_reason": "automatic_clean_preflight_infrastructure_exhausted",
                "stage_history": [],
                "infrastructure_failures": infrastructure,
                "automatic_clean_preflight": None,
            }
        if not automatic_clean_preflight["eligible"]:
            return {
                "status": "qc_hold",
                "failure_reason": "automatic_clean_preflight_hold",
                "stage_history": [],
                "infrastructure_failures": infrastructure,
                "automatic_clean_preflight": automatic_clean_preflight,
            }

    def with_clean_preflight(payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "automatic_clean_preflight": automatic_clean_preflight}

    current = row["clean_response"]
    history: list[dict[str, Any]] = []
    final_grade = None
    for stage, axis in enumerate(item["intended_axes"], 1):
        feedback = None
        stage_ok = False
        for semantic_attempt in (1, 2):
            request = StageRequest(
                row["canonical_id"], row["split"], row["question"], row["clean_response"], current,
                tuple(item["intended_axes"]), tuple(item["intended_axes"][: stage - 1]), axis, stage,
                _selection_generation_seed(item, stage, semantic_attempt), semantic_attempt, feedback,
                item.get("generation_objective", "response_pair"),
            )
            candidate = None
            allowance = None
            for infrastructure_attempt in range(1, ACCEPTANCE_RULES["max_infrastructure_attempts"] + 1):
                try:
                    generation_started = time.monotonic()
                    candidate = backend.generate([request], allowance)[0]
                    judge.record_generation({
                        "canonical_id": row["canonical_id"], "call_purpose": "sequential_corruption",
                        "stage_index": stage, "target_axis": axis,
                        "semantic_attempt": semantic_attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                        "generation_seed": request.generation_seed,
                        "generation_objective": request.generation_objective,
                    }, candidate, time.monotonic() - generation_started)
                except Exception as exc:
                    infrastructure.append({
                        "kind": "generation_failure", "stage": stage,
                        "semantic_attempt": semantic_attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    candidate = None
                    continue
                if candidate.truncation_reason:
                    infrastructure.append({
                        "kind": "generation_truncation", "stage": stage,
                        "semantic_attempt": semantic_attempt,
                        "infrastructure_attempt": infrastructure_attempt,
                        "current_response_tokens": candidate.current_response_tokens,
                        "max_new_tokens": candidate.max_new_tokens,
                        "generated_tokens": candidate.generated_tokens,
                        "eos_reached": candidate.eos_reached,
                        "reason": candidate.truncation_reason,
                    })
                    if candidate.max_new_tokens >= ACCEPTANCE_RULES["truncation_ceiling"]:
                        candidate = None
                        break
                    allowance = min(
                        max(candidate.max_new_tokens + 256, candidate.max_new_tokens * 2),
                        ACCEPTANCE_RULES["truncation_ceiling"],
                    )
                    candidate = None
                    continue
                break
            if candidate is None:
                return with_clean_preflight({
                    "status": "technical_failure",
                    "failure_reason": "generation_infrastructure_exhausted",
                    "stage_history": history,
                    "infrastructure_failures": infrastructure,
                })
            immediate = None
            if not candidate.text:
                immediate = "empty_generation"
            elif normalize_text(candidate.text) == normalize_text(current):
                immediate = "unchanged_response"
            elif editing_meta(candidate.text):
                immediate = "editing_task_refusal_or_meta"
            if immediate:
                history.append({
                    "stage_index": stage, "target_axis": axis,
                    "semantic_attempt": semantic_attempt,
                    "generation_seed": request.generation_seed,
                    "generation_objective": request.generation_objective,
                    "candidate_response": candidate.text,
                    "edit_trace": text_edit_trace(current, candidate.text),
                    "generation": candidate.__dict__, "grade": None,
                    "accepted": False, "reason": immediate,
                })
                feedback = immediate
                continue
            grade, judge_infrastructure = paired_grade_local(
                judge, {**row, "intended_axes": item["intended_axes"]},
                candidate.text, stage, semantic_attempt,
            )
            infrastructure.extend({
                **event, "stage": stage, "semantic_attempt": semantic_attempt,
            } for event in judge_infrastructure)
            if grade is None:
                return with_clean_preflight({
                    "status": "technical_failure",
                    "failure_reason": "local_paired_judge_infrastructure_exhausted",
                    "stage_history": history,
                    "infrastructure_failures": infrastructure,
                })
            conflict = []
            if grade["clean_scores"]["specificity"]["score"] == 1:
                conflict.append("clean_specificity_score_1")
            if grade["clean_scores"]["medical_boundary"]["violation"]:
                conflict.append("clean_medical_boundary_violation")
            if grade["text_reason_contradiction"]["detected"]:
                conflict.append("judge_text_reason_contradiction")
            if conflict:
                recheck = None
                try:
                    recheck = judge.eligibility(row)
                except Exception as exc:
                    infrastructure.append({
                        "kind": "bounded_local_clean_recheck_failure", "stage": stage,
                        "semantic_attempt": semantic_attempt,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                history.append({
                    "stage_index": stage, "target_axis": axis,
                    "semantic_attempt": semantic_attempt,
                    "generation_seed": request.generation_seed,
                    "generation_objective": request.generation_objective,
                    "candidate_response": candidate.text,
                    "edit_trace": text_edit_trace(current, candidate.text),
                    "generation": candidate.__dict__, "grade": grade,
                    "accepted": False, "reason": "qc_conflict",
                    "conflict_flags": conflict,
                    "bounded_clean_recheck": recheck,
                })
                return with_clean_preflight({
                    "status": "qc_conflict", "failure_reason": ";".join(conflict),
                    "stage_history": history, "infrastructure_failures": infrastructure,
                    "conflict_recheck": recheck,
                })
            disposition, reason = content_disposition(
                row["question"], candidate.text, grade, clean=row["clean_response"],
                contextual_clean_review=contextual_clean_review,
            )
            missing = [
                intended for intended in item["intended_axes"][:stage]
                if intended not in grade["realized_axes"]
            ]
            if disposition == "pass" and missing:
                reason = "missing_intended_axes:" + ",".join(missing)
                disposition = "reject"
            history.append({
                "stage_index": stage, "target_axis": axis,
                "semantic_attempt": semantic_attempt,
                "generation_seed": request.generation_seed,
                "generation_objective": request.generation_objective,
                "candidate_response": candidate.text,
                "edit_trace": text_edit_trace(current, candidate.text),
                "generation": candidate.__dict__, "grade": grade,
                "accepted": disposition == "pass", "reason": reason,
            })
            if disposition == "hold":
                return with_clean_preflight({
                    "status": "qc_hold", "failure_reason": reason,
                    "stage_history": history, "infrastructure_failures": infrastructure,
                })
            if disposition == "pass":
                current = candidate.text
                final_grade = grade
                stage_ok = True
                break
            feedback = reason
        if not stage_ok:
            return with_clean_preflight({
                "status": "rejected", "failure_reason": feedback or "semantic_stage_failed",
                "stage_history": history, "infrastructure_failures": infrastructure,
            })
    return with_clean_preflight({
        "status": "accepted", "corrupted_response": current,
        "realized_axes": final_grade["realized_axes"],
        "unintended_axes": final_grade["unintended_axes"],
        "final_grade": final_grade,
        "stage_history": history, "infrastructure_failures": infrastructure,
    })


def main() -> None:
    load_repo_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-file", required=True)
    parser.add_argument("--max-inputs", type=int, required=True)
    parser.add_argument("--splits", default="train", help="Comma-separated canonical splits: train, valid, or train,valid")
    parser.add_argument("--known-holds-file", default=str(DEFAULT_HOLDS))
    parser.add_argument(
        "--additional-known-hold-ids-file", action="append", default=[],
        help="Additional newline-delimited canonical IDs to exclude; may be repeated.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--selected-gpu", type=int, default=0)
    parser.add_argument("--required-free-vram-mib", type=int, default=DEFAULT_REQUIRED_FREE_VRAM_MIB)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--generator-model-dir", help="Relocated local snapshot for the frozen Qwen generator")
    parser.add_argument("--budget-ledger", required=True)
    parser.add_argument("--max-api-requests", type=int, required=True)
    parser.add_argument("--max-api-usd", type=float, required=True)
    parser.add_argument("--input-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--output-usd-per-million-tokens", type=float, required=True)
    parser.add_argument("--budget-max-input-tokens-per-request", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--judge-backend", choices=("external-gpt", "local-qwen35-27b"), default="local-qwen35-27b")
    parser.add_argument("--local-judge-model-dir")
    parser.add_argument("--local-judge-max-new-tokens", type=int, default=3200)
    parser.add_argument("--local-judge-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--carry-forward-checkpoint")
    parser.add_argument(
        "--clean-context-review-file",
        help=(
            "Optional JSONL of hash-bound complete-question/clean reviews. "
            "Review data is applied generically; missing IDs continue through automated QC."
        ),
    )
    parser.add_argument(
        "--automatic-clean-preflight",
        action="store_true",
        help="Run bounded local-Qwen clean eligibility QC before any generator call.",
    )
    args = parser.parse_args()
    if args.judge_backend != "local-qwen35-27b":
        raise RuntimeError(
            "The typed-v4 production contract requires the local Qwen judge; paid/API QC is disabled"
        )
    if args.max_inputs < 1 or not 1 <= args.worker_count <= 3 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("Invalid processing or worker cap")
    allowed_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    if not allowed_splits or not allowed_splits <= {"train", "valid"}:
        raise ValueError("--splits must contain only train and/or valid")
    generator_model_dir = Path(
        args.generator_model_dir or GENERATORS["qwen"]["snapshot"]
    ).resolve()
    if not generator_model_dir.is_dir():
        raise FileNotFoundError(f"Frozen local generator snapshot is missing: {generator_model_dir}")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonical_path = DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"
    qc_path = DEFAULT_OUTPUT_DIR / "clean_target_qc/corpus_qc.jsonl"
    canonical_rows = list(read_jsonl(canonical_path))
    canonical = {row["canonical_id"]: row for row in canonical_rows}
    if len(canonical) != len(canonical_rows):
        raise RuntimeError("Canonical data contains duplicate canonical IDs")
    clean_qc_rows = list(read_jsonl(qc_path))
    clean_qc = {row["canonical_id"]: row for row in clean_qc_rows}
    holds_path = Path(args.known_holds_file).resolve()
    held_ids = {row["canonical_id"] for row in read_jsonl(holds_path)}
    additional_hold_paths = [Path(path).resolve() for path in args.additional_known_hold_ids_file]
    for path in additional_hold_paths:
        held_ids.update(
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    selected = load_selection(
        Path(args.selection_file).resolve(), args.max_inputs, canonical, clean_qc, held_ids, allowed_splits
    )
    for item in selected:
        item["automatic_clean_preflight"] = bool(args.automatic_clean_preflight)
    clean_context_review_path = (
        Path(args.clean_context_review_file).resolve()
        if args.clean_context_review_file else None
    )
    clean_context_reviews: dict[str, dict[str, Any]] = {}
    if clean_context_review_path is not None:
        review_rows = list(read_jsonl(clean_context_review_path))
        clean_context_reviews = {str(row.get("canonical_id") or ""): row for row in review_rows}
        if "" in clean_context_reviews or len(clean_context_reviews) != len(review_rows):
            raise ValueError("Contextual clean review file needs unique canonical IDs")
        selected_ids_for_review = {item["clean"]["canonical_id"] for item in selected}
        unknown_review_ids = set(clean_context_reviews) - selected_ids_for_review
        if unknown_review_ids:
            raise ValueError(
                f"Contextual clean review references IDs outside selection: {sorted(unknown_review_ids)[:5]}"
            )
        for item in selected:
            review = clean_context_reviews.get(item["clean"]["canonical_id"])
            if review is not None:
                item["clean"] = {**item["clean"], "contextual_clean_review": review}
    frozen_rows = [
        {
            "canonical_id": item["clean"]["canonical_id"],
            "split": item["split"],
            "question_normalized_sha256": item["clean"]["question_normalized_sha256"],
            "duplicate_cluster_id": item["clean"]["duplicate_cluster_id"],
            "source_group_id": item["clean"]["source_group_id"],
            "intended_axes": item["intended_axes"],
            "generation_seeds": item["generation_seeds"],
            "generation_objective": item["generation_objective"],
        }
        for item in selected
    ]
    local_judge = args.judge_backend == "local-qwen35-27b"
    if local_judge and not args.local_judge_model_dir:
        raise ValueError("--local-judge-model-dir is required for the local Qwen judge")
    config = BudgetConfig(
        max_requests=args.max_api_requests,
        max_usd=args.max_api_usd,
        input_usd_per_million_tokens=args.input_usd_per_million_tokens,
        output_usd_per_million_tokens=args.output_usd_per_million_tokens,
    )
    budget = None if local_judge else SharedApiBudget(args.budget_ledger, config)
    carry_path = Path(args.carry_forward_checkpoint).resolve() if args.carry_forward_checkpoint else None
    carry_rows = list(read_jsonl(carry_path)) if carry_path else []
    carry_ids = [row["canonical_id"] for row in carry_rows]
    selected_ids = [item["clean"]["canonical_id"] for item in selected]
    if carry_ids != selected_ids[:len(carry_ids)]:
        raise RuntimeError("Carry-forward checkpoint must be an exact prefix of the frozen selection")
    if len(carry_ids) != len(set(carry_ids)):
        raise RuntimeError("Carry-forward checkpoint contains duplicate canonical IDs")
    if any(row.get("status") not in {"accepted", "rejected", "qc_conflict", "technical_failure"} for row in carry_rows):
        raise RuntimeError("Carry-forward checkpoint contains a non-terminal historical status")
    remaining_selected = selected[len(carry_rows):]
    if local_judge:
        from local_qwen_production_qc_v4 import (
            JUDGE_REPO, JUDGE_REVISION, SYSTEM_PROMPT as LOCAL_SYSTEM_PROMPT,
            VERSION as local_qc_version, prompt_sha256 as local_prompt_sha256,
        )
        judge_contract: dict[str, Any] = {
            "backend": "local_transformers",
            "repo": JUDGE_REPO,
            "revision": JUDGE_REVISION,
            "local_path": str(Path(args.local_judge_model_dir).resolve()),
            "qc_version": local_qc_version,
            "paired_prompt_sha256": local_prompt_sha256(),
            "clean_preflight_prompt_sha256": hashlib.sha256(
                (
                    LOCAL_SYSTEM_PROMPT + "\n" + ELIGIBILITY_PROMPT + "\n"
                    + CLEAN_PREFLIGHT_RETRY_INSTRUCTION
                ).encode("utf-8")
            ).hexdigest(),
            "max_new_tokens": args.local_judge_max_new_tokens,
            "call_timeout_seconds": args.local_judge_timeout_seconds,
            "enable_thinking": False,
            "do_sample": False,
            "openai_fallback": False,
        }
    else:
        judge_contract = {
            "backend": "openai_chat_completions", "model": JUDGE_MODEL,
            "max_tokens": BUILD_JUDGE_MAX_TOKENS,
        }
    run_contract = {
        "version": LOCAL_JUDGE_VERSION if local_judge else VERSION,
        "selection_file": str(Path(args.selection_file).resolve()),
        "selection_file_sha256": sha256_file(Path(args.selection_file).resolve()),
        "canonical_file_sha256": sha256_file(canonical_path),
        "clean_qc_file_sha256": sha256_file(qc_path),
        "known_holds_file_sha256": sha256_file(holds_path),
        "additional_known_hold_ids": [
            {"path": str(path), "sha256": sha256_file(path)} for path in additional_hold_paths
        ],
        "clean_context_review": (
            {
                "path": str(clean_context_review_path),
                "sha256": sha256_file(clean_context_review_path),
                "reviewed_ids": len(clean_context_reviews),
            }
            if clean_context_review_path is not None else None
        ),
        "automatic_clean_preflight": bool(args.automatic_clean_preflight),
        "known_hold_ids_after_union": len(held_ids),
        "selected_after_cap": len(selected),
        "selection_execution_contract_sha256": selection_hash(frozen_rows),
        # Deprecated alias retained for readers of the v5 manifest.  This is a
        # hash of the expanded execution projection, not the selection file.
        "selection_sha256": selection_hash(frozen_rows),
        "processing_cap": args.max_inputs,
        "worker_count": args.worker_count,
        "allowed_splits": sorted(allowed_splits),
        "axis_count_distribution": dict(Counter(item["axis_count"] for item in selected)),
        "generation_objective_distribution": dict(
            Counter(item["generation_objective"] for item in selected)
        ),
        "generator": {**GENERATORS["qwen"], "snapshot": str(generator_model_dir)},
        "judge": judge_contract,
        "seed": BUILD_SEED,
        "acceptance_rules": ACCEPTANCE_RULES,
        "hashes": {
            "generation_prompt": generation_prompt_hash(),
            "clean_qc_prompt": clean_qc_hash(),
            "paired_judge_prompt": (
                judge_contract["paired_prompt_sha256"] if local_judge else paired_prompt_hash()
            ),
            "acceptance_rules": canonical_json_hash(ACCEPTANCE_RULES),
        },
        "budget_contract": (
            {"paid_api_disabled": True, "openai_fallback": False}
            if local_judge else {
                "ledger": str(Path(args.budget_ledger).resolve()),
                "config": {
                    "max_requests": args.max_api_requests,
                    "max_usd": args.max_api_usd,
                    "input_usd_per_million_tokens": args.input_usd_per_million_tokens,
                    "output_usd_per_million_tokens": args.output_usd_per_million_tokens,
                },
            }
        ),
        "budget_max_input_tokens_per_request": args.budget_max_input_tokens_per_request,
        "retry_policy": ACCEPTANCE_RULES,
        "implementation_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "local_qc_contract": (
                sha256_file(Path(__file__).resolve().parent / "local_qwen_production_qc_v4.py")
                if local_judge else None
            ),
        },
        "checkpoint_policy": "reuse terminal canonical_ids from this output's per-worker checkpoint",
        "carry_forward": ({
            "path": str(carry_path), "sha256": sha256_file(carry_path),
            "terminal_rows": len(carry_rows),
            "policy": "immutable historical terminal outcomes; not rejudged by local backend",
        } if carry_path else None),
        "remaining_after_carry_forward": len(remaining_selected),
    }
    run_fingerprint = initialize_run_contract(output, run_contract)
    manifest = {
        **run_contract,
        "run_fingerprint": run_fingerprint,
        "worker_index": args.worker_index,
        "status": "dry_run" if args.dry_run else "running",
        "budget": budget.status() if budget is not None else {"paid_api_calls": 0, "paid_api_disabled": True},
    }
    if args.dry_run:
        atomic_write_json(output / "production_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    completed_manifest = output / "production_manifest.json"
    if completed_manifest.exists():
        completed_payload = json.loads(completed_manifest.read_text(encoding="utf-8"))
        if completed_payload.get("status") == "complete":
            if completed_payload.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError("Completed manifest fingerprint differs from the current run")
            return

    gpu = gpu_status(args.selected_gpu, args.required_free_vram_mib)
    if not gpu["sufficient_free_vram"]:
        raise RuntimeError("Insufficient free VRAM")
    width = (len(remaining_selected) + args.worker_count - 1) // args.worker_count
    worker_items = remaining_selected[args.worker_index * width : min((args.worker_index + 1) * width, len(remaining_selected))]
    checkpoint = output / f"results_checkpoint.part{args.worker_index}.jsonl"
    done = {row["canonical_id"]: row for row in read_jsonl(checkpoint)} if checkpoint.exists() else {}
    if any(row.get("run_fingerprint") != run_fingerprint for row in done.values()):
        raise RuntimeError("Checkpoint fingerprint differs from the current run contract; refusing reuse")
    if local_judge:
        judge = LocalProductionJudge(
            Path(args.local_judge_model_dir), args.local_judge_max_new_tokens,
            args.local_judge_timeout_seconds, output / "local_calls",
        )
    else:
        judge = ExternalJudge(
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com"),
            max_tokens=BUILD_JUDGE_MAX_TOKENS,
            budget_guard=budget,
            budget_max_input_tokens=args.budget_max_input_tokens_per_request,
            budget_worker_id=f"worker-{args.worker_index}",
        )
    backend = LocalGenerator("qwen", 1, args.max_input_tokens, snapshot_override=generator_model_dir)
    started = time.monotonic()
    try:
        for item in worker_items:
            canonical_id = item["clean"]["canonical_id"]
            if canonical_id in done:
                continue
            before_generator = (backend.calls, backend.prompt_tokens, backend.completion_tokens, backend.elapsed)
            before_judge = (judge.calls, judge.prompt_tokens, judge.completion_tokens, judge.elapsed)
            row_started = time.monotonic()
            try:
                result = process_one_local(item, backend, judge) if local_judge else process_one(item, backend, judge)
            except BudgetExceeded as exc:
                status = {
                    **manifest,
                    "status": "budget_exhausted",
                    "completed_in_worker": len(done),
                    "budget": budget.status() if budget is not None else {"paid_api_calls": 0, "paid_api_disabled": True},
                    "reason": str(exc),
                }
                atomic_write_json(output / f"worker_status.part{args.worker_index}.json", status)
                with exclusive_lock(output / ".finalize.lock"):
                    current_path = output / "production_manifest.json"
                    current = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else {}
                    if current.get("status") != "complete":
                        atomic_write_json(current_path, status)
                return
            usage = {
                "generator_calls": backend.calls - before_generator[0],
                "generator_prompt_tokens": backend.prompt_tokens - before_generator[1],
                "generator_completion_tokens": backend.completion_tokens - before_generator[2],
                "generator_seconds": backend.elapsed - before_generator[3],
                "judge_calls": judge.calls - before_judge[0],
                "judge_prompt_tokens": judge.prompt_tokens - before_judge[1],
                "judge_completion_tokens": judge.completion_tokens - before_judge[2],
                "judge_seconds": judge.elapsed - before_judge[3],
                "wall_seconds": time.monotonic() - row_started,
            }
            done[canonical_id] = {
                **base_record(item),
                "split": item["split"],
                "judge_model": (
                    f"{judge.repo}@{judge.revision}" if local_judge else JUDGE_MODEL
                ),
                "judge_repo": judge.repo if local_judge else JUDGE_MODEL,
                "judge_revision": judge.revision if local_judge else JUDGE_MODEL,
                "question_normalized_sha256": item["clean"]["question_normalized_sha256"],
                "duplicate_cluster_id": item["clean"]["duplicate_cluster_id"],
                "source_group_id": item["clean"]["source_group_id"],
                "generation_objective": item["generation_objective"],
                **result,
                "usage": usage,
                "run_fingerprint": run_fingerprint,
                "generator_prompt_version": "corruption-contract-v2",
                "judge_prompt_version": LOCAL_JUDGE_VERSION if local_judge else "historical-gpt-paired-v1",
            }
            atomic_write_jsonl(
                checkpoint,
                [done[x["clean"]["canonical_id"]] for x in worker_items if x["clean"]["canonical_id"] in done],
            )
            local_completed = len(done)
            statuses = Counter(value["status"] for value in done.values())
            elapsed = time.monotonic() - started
            remaining_axes = Counter(
                pending["axis_count"] for pending in worker_items
                if pending["clean"]["canonical_id"] not in done
            )
            eta_seconds = elapsed / local_completed * (len(worker_items) - local_completed) if local_completed else None
            print(json.dumps({
                "local_progress": f"{local_completed}/{len(worker_items)}",
                "total_progress_with_carry": f"{len(carry_rows) + local_completed}/{len(selected)}",
                "status_counts": dict(statuses),
                "generator_seconds": usage["generator_seconds"],
                "judge_seconds": usage["judge_seconds"],
                "judge_calls": usage["judge_calls"],
                "remaining_axis_count_composition": dict(remaining_axes),
                "eta_seconds_from_observed_rows": eta_seconds,
            }, ensure_ascii=False), flush=True)
    finally:
        backend.close()
        if local_judge:
            judge.close()
    manifest["elapsed_seconds_this_invocation"] = time.monotonic() - started
    manifest["budget"] = budget.status() if budget is not None else {"paid_api_calls": 0, "paid_api_disabled": True}
    if local_judge:
        manifest["local_judge_runtime"] = judge.metadata()
    atomic_write_json(output / f"worker_status.part{args.worker_index}.json", {**manifest, "status": "worker_complete", "completed_in_worker": len(done)})

    with exclusive_lock(output / ".finalize.lock"):
        final_manifest = output / "production_manifest.json"
        if final_manifest.exists() and json.loads(final_manifest.read_text(encoding="utf-8")).get("status") == "complete":
            return
        combined = {row["canonical_id"]: row for row in carry_rows}
        for worker_index in range(args.worker_count):
            part = output / f"results_checkpoint.part{worker_index}.jsonl"
            if not part.exists():
                return
            part_rows = list(read_jsonl(part))
            if any(row.get("run_fingerprint") != run_fingerprint for row in part_rows):
                raise RuntimeError("A worker checkpoint has a different run fingerprint")
            combined.update({row["canonical_id"]: row for row in part_rows})
        if len(combined) != len(selected):
            return
        export(output, [combined[item["clean"]["canonical_id"]] for item in selected], manifest)


if __name__ == "__main__":
    main()
