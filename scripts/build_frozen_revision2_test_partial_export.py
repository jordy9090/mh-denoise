#!/usr/bin/env python3
"""Freeze the first accepted TEST rows from an in-progress production checkpoint.

The production runner atomically rewrites ``results_checkpoint.part*.jsonl``
after every terminal row.  This exporter reads those checkpoint files directly;
it never waits for, or falls back to, the final ``accepted.jsonl`` artifact.
Only the contiguous completed prefix of the precommitted selection is eligible,
so a later worker result can never jump over an earlier pending row.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


VERSION = "frozen-revision2-test-partial-export-v1-20261008"
DEFAULT_PRODUCTION_DIR = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008"
)
DEFAULT_SELECTION_FILE = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008/"
    "test_generation_selection.json"
)
DEFAULT_CANONICAL_FILE = Path("data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl")
DEFAULT_OUTPUT_DIR = Path(
    "data/fullpaper_acl_pipeline/frozen_revision2_test_partial15_v1_20261008"
)
TERMINAL_STATUSES = {
    "accepted",
    "rejected",
    "qc_conflict",
    "qc_hold",
    "technical_failure",
}
CHECKPOINT_PATTERN = re.compile(r"^results_checkpoint\.part([0-9]+)\.jsonl$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl_with_locations(path: Path) -> list[tuple[dict[str, Any], int]]:
    rows: list[tuple[dict[str, Any], int]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append((payload, line_number))
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".tmp.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def load_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Frozen selection must be an object containing rows: {path}")
    rows = payload["rows"]
    ids = [str(row.get("canonical_id") or "") for row in rows]
    if "" in ids or len(ids) != len(set(ids)):
        raise ValueError("Frozen selection needs nonempty, unique canonical IDs")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("Partial export accepts only a frozen split=test selection")
    question_hashes = [str(row.get("question_normalized_sha256") or "") for row in rows]
    if "" in question_hashes or len(question_hashes) != len(set(question_hashes)):
        raise ValueError("Frozen selection needs nonempty, unique normalized-question hashes")
    return payload, rows


def _pilot_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Pilot selection has no rows list: {path}")
    return [row for row in rows if row.get("split") == "test"]


def load_pilot_exclusions(
    selection_file: Path,
    canonical_file: Path | None,
    pilot_selection_files: list[Path] | None,
    explicit_question_hashes: Iterable[str],
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    """Return historical pilot TEST IDs and all of their question hashes."""

    question_hashes = {str(value) for value in explicit_question_hashes if str(value)}
    pilot_ids: set[str] = set()
    provenance: list[dict[str, Any]] = []
    selection_manifest_path = selection_file.parent / "manifest.json"
    selection_manifest: dict[str, Any] = {}
    if selection_manifest_path.exists():
        selection_manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
        declared = selection_manifest.get(
            "pilot_exposed_question_groups_retained_via_nonpilot_duplicate_id", []
        )
        if not isinstance(declared, list):
            raise ValueError("Selection manifest pilot-question exclusion field is not a list")
        question_hashes.update(str(value) for value in declared if str(value))
        provenance.append(
            {
                "role": "selection_manifest_declared_pilot_question_groups",
                "path": str(selection_manifest_path),
                "sha256": sha256_file(selection_manifest_path),
                "declared_hashes": sorted(str(value) for value in declared),
            }
        )

    if pilot_selection_files is None:
        discovered: list[Path] = []
        for record in selection_manifest.get("pilot_exclusion_sources", []):
            if isinstance(record, dict) and record.get("path"):
                discovered.append(Path(str(record["path"])).resolve())
        pilot_selection_files = discovered

    for path in pilot_selection_files:
        path = path.resolve()
        rows = _pilot_rows(path)
        ids = {str(row.get("canonical_id") or "") for row in rows}
        if "" in ids:
            raise ValueError(f"Pilot selection contains an empty canonical ID: {path}")
        pilot_ids.update(ids)
        provenance.append(
            {
                "role": "historical_pilot_selection",
                "path": str(path),
                "sha256": sha256_file(path),
                "test_ids": sorted(ids),
            }
        )

    if pilot_ids:
        if canonical_file is None or not canonical_file.exists():
            raise FileNotFoundError(
                "Canonical rows are required to exclude different IDs with pilot-identical questions"
            )
        wanted = set(pilot_ids)
        found: dict[str, str] = {}
        for row, _ in read_jsonl_with_locations(canonical_file):
            canonical_id = str(row.get("canonical_id") or "")
            if canonical_id in wanted:
                question_hash = str(row.get("question_normalized_sha256") or "")
                if not question_hash:
                    raise ValueError(f"Pilot canonical row lacks question hash: {canonical_id}")
                found[canonical_id] = question_hash
        missing = wanted - set(found)
        if missing:
            raise ValueError(f"Historical pilot IDs missing from canonical data: {sorted(missing)}")
        question_hashes.update(found.values())
        provenance.append(
            {
                "role": "canonical_pilot_question_hash_lookup",
                "path": str(canonical_file),
                "sha256": sha256_file(canonical_file),
                "resolved_test_ids": len(found),
                "resolved_unique_question_hashes": len(set(found.values())),
            }
        )
    return pilot_ids, question_hashes, provenance


def load_checkpoint_rows(
    production_dir: Path,
    selection_rows: list[dict[str, Any]],
    run_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load checkpoint/carry-forward rows and prove each file is an ordered prefix."""

    selection_ids = [str(row["canonical_id"]) for row in selection_rows]
    completed: dict[str, dict[str, Any]] = {}
    locators: dict[str, dict[str, Any]] = {}
    input_files: list[dict[str, Any]] = []

    carry = run_contract.get("carry_forward")
    carry_count = 0
    if isinstance(carry, dict) and carry.get("path"):
        carry_path = Path(str(carry["path"])).resolve()
        carry_rows = read_jsonl_with_locations(carry_path)
        carry_ids = [str(row.get("canonical_id") or "") for row, _ in carry_rows]
        if carry_ids != selection_ids[: len(carry_ids)]:
            raise RuntimeError("Carry-forward rows are not an exact frozen-selection prefix")
        carry_count = len(carry_ids)
        file_hash = sha256_file(carry_path)
        input_files.append(
            {"role": "carry_forward", "path": str(carry_path), "sha256": file_hash, "rows": carry_count}
        )
        for row, line_number in carry_rows:
            canonical_id = str(row["canonical_id"])
            completed[canonical_id] = row
            locators[canonical_id] = {
                "artifact_role": "carry_forward",
                "path": str(carry_path),
                "line": line_number,
                "file_sha256": file_hash,
            }

    remaining_ids = selection_ids[carry_count:]
    worker_count = int(run_contract.get("worker_count") or 1)
    if worker_count < 1:
        raise ValueError("Run contract worker_count must be positive")
    width = (len(remaining_ids) + worker_count - 1) // worker_count
    expected_paths = {
        index: production_dir / f"results_checkpoint.part{index}.jsonl"
        for index in range(worker_count)
    }
    for path in production_dir.glob("results_checkpoint.part*.jsonl"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match is None or int(match.group(1)) not in expected_paths:
            raise RuntimeError(f"Unexpected production checkpoint part: {path}")

    run_fingerprint = str(run_contract.get("run_fingerprint") or "")
    for worker_index, path in expected_paths.items():
        if not path.exists():
            continue
        located_rows = read_jsonl_with_locations(path)
        ids = [str(row.get("canonical_id") or "") for row, _ in located_rows]
        expected_block = remaining_ids[
            worker_index * width : min((worker_index + 1) * width, len(remaining_ids))
        ]
        if ids != expected_block[: len(ids)]:
            raise RuntimeError(
                f"Checkpoint part {worker_index} is not an ordered prefix of its frozen worker block"
            )
        file_hash = sha256_file(path)
        input_files.append(
            {
                "role": f"checkpoint_part_{worker_index}",
                "path": str(path),
                "sha256": file_hash,
                "rows": len(ids),
            }
        )
        for row, line_number in located_rows:
            canonical_id = str(row["canonical_id"])
            if canonical_id in completed:
                raise RuntimeError(f"Duplicate terminal checkpoint ID: {canonical_id}")
            if row.get("status") not in TERMINAL_STATUSES:
                raise RuntimeError(f"Checkpoint row has nonterminal/unknown status: {canonical_id}")
            if str(row.get("run_fingerprint") or "") != run_fingerprint:
                raise RuntimeError(f"Checkpoint fingerprint mismatch: {canonical_id}")
            completed[canonical_id] = row
            locators[canonical_id] = {
                "artifact_role": f"checkpoint_part_{worker_index}",
                "path": str(path),
                "line": line_number,
                "file_sha256": file_hash,
            }
    return completed, locators, input_files


def failure_reason(row: dict[str, Any]) -> str | None:
    for key in ("failure_reason", "conflict_reason", "hold_reason"):
        if row.get(key):
            return str(row[key])
    return None


def pair_row(
    row: dict[str, Any],
    selection_row: dict[str, Any],
    partial_rank: int,
    frozen_rank: int,
    locator: dict[str, Any],
    run_fingerprint: str,
) -> dict[str, Any]:
    for field in ("question", "clean_response", "corrupted_response"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Accepted checkpoint row has empty {field}: {row.get('canonical_id')}")
    if row["clean_response"] == row["corrupted_response"]:
        raise ValueError(f"Accepted clean/corrupted responses are identical: {row['canonical_id']}")
    canonical_id = str(row["canonical_id"])
    for field in (
        "question_normalized_sha256",
        "duplicate_cluster_id",
        "source_group_id",
    ):
        if str(row.get(field) or "") != str(selection_row.get(field) or ""):
            raise RuntimeError(f"Checkpoint/selection {field} mismatch: {canonical_id}")
    intended = list(row.get("intended_axes") or selection_row.get("intended_axes") or [])
    metadata = {
        "canonical_id": canonical_id,
        "split": "test",
        "source": row.get("source"),
        "source_component": row.get("source_component"),
        "question_normalized_sha256": row["question_normalized_sha256"],
        "duplicate_cluster_id": row["duplicate_cluster_id"],
        "source_group_id": row["source_group_id"],
        "intended_axes": intended,
        "realized_axes": row.get("realized_axes", []),
        "unintended_axes": row.get("unintended_axes", []),
        "axis_count": row.get("axis_count", selection_row.get("axis_count")),
        "generation_objective": row.get(
            "generation_objective", selection_row.get("generation_objective")
        ),
        "generator_repo": row.get("generator_repo"),
        "generator_revision": row.get("generator_revision"),
        "generator_prompt_version": row.get("generator_prompt_version"),
        "judge_model": row.get("judge_model"),
        "judge_repo": row.get("judge_repo"),
        "judge_revision": row.get("judge_revision"),
        "judge_prompt_version": row.get("judge_prompt_version"),
        "production_run_fingerprint": run_fingerprint,
        "frozen_selection_rank": frozen_rank,
        "partial_selection_rank": partial_rank,
        "checkpoint_source": locator,
        "partial_export_version": VERSION,
    }
    return {
        "canonical_id": canonical_id,
        "id": canonical_id,
        "split": "test",
        "source": row.get("source"),
        "source_component": row.get("source_component"),
        "question_normalized_sha256": row["question_normalized_sha256"],
        "duplicate_cluster_id": row["duplicate_cluster_id"],
        "source_group_id": row["source_group_id"],
        "question": row["question"],
        "unsafe_response": row["corrupted_response"],
        "safe_response": row["clean_response"],
        "intended_axes": intended,
        "realized_axes": row.get("realized_axes", []),
        "unintended_axes": row.get("unintended_axes", []),
        "metadata": metadata,
    }


def _validate_existing_manifest(
    manifest: dict[str, Any], selection_sha256: str, run_fingerprint: str, target_count: int
) -> None:
    if manifest.get("status") != "complete_frozen_partial_test_export":
        raise RuntimeError("Existing partial manifest is not a completed immutable export")
    if int(manifest.get("target_count") or 0) != target_count:
        raise RuntimeError("Existing partial export target count differs")
    if manifest.get("inputs", {}).get("selection", {}).get("sha256") != selection_sha256:
        raise RuntimeError("Existing partial export uses a different frozen selection")
    if manifest.get("production_run_fingerprint") != run_fingerprint:
        raise RuntimeError("Existing partial export uses a different production run")
    for record in manifest.get("outputs", {}).values():
        path = Path(str(record["path"]))
        if not path.exists() or sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Completed partial export artifact was modified or removed: {path}")


def build_partial_export(
    production_dir: Path,
    selection_file: Path,
    output_dir: Path,
    *,
    target_count: int = 15,
    canonical_file: Path | None = None,
    pilot_selection_files: list[Path] | None = None,
    explicit_pilot_question_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    production_dir = production_dir.resolve()
    selection_file = selection_file.resolve()
    output_dir = output_dir.resolve()
    canonical_file = canonical_file.resolve() if canonical_file is not None else None
    if target_count < 1:
        raise ValueError("target_count must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / ".partial_export.lock").open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        selection_payload, selection_rows = load_selection(selection_file)
        selection_sha = sha256_file(selection_file)
        contract_path = production_dir / "run_contract.json"
        run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        run_fingerprint = str(run_contract.get("run_fingerprint") or "")
        if not run_fingerprint:
            raise ValueError("Production run contract has no fingerprint")
        if run_contract.get("allowed_splits") != ["test"]:
            raise ValueError("Production run contract is not TEST-only")
        if run_contract.get("selection_file_sha256") != selection_sha:
            raise RuntimeError("Frozen selection hash differs from the production run contract")

        final_manifest_path = output_dir / "manifest.json"
        if final_manifest_path.exists():
            existing = json.loads(final_manifest_path.read_text(encoding="utf-8"))
            _validate_existing_manifest(existing, selection_sha, run_fingerprint, target_count)
            return existing

        pilot_ids, pilot_hashes, pilot_provenance = load_pilot_exclusions(
            selection_file,
            canonical_file,
            pilot_selection_files,
            explicit_pilot_question_hashes,
        )
        completed, locators, checkpoint_files = load_checkpoint_rows(
            production_dir, selection_rows, run_contract
        )
        selection_by_id = {str(row["canonical_id"]): row for row in selection_rows}
        selection_ids = [str(row["canonical_id"]) for row in selection_rows]
        completed_prefix_ids: list[str] = []
        for canonical_id in selection_ids:
            if canonical_id not in completed:
                break
            completed_prefix_ids.append(canonical_id)
        prefix_set = set(completed_prefix_ids)

        def pilot_reasons(selection_row: dict[str, Any]) -> list[str]:
            reasons: list[str] = []
            if str(selection_row["canonical_id"]) in pilot_ids:
                reasons.append("historical_pilot_test_id")
            if str(selection_row["question_normalized_sha256"]) in pilot_hashes:
                reasons.append("historical_pilot_same_normalized_question")
            return reasons

        eligible_accepted_ids = [
            canonical_id
            for canonical_id in completed_prefix_ids
            if completed[canonical_id].get("status") == "accepted"
            and not pilot_reasons(selection_by_id[canonical_id])
        ]
        selected_ids = eligible_accepted_ids[:target_count]
        selected_set = set(selected_ids)

        status_rows: list[dict[str, Any]] = []
        for frozen_index, selection_row in enumerate(selection_rows, start=1):
            canonical_id = str(selection_row["canonical_id"])
            result = completed.get(canonical_id)
            reasons = pilot_reasons(selection_row)
            if result is None:
                decision_state = "pending"
            elif result.get("status") == "accepted" and reasons:
                decision_state = "accepted_but_pilot_question_excluded"
            elif canonical_id in selected_set:
                decision_state = "accepted_selected"
            elif result.get("status") == "accepted":
                decision_state = "accepted_not_selected"
            else:
                decision_state = str(result.get("status"))
            status_rows.append(
                {
                    "frozen_selection_rank": frozen_index,
                    "canonical_id": canonical_id,
                    "question_normalized_sha256": selection_row[
                        "question_normalized_sha256"
                    ],
                    "processed": result is not None,
                    "within_contiguous_completed_prefix": canonical_id in prefix_set,
                    "terminal_status": result.get("status") if result else None,
                    "decision_state": decision_state,
                    "selected_for_partial_export": canonical_id in selected_set,
                    "pilot_exclusion_reasons": reasons,
                    "failure_reason": failure_reason(result) if result else None,
                    "checkpoint_source": locators.get(canonical_id),
                }
            )

        status_path = output_dir / "selection_status_snapshot.jsonl"
        atomic_write_jsonl(status_path, status_rows)
        terminal_counts = Counter(str(row.get("status")) for row in completed.values())
        prefix_terminal_counts = Counter(
            str(completed[canonical_id].get("status")) for canonical_id in completed_prefix_ids
        )
        progress: dict[str, Any] = {
            "version": VERSION,
            "status": (
                "ready_to_freeze" if len(selected_ids) == target_count else "waiting_for_accepted_prefix"
            ),
            "target_count": target_count,
            "production_run_fingerprint": run_fingerprint,
            "selection_rows": len(selection_rows),
            "processed_rows_observed": len(completed),
            "contiguous_completed_prefix_rows": len(completed_prefix_ids),
            "pending_rows": len(selection_rows) - len(completed),
            "eligible_accepted_in_completed_prefix": len(eligible_accepted_ids),
            "selected_if_frozen_now": len(selected_ids),
            "terminal_status_counts_observed": dict(sorted(terminal_counts.items())),
            "terminal_status_counts_in_completed_prefix": dict(
                sorted(prefix_terminal_counts.items())
            ),
            "pilot_excluded_selection_rows": sum(
                bool(pilot_reasons(row)) for row in selection_rows
            ),
            "inputs": {
                "selection": {"path": str(selection_file), "sha256": selection_sha},
                "run_contract": {
                    "path": str(contract_path),
                    "sha256": sha256_file(contract_path),
                },
                "checkpoint_artifacts": checkpoint_files,
                "pilot_exclusion_sources": pilot_provenance,
            },
            "status_snapshot": {
                "path": str(status_path),
                "sha256": sha256_file(status_path),
            },
        }
        progress_path = output_dir / "progress_manifest.json"
        atomic_write_json(progress_path, progress)
        if len(selected_ids) < target_count:
            return progress

        pairs = [
            pair_row(
                completed[canonical_id],
                selection_by_id[canonical_id],
                partial_rank,
                selection_ids.index(canonical_id) + 1,
                locators[canonical_id],
                run_fingerprint,
            )
            for partial_rank, canonical_id in enumerate(selected_ids, start=1)
        ]
        pair_path = output_dir / "test_pairs.jsonl"
        sft_path = output_dir / "sft_test.jsonl"
        dpo_path = output_dir / "dpo_test.jsonl"
        partial_selection_path = output_dir / "partial_selection.json"
        atomic_write_jsonl(pair_path, pairs)
        atomic_write_jsonl(
            sft_path,
            [{**row, "target": row["safe_response"]} for row in pairs],
        )
        atomic_write_jsonl(
            dpo_path,
            [
                {
                    **row,
                    "chosen": row["safe_response"],
                    "rejected": row["unsafe_response"],
                }
                for row in pairs
            ],
        )
        partial_selection = {
            "version": VERSION,
            "status": "frozen",
            "target_count": target_count,
            "selection_seed": selection_payload.get("seed"),
            "selection_file_sha256": selection_sha,
            "production_run_fingerprint": run_fingerprint,
            "rows": [
                {
                    "partial_selection_rank": index,
                    "frozen_selection_rank": selection_ids.index(canonical_id) + 1,
                    "canonical_id": canonical_id,
                    "question_normalized_sha256": selection_by_id[canonical_id][
                        "question_normalized_sha256"
                    ],
                    "checkpoint_source": locators[canonical_id],
                }
                for index, canonical_id in enumerate(selected_ids, start=1)
            ],
        }
        atomic_write_json(partial_selection_path, partial_selection)

        checks = {
            "production_selection_hash_matches": True,
            "selection_test_only": True,
            "checkpoint_rows_are_ordered_worker_prefixes": True,
            "selection_uses_contiguous_completed_prefix_only": True,
            "exact_target_accepted_rows": len(pairs) == target_count,
            "selected_ids_follow_frozen_order": selected_ids
            == sorted(selected_ids, key=selection_ids.index),
            "selected_rows_have_no_historical_pilot_id_or_question": all(
                not pilot_reasons(selection_by_id[canonical_id]) for canonical_id in selected_ids
            ),
            "selected_normalized_questions_unique": len(
                {row["question_normalized_sha256"] for row in pairs}
            )
            == target_count,
            "selected_text_fields_nonempty": all(
                row["question"].strip()
                and row["unsafe_response"].strip()
                and row["safe_response"].strip()
                for row in pairs
            ),
            "sft_and_dpo_are_flat_inference_rows": all(
                all(field in row for field in ("canonical_id", "question", "unsafe_response", "safe_response"))
                for row in pairs
            ),
        }
        if not all(checks.values()):
            raise AssertionError(checks)
        progress["status"] = "frozen"
        atomic_write_json(progress_path, progress)
        outputs = {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                pair_path,
                sft_path,
                dpo_path,
                partial_selection_path,
                status_path,
                progress_path,
            )
        }
        manifest = {
            **progress,
            "status": "complete_frozen_partial_test_export",
            "purpose": "counselor_partial_test_inference_not_training",
            "target_count": target_count,
            "selected_ids_ordered": selected_ids,
            "freeze_frontier": {
                "contiguous_completed_prefix_rows": len(completed_prefix_ids),
                "last_prefix_canonical_id": completed_prefix_ids[-1],
                "selected_15th_frozen_rank": selection_ids.index(selected_ids[-1]) + 1,
            },
            "checks": checks,
            "outputs": outputs,
        }
        atomic_write_json(final_manifest_path, manifest)
        return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", default=str(DEFAULT_PRODUCTION_DIR))
    parser.add_argument("--selection-file", default=str(DEFAULT_SELECTION_FILE))
    parser.add_argument("--canonical-file", default=str(DEFAULT_CANONICAL_FILE))
    parser.add_argument("--pilot-selection", action="append", default=None)
    parser.add_argument("--pilot-question-hash", action="append", default=[])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-count", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_partial_export(
        Path(args.production_dir),
        Path(args.selection_file),
        Path(args.output_dir),
        target_count=args.target_count,
        canonical_file=Path(args.canonical_file) if args.canonical_file else None,
        pilot_selection_files=(
            [Path(path) for path in args.pilot_selection]
            if args.pilot_selection is not None
            else None
        ),
        explicit_pilot_question_hashes=args.pilot_question_hash,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
