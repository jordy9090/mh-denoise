#!/usr/bin/env python3
"""Write a reproducible final summary for production reuse/QC artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from fullpaper_acl_pipeline import read_jsonl, sha256_file, write_json


def rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejudge-dir", required=True)
    parser.add_argument("--regeneration-dir", required=True)
    parser.add_argument("--final-export-dir", required=True)
    args = parser.parse_args()
    rejudge = Path(args.rejudge_dir).resolve()
    regen = Path(args.regeneration_dir).resolve()
    final = Path(args.final_export_dir).resolve()
    checkpoint = rows(rejudge / "rejudged_checkpoint.jsonl")
    final_manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    regen_manifest = json.loads((regen / "production_manifest.json").read_text(encoding="utf-8"))

    new_rows = [row for row in checkpoint if not row.get("reused_without_model_call")]
    calls = sum(int(row.get("usage", {}).get("judge_calls", 0)) for row in new_rows)
    judge_seconds = sum(float(row.get("usage", {}).get("judge_seconds", 0)) for row in new_rows)
    wall_seconds = sum(float(row.get("usage", {}).get("wall_seconds_allocated", 0)) for row in new_rows)
    retry_calls = sum(max(0, int(row.get("usage", {}).get("judge_calls", 0)) - 1) for row in new_rows)

    location_counts = Counter()
    location_examples: dict[str, list[str]] = {}
    for row in checkpoint:
        if row.get("status") != "complete":
            category = "unreviewed_technical_failure"
        else:
            candidate_positive = [
                item for item in row.get("grade", {}).get("local_supervision", [])
                if item.get("side") == "candidate"
                and item.get("scope") == "local_defect"
                and item.get("label") == 1
            ]
            response_only = [
                item for item in row.get("grade", {}).get("response_level_evidence", [])
                if item.get("side") == "candidate"
                and item.get("scope") in {"holistic", "omission"}
            ]
            if candidate_positive:
                category = "confirmed_candidate_local_defect_position"
            elif response_only:
                category = "response_level_issue_without_local_position"
            else:
                category = "no_representative_local_mark_not_proof_of_no_local_defect"
        location_counts[category] += 1
        location_examples.setdefault(category, [])
        if len(location_examples[category]) < 5:
            location_examples[category].append(row["canonical_id"])

    regen_checkpoint = rows(regen / "results_checkpoint.part0.jsonl")
    regen_usage = {
        key: sum(float(row.get("usage", {}).get(key, 0)) for row in regen_checkpoint)
        for key in ("generator_calls", "judge_calls", "generator_seconds", "judge_seconds", "wall_seconds")
    }
    decisions = rows(final / "adjudication_ledger.jsonl")
    summary = {
        "status": "production_only_reuse_export_complete_not_main_training_approved",
        "rejudge": {
            "selection_rows": len(checkpoint),
            "reused_same_hash_without_call": sum(bool(row.get("reused_without_model_call")) for row in checkpoint),
            "newly_called_rows": len(new_rows),
            "terminal_status": dict(Counter(row.get("status") for row in checkpoint)),
            "judge_calls": calls,
            "retry_calls_after_first": retry_calls,
            "retry_rate_per_new_row": retry_calls / len(new_rows) if new_rows else 0.0,
            "judge_seconds_sum": judge_seconds,
            "allocated_wall_seconds_sum": wall_seconds,
            "seconds_per_new_row": wall_seconds / len(new_rows) if new_rows else 0.0,
        },
        "candidate_regeneration": {
            "processed": regen_manifest.get("processed_inputs"),
            "accepted": regen_manifest.get("accepted"),
            "rejected": regen_manifest.get("rejected"),
            "qc_holds": regen_manifest.get("qc_holds"),
            "qc_conflicts": regen_manifest.get("qc_conflicts"),
            "elapsed_seconds": regen_manifest.get("elapsed_seconds_this_invocation"),
            "usage": regen_usage,
        },
        "final_pair_dispositions": {
            "total": len(decisions),
            "counts": dict(Counter(row["reuse_disposition"] for row in decisions)),
            "by_split": final_manifest["counts"]["by_split"],
        },
        "router": final_manifest["router"],
        "scorer": final_manifest["scorer"],
        "integrity": final_manifest["integrity"],
        "legacy_collision_accounting": final_manifest["legacy_collision_accounting"],
        "local_defect_location_interpretation": {
            "denominator": len(checkpoint),
            "counts": dict(location_counts),
            "representative_ids": location_examples,
            "caveat": (
                "The QC schema stores representative evidence, not an exhaustive defect inventory; "
                "absence of a local mark is retained as unknown rather than asserted no-local-defect."
            ),
        },
        "corrected57_included": False,
        "training_ready": final_manifest["readiness"]["training_ready"],
        "artifact_sha256": {
            "rejudge_checkpoint": sha256_file(rejudge / "rejudged_checkpoint.jsonl"),
            "regeneration_manifest": sha256_file(regen / "production_manifest.json"),
            "final_manifest": sha256_file(final / "manifest.json"),
        },
    }
    write_json(final / "completion_summary.json", summary)
    md = [
        "# Production 652 reuse completion summary",
        "",
        f"- Final disposition: `{json.dumps(summary['final_pair_dispositions']['counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- Rejudge: {len(new_rows)} new rows, {calls} calls, {retry_calls} retry calls, {wall_seconds / 3600:.2f} allocated wall hours.",
        f"- Candidate regeneration: {regen_manifest.get('accepted')} accepted / {regen_manifest.get('processed_inputs')} processed.",
        f"- Final active Scorer conflicts: {final_manifest['scorer']['final_collision_groups']}.",
        f"- Training ready: `{final_manifest['readiness']['training_ready']}` (approval is intentionally not inferred).",
        "- corrected57 is not included in this production-only version.",
        "",
        "The local-position counts in completion_summary.json distinguish confirmed local spans, response-only holistic/omission evidence, and missing representative marks. Missing marks are not interpreted as proof that a response has no local defect.",
    ]
    (final / "completion_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
