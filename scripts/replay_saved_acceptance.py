"""Replay saved candidates on CPU; no generation, network, or model imports.

Run from the repository root after applying acceptance_policy.patch.
The source manifest is mandatory; old decisions must reproduce before outputs
are written. An existing output directory is never reused.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy_functions(runner):
    tree = ast.parse(Path(runner).read_text(encoding="utf-8"))
    # Definitions are compiled, never main() or GPU/import statements.
    wanted = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name != "main" and not node.decorator_list]
    scope = {"math": math, "clean_text": lambda text: re.sub(r"\s+", " ", str(text or "")).strip()}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(runner), "exec"), scope)
    return scope["acceptance_reasons"]


def metrics(row, prefix):
    response = row[prefix + "_response"]
    return {
        "response": response,
        "word_count": len(re.findall(r"\b[\w']+\b", response)),
        "risk_score": row[prefix + "_risk_score"],
        "focus_risk_score": row[prefix + "_focus_risk_score"],
        "bad_safety_count": row[prefix + "_bad_safety_count"],
        "generic_count": row[prefix + "_generic_count"],
        "keyword_overlap": row[prefix + "_keyword_overlap"],
        "specificity_ratio": row["specificity_ratio"] if prefix == "denoiser" else 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_gemma_selective_risk_refinement.py"))
    args = ap.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Output directory already exists; choose a fresh versioned path.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    source = manifest["sources"]["proposed"] if "sources" in manifest else manifest
    settings = source["settings"]
    expected_hash = source.get("sha256") or source.get("output", {}).get("sha256")
    if not expected_hash:
        raise SystemExit("Source manifest must bind the saved output SHA-256.")
    if sha256(args.input) != expected_hash:
        raise SystemExit("Input hash differs from source manifest; no outputs written.")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row.get("canonical_id") or row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate canonical IDs; no outputs written.")
    acceptance = load_policy_functions(args.runner)
    original_policy = settings.get("acceptance_policy", "legacy")
    if original_policy != "legacy":
        raise SystemExit("This comparison requires legacy source outputs.")
    policies = ("legacy", "surface_relaxed_v1", "always_accept_called")
    outputs = {policy: [] for policy in policies}
    audit = []
    for row in rows:
        used = row["used_denoiser"]
        if type(used) is not bool or type(row["accepted_denoiser"]) is not bool:
            raise ValueError("Call and acceptance flags must be booleans")
        sft = metrics(row, "sft")
        den = metrics(row, "denoiser") if used else None
        if used and not isinstance(den["response"], str):
            raise ValueError("Called row is missing its candidate text")
        legacy_reasons = acceptance(sft, den, SimpleNamespace(**dict(settings, acceptance_policy="legacy"))) if used else []
        legacy_accept = used and not legacy_reasons
        if legacy_accept != row["accepted_denoiser"]:
            raise ValueError("Legacy decision mismatch: " + str(row["id"]))
        if used and (",".join(legacy_reasons) or None) != row.get("reject_reason"):
            raise ValueError("Legacy rejection-reason mismatch: " + str(row["id"]))
        expected_final = row["denoiser_response"] if legacy_accept else row["sft_response"]
        if expected_final != row["final_response"]:
            raise ValueError("Legacy final-response mismatch: " + str(row["id"]))
        entry = {"id": row["id"], "called": used, "legacy_reject_reason": row.get("reject_reason"),
                 "sft_risk": row["sft_risk_score"], "candidate_risk": row.get("denoiser_risk_score"),
                 "word_count_ratio": row["specificity_ratio"]}
        for policy in policies:
            reasons = [] if not used or policy == "always_accept_called" else acceptance(sft, den, SimpleNamespace(**dict(settings, acceptance_policy=policy)))
            accepted = bool(used and not reasons)
            entry[policy + "_accepted"] = accepted
            entry[policy + "_reasons"] = ",".join(reasons)
            updated = copy.deepcopy(row)
            updated.update({
                "final_response": row["denoiser_response"] if accepted else row["sft_response"],
                "final_risk_score": row["denoiser_risk_score"] if accepted else row["sft_risk_score"],
                "final_focus_risk_score": row["denoiser_focus_risk_score"] if accepted else row["sft_focus_risk_score"],
                "accepted_denoiser": accepted,
                "reject_reason": (",".join(reasons) or None),
                "acceptance_policy": policy,
                "acceptance_settings": {key: settings[key] for key in ("gate_strategy", "min_risk_delta", "min_focus_risk_delta", "specificity_min_ratio", "keyword_overlap_slack", "min_word_count")},
                "word_count_ratio": row["specificity_ratio"],
                "acceptance_replay": {
                    "source_sha256": expected_hash,
                    "original_accepted_denoiser": row["accepted_denoiser"],
                    "original_reject_reason": row.get("reject_reason"),
                    "generation_reused": True,
                    "diagnostic_only": policy == "always_accept_called",
                },
            })
            outputs[policy].append(updated)
        audit.append(entry)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "complete_offline_replay_not_quality_validation",
        "input_sha256": expected_hash, "source_manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(args.runner), "rows": len(rows),
        "called": sum(row["used_denoiser"] for row in rows),
        "legacy_decisions_and_final_text_reproduced": len(rows),
        "generation_calls": 0, "judge_calls": 0, "settings": settings,
        "policies": {},
    }
    for policy, policy_rows in outputs.items():
        destination = args.output_dir / (policy + ".jsonl")
        destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in policy_rows), encoding="utf-8")
        summary["policies"][policy] = {
            "accepted": sum(row["accepted_denoiser"] for row in policy_rows),
            "changed_from_legacy": sum(a["final_response"] != b["final_response"] for a, b in zip(rows, policy_rows)),
            "sft_identical": sum(row["final_response"] == row["sft_response"] for row in policy_rows),
            "output_sha256": sha256(destination),
        }
    with (args.output_dir / "decisions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    (args.output_dir / "replay_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "settings"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
