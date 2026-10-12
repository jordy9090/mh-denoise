#!/usr/bin/env python3
"""Package the completed VALID43 base-first residual diagnostics."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION = "base-first-residual-valid-v1-20261011"
ROOT = Path("/mnt/ssd00/hsoh/base_first_residual_valid_20261011_v1")
REPO = Path("/home/user/hsoh/mh-denoise")
DOWNLOADS = Path("/home/user/Downloads")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    required = (
        ROOT / "VALID43_BASE_FIRST_DIAGNOSTIC.json",
        ROOT / "PAIRWISE_RESIDUAL_TARGET_DIAGNOSTIC.json",
        ROOT / "VALID43_BASE_DRAFT_CONTROL_RESULTS.json",
        ROOT / "VALID43_BASE_DRAFT_CONTROL_LLAMA_RESULTS.json",
        ROOT / "generation/base_prompt_rewrite_valid43.manifest.json",
        ROOT / "generation/base_draft_second_pass_controls_valid43.manifest.json",
        ROOT / "SECOND_PASS_PIPELINE_COMPLETE.txt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot package incomplete experiment; missing: {missing}")

    initial = read_json(ROOT / "VALID43_BASE_FIRST_DIAGNOSTIC.json")
    initial_pairwise = read_json(ROOT / "PAIRWISE_RESIDUAL_TARGET_DIAGNOSTIC.json")
    controls = read_json(ROOT / "VALID43_BASE_DRAFT_CONTROL_RESULTS.json")
    llama = read_json(ROOT / "VALID43_BASE_DRAFT_CONTROL_LLAMA_RESULTS.json")
    initial_methods = {row["method"]: row for row in initial["method_summary"]}
    candidate_methods = {row["method"]: row for row in controls["candidate_quality"]}
    final_methods = {row["method"]: row for row in controls["selected_final_quality"]}
    llama_methods = {row["method"]: row for row in llama["candidate_quality"]}
    llama_final = {row["method"]: row for row in llama["gpt_selected_final_quality_under_llama"]}

    api_manifests = (
        ROOT / "results/new_pointwise_results.manifest.json",
        ROOT / "results/new_pairwise_results.manifest.json",
        ROOT / "results/second_pass_new_pointwise.manifest.json",
        ROOT / "results/second_pass_new_pairwise.manifest.json",
    )
    api = [read_json(path) for path in api_manifests]
    api_usage = {
        "successful_paid_requests": sum(int(row["completed"]) for row in api),
        "failed_latest": sum(int(row["failed_latest"]) for row in api),
        "input_tokens": sum(int(row["input_tokens"]) for row in api),
        "output_tokens": sum(int(row["output_tokens"]) for row in api),
        "estimated_cost_usd": round(sum(float(row["estimated_cost_usd"]) for row in api), 6),
        "manifests": [str(path) for path in api_manifests],
    }
    write_json(ROOT / "API_USAGE_TOTAL.json", api_usage)

    refusal_rows = read_jsonl(ROOT / "tables/second_pass_generation_failure_signals.jsonl")
    refusal_by_method = {row["method"]: row for row in refusal_rows}
    signal_ids_by_candidate = {
        method: set(row.get("refusal_or_meta_refusal_ids", []))
        | set(row.get("unexpected_non_latin_script_ids", []))
        | set(row.get("placeholder_or_bracket_corruption_ids", []))
        | set(row.get("malformed_surface_token_or_markup_ids", []))
        for method, row in refusal_by_method.items()
    }
    control_generation = read_jsonl(ROOT / "generation/base_draft_second_pass_controls_valid43.jsonl")
    generation_by_method_id = {
        (
            "BaseSelfRefineCandidate" if row["arm"] == "base_self_refine" else "TrainedDenoiserOnBaseCandidate",
            row["canonical_id"],
        ): row
        for row in control_generation
    }
    pairwise_controls = {row["comparison"]: row for row in controls["pairwise"]}
    wtl = {row["method"]: row for row in controls["selected_final_wtl"]}
    control_risk = {(row["scope"], row["method"]): row for row in controls["risk"]}

    paired_rows = []
    for judge, score_file in (
        ("GPT-4.1", "second_pass_pointwise_scores.jsonl"),
        ("Llama", "second_pass_llama_scores.jsonl"),
    ):
        scores_by_method: dict[str, dict[str, float]] = {}
        for scored in read_jsonl(ROOT / "results" / score_file):
            scores_by_method.setdefault(scored["method"], {})[scored["canonical_id"]] = scored["quality_3axis_mean"]
        for final_row in read_jsonl(ROOT / "results/second_pass_final_outputs.jsonl"):
            if final_row["method"] == "BasePromptRewrite":
                continue
            source = scores_by_method[final_row["source_method"]]
            if final_row["canonical_id"] in source:
                scores_by_method.setdefault(final_row["method"], {})[final_row["canonical_id"]] = source[final_row["canonical_id"]]
        for right in sorted(method for method in scores_by_method if method != "BasePromptRewrite"):
            left_scores, right_scores = scores_by_method["BasePromptRewrite"], scores_by_method[right]
            ids = sorted(left_scores.keys() & right_scores.keys())
            deltas = [right_scores[cid] - left_scores[cid] for cid in ids]
            paired_rows.append({
                "judge": judge, "left": "BasePromptRewrite", "right": right,
                "N": len(ids), "missing_pairs": 43 - len(ids),
                "left_mean": statistics.mean(left_scores[cid] for cid in ids),
                "right_mean": statistics.mean(right_scores[cid] for cid in ids),
                "right_minus_left": statistics.mean(deltas),
                "right_wins": sum(delta > 1e-9 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-9 for delta in deltas),
                "right_losses": sum(delta < -1e-9 for delta in deltas),
                "canonical_ids": ids,
            })
    write_json(ROOT / "tables/second_pass_matched_pointwise_comparisons.json", paired_rows)
    with (ROOT / "tables/second_pass_matched_pointwise_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in paired_rows[0] if key != "canonical_ids"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(paired_rows)

    report_lines = [
        "# VALID43 Base-First Residual Diagnostic",
        "",
        "## Scope",
        "",
        "This development-only experiment used the frozen VALID43 questions and did not regenerate or tune on TEST. "
        "It first evaluated the no-adapter prompted rewrite, then held that BasePromptRewrite draft and the rendered "
        "second-pass prompt fixed while switching only the generation adapter.",
        "This diagnostic was initiated after observing the earlier TEST baseline comparison. "
        "The denoiser was trained on SFT drafts; applying it to Base drafts changes its input distribution. "
        "Quality-3 is the mean of Overall Quality, Empathy, and Specificity; Medical Advice remains categorical.",
        "",
        "## First-pass finding",
        "",
        "| Method | N | Overall | Empathy | Specificity | Quality-3 | Mean risk |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    risk_initial = {row["method"]: row for row in read_jsonl(ROOT / "tables/full_scorer_risk_summary.jsonl")}
    for method in ("BasePromptRewrite", "ReferenceResponse", "SFT", "DPO", "StoredMaskOffFinal", "OriginalDenoiserCandidate"):
        row = initial_methods[method]
        report_lines.append(
            f"| {method} | {row['N']} | {fmt(row['overall_quality'])} | {fmt(row['empathy'])} | "
            f"{fmt(row['specificity'])} | {fmt(row['quality_3axis_mean'])} | {fmt(risk_initial[method]['mean_risk'])} |"
        )
    report_lines += [
        "",
        f"BasePromptRewrite reached an exact 5/5 Quality-3 ceiling on {initial['counts']['base_ceiling_5_of_5']}/43 items. "
        f"The stored reference was better on {initial['counts']['reference_better_than_base']}, tied on "
        f"{initial['counts']['reference_tied_with_base']}, and worse on {initial['counts']['reference_worse_than_base']} "
        "under pointwise GPT-4.1 scoring.",
        "",
        "Under strict two-order pairwise comparison, BasePromptRewrite beat the stored reference on "
        f"{next(row for row in initial_pairwise['summaries'] if row['comparison'].endswith('ReferenceResponse'))['base_wins_consensus']}/43 "
        "items; the stored reference won 0. This does not support automatically treating these VALID references as superior residual-training targets; independent target review is needed.",
        "",
        "## Second-pass adapter-only control",
        "",
        "Both candidate arms received the same question, unsafe response, BasePromptRewrite draft, aspect scores, rendered "
        "risk prompt, token IDs, and decoding settings. BaseSelfRefineCandidate disabled adapters; "
        "TrainedDenoiserOnBaseCandidate loaded proposed_mask_off/final.",
        "",
        "| Candidate | N | Overall | Empathy | Specificity | Quality-3 | Mean words | Refusal signals | Foreign-script signals | Placeholder corruption | Malformed surface/markup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("BasePromptRewrite", "BaseSelfRefineCandidate", "TrainedDenoiserOnBaseCandidate"):
        row = candidate_methods[method]
        refusal = refusal_by_method.get(method, {}).get("refusal_or_meta_refusal_count", "not counted")
        foreign_script = refusal_by_method.get(method, {}).get("unexpected_non_latin_script_count", "not counted")
        placeholder = refusal_by_method.get(method, {}).get("placeholder_or_bracket_corruption_count", "not counted")
        malformed = refusal_by_method.get(method, {}).get("malformed_surface_token_or_markup_count", "not counted")
        report_lines.append(
            f"| {method} | {row['N']} | {fmt(row['overall_quality'])} | {fmt(row['empathy'])} | "
            f"{fmt(row['specificity'])} | {fmt(row['quality_3axis_mean'])} | {row['mean_words']:.1f} | "
            f"{refusal} | {foreign_script} | {placeholder} | {malformed} |"
        )
    report_lines += ["", "### Strict two-order comparisons", ""]
    for row in controls["pairwise"]:
        report_lines.append(
            f"- {row['comparison']}: left/right/tie/disagreement/failure = "
            f"{row['left_wins_consensus']}/{row['right_wins_consensus']}/{row['tie_consensus']}/"
            f"{row['order_disagreement']}/{row['technical_failure']} (N={row['N']})."
        )
    failure_examples: dict[tuple[str, str], dict[str, Any]] = {}
    for method, signal_row in refusal_by_method.items():
        for field, label in (
            ("refusal_or_meta_refusal_ids", "meta-refusal"),
            ("unexpected_non_latin_script_ids", "mixed-script generation"),
            ("placeholder_or_bracket_corruption_ids", "placeholder/bracket corruption"),
            ("malformed_surface_token_or_markup_ids", "malformed surface token/markup"),
        ):
            ids = signal_row.get(field) or []
            if not ids:
                continue
            item = generation_by_method_id[(method, ids[0])]
            normalized_response = " ".join(item["candidate_response"].split())
            excerpt = normalized_response[:320] + ("..." if len(normalized_response) > 320 else "")
            key = (method, ids[0])
            if key not in failure_examples:
                failure_examples[key] = {
                    "labels": [],
                    "method": method,
                    "canonical_id": ids[0],
                    "excerpt": excerpt,
                }
            if label not in failure_examples[key]["labels"]:
                failure_examples[key]["labels"].append(label)
    if failure_examples:
        report_lines += ["", "### Representative raw failure evidence", ""]
        for example in failure_examples.values():
            labels = ", ".join(example["labels"])
            report_lines.append(
                f"- {labels}; {example['method']}; `{example['canonical_id']}`: {example['excerpt']}"
            )
    report_lines += [
        "",
        "### Selection-stage results",
        "",
        "| Final policy | N | Gate calls | Accepted | Accepted with surface signal | Quality-3 | Mean risk | W/T/L vs Base |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in sorted(final_methods.items()):
        compare = wtl.get(method, {"wins_vs_base": 0, "ties_vs_base": 43, "losses_vs_base": 0})
        if method.startswith("BaseSelfRefine"):
            candidate_method = "BaseSelfRefineCandidate"
        elif method.startswith("TrainedDenoiserOnBase"):
            candidate_method = "TrainedDenoiserOnBaseCandidate"
        else:
            candidate_method = None
        accepted_with_signal = (
            len(set(compare.get("accepted_ids", [])) & signal_ids_by_candidate.get(candidate_method, set()))
            if candidate_method
            else 0
        )
        mean_risk = control_risk[("final", method)]["mean_risk"]
        report_lines.append(
            f"| {method} | {row['N']} | {row['gate_called']} | {row['accepted']} | "
            f"{accepted_with_signal} | {fmt(row['quality_3axis_mean'])} | {fmt(mean_risk)} | "
            f"{compare['wins_vs_base']}/{compare['ties_vs_base']}/{compare['losses_vs_base']} |"
        )
    report_lines += [
        "",
        "## Auxiliary Llama check",
        "",
        "| Method | N | Failures | Quality-3 |",
        "|---|---:|---:|---:|",
    ]
    for method in ("BasePromptRewrite", "BaseSelfRefineCandidate", "TrainedDenoiserOnBaseCandidate"):
        row = llama_methods[method]
        report_lines.append(f"| {method} | {row['N']} | {row['failures']} | {fmt(row['quality_3axis_mean'])} |")
    for method, row in sorted(llama_final.items()):
        report_lines.append(f"| {method} | {row['N']} | {row['pointwise_failures']} | {fmt(row['quality_3axis_mean'])} |")
    report_lines += [
        "", "### Matched pointwise comparisons", "",
        "Each row includes only IDs with valid scores for both responses. W/T/L refers to the right method versus Base.", "",
        "| Judge | Right method | Paired N | Base mean | Right mean | W/T/L |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in paired_rows:
        report_lines.append(
            f"| {row['judge']} | {row['right']} | {row['N']} | {fmt(row['left_mean'])} | "
            f"{fmt(row['right_mean'])} | {row['right_wins']}/{row['ties']}/{row['right_losses']} |"
        )
    report_lines += [
        "",
        "## Interpretation",
        "",
        "- The stored VALID references scored below the no-adapter prompted rewrite on average under this rubric. This does not establish the cause of the trained checkpoints' lower scores.",
        f"- The no-adapter second-pass candidate scored {fmt(candidate_methods['BaseSelfRefineCandidate']['quality_3axis_mean'])} "
        f"versus {fmt(candidate_methods['BasePromptRewrite']['quality_3axis_mean'])} for BasePromptRewrite, but its strict "
        f"Base comparison had {pairwise_controls['BasePromptRewrite_vs_BaseSelfRefineCandidate']['order_disagreement']}/43 "
        "order disagreements and visible surface artifacts.",
        f"- With strict two-order GPT selection, the no-adapter all-generated final scored "
        f"{fmt(final_methods['BaseSelfRefineAllGeneratedGPTSelected']['quality_3axis_mean'])}; this is a small development-set gain, "
        "not evidence of robust TEST superiority.",
        f"- The trained-denoiser candidate scored {fmt(candidate_methods['TrainedDenoiserOnBaseCandidate']['quality_3axis_mean'])}; "
        f"in its direct adapter-only comparison, the no-adapter candidate won "
        f"{pairwise_controls['BaseSelfRefineCandidate_vs_TrainedDenoiserOnBaseCandidate']['left_wins_consensus']} cases and the "
        f"trained adapter won {pairwise_controls['BaseSelfRefineCandidate_vs_TrainedDenoiserOnBaseCandidate']['right_wins_consensus']}.",
        f"- GPT-4.1 scored the no-adapter second pass above Base, whereas auxiliary Llama scored it "
        f"{fmt(llama_methods['BaseSelfRefineCandidate']['quality_3axis_mean'])} versus "
        f"{fmt(llama_methods['BasePromptRewrite']['quality_3axis_mean'])} for Base (with "
        f"{llama_methods['BaseSelfRefineCandidate']['failures']} and {llama_methods['BasePromptRewrite']['failures']} parse failures, respectively). "
        "The judges therefore agree that the trained adapter is weak on Base drafts, but do not establish a robust gain over BasePromptRewrite.",
        "- A new residual adapter should not be trained on all stored safe_response targets. It requires independently verified edits that are better than each BasePromptRewrite draft.",
        "- Preservation examples are a possible future training intervention; their benefit has not been tested here.",
        "- Failure-pattern counts are exploratory signals devised during inspection, can overlap, and are not exhaustive human error annotations.",
        "- GPT-4.1 served as both selector and primary evaluator; the auxiliary Llama result is reported separately and expert validation remains independent.",
        "- GPT-4.1 assigned ceiling scores to some responses with visible surface corruption; automatic scores therefore do not replace response-level or expert review.",
        "- The SFT-saved and base tokenizer files have different raw hashes, but their vocabulary/token-ID mapping and chat template are identical; see TOKENIZER_EQUIVALENCE_AUDIT.json.",
        "- All analyses here are VALID development evidence, not untouched TEST confirmation.",
        "",
        "## API usage",
        "",
        f"Paid GPT-4.1 calls: {api_usage['successful_paid_requests']}; failures: {api_usage['failed_latest']}; "
        f"prompt tokens: {api_usage['input_tokens']}; completion tokens: {api_usage['output_tokens']}; "
        f"estimated cost: ${api_usage['estimated_cost_usd']:.4f}.",
        "",
    ]
    report = "\n".join(report_lines)
    (ROOT / "BASE_FIRST_RESIDUAL_RESEARCH_REPORT.md").write_text(report, encoding="utf-8")

    latex_lines = [
        "# LaTeX insertion: VALID43 base-first residual diagnostic",
        "",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Overall & Empathy & Specificity & Quality-3 & $N$ \\\\",
        "\\midrule",
    ]
    for method, label in (
        ("BasePromptRewrite", "Prompt rewrite (base)"),
        ("BaseSelfRefineCandidate", "Base self-refine candidate"),
        ("TrainedDenoiserOnBaseCandidate", "Trained denoiser candidate"),
    ):
        row = candidate_methods[method]
        latex_lines.append(
            f"{label} & {fmt(row['overall_quality'])} & {fmt(row['empathy'])} & "
            f"{fmt(row['specificity'])} & {fmt(row['quality_3axis_mean'])} & {row['N']} \\\\"
        )
    latex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Development-only VALID43 comparison using the expert-aligned GPT-4.1 rubric. "
        "The two second-pass candidates use identical Base drafts and rendered prompts; only the generation adapter differs.}",
        "\\label{tab:valid43_base_draft_control}",
        "\\end{table}",
        "```",
        "",
        "```latex",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Final policy & Quality-3 & Mean risk & Accepted & $N$ \\\\",
        "\\midrule",
    ]
    for method, label in (
        ("BasePromptRewrite", "Prompt rewrite (base)"),
        ("BaseSelfRefineAllGeneratedGPTSelected", "Base self-refine + selector"),
        ("TrainedDenoiserOnBaseAllGeneratedGPTSelected", "Trained denoiser + selector"),
    ):
        row = final_methods[method]
        risk = control_risk[("final", method)]["mean_risk"]
        latex_lines.append(
            f"{label} & {fmt(row['quality_3axis_mean'])} & {fmt(risk)} & {row['accepted']} & {row['N']} \\\\"
        )
    latex_lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Development-only final-output comparison on VALID43. Candidates were generated for every item for diagnosis; "
        "acceptance required agreement under both A/B presentation orders.}",
        "\\label{tab:valid43_base_first_selection}",
        "\\end{table}",
        "```",
        "",
        "```latex",
        "\\paragraph{Base-first residual diagnostic.}",
        f"On VALID43, the no-adapter prompted rewrite achieved a Quality-3 mean of "
        f"{fmt(initial_methods['BasePromptRewrite']['quality_3axis_mean'])}, compared with "
        f"{fmt(initial_methods['SFT']['quality_3axis_mean'])} for SFT and "
        f"{fmt(initial_methods['StoredMaskOffFinal']['quality_3axis_mean'])} for the stored denoiser policy. "
        "The stored reference did not win any strict two-order comparison against the prompted rewrite, so we do not treat it as an automatic residual-training target. "
        "We further held the prompted draft and second-pass input fixed while switching only the generation adapter. "
        f"The no-adapter candidate beat the trained-denoiser candidate in "
        f"{pairwise_controls['BaseSelfRefineCandidate_vs_TrainedDenoiserOnBaseCandidate']['left_wins_consensus']} cases under strict two-order comparison, "
        f"whereas the trained candidate won {pairwise_controls['BaseSelfRefineCandidate_vs_TrainedDenoiserOnBaseCandidate']['right_wins_consensus']}. "
        f"However, GPT-4.1 had {pairwise_controls['BasePromptRewrite_vs_BaseSelfRefineCandidate']['order_disagreement']}/43 order disagreements for Base versus self-refinement, "
        f"and auxiliary Llama scored self-refinement {fmt(llama_methods['BaseSelfRefineCandidate']['quality_3axis_mean'])} versus "
        f"{fmt(llama_methods['BasePromptRewrite']['quality_3axis_mean'])} for Base, with valid counts of "
        f"{llama_methods['BaseSelfRefineCandidate']['N']} and {llama_methods['BasePromptRewrite']['N']}, respectively. "
        "GPT-4.1 served as both selector and primary evaluator, and the denoiser was originally trained on SFT drafts. "
        "These development results are diagnostic evidence and were not used to alter TEST outputs.",
        "```",
        "",
    ]
    publication = ROOT / "publication/BASE_FIRST_RESIDUAL_LATEX.md"
    publication.parent.mkdir(parents=True, exist_ok=True)
    publication.write_text("\n".join(latex_lines), encoding="utf-8")

    residual = read_jsonl(ROOT / "inputs/residual_development_candidates.jsonl")
    residual_by_id = {row["canonical_id"]: {**row, "review_trigger": "automatic_quality_below_5"} for row in residual}
    base_rows = read_jsonl(ROOT / "generation/base_prompt_rewrite_valid43_extraction_recovered.jsonl")
    base_scores = {
        row["canonical_id"]: row
        for row in read_jsonl(ROOT / "results/pointwise_assignment_scores.jsonl")
        if row["method"] == "BasePromptRewrite"
    }
    surface_artifact_pattern = re.compile(
        r"[\u0400-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]|\bvarsigma\b|\bend-oflife\b",
        re.I,
    )
    for row in base_rows:
        response = row["base_prompt_response_evaluation"]
        if not surface_artifact_pattern.search(response) or row["canonical_id"] in residual_by_id:
            continue
        scored = base_scores[row["canonical_id"]]
        residual_by_id[row["canonical_id"]] = {
            "source_order": row["source_order"],
            "canonical_id": row["canonical_id"],
            "question": row["question"],
            "base_response": response,
            "base_scores": {
                "overall_quality": scored["overall_quality"],
                "empathy": scored["empathy"],
                "specificity": scored["specificity"],
                "quality_3axis_mean": scored["quality_3axis_mean"],
            },
            "base_rationale": scored["rationale"],
            "review_trigger": "surface_artifact_missed_by_automatic_score",
        }
    review_rows = sorted(residual_by_id.values(), key=lambda row: row["source_order"])
    review_csv = ROOT / "expert_followup/VALID43_base_residual_review_candidates.csv"
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with review_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "source_order",
            "canonical_id",
            "question",
            "base_response",
            "base_quality_3axis",
            "automatic_rationale",
            "review_trigger",
            "expert_revision_needed",
            "expert_edited_response",
            "expert_reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(
                {
                    "source_order": row["source_order"],
                    "canonical_id": row["canonical_id"],
                    "question": row["question"],
                    "base_response": row["base_response"],
                    "base_quality_3axis": row["base_scores"]["quality_3axis_mean"],
                    "automatic_rationale": row["base_rationale"],
                    "review_trigger": row["review_trigger"],
                    "expert_revision_needed": "",
                    "expert_edited_response": "",
                    "expert_reason": "",
                }
            )
    html_rows = []
    for row in review_rows:
        html_rows.append(
            "<article><h2>#{order} {cid}</h2><h3>Question</h3><p>{question}</p>"
            "<h3>Base response</h3><p>{response}</p><p><strong>Review trigger:</strong> {trigger}</p>"
            "<p><strong>Automatic diagnostic only:</strong> {rationale}</p></article>".format(
                order=row["source_order"] + 1,
                cid=html.escape(row["canonical_id"]),
                question=html.escape(row["question"]).replace("\n", "<br>"),
                response=html.escape(row["base_response"]).replace("\n", "<br>"),
                trigger=html.escape(row["review_trigger"]),
                rationale=html.escape(row["base_rationale"]),
            )
        )
    (ROOT / "expert_followup/VALID43_base_residual_review_candidates.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>VALID43 residual review candidates</title>"
        "<style>body{font-family:Arial,sans-serif;max-width:960px;margin:40px auto;line-height:1.55}article{border-bottom:1px solid #ccc;padding:24px 0}h2{font-size:20px}h3{font-size:15px;margin-bottom:4px}p{white-space:normal}</style>"
        "</head><body><h1>VALID43 Base residual review candidates</h1>"
        "<p>Development-only review set. Inclusion does not mean the stored reference is a better training target.</p>"
        + "".join(html_rows)
        + "</body></html>",
        encoding="utf-8",
    )

    code_dir = ROOT / "code"
    code_dir.mkdir(exist_ok=True)
    code_names = (
        "run_valid43_base_prompt_rewrite.py",
        "run_base_first_residual_valid_analysis.py",
        "run_base_first_residual_valid_pipeline.sh",
        "run_valid43_base_draft_controls.py",
        "run_valid43_base_draft_control_analysis.py",
        "run_valid43_base_draft_control_pipeline.sh",
        "run_valid43_base_draft_control_llama.py",
        "run_gpt41_checkpoint_queue.py",
        "package_base_first_residual_valid.py",
    )
    for name in code_names:
        shutil.copy2(REPO / "scripts" / name, code_dir / name)

    generation_code = code_dir / "at_generation"
    generation_code.mkdir(exist_ok=True)
    source_hashes = []
    for name, expected_hash in (
        ("run_valid43_base_prompt_rewrite.py", read_json(ROOT / "run_contract.json")["frozen_code"]["runner_sha256"]),
        ("run_valid43_base_draft_controls.py", read_json(ROOT / "base_draft_second_pass_contract.json")["artifacts"]["runner_sha256"]),
    ):
        current_source = (REPO / "scripts" / name).read_text(encoding="utf-8")
        original_source = current_source.replace(
            'path.read_text(encoding="utf-8").split("\\n")',
            'path.read_text(encoding="utf-8").splitlines()',
        )
        original_hash = hashlib.sha256(original_source.encode("utf-8")).hexdigest()
        if original_hash != expected_hash:
            raise RuntimeError(f"Cannot recover frozen generation runner: {name}")
        (generation_code / name).write_text(original_source, encoding="utf-8")
        source_hashes.append({"file": name, "generation_sha256": original_hash, "current_sha256": sha256_file(REPO / "scripts" / name)})
    write_json(code_dir / "GENERATION_CODE_PROVENANCE.json", {
        "runners": source_hashes,
        "post_generation_change": "JSONL readers now split only on LF, preserving Unicode separators within JSON strings. Frozen generation source reconstructed and verified against pre-generation hashes.",
    })

    bundle = ROOT / "VALID43_base_first_residual_research_bundle.zip"
    excluded = {bundle, ROOT / "FINAL_MANIFEST.json", ROOT / "BUNDLE_CONTENT_MANIFEST.json"}
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and path not in excluded and not path.name.endswith(".lock"))
    inventory = [
        {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in files
    ]
    write_json(ROOT / "BUNDLE_CONTENT_MANIFEST.json", {"version": VERSION, "files": len(inventory), "artifacts": inventory})
    files.append(ROOT / "BUNDLE_CONTENT_MANIFEST.json")
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT))
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    download_copy = DOWNLOADS / bundle.name
    shutil.copy2(bundle, download_copy)
    scp_command = (
        f'scp "user@10.201.31.127:{download_copy}" '
        f'"$env:USERPROFILE\\Downloads\\{bundle.name}"'
    )
    (ROOT / "DOWNLOAD_COMMAND_POWERSHELL.txt").write_text(scp_command + "\n", encoding="utf-8")
    final_manifest = {
        "version": VERSION,
        "status": "complete_with_auxiliary_evaluation_failures" if llama["failures"] else "complete",
        "llama": {"attempted": 129, "valid": 129 - len(llama["failures"]), "failed": len(llama["failures"])},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": {"path": str(bundle), "bytes": bundle.stat().st_size, "sha256": sha256_file(bundle)},
        "server_download_copy": {"path": str(download_copy), "bytes": download_copy.stat().st_size, "sha256": sha256_file(download_copy)},
        "report": str(ROOT / "BASE_FIRST_RESIDUAL_RESEARCH_REPORT.md"),
        "latex": str(publication),
        "expert_followup_candidates": len(review_rows),
        "expert_followup_automatic_below_ceiling": sum(
            row["review_trigger"] == "automatic_quality_below_5" for row in review_rows
        ),
        "expert_followup_surface_artifacts": sum(
            row["review_trigger"] == "surface_artifact_missed_by_automatic_score" for row in review_rows
        ),
        "api_usage": api_usage,
        "scp_command": scp_command,
    }
    write_json(ROOT / "FINAL_MANIFEST.json", final_manifest)
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
