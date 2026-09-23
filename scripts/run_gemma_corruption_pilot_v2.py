#!/usr/bin/env python3
"""Run a six-axis, sequentially QC-gated Gemma corruption pilot of 100 rows."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from corruption_contract_v2 import (
    AXES,
    CorruptionJudge,
    GeneratedStage,
    JudgeResult,
    SequentialCorruptionGenerator,
    StageRequest,
    acceptance_decision,
    build_judge_prompt,
    build_stage_prompt,
    judge_result_dict,
    parse_json_object,
    parse_judge_result,
    validate_judge_evidence,
)
from fullpaper_acl_pipeline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    SPLIT_RATIOS,
    largest_remainder_counts,
    normalize_text,
    read_jsonl,
    sha256_file,
    stable_random_key,
    write_json,
    write_jsonl,
)
from run_gemma_corruption_pilot import GemmaBackend, MODEL_REPO, MODEL_REVISION, MODEL_SNAPSHOT


PILOT_SEED = DEFAULT_SEED + 2
AXIS_COUNT_RATIOS = {1: 0.50, 2: 0.35, 3: 0.15}


def balanced_specs(total: int, seed: int) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
    quotas = largest_remainder_counts(total, AXIS_COUNT_RATIOS)
    total_axis_slots = sum(k * count for k, count in quotas.items())
    total_pair_slots = sum(math.comb(k, 2) * count for k, count in quotas.items())
    axis_target = total_axis_slots / len(AXES)
    pair_target = total_pair_slots / math.comb(len(AXES), 2)
    marginal = Counter()
    pairwise = Counter()
    combination_counts: dict[int, Counter[tuple[str, ...]]] = defaultdict(Counter)
    specs: list[tuple[str, ...]] = []

    # Place high-order combinations first so singleton choices can finish exact
    # marginal balancing without disturbing pairwise balance.
    for axis_count in sorted(quotas, reverse=True):
        choices = list(itertools.combinations(AXES, axis_count))
        for slot in range(quotas[axis_count]):
            def score(combo: tuple[str, ...]) -> tuple[float, float, int, str]:
                combo_pairs = set(itertools.combinations(combo, 2))
                marginal_loss = sum(
                    (marginal[axis] + (axis in combo) - axis_target) ** 2
                    for axis in AXES
                )
                pair_loss = sum(
                    (pairwise[pair] + (pair in combo_pairs) - pair_target) ** 2
                    for pair in itertools.combinations(AXES, 2)
                )
                reuse = combination_counts[axis_count][combo]
                tie = stable_random_key(seed, f"{axis_count}:{slot}:{'+'.join(combo)}")
                return marginal_loss, pair_loss, reuse, tie

            chosen = min(choices, key=score)
            specs.append(chosen)
            marginal.update(chosen)
            pairwise.update(itertools.combinations(chosen, 2))
            combination_counts[axis_count][chosen] += 1

    indexed = list(enumerate(specs))
    indexed.sort(key=lambda item: stable_random_key(seed + 2, f"{item[0]}:{'+'.join(item[1])}"))
    specs = [combo for _, combo in indexed]
    stats = {
        "axis_count_counts": {str(k): v for k, v in sorted(quotas.items())},
        "marginal_counts": {axis: marginal[axis] for axis in AXES},
        "marginal_range": max(marginal.values()) - min(marginal.values()),
        "pairwise_counts": {
            "+".join(pair): pairwise[pair]
            for pair in itertools.combinations(AXES, 2)
        },
        "pairwise_range": max(pairwise.values()) - min(pairwise.values()),
    }
    return specs, stats


def select_rows(canonical: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size != 100:
        raise ValueError("This guarded pilot runner only permits --pilot-size 100")
    source_counts = Counter(row["source"] for row in canonical)
    source_quotas = largest_remainder_counts(
        size,
        {source: count / len(canonical) for source, count in source_counts.items()},
    )
    selected = []
    for source in sorted(source_quotas):
        source_rows = [row for row in canonical if row["source"] == source]
        split_quotas = largest_remainder_counts(source_quotas[source], SPLIT_RATIOS)
        for split in SPLIT_RATIOS:
            candidates = [row for row in source_rows if row["split"] == split]
            candidates.sort(key=lambda row: stable_random_key(seed, row["canonical_id"]))
            selected.extend(candidates[: split_quotas[split]])
    if len(selected) != size or len({row["canonical_id"] for row in selected}) != size:
        raise AssertionError("Balanced pilot selection did not produce 100 unique rows")
    selected.sort(key=lambda row: stable_random_key(seed + 1, row["canonical_id"]))
    return selected


def attach_specs(
    rows: list[dict[str, Any]], specs: list[tuple[str, ...]], seed: int
) -> list[dict[str, Any]]:
    remaining = list(specs)
    source_marginals: dict[str, Counter[str]] = defaultdict(Counter)
    joined = []
    for row in rows:
        source = row["source"]
        def score(combo: tuple[str, ...]) -> tuple[int, int, str]:
            after = source_marginals[source].copy()
            after.update(combo)
            values = [after[axis] for axis in AXES]
            return max(values) - min(values), sum(value * value for value in values), stable_random_key(
                seed, row["canonical_id"] + ":" + "+".join(combo)
            )
        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        source_marginals[source].update(chosen)
        joined.append({"clean": row, "intended_axes": list(chosen), "axis_count": len(chosen)})
    return joined


class GemmaSequentialGenerator(SequentialCorruptionGenerator):
    repo = MODEL_REPO
    revision = MODEL_REVISION

    def __init__(self, backend: GemmaBackend) -> None:
        self.backend = backend

    def generate_batch(self, requests: Sequence[StageRequest]) -> list[GeneratedStage]:
        seeds = {request.generation_seed for request in requests}
        if len(seeds) != 1:
            raise ValueError("Generation batch must share one recorded seed")
        outputs = self.backend.complete(
            [build_stage_prompt(request) for request in requests],
            seed=next(iter(seeds)),
            max_new_tokens=self.backend.generation_max_new_tokens,
            sample=True,
        )
        return [GeneratedStage(text=text, raw_output=text) for text in outputs]


class GemmaEvidenceJudge(CorruptionJudge):
    """Distinct QC interface and call path; model identity is recorded explicitly."""

    repo = MODEL_REPO
    revision = MODEL_REVISION

    def __init__(self, backend: GemmaBackend) -> None:
        self.backend = backend

    def judge_batch(
        self,
        requests: Sequence[StageRequest],
        candidates: Sequence[GeneratedStage],
    ) -> list[JudgeResult | None]:
        outputs = self.backend.complete(
            [
                build_judge_prompt(request.question, request.clean_response, candidate.text)
                for request, candidate in zip(requests, candidates, strict=True)
            ],
            seed=0,
            max_new_tokens=self.backend.qc_max_new_tokens,
            sample=False,
        )
        results: list[JudgeResult | None] = []
        for output, candidate, request in zip(outputs, candidates, requests, strict=True):
            try:
                result = parse_judge_result(parse_json_object(output), output)
                validate_judge_evidence(
                    result,
                    candidate=candidate.text,
                    clean_response=request.clean_response,
                    normalize=normalize_text,
                )
                results.append(result)
            except Exception:
                results.append(None)
        return results


@dataclass
class PilotState:
    clean: dict[str, Any]
    intended_axes: tuple[str, ...]
    current_response: str
    completed_axes: list[str] = field(default_factory=list)
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    total_generation_attempts: int = 0
    last_qc: JudgeResult | None = None
    failed: bool = False
    failure_reason: str | None = None


def stage_record(
    request: StageRequest,
    candidate: GeneratedStage,
    judge: JudgeResult | None,
    accepted: bool,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "stage_index": request.stage_index,
        "target_axis": request.target_axis,
        "required_axes_after_stage": list(request.completed_axes) + [request.target_axis],
        "generation_attempt": request.generation_attempt,
        "generation_seed": request.generation_seed,
        "input_response": request.current_response,
        "candidate_response": candidate.text,
        "judge": judge_result_dict(judge) if judge else None,
        "accepted": accepted,
        "rejection_reason": reason,
    }


def run_stages(
    states: list[PilotState],
    generator: SequentialCorruptionGenerator,
    judge: CorruptionJudge,
    *,
    batch_size: int,
    max_attempts: int,
    seed: int,
) -> None:
    for stage_index in range(3):
        active = [state for state in states if not state.failed and len(state.intended_axes) > stage_index]
        pending = active
        feedback: dict[str, str] = {}
        for attempt in range(1, max_attempts + 1):
            if not pending:
                break
            retry: list[PilotState] = []
            for batch_start in range(0, len(pending), batch_size):
                batch = pending[batch_start : batch_start + batch_size]
                batch_seed = seed + stage_index * 1_000_000 + attempt * 100_000 + batch_start
                requests = [
                    StageRequest(
                        canonical_id=state.clean["canonical_id"],
                        split=state.clean["split"],
                        question=state.clean["question"],
                        clean_response=state.clean["clean_response"],
                        current_response=state.current_response,
                        intended_axes=state.intended_axes,
                        completed_axes=tuple(state.completed_axes),
                        target_axis=state.intended_axes[stage_index],
                        stage_index=stage_index + 1,
                        generation_seed=batch_seed,
                        generation_attempt=attempt,
                        retry_feedback=feedback.get(state.clean["canonical_id"]),
                    )
                    for state in batch
                ]
                try:
                    candidates = generator.generate_batch(requests)
                    judgments = judge.judge_batch(requests, candidates)
                except Exception as exc:
                    reason = f"batch runtime failure: {type(exc).__name__}: {exc}"
                    for state in batch:
                        state.total_generation_attempts += 1
                        feedback[state.clean["canonical_id"]] = reason
                        retry.append(state)
                    continue
                for state, request, candidate, result in zip(
                    batch, requests, candidates, judgments, strict=True
                ):
                    state.total_generation_attempts += 1
                    required = tuple(state.completed_axes) + (request.target_axis,)
                    if not candidate.text.strip():
                        accepted, reason = False, "generator returned an empty response"
                    elif normalize_text(candidate.text) == normalize_text(state.current_response):
                        accepted, reason = False, "generator left the current response unchanged"
                    else:
                        accepted, reason = acceptance_decision(result, required)
                    state.stage_history.append(stage_record(request, candidate, result, accepted, reason))
                    if accepted:
                        state.current_response = candidate.text
                        state.completed_axes.append(request.target_axis)
                        state.last_qc = result
                        feedback.pop(state.clean["canonical_id"], None)
                    else:
                        feedback[state.clean["canonical_id"]] = reason or "automated QC rejection"
                        retry.append(state)
                print(
                    f"stage={stage_index + 1} attempt={attempt} "
                    f"processed={min(batch_start + len(batch), len(pending))}/{len(pending)} "
                    f"retry={len(retry)}",
                    flush=True,
                )
            pending = retry
        for state in pending:
            state.failed = True
            state.failure_reason = feedback.get(state.clean["canonical_id"], "stage failed after retries")


def final_record(state: PilotState) -> dict[str, Any]:
    if state.failed or state.last_qc is None:
        raise ValueError("Cannot serialize failed pilot state as accepted")
    realized = list(state.last_qc.realized_axes)
    if not set(state.intended_axes).issubset(realized):
        raise AssertionError("Accepted output lacks an intended axis")
    return {
        "schema_version": "sequential-six-axis-corruption-v2",
        "canonical_id": state.clean["canonical_id"],
        "source": state.clean["source"],
        "source_component": state.clean["source_component"],
        "split": state.clean["split"],
        "question": state.clean["question"],
        "clean_response": state.clean["clean_response"],
        "intended_axes": list(state.intended_axes),
        "axis_count": len(state.intended_axes),
        "generator_repo": MODEL_REPO,
        "generator_revision": MODEL_REVISION,
        "judge_repo": MODEL_REPO,
        "judge_revision": MODEL_REVISION,
        "generation_seed": state.stage_history[-1]["generation_seed"],
        "generation_attempt": state.stage_history[-1]["generation_attempt"],
        "total_generation_attempts": state.total_generation_attempts,
        "corrupted_response": state.current_response,
        "realized_axes": realized,
        "axis_judgments": judge_result_dict(state.last_qc)["axes"],
        "qc_global": {
            key: value
            for key, value in judge_result_dict(state.last_qc).items()
            if key not in {"axes", "realized_axes"}
        },
        "qc_pass": True,
        "qc_failure_reason": None,
        "stage_history": state.stage_history,
    }


def build_report(
    states: list[PilotState],
    records: list[dict[str, Any]],
    selection_stats: dict[str, Any],
    output: Path,
) -> None:
    intended = Counter(axis for state in states for axis in state.intended_axes)
    realized_intended = Counter(
        axis
        for state in states
        if not state.failed and state.last_qc
        for axis in state.intended_axes
        if axis in state.last_qc.realized_axes
    )
    full_by_count = Counter()
    total_by_count = Counter(len(state.intended_axes) for state in states)
    unintended_events = 0
    accepted_axis_slots = 0
    for state in states:
        if state.failed or not state.last_qc:
            continue
        realized = set(state.last_qc.realized_axes)
        if set(state.intended_axes).issubset(realized):
            full_by_count[len(state.intended_axes)] += 1
        unintended_events += len(realized - set(state.intended_axes))
        accepted_axis_slots += len(AXES) - len(state.intended_axes)
    failures = [state for state in states if state.failed]
    total_attempts = sum(state.total_generation_attempts for state in states)
    retries = sum(
        event["generation_attempt"] > 1
        for state in states
        for event in state.stage_history
    )
    judge_parse_failures = sum(
        event["rejection_reason"] == "judge output could not be parsed"
        for state in states
        for event in state.stage_history
    )
    factual_states = [state for state in states if "factual_consistency" in state.intended_axes]
    factual_accepted = [state for state in factual_states if not state.failed]
    factual_stage_attempts = [
        event
        for state in factual_states
        for event in state.stage_history
        if event["target_axis"] == "factual_consistency"
    ]

    lines = [
        "# Gemma v2 balanced 100-example six-axis pilot",
        "",
        "This report gates any full corruption run. Generation used sequential single-axis stages, and each",
        "stage was accepted only after a separate evidence-based QC call detected every intended axis added so far.",
        "No counselor or expert annotation was used.",
        "",
        "## Pilot design and accounting",
        "",
        f"- Selected examples: {len(states)}",
        f"- Accepted final corruptions: {len(records)}",
        f"- Failed/dropped after stage retries: {len(failures)}",
        f"- Requested axis counts: `{selection_stats['axis_count_counts']}`",
        f"- Requested marginal counts: `{selection_stats['marginal_counts']}`",
        f"- Requested pairwise range: {selection_stats['pairwise_range']}",
        f"- Total generation attempts: {total_attempts}",
        f"- Retry generation calls (attempt number > 1): {retries}",
        f"- Judge parse/evidence-validation failures: {judge_parse_failures}",
        "",
        "## Realization rate per axis",
        "",
        "| Axis | Requested | Fully realized in final accepted output | Rate |",
        "|---|---:|---:|---:|",
    ]
    for axis in AXES:
        rate = realized_intended[axis] / intended[axis] if intended[axis] else 0
        lines.append(f"| `{axis}` | {intended[axis]} | {realized_intended[axis]} | {rate:.3f} |")
    lines.extend(
        [
            "",
            "## Full realization by requested axis count",
            "",
            "| Requested axes | Examples | Fully realized | Rate |",
            "|---:|---:|---:|---:|",
        ]
    )
    for count in (1, 2, 3):
        rate = full_by_count[count] / total_by_count[count] if total_by_count[count] else 0
        lines.append(f"| {count} | {total_by_count[count]} | {full_by_count[count]} | {rate:.3f} |")
    unintended_rate = unintended_events / accepted_axis_slots if accepted_axis_slots else 0
    lines.extend(
        [
            "",
            "## Unintended axes",
            "",
            f"- Additional realized-axis events: {unintended_events}",
            f"- Available unselected-axis slots among accepted outputs: {accepted_axis_slots}",
            f"- Unintended-axis rate: {unintended_rate:.3f}",
            "",
            "## Factual-consistency analysis",
            "",
            f"- Requested factual-consistency examples: {len(factual_states)}",
            f"- Accepted with factual inconsistency detected in final output: {len(factual_accepted)}",
            f"- Factual stage generation attempts: {len(factual_stage_attempts)}",
            f"- Factual stage first-attempt acceptances: {sum(e['accepted'] and e['generation_attempt'] == 1 for e in factual_stage_attempts)}",
            f"- Factual stage retry acceptances: {sum(e['accepted'] and e['generation_attempt'] > 1 for e in factual_stage_attempts)}",
            f"- Factual examples ultimately dropped: {sum(state.failed for state in factual_states)}",
            "",
            "The v2 prompt requires one explicit, confidently stated false general claim and the judge must return",
            "a yes/no decision plus a verbatim evidence span. This directly targets the v1 0/17 realization failure.",
            "Factual stage acceptances may exceed final factual examples because an example can pass its factual stage",
            "and later be dropped when another intended axis cannot be realized.",
            "",
            "## Calibration limitations",
            "",
            "- The generator and judge use separate interfaces and calls, but both use the mandated fixed Gemma repository.",
            "  This is automated self-QC, not independent-model validation.",
            "- Axis order follows the deterministic combination order in this pilot. `overall_quality`, when requested,",
            "  is therefore introduced first; its 0% result is a production blocker and also an order confound to fix",
            "  before a subsequent pilot.",
            "- Categorical acceptance rules were enforced; no global numerical score threshold was selected.",
            "",
            "## Failure examples",
            "",
        ]
    )
    if not failures:
        lines.append("No examples were dropped. The diagnostic examples below are retry events that initially failed QC.")
    retry_events = [
        (state, event)
        for state in states
        for event in state.stage_history
        if not event["accepted"]
    ][:8]
    if retry_events:
        lines.extend(["", "| ID | Stage axis | Attempt | Failure | Candidate excerpt |", "|---|---|---:|---|---|"])
        for state, event in retry_events:
            excerpt = event["candidate_response"][:260].replace("\n", " ").replace("|", "\\|")
            reason = str(event["rejection_reason"] or "").replace("|", "\\|")
            lines.append(
                f"| `{state.clean['canonical_id']}` | `{event['target_axis']}` | {event['generation_attempt']} | {reason} | {excerpt} |"
            )
    else:
        lines.append("No generation or QC retry events occurred.")
    lines.extend(
        [
            "",
            "## Production gate",
            "",
            "The full 18,224-row corruption run was not started. Review the automated realization and unintended-axis",
            "rates in this report before authorizing full generation. Experts remain reserved for final held-out evaluation.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--split-manifest", default=str(DEFAULT_OUTPUT_DIR / "split_manifest.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_corruption_pilot_100_v2.jsonl"))
    parser.add_argument("--failures", default=str(DEFAULT_OUTPUT_DIR / "gemma_corruption_pilot_100_v2.failures.jsonl"))
    parser.add_argument("--selection", default=str(DEFAULT_OUTPUT_DIR / "gemma_pilot_100_v2_selection.json"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_DIR / "qc_pilot_100_v2_report.md"))
    parser.add_argument("--model-snapshot", default=str(MODEL_SNAPSHOT))
    parser.add_argument("--pilot-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--qc-max-new-tokens", type=int, default=640)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    canonical_path = Path(args.canonical).resolve()
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    if sha256_file(canonical_path) != split_manifest["canonical_sha256"]:
        raise AssertionError("Canonical data differs from the frozen clean split")
    canonical = list(read_jsonl(canonical_path))
    selected_rows = select_rows(canonical, args.pilot_size, args.seed)
    specs, stats = balanced_specs(args.pilot_size, args.seed)
    selected = attach_specs(selected_rows, specs, args.seed)
    selection = {
        "selection_version": "balanced-six-axis-sequential-pilot-v2",
        "seed": args.seed,
        "pilot_size": args.pilot_size,
        "canonical_sha256": split_manifest["canonical_sha256"],
        "source_counts": dict(Counter(item["clean"]["source"] for item in selected)),
        "split_counts": dict(Counter(item["clean"]["split"] for item in selected)),
        **stats,
        "rows": [
            {
                "canonical_id": item["clean"]["canonical_id"],
                "source": item["clean"]["source"],
                "split": item["clean"]["split"],
                "intended_axes": item["intended_axes"],
                "axis_count": item["axis_count"],
            }
            for item in selected
        ],
    }
    write_json(Path(args.selection).resolve(), selection)
    print(json.dumps({key: selection[key] for key in ("pilot_size", "source_counts", "split_counts", "axis_count_counts", "marginal_counts", "pairwise_range")}), flush=True)
    if args.select_only:
        return

    backend = GemmaBackend(
        snapshot=Path(args.model_snapshot).resolve(),
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        generation_max_new_tokens=args.generation_max_new_tokens,
        qc_max_new_tokens=args.qc_max_new_tokens,
    )
    generator = GemmaSequentialGenerator(backend)
    judge = GemmaEvidenceJudge(backend)
    states = [
        PilotState(
            clean=item["clean"],
            intended_axes=tuple(item["intended_axes"]),
            current_response=item["clean"]["clean_response"],
        )
        for item in selected
    ]
    run_stages(
        states,
        generator,
        judge,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )
    records = [final_record(state) for state in states if not state.failed]
    failures = [
        {
            "canonical_id": state.clean["canonical_id"],
            "source": state.clean["source"],
            "split": state.clean["split"],
            "intended_axes": list(state.intended_axes),
            "completed_axes": state.completed_axes,
            "total_generation_attempts": state.total_generation_attempts,
            "failure_reason": state.failure_reason,
            "stage_history": state.stage_history,
        }
        for state in states
        if state.failed
    ]
    write_jsonl(Path(args.output).resolve(), records)
    write_jsonl(Path(args.failures).resolve(), failures)
    build_report(states, records, stats, Path(args.report).resolve())
    print(json.dumps({"selected": len(states), "accepted": len(records), "failed": len(failures), "output": str(Path(args.output).resolve()), "report": str(Path(args.report).resolve())}), flush=True)


if __name__ == "__main__":
    main()
