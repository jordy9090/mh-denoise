#!/usr/bin/env python3
"""Run the 50-row Gemma corruption pilot and a separate QC pass."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from corruption_contract import (
    AXES,
    CorruptionGenerator,
    CorruptionQC,
    CorruptionRequest,
    GeneratedCorruption,
    QCResult,
    apply_calibrated_thresholds,
    build_corruption_prompt,
    build_qc_prompt,
    parse_json_object,
    qc_result_from_payload,
    validate_output_record,
)
from fullpaper_acl_pipeline import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    largest_remainder_counts,
    normalize_text,
    read_jsonl,
    sha256_file,
    stable_random_key,
    write_json,
    write_jsonl,
)


MODEL_REPO = "google/gemma-4-E4B-it"
MODEL_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
MODEL_SNAPSHOT = Path(
    "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)


def source_axis_count_quotas(
    source_counts: Counter[str],
    axis_count_quotas: dict[int, int],
    pilot_size: int,
) -> dict[tuple[str, int], int]:
    sources = sorted(source_counts)
    source_quotas = largest_remainder_counts(
        pilot_size,
        {source: count / sum(source_counts.values()) for source, count in source_counts.items()},
    )
    if len(sources) == 1:
        return {(sources[0], count): quota for count, quota in axis_count_quotas.items()}
    if len(sources) != 2:
        raise ValueError("Pilot quota solver currently expects one or two immediate sources")

    left, right = sources
    best: tuple[float, tuple[int, ...]] | None = None
    counts = sorted(axis_count_quotas)
    for allocation in itertools.product(*(range(axis_count_quotas[k] + 1) for k in counts)):
        if sum(allocation) != source_quotas[left]:
            continue
        expected = [axis_count_quotas[k] * source_quotas[left] / pilot_size for k in counts]
        loss = sum((actual - target) ** 2 for actual, target in zip(allocation, expected, strict=True))
        candidate = (loss, allocation)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise AssertionError("Could not solve source-by-axis-count pilot quotas")
    allocation = best[1]
    quotas: dict[tuple[str, int], int] = {}
    for index, axis_count in enumerate(counts):
        quotas[(left, axis_count)] = allocation[index]
        quotas[(right, axis_count)] = axis_count_quotas[axis_count] - allocation[index]
    return quotas


def select_stratified_pilot(
    canonical_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if size > len(canonical_rows):
        raise ValueError("Pilot size exceeds canonical dataset")
    canonical_by_id = {row["canonical_id"]: row for row in canonical_rows}
    joined = []
    for assignment in assignments:
        clean = canonical_by_id.get(assignment["canonical_id"])
        if clean is None:
            raise AssertionError(f"Assignment lacks canonical row: {assignment['canonical_id']}")
        joined.append({**assignment, "clean": clean})

    source_counts = Counter(item["source"] for item in joined)
    axis_count_quotas = largest_remainder_counts(size, {1: 0.50, 2: 0.35, 3: 0.15})
    cell_quotas = source_axis_count_quotas(source_counts, axis_count_quotas, size)
    target_axis_slots = sum(count * quota for count, quota in axis_count_quotas.items()) / len(AXES)
    total_pair_slots = sum(math.comb(count, 2) * quota for count, quota in axis_count_quotas.items())
    target_pair_slots = total_pair_slots / math.comb(len(AXES), 2)
    selected: list[dict[str, Any]] = []
    marginal = Counter()
    pairwise = Counter()

    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in joined:
        cells[(item["source"], item["axis_count"])].append(item)
    for cell in cells:
        cells[cell].sort(key=lambda item: stable_random_key(seed, item["canonical_id"]))

    for cell in sorted(cell_quotas, key=lambda value: (cell_quotas[value], value)):
        quota = cell_quotas[cell]
        available = list(cells[cell])
        for _ in range(quota):
            if not available:
                raise AssertionError(f"Insufficient candidates for pilot stratum {cell}")

            def selection_loss(item: dict[str, Any]) -> tuple[float, str]:
                axes = item["intended_axes"]
                marginal_loss = sum(
                    (marginal[axis] + (1 if axis in axes else 0) - target_axis_slots) ** 2
                    for axis in AXES
                )
                item_pairs = set(itertools.combinations(axes, 2))
                pair_loss = sum(
                    (pairwise[pair] + (1 if pair in item_pairs else 0) - target_pair_slots) ** 2
                    for pair in itertools.combinations(AXES, 2)
                )
                return marginal_loss + pair_loss, stable_random_key(seed + 1, item["canonical_id"])

            chosen = min(available, key=selection_loss)
            available.remove(chosen)
            selected.append(chosen)
            marginal.update(chosen["intended_axes"])
            pairwise.update(itertools.combinations(chosen["intended_axes"], 2))

    selected.sort(key=lambda item: stable_random_key(seed + 2, item["canonical_id"]))
    if len(selected) != size or len({item["canonical_id"] for item in selected}) != size:
        raise AssertionError("Pilot selection is not the requested number of unique rows")
    return selected


class GemmaBackend:
    def __init__(
        self,
        *,
        snapshot: Path,
        batch_size: int,
        max_input_tokens: int,
        generation_max_new_tokens: int,
        qc_max_new_tokens: int,
    ) -> None:
        if not snapshot.is_dir():
            raise FileNotFoundError(f"Pinned Gemma snapshot not found: {snapshot}")
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.generation_max_new_tokens = generation_max_new_tokens
        self.qc_max_new_tokens = qc_max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def complete(self, prompts: Sequence[str], *, seed: int, max_new_tokens: int, sample: bool) -> list[str]:
        if not prompts:
            return []
        chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
        rendered = self.tokenizer.apply_chat_template(
            chats,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        ).to(self.model.device)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if sample:
            kwargs.update({"temperature": 0.75, "top_p": 0.9})
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(output[prompt_width:], skip_special_tokens=True).strip()
            for output in outputs
        ]


class GemmaCorruptionGenerator(CorruptionGenerator):
    repo = MODEL_REPO
    revision = MODEL_REVISION

    def __init__(self, backend: GemmaBackend) -> None:
        self.backend = backend

    def generate_batch(self, requests: Sequence[CorruptionRequest]) -> list[GeneratedCorruption]:
        if not requests:
            return []
        seeds = {request.generation_seed for request in requests}
        if len(seeds) != 1:
            raise ValueError("One deterministic seed is required per generation batch")
        raw = self.backend.complete(
            [build_corruption_prompt(request) for request in requests],
            seed=next(iter(seeds)),
            max_new_tokens=self.backend.generation_max_new_tokens,
            sample=True,
        )
        return [GeneratedCorruption(text=text, raw_output=text) for text in raw]


class GemmaCorruptionQC(CorruptionQC):
    def __init__(self, backend: GemmaBackend, thresholds: dict[str, float] | None) -> None:
        self.backend = backend
        self.thresholds = thresholds

    def evaluate_batch(
        self,
        requests: Sequence[CorruptionRequest],
        corruptions: Sequence[GeneratedCorruption],
    ) -> list[QCResult]:
        raw_outputs = self.backend.complete(
            [build_qc_prompt(request, corruption.text) for request, corruption in zip(requests, corruptions, strict=True)],
            seed=0,
            max_new_tokens=self.backend.qc_max_new_tokens,
            sample=False,
        )
        results = []
        for request, raw in zip(requests, raw_outputs, strict=True):
            try:
                parsed = parse_json_object(raw)
                result = qc_result_from_payload(parsed, raw)
                results.append(apply_calibrated_thresholds(request, result, self.thresholds))
            except Exception as exc:
                results.append(
                    QCResult(
                        realized_axes=(),
                        scores={},
                        qc_pass=False,
                        failure_reason=f"QC parse failure: {type(exc).__name__}: {exc}",
                        raw_output=raw,
                    )
                )
        return results


def output_record(
    request: CorruptionRequest,
    corruption: GeneratedCorruption,
    qc: QCResult,
) -> dict[str, Any]:
    scores = dict(qc.scores)
    clean_words = len(request.clean_response.split())
    corrupted_words = len(corruption.text.split())
    scores.update(
        {
            "clean_response_words": clean_words,
            "corrupted_response_words": corrupted_words,
            "response_length_ratio": corrupted_words / max(1, clean_words),
        }
    )
    record = {
        "canonical_id": request.canonical_id,
        "split": request.split,
        "question": request.question,
        "clean_response": request.clean_response,
        "intended_axes": list(request.intended_axes),
        "axis_count": len(request.intended_axes),
        "generator_repo": MODEL_REPO,
        "generator_revision": MODEL_REVISION,
        "generation_seed": request.generation_seed,
        "generation_attempt": request.generation_attempt,
        "corrupted_response": corruption.text,
        "realized_axes": list(qc.realized_axes),
        "qc_scores": scores,
        "qc_pass": qc.qc_pass,
        "qc_failure_reason": qc.failure_reason,
    }
    validate_output_record(record)
    return record


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build_calibration_report(
    records: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    *,
    selected_count: int,
    output_path: Path,
) -> None:
    axis_count_counts = Counter(record["axis_count"] for record in records)
    source_counts = Counter(record.get("source", "unknown") for record in records)
    intended = Counter(axis for record in records for axis in record["intended_axes"])
    realized = Counter(axis for record in records for axis in record["realized_axes"])
    intended_realized = Counter(
        axis
        for record in records
        for axis in record["intended_axes"]
        if axis in record["realized_axes"]
    )

    metric_values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        scores = record["qc_scores"]
        for axis, score in scores["axis_degradation"].items():
            metric_values[f"degradation:{axis}"].append(float(score))
        for field in ("topic_relevance", "fluency", "unintended_catastrophic_degradation"):
            metric_values[field].append(float(scores[field]))

    def concern_key(record: dict[str, Any]) -> tuple[float, float, float, str]:
        scores = record["qc_scores"]
        intended_strength = min(scores["axis_degradation"][axis] for axis in record["intended_axes"])
        return (
            intended_strength,
            scores["topic_relevance"] + scores["fluency"],
            -scores["unintended_catastrophic_degradation"],
            record["canonical_id"],
        )

    concerns = sorted(records, key=concern_key)[:5]
    length_ratios = [record["qc_scores"]["response_length_ratio"] for record in records]
    missing_all_intended = sum(
        not set(record["intended_axes"]).issubset(record["realized_axes"])
        for record in records
    )
    lines = [
        "# QC calibration report: Gemma corruption pilot 50",
        "",
        "This is a calibration report, not a threshold-selection result. No final score thresholds are hard-coded,",
        "and `qc_pass` remains `null` for unthresholded pilot rows. Generation and QC were separate model calls.",
        "",
        "## Pilot accounting",
        "",
        f"- Stratified clean rows selected: {selected_count}",
        f"- Rows retained for calibration: {len(records)}",
        f"- Rows dropped after at most three hard-failure attempts: {len(dropped)}",
        f"- Axis-count distribution: `{dict(sorted(axis_count_counts.items()))}`",
        f"- Source distribution: `{dict(sorted(source_counts.items()))}`",
        f"- Rows missing at least one intended realized dimension: {missing_all_intended}",
        f"- Rows within the prompt's 0.8-1.2 approximate word-length band: "
        f"{sum(0.8 <= ratio <= 1.2 for ratio in length_ratios)}/{len(length_ratios)}",
        "",
        "## Intended versus realized dimensions",
        "",
        "| Dimension | Intended | Realized | Intended realized | Pilot recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis in AXES:
        recall = intended_realized[axis] / intended[axis] if intended[axis] else 0.0
        lines.append(f"| `{axis}` | {intended[axis]} | {realized[axis]} | {intended_realized[axis]} | {recall:.3f} |")
    lines.extend(
        [
            "",
            "## Score summaries",
            "",
            "| QC score | Mean | Median | Min | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in sorted(metric_values):
        values = metric_values[metric]
        if values:
            lines.append(
                f"| `{metric}` | {statistics.fmean(values):.3f} | {statistics.median(values):.3f} | "
                f"{min(values):.1f} | {max(values):.1f} |"
            )
    lines.extend(
        [
            "",
            "## Failure and boundary examples for calibration",
            "",
            "These are the five lowest-ranked rows under a diagnostic ordering (weakest intended-axis score first,",
            "then relevance/fluency, then catastrophic degradation). This ordering is descriptive and is not a pass threshold.",
            "",
            "| ID | Intended | Realized | QC note | Corrupted-response excerpt |",
            "|---|---|---|---|---|",
        ]
    )
    for record in concerns:
        excerpt = markdown_escape(record["corrupted_response"][:320])
        note = markdown_escape(str(record["qc_scores"].get("notes") or "")[:240])
        lines.append(
            f"| `{record['canonical_id']}` | `{','.join(record['intended_axes'])}` | "
            f"`{','.join(record['realized_axes'])}` | {note} | {excerpt} |"
        )
    if dropped:
        lines.extend(["", "### Dropped hard failures", ""])
        for item in dropped:
            lines.append(f"- `{item['canonical_id']}`: {item['failure_reason']}")
    lines.extend(
        [
            "",
            "## Calibration cautions",
            "",
            "- Gemma generated and judged this pilot; self-judging can inflate apparent realization and fluency.",
            "- Human double-annotation should calibrate intended-axis minima, relevance/fluency floors, and the",
            "  catastrophic-degradation ceiling before a production run.",
            "- Medical-boundary and toxicity-or-harm booleans are reported explicitly and should be compared with",
            "  human labels rather than inferred only from generic axis scores.",
            "- CounselBench Overall remains an evaluation dimension and was not used as a corruption operator.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", default=str(DEFAULT_OUTPUT_DIR / "canonical_clean_qa.jsonl"))
    parser.add_argument("--split-manifest", default=str(DEFAULT_OUTPUT_DIR / "split_manifest.json"))
    parser.add_argument("--assignments", default=str(DEFAULT_OUTPUT_DIR / "corruption_assignment.jsonl"))
    parser.add_argument("--assignment-manifest", default=str(DEFAULT_OUTPUT_DIR / "corruption_assignment.manifest.json"))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR / "gemma_corruption_pilot_50.jsonl"))
    parser.add_argument("--report", default=str(DEFAULT_OUTPUT_DIR / "qc_calibration_report.md"))
    parser.add_argument("--failures", default=str(DEFAULT_OUTPUT_DIR / "gemma_corruption_pilot_50.failures.jsonl"))
    parser.add_argument("--selection-manifest", default=str(DEFAULT_OUTPUT_DIR / "gemma_pilot_selection_manifest.json"))
    parser.add_argument("--model-snapshot", default=str(MODEL_SNAPSHOT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pilot-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--generation-max-new-tokens", type=int, default=512)
    parser.add_argument("--qc-max-new-tokens", type=int, default=320)
    parser.add_argument("--thresholds", help="JSON file created after calibration; omit for this pilot")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()

    canonical_path = Path(args.canonical).resolve()
    assignment_path = Path(args.assignments).resolve()
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    assignment_manifest = json.loads(Path(args.assignment_manifest).read_text(encoding="utf-8"))
    if sha256_file(canonical_path) != split_manifest["canonical_sha256"]:
        raise AssertionError("Canonical data no longer matches the frozen split manifest")
    if sha256_file(assignment_path) != assignment_manifest["assignment_sha256"]:
        raise AssertionError("Assignments no longer match their manifest")

    canonical = list(read_jsonl(canonical_path))
    assignments = list(read_jsonl(assignment_path))
    selected = select_stratified_pilot(canonical, assignments, size=args.pilot_size, seed=args.seed)
    selection_rows = []
    for item in selected:
        selection_rows.append(
            {
                "canonical_id": item["canonical_id"],
                "source": item["source"],
                "split": item["split"],
                "axis_count": item["axis_count"],
                "intended_axes": item["intended_axes"],
                "clean_response_words": len(item["clean"]["clean_response"].split()),
            }
        )
    selection_manifest_path = Path(args.selection_manifest).resolve()
    write_json(
        selection_manifest_path,
        {
            "selection_version": "gemma-corruption-pilot-stratified-v1",
            "seed": args.seed,
            "pilot_size": len(selected),
            "source_counts": dict(Counter(item["source"] for item in selected)),
            "axis_count_counts": dict(Counter(str(item["axis_count"]) for item in selected)),
            "axis_marginal_counts": dict(Counter(axis for item in selected for axis in item["intended_axes"])),
            "clean_split_sha256": split_manifest["canonical_sha256"],
            "assignment_sha256": assignment_manifest["assignment_sha256"],
            "rows": selection_rows,
        },
    )
    print(f"Selected {len(selected)} rows; wrote {selection_manifest_path}", flush=True)
    if args.select_only:
        return

    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    backend = GemmaBackend(
        snapshot=Path(args.model_snapshot).resolve(),
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        generation_max_new_tokens=args.generation_max_new_tokens,
        qc_max_new_tokens=args.qc_max_new_tokens,
    )
    generator = GemmaCorruptionGenerator(backend)
    qc = GemmaCorruptionQC(backend, thresholds)

    pending = list(selected)
    records_by_id: dict[str, dict[str, Any]] = {}
    hard_errors: dict[str, str] = {}
    for attempt in range(1, args.max_attempts + 1):
        if not pending:
            break
        next_pending = []
        next_pending_ids: set[str] = set()

        def queue_retry(item: dict[str, Any], message: str) -> None:
            canonical_id = item["canonical_id"]
            hard_errors[canonical_id] = message
            if canonical_id not in next_pending_ids:
                next_pending.append(item)
                next_pending_ids.add(canonical_id)

        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            batch_seed = args.seed + attempt * 100_000 + batch_start
            requests = [
                CorruptionRequest(
                    canonical_id=item["canonical_id"],
                    split=item["split"],
                    question=item["clean"]["question"],
                    clean_response=item["clean"]["clean_response"],
                    intended_axes=tuple(item["intended_axes"]),
                    generation_seed=batch_seed,
                    generation_attempt=attempt,
                )
                for item in batch
            ]
            try:
                corruptions = generator.generate_batch(requests)
            except Exception as exc:
                message = f"Generation failure: {type(exc).__name__}: {exc}"
                for item in batch:
                    queue_retry(item, message)
                print(
                    f"attempt={attempt} processed={min(batch_start + len(batch), len(pending))}/{len(pending)} "
                    f"retained={len(records_by_id)} retry={len(next_pending)}",
                    flush=True,
                )
                continue

            try:
                valid_items = []
                valid_requests = []
                valid_corruptions = []
                for item, request, corruption in zip(batch, requests, corruptions, strict=True):
                    if not corruption.text.strip():
                        queue_retry(item, "Generator returned an empty response")
                    elif normalize_text(corruption.text) == normalize_text(request.clean_response):
                        queue_retry(item, "Generator returned the clean response unchanged")
                    else:
                        valid_items.append(item)
                        valid_requests.append(request)
                        valid_corruptions.append(corruption)
                qc_results = qc.evaluate_batch(valid_requests, valid_corruptions)
                for item, request, corruption, qc_result in zip(
                    valid_items, valid_requests, valid_corruptions, qc_results, strict=True
                ):
                    if qc_result.qc_pass is False:
                        queue_retry(item, qc_result.failure_reason or "QC failed")
                    else:
                        record = output_record(request, corruption, qc_result)
                        record["source"] = item["source"]
                        record["source_component"] = item["clean"]["source_component"]
                        records_by_id[request.canonical_id] = record
                        hard_errors.pop(request.canonical_id, None)
            except Exception as exc:
                message = f"QC batch failure: {type(exc).__name__}: {exc}"
                for item in valid_items:
                    queue_retry(item, message)
            print(
                f"attempt={attempt} processed={min(batch_start + len(batch), len(pending))}/{len(pending)} "
                f"retained={len(records_by_id)} retry={len(next_pending)}",
                flush=True,
            )
        pending = next_pending

    dropped = [
        {
            "canonical_id": item["canonical_id"],
            "source": item["source"],
            "split": item["split"],
            "intended_axes": item["intended_axes"],
            "axis_count": item["axis_count"],
            "attempts": args.max_attempts,
            "failure_reason": hard_errors.get(item["canonical_id"], "unknown hard failure"),
        }
        for item in pending
    ]
    records = [records_by_id[item["canonical_id"]] for item in selected if item["canonical_id"] in records_by_id]
    output_path = Path(args.output).resolve()
    failure_path = Path(args.failures).resolve()
    write_jsonl(output_path, records)
    write_jsonl(failure_path, dropped)
    build_calibration_report(
        records,
        dropped,
        selected_count=len(selected),
        output_path=Path(args.report).resolve(),
    )
    print(
        json.dumps(
            {
                "selected": len(selected),
                "retained": len(records),
                "dropped": len(dropped),
                "output": str(output_path),
                "report": str(Path(args.report).resolve()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
