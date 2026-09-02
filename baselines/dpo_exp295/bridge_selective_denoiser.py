#!/usr/bin/env python3
"""Bridge q+d DPO outputs to the repository's existing denoiser schema.

Subcommands
-----------
prepare
    Replace ``unsafe_response`` with an upstream SFT/DPO response while preserving
    the original exp295 flawed draft. The resulting file can be passed to
    ``scripts/run_gemma_peft_real_inference.py``.
merge
    Merge the denoiser candidate with an upstream response using explicit gate
    decisions. This command refuses to label an unfiltered candidate as a
    selective result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


TRUTHY = {True, 1, "1", "true", "yes", "accept", "accepted"}
FALSY = {False, 0, "0", "false", "no", "reject", "rejected"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def row_key(row: Mapping[str, Any], index: int) -> str:
    for field in ("source_id", "base_example_id", "id", "example_id"):
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return f"row_{index:06d}"


def required_text(row: Mapping[str, Any], field: str, *, path: Path, index: int) -> str:
    value = row.get(field)
    if value is None or not str(value).strip():
        raise ValueError(f"{path}: row {index} missing non-empty field {field!r}")
    return str(value).strip()


def parse_decision(value: Any) -> bool:
    normalized = value.casefold().strip() if isinstance(value, str) else value
    if normalized in TRUTHY:
        return True
    if normalized in FALSY:
        return False
    raise ValueError(f"unrecognized accept/reject decision: {value!r}")


def command_prepare(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        upstream = required_text(row, args.upstream_response_field, path=args.input, index=index)
        prepared = dict(row)
        original = str(row.get("unsafe_response", row.get("initial_draft", ""))).strip()
        prepared["source_id"] = row_key(row, index)
        prepared["original_unsafe_response"] = original
        prepared["upstream_response"] = upstream
        prepared["upstream_system"] = args.upstream_system
        prepared["unsafe_response"] = upstream
        prepared["denoiser_bridge_contract"] = (
            "denoiser input unsafe_response is the upstream model output; "
            "original exp295 draft is preserved in original_unsafe_response"
        )
        output.append(prepared)
    write_jsonl(output, args.output)
    print(json.dumps({"command": "prepare", "rows": len(output), "output": str(args.output)}, indent=2))


def decision_map(path: Path, field: str) -> dict[str, bool]:
    rows = read_jsonl(path)
    mapping: dict[str, bool] = {}
    for index, row in enumerate(rows):
        key = row_key(row, index)
        if key in mapping:
            raise ValueError(f"duplicate decision key {key!r} in {path}")
        if field not in row:
            raise ValueError(f"{path}: row {index} missing decision field {field!r}")
        mapping[key] = parse_decision(row[field])
    return mapping


def command_merge(args: argparse.Namespace) -> None:
    upstream_rows = read_jsonl(args.upstream_input)
    candidate_rows = read_jsonl(args.denoiser_input)
    upstream = {row_key(row, index): row for index, row in enumerate(upstream_rows)}
    candidates = {row_key(row, index): row for index, row in enumerate(candidate_rows)}
    if set(upstream) != set(candidates):
        missing_candidates = sorted(set(upstream) - set(candidates))[:10]
        missing_upstream = sorted(set(candidates) - set(upstream))[:10]
        raise ValueError(
            "upstream/denoiser key mismatch: "
            f"missing_candidate={missing_candidates}, missing_upstream={missing_upstream}"
        )

    if args.decisions is None:
        raise ValueError(
            "explicit selective gate decisions are required. The committed repository exposes a denoiser candidate "
            "generator, but no complete invocation/acceptance finalizer. Supply --decisions from the exact gate used "
            "for Proposed so both systems share the same policy."
        )
    decisions = decision_map(args.decisions, args.decision_field)
    if set(decisions) != set(upstream):
        missing = sorted(set(upstream) - set(decisions))[:10]
        extra = sorted(set(decisions) - set(upstream))[:10]
        raise ValueError(f"decision key mismatch: missing={missing}, extra={extra}")

    output: list[dict[str, Any]] = []
    accepted_count = 0
    for key in upstream:
        source = upstream[key]
        candidate = candidates[key]
        upstream_text = required_text(
            source, args.upstream_response_field, path=args.upstream_input, index=0
        )
        candidate_text = required_text(
            candidate, args.denoiser_response_field, path=args.denoiser_input, index=0
        )
        accepted = decisions[key]
        accepted_count += int(accepted)
        merged = dict(source)
        merged["source_id"] = key
        merged["upstream_response"] = upstream_text
        merged["denoiser_candidate"] = candidate_text
        merged["denoiser_invoked"] = bool(candidate.get("denoiser_invoked", True))
        merged["denoiser_accepted"] = accepted
        merged[args.output_response_field] = candidate_text if accepted else upstream_text
        merged["system"] = args.output_system
        merged["selective_final"] = True
        output.append(merged)

    write_jsonl(output, args.output)
    print(
        json.dumps(
            {
                "command": "merge",
                "rows": len(output),
                "accepted": accepted_count,
                "acceptance_rate": accepted_count / max(1, len(output)),
                "output": str(args.output),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Build input for the existing denoiser script.")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--upstream_response_field", default="response")
    prepare.add_argument("--upstream_system", required=True)
    prepare.set_defaults(func=command_prepare)

    merge = sub.add_parser("merge", help="Apply explicit shared gate decisions to denoiser candidates.")
    merge.add_argument("--upstream_input", type=Path, required=True)
    merge.add_argument("--denoiser_input", type=Path, required=True)
    merge.add_argument("--decisions", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--upstream_response_field", default="response")
    merge.add_argument("--denoiser_response_field", default="peft_response")
    merge.add_argument("--decision_field", default="denoiser_accepted")
    merge.add_argument("--output_response_field", default="response")
    merge.add_argument("--output_system", required=True)
    merge.set_defaults(func=command_merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
