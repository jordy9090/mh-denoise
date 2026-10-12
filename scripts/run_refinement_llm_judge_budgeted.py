#!/usr/bin/env python3
"""Run the frozen refinement rubric with exact request/USD caps and usage logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import run_refinement_llm_judge as rubric
from shared_api_budget import BudgetConfig, BudgetExceeded, SharedApiBudget


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(rubric.read_jsonl(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_messages(row: dict[str, Any], rubric_style: str) -> list[dict[str, str]]:
    system = (
        rubric.COUNSELBENCH_SYSTEM_PROMPT
        if rubric_style == "counselbench"
        else rubric.SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": rubric.build_prompt(row, rubric_style)},
    ]


def token_upper_bound(model: str, messages: list[dict[str, str]]) -> int:
    try:
        import tiktoken
    except ImportError:
        # A byte count is a conservative tokenizer-independent upper bound for
        # UTF-8 text, with additional room for chat framing.
        return sum(len((item["role"] + item["content"]).encode("utf-8")) for item in messages) + 512
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    content_tokens = sum(len(encoding.encode(item["role"] + item["content"])) for item in messages)
    # Conservative allowance for chat framing and any provider-side serialization.
    return content_tokens + 256


def budget_projection(
    rows: list[dict[str, Any]],
    *,
    model: str,
    rubric_style: str,
    max_tokens: int,
    max_retries: int,
    input_price: float,
    output_price: float,
) -> dict[str, Any]:
    bounds = [token_upper_bound(model, prompt_messages(row, rubric_style)) for row in rows]
    one_attempt = sum(
        (bound * input_price + max_tokens * output_price) / 1_000_000
        for bound in bounds
    )
    return {
        "unique_rows": len(rows),
        "max_retries_per_row": max_retries,
        "absolute_request_cap": len(rows) * max_retries,
        "input_token_upper_bound": {
            "min": min(bounds) if bounds else 0,
            "max": max(bounds) if bounds else 0,
            "sum_one_attempt": sum(bounds),
        },
        "max_output_tokens_per_request": max_tokens,
        "input_usd_per_million_tokens": input_price,
        "output_usd_per_million_tokens": output_price,
        "one_attempt_ceiling_usd": one_attempt,
        "absolute_three_attempt_ceiling_usd": one_attempt * max_retries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--rubric_style", choices=("json", "counselbench"), default="json")
    parser.add_argument("--input-usd-per-million", type=float, default=0.40)
    parser.add_argument("--output-usd-per-million", type=float, default=1.60)
    parser.add_argument(
        "--hard-max-usd",
        type=float,
        default=None,
        help="Optional external hard cap; may be lower than the computed all-retries ceiling",
    )
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(input_path)
    projection = budget_projection(
        rows,
        model=args.model,
        rubric_style=args.rubric_style,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        input_price=args.input_usd_per_million,
        output_price=args.output_usd_per_million,
    )
    base_manifest = {
        "status": "estimate_only" if args.estimate_only else "running",
        "input": {"path": str(input_path), "sha256": sha256(input_path), "rows": len(rows)},
        "output": str(output_path),
        "model": args.model,
        "seed": args.seed,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "max_retries": args.max_retries,
        "rubric_style": args.rubric_style,
        "rubric_sha256": hashlib.sha256(
            (rubric.SYSTEM_PROMPT + "\n" + rubric.RUBRIC_TEMPLATE).encode("utf-8")
        ).hexdigest(),
        "budget_projection": projection,
        "external_hard_max_usd": args.hard_max_usd,
        "pricing_source": "official OpenAI GPT-4.1 mini model page checked 2026-10-08",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(base_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.estimate_only:
        print(json.dumps(base_manifest, ensure_ascii=False, indent=2))
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    # Numerical slack only; the semantic cap remains the computed all-retries ceiling.
    budget_cap = math.nextafter(
        float(projection["absolute_three_attempt_ceiling_usd"]) * 1.000001 + 1e-9,
        math.inf,
    )
    if args.hard_max_usd is not None:
        if args.hard_max_usd <= 0:
            raise ValueError("--hard-max-usd must be positive")
        budget_cap = min(budget_cap, args.hard_max_usd)
    budget = SharedApiBudget(
        args.ledger,
        BudgetConfig(
            max_requests=int(projection["absolute_request_cap"]),
            max_usd=budget_cap,
            input_usd_per_million_tokens=args.input_usd_per_million,
            output_usd_per_million_tokens=args.output_usd_per_million,
        ),
    )

    done = rubric.load_done(output_path)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    random.seed(args.seed)

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    for row in rows:
        if row.get("example_id") in done:
            continue
        raw = ""
        normalized = None
        error = ""
        attempts: list[dict[str, Any]] = []
        messages = prompt_messages(row, args.rubric_style)
        input_bound = token_upper_bound(args.model, messages)
        for attempt in range(1, args.max_retries + 1):
            reservation_id = budget.reserve(
                max_input_tokens=input_bound,
                max_output_tokens=args.max_tokens,
                worker_id="frozen-revision2-test-judge",
                purpose="held_out_test_quality_evaluation",
            )
            try:
                kwargs: dict[str, Any] = {
                    "model": args.model,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "messages": messages,
                    "seed": args.seed,
                }
                response = client.chat.completions.create(**kwargs)
                usage = response.usage
                prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                budget.settle(
                    reservation_id,
                    actual_input_tokens=prompt_tokens,
                    actual_output_tokens=completion_tokens,
                )
                raw = response.choices[0].message.content or ""
                if args.rubric_style == "counselbench":
                    normalized, error = rubric.normalize_counselbench_scores(raw)
                else:
                    normalized, error = rubric.normalize_scores(rubric.extract_json_block(raw))
                attempts.append(
                    {
                        "attempt": attempt,
                        "reservation_id": reservation_id,
                        "returned_model": response.model,
                        "system_fingerprint": getattr(response, "system_fingerprint", None),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "parse_ok": normalized is not None,
                        "parse_error": error,
                    }
                )
                if normalized is not None:
                    break
            except BudgetExceeded:
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                budget.fail(reservation_id, error)
                attempts.append(
                    {
                        "attempt": attempt,
                        "reservation_id": reservation_id,
                        "parse_ok": False,
                        "request_error": error,
                    }
                )
            if attempt < args.max_retries:
                time.sleep(max(args.sleep, min(60.0, 2.0 ** (attempt - 1))))

        out = {
            **row,
            "judge_model": args.model,
            "judge_raw": raw,
            "judge_ok": normalized is not None,
            "judge_scores": None if normalized is None else normalized["scores"],
            "judge_rationale": "" if normalized is None else normalized["rationale"],
            "judge_error": (error or "failed_to_parse_judge_output") if normalized is None else "",
            "judge_attempts": attempts,
        }
        rubric.append_jsonl(out, output_path)
        if args.sleep > 0:
            time.sleep(args.sleep)

    judged = read_jsonl(output_path)
    complete_manifest = {
        **base_manifest,
        "status": "complete" if len(judged) == len(rows) else "incomplete",
        "output": {
            "path": str(output_path),
            "sha256": sha256(output_path),
            "rows": len(judged),
            "judge_ok": sum(row.get("judge_ok") is True for row in judged),
        },
        "budget": budget.status(),
    }
    manifest_path.write_text(
        json.dumps(complete_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(complete_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
