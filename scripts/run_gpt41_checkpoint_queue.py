#!/usr/bin/env python3
"""Run a frozen GPT-4.1 JSON request queue with resumable checkpoints."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


INPUT_USD_PER_MILLION = 2.00
OUTPUT_USD_PER_MILLION = 8.00
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


def load_repo_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_sha256(row: dict[str, Any]) -> str:
    selected = {
        key: row[key]
        for key in (
            "request_id",
            "model",
            "temperature",
            "max_tokens",
            "response_format",
            "parser",
            "system_prompt",
            "user_prompt",
            "metadata",
        )
    }
    material = json.dumps(selected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def parse_payload(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("API response is not a JSON object")
    return value


def validate_payload(parser: str, payload: dict[str, Any]) -> dict[str, Any]:
    if parser == "pointwise_expert_rubric":
        scores = {}
        for axis in ("overall_quality", "empathy", "specificity"):
            value = payload.get(axis)
            if isinstance(value, bool):
                raise ValueError(f"invalid boolean score for {axis}")
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid score for {axis}") from exc
            if not 1 <= numeric <= 5:
                raise ValueError(f"out-of-range score for {axis}: {numeric}")
            scores[axis] = numeric
        medical_advice = str(payload.get("medical_advice") or "").strip().lower()
        if medical_advice not in {"yes", "no", "unsure"}:
            raise ValueError(f"invalid medical_advice: {medical_advice!r}")
        evidence = str(payload.get("medical_advice_evidence") or "").strip()
        medical_rationale = str(payload.get("medical_advice_rationale") or "").strip()
        if medical_advice in {"yes", "unsure"} and not evidence:
            raise ValueError(f"medical_advice={medical_advice} requires evidence")
        if not medical_rationale:
            raise ValueError("medical_advice_rationale is required")
        return {
            "scores": scores,
            "medical_advice": medical_advice,
            "rationale": str(payload.get("rationale") or "").strip(),
            "medical_advice_evidence": evidence,
            "medical_advice_rationale": medical_rationale,
        }
    if parser == "pointwise_four_axis":
        scores = {}
        for axis in ("overall_quality", "empathy", "specificity", "medical_advice"):
            value = payload.get(axis)
            if isinstance(value, bool):
                raise ValueError(f"invalid boolean score for {axis}")
            try:
                numeric = int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid score for {axis}") from exc
            if not 1 <= numeric <= 5:
                raise ValueError(f"out-of-range score for {axis}: {numeric}")
            scores[axis] = numeric
        return {"scores": scores, "rationale": str(payload.get("rationale") or "").strip()}
    if parser == "blind_selector":
        choice = str(payload.get("choice") or "").strip().lower()
        choice = {"response a": "a", "response b": "b"}.get(choice, choice)
        if choice not in {"a", "b", "tie"}:
            raise ValueError(f"invalid selector choice: {choice!r}")
        return {"choice": choice.upper() if choice in {"a", "b"} else "tie", "rationale": str(payload.get("rationale") or "").strip()}
    if parser == "pairwise_four_axis":
        normalized = {}
        for side in ("response_A", "response_B"):
            item = payload.get(side)
            if not isinstance(item, dict):
                raise ValueError(f"missing {side}")
            scores = {}
            for axis in ("overall_quality", "empathy", "specificity", "medical_advice"):
                value = item.get(axis)
                if isinstance(value, bool):
                    raise ValueError(f"invalid boolean score for {side}.{axis}")
                try:
                    numeric = int(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid score for {side}.{axis}") from exc
                if not 1 <= numeric <= 5:
                    raise ValueError(f"out-of-range score for {side}.{axis}: {numeric}")
                scores[axis] = numeric
            normalized[side] = {**item, **scores}
        preference = str(payload.get("preference") or "").strip().lower()
        preference = {"response a": "A", "response b": "B", "a": "A", "b": "B", "tie": "tie"}.get(preference)
        if preference is None:
            raise ValueError("invalid pairwise preference")
        normalized["preference"] = preference
        normalized["rationale"] = str(payload.get("rationale") or "").strip()
        return normalized
    raise ValueError(f"Unknown parser: {parser}")


def retry_after_seconds(exc: Exception) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
    if not headers:
        return 0.0
    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            parsed = parsedate_to_datetime(str(value))
            return max(0.0, parsed.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def is_transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status in TRANSIENT_STATUS_CODES or type(exc).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


async def call_one(client: AsyncOpenAI, row: dict[str, Any], max_retries: int) -> tuple[dict[str, Any], bool]:
    retry_events = []
    rate_limited = False
    started_all = time.monotonic()
    for attempt in range(1, max_retries + 1):
        started = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=row["model"],
                temperature=row["temperature"],
                max_tokens=row["max_tokens"],
                response_format=row["response_format"],
                messages=[
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ],
            )
            raw = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason
            if finish_reason != "stop":
                raise RuntimeError(f"non-stop finish_reason: {finish_reason}")
            payload = parse_payload(raw)
            normalized = validate_payload(row["parser"], payload)
            dumped = response.model_dump(mode="json")
            usage = dumped.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            result = {
                **row,
                "request_sha256": request_sha256(row),
                "ok": True,
                "attempt": attempt,
                "retry_events": retry_events,
                "raw_json_text": raw,
                "parsed_payload": payload,
                "normalized": normalized,
                "finish_reason": finish_reason,
                "returned_model": dumped.get("model"),
                "system_fingerprint": dumped.get("system_fingerprint"),
                "usage": usage,
                "estimated_cost_usd": round(
                    input_tokens * INPUT_USD_PER_MILLION / 1_000_000
                    + output_tokens * OUTPUT_USD_PER_MILLION / 1_000_000,
                    8,
                ),
                "latency_seconds": round(time.monotonic() - started, 6),
                "total_elapsed_seconds": round(time.monotonic() - started_all, 6),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "api_response": dumped,
            }
            return result, rate_limited
        except Exception as exc:
            transient = is_transient(exc)
            status = getattr(exc, "status_code", None)
            if status == 429 or type(exc).__name__ == "RateLimitError":
                rate_limited = True
            event = {
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "status_code": status,
                "transient": transient,
            }
            if not transient or attempt >= max_retries:
                return {
                    **row,
                    "request_sha256": request_sha256(row),
                    "ok": False,
                    "attempt": attempt,
                    "retry_events": retry_events + [event],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "total_elapsed_seconds": round(time.monotonic() - started_all, 6),
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }, rate_limited
            delay = max(retry_after_seconds(exc), min(2 ** (attempt - 1), 30.0))
            delay += random.uniform(0.0, min(1.0, delay * 0.1))
            event["sleep_seconds"] = round(delay, 3)
            retry_events.append(event)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def aggregate(rows: list[dict[str, Any]], request_ids: set[str]) -> dict[str, Any]:
    latest = {}
    for row in rows:
        if row.get("request_id") in request_ids:
            latest[row["request_id"]] = row
    ok = [row for row in latest.values() if row.get("ok") is True]
    failed = [row for row in latest.values() if row.get("ok") is not True]
    input_tokens = sum(int((row.get("usage") or {}).get("prompt_tokens") or 0) for row in ok)
    output_tokens = sum(int((row.get("usage") or {}).get("completion_tokens") or 0) for row in ok)
    return {
        "completed": len(ok),
        "failed_latest": len(failed),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": round(
            input_tokens * INPUT_USD_PER_MILLION / 1_000_000
            + output_tokens * OUTPUT_USD_PER_MILLION / 1_000_000,
            8,
        ),
        "returned_models": sorted({str(row.get("returned_model")) for row in ok}),
        "system_fingerprints": sorted({str(row.get("system_fingerprint")) for row in ok}),
    }


async def run_phase(
    client: AsyncOpenAI,
    rows: list[dict[str, Any]],
    output: Path,
    concurrency: int,
    max_retries: int,
    write_lock: asyncio.Lock,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_seen = False

    async def wrapped(row: dict[str, Any]):
        nonlocal rate_limit_seen
        async with semaphore:
            result, limited = await call_one(client, row, max_retries)
        rate_limit_seen = rate_limit_seen or limited
        async with write_lock:
            append_jsonl(output, result)
        print(
            json.dumps(
                {
                    "queue": row["queue"],
                    "request_id": row["request_id"],
                    "ok": result["ok"],
                    "attempt": result["attempt"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    started = time.monotonic()
    results = await asyncio.gather(*(wrapped(row) for row in rows))
    return {
        "requests": len(rows),
        "ok": sum(row.get("ok") is True for row in results),
        "failed": sum(row.get("ok") is not True for row in results),
        "elapsed_seconds": time.monotonic() - started,
        "rate_limit_seen": rate_limit_seen,
    }


async def async_main(args) -> int:
    load_repo_env(Path("/home/user/hsoh/mh-denoise/.env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is unavailable")
    requests = read_jsonl(args.requests)
    if not requests:
        raise RuntimeError("Request queue is empty")
    ids = [row["request_id"] for row in requests]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Request queue contains duplicate request IDs")
    for row in requests:
        if row["model"] != "gpt-4.1-2025-04-14":
            raise RuntimeError(f"Non-GPT-4.1 request refused: {row['model']}")
        if row["queue"] != args.queue_name:
            raise RuntimeError(f"Queue name mismatch: {row['queue']} != {args.queue_name}")
    existing_rows = read_jsonl(args.output)
    latest = {}
    for row in existing_rows:
        latest[row.get("request_id")] = row
    complete = {
        rid
        for rid, row in latest.items()
        if rid in set(ids)
        and row.get("ok") is True
        and row.get("request_sha256") == request_sha256(next(item for item in requests if item["request_id"] == rid))
    }
    pending = [row for row in requests if row["request_id"] not in complete]
    print(
        json.dumps(
            {
                "queue": args.queue_name,
                "total": len(requests),
                "already_complete": len(complete),
                "pending": len(pending),
                "initial_concurrency": args.initial_concurrency,
                "max_concurrency": args.max_concurrency,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.validate_only:
        return 0

    lock_path = args.output.with_suffix(args.output.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"Queue already running: {args.queue_name}") from exc

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        timeout=args.timeout,
        max_retries=0,
    )
    write_lock = asyncio.Lock()
    run_started = time.monotonic()
    first = pending[: min(10, len(pending))]
    rest = pending[len(first) :]
    first_stats = None
    selected_concurrency = args.initial_concurrency
    if first:
        first_stats = await run_phase(
            client,
            first,
            args.output,
            args.initial_concurrency,
            args.max_retries,
            write_lock,
        )
        elapsed = max(first_stats["elapsed_seconds"], 1e-9)
        throughput = first_stats["ok"] / elapsed * 60
        selected_concurrency = (
            args.max_concurrency
            if not first_stats["rate_limit_seen"] and first_stats["failed"] == 0
            else args.initial_concurrency
        )
        remaining = len(rest)
        eta_minutes = remaining / throughput if throughput > 0 else None
        milestone = {
            "queue": args.queue_name,
            "milestone_completed": first_stats["ok"],
            "measured_throughput_requests_per_minute": round(throughput, 3),
            "remaining_unique_requests": remaining,
            "eta_minutes": round(eta_minutes, 3) if eta_minutes is not None else None,
            "rate_limit_seen": first_stats["rate_limit_seen"],
            "next_concurrency": selected_concurrency,
        }
        atomic_json(args.output.with_suffix(".first10.json"), milestone)
        print(json.dumps({"first10_report": milestone}, sort_keys=True), flush=True)
    if rest:
        await run_phase(
            client,
            rest,
            args.output,
            selected_concurrency,
            args.max_retries,
            write_lock,
        )
    await client.close()

    all_results = read_jsonl(args.output)
    summary = aggregate(all_results, set(ids))
    summary.update(
        {
            "queue": args.queue_name,
            "status": "complete" if summary["completed"] == len(requests) else "partial",
            "requests": len(requests),
            "remaining": len(requests) - summary["completed"],
            "initial_concurrency": args.initial_concurrency,
            "final_concurrency": selected_concurrency,
            "elapsed_seconds_this_run": round(time.monotonic() - run_started, 3),
            "pricing_usd_per_million_tokens": {
                "input": INPUT_USD_PER_MILLION,
                "output": OUTPUT_USD_PER_MILLION,
            },
            "request_file": str(args.requests.resolve()),
            "result_file": str(args.output.resolve()),
            "first10": first_stats,
        }
    )
    atomic_json(args.output.with_suffix(".manifest.json"), summary)
    print(json.dumps({"final": summary}, sort_keys=True), flush=True)
    return 0 if summary["status"] == "complete" else 2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue-name", required=True)
    parser.add_argument("--initial-concurrency", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(async_main(parse_args())))


if __name__ == "__main__":
    main()
