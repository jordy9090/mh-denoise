#!/usr/bin/env python3
"""Generate exp295 refinements under the fixed q+d prompt and record efficiency."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from prepare_preferences import build_prompt_messages, get_chosen, get_question, get_rejected


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_id(row: Mapping[str, Any], index: int) -> str:
    for name in ("source_id", "id", "example_id", "pair_id"):
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    payload = f"{index}\0{get_question(row)}\0{get_rejected(row)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def cleanup(text: str) -> str:
    text = str(text or "").strip()
    prefixes = (
        "Final response:",
        "Safe response:",
        "Refined response:",
        "Assistant:",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if text.casefold().startswith(prefix.casefold()):
                text = text[len(prefix) :].strip()
                changed = True
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip('"').strip()


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(x) for x in values)
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original exp295 split JSONL.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter_dir", type=Path, required=True)
    parser.add_argument("--system_name", required=True)
    parser.add_argument("--response_field", default="response")
    parser.add_argument("--expected_rows", type=int, default=-1)
    parser.add_argument("--max_source_len", type=int, default=896)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.08)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite to replace it")
    if not args.adapter_dir.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {args.adapter_dir}")

    rows = read_jsonl(args.input)
    if args.expected_rows >= 0 and len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {len(rows)} in {args.input}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 4-bit Gemma generation")
    set_seed(args.seed)

    tokenizer_source = (
        str(args.adapter_dir)
        if (args.adapter_dir / "tokenizer_config.json").exists()
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir), is_trainable=False)
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device

    torch.cuda.reset_peak_memory_stats()
    latencies: list[float] = []
    generated_token_counts: list[int] = []
    output_rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for index, row in enumerate(rows):
        question = get_question(row)
        initial_draft = get_rejected(row)
        messages = build_prompt_messages(question, initial_draft)
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        if encoded["input_ids"].shape[-1] > args.max_source_len:
            # Re-tokenize the rendered chat with explicit truncation to make the policy transparent.
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_len,
                add_special_tokens=False,
            ).to(device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = generated[0, encoded["input_ids"].shape[-1] :]
        response = cleanup(tokenizer.decode(new_tokens, skip_special_tokens=True))
        latencies.append(elapsed)
        generated_token_counts.append(int(new_tokens.numel()))

        output = dict(row)
        output["source_id"] = source_id(row, index)
        output["question"] = question
        output["initial_draft"] = initial_draft
        output["safe_response"] = get_chosen(row)
        output[args.response_field] = response
        output["system"] = args.system_name
        output["adapter_dir"] = str(args.adapter_dir)
        output["generation_seed"] = args.seed
        output["generation_latency_seconds"] = elapsed
        output["generated_tokens"] = int(new_tokens.numel())
        output_rows.append(output)

    total_elapsed = time.perf_counter() - total_start
    write_jsonl(output_rows, args.output)

    max_allocated = int(torch.cuda.max_memory_allocated())
    max_reserved = int(torch.cuda.max_memory_reserved())
    metrics = {
        "system": args.system_name,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "base_model": args.base_model,
        "adapter_dir": str(args.adapter_dir),
        "response_field": args.response_field,
        "prompt_contract": "x = question + potentially unsafe initial draft",
        "decoding": {
            "do_sample": False,
            "max_source_len": args.max_source_len,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
        },
        "latency_seconds": {
            "total": total_elapsed,
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": quantile(latencies, 0.95),
        },
        "generated_tokens": {
            "total": sum(generated_token_counts),
            "mean": statistics.mean(generated_token_counts) if generated_token_counts else None,
        },
        "throughput": {
            "examples_per_second": len(output_rows) / total_elapsed if total_elapsed else None,
            "tokens_per_second": sum(generated_token_counts) / sum(latencies) if sum(latencies) else None,
        },
        "peak_cuda_allocated_bytes": max_allocated,
        "peak_cuda_reserved_bytes": max_reserved,
        "peak_cuda_allocated_gib": max_allocated / (1024**3),
        "peak_cuda_reserved_gib": max_reserved / (1024**3),
        "gpu": torch.cuda.get_device_name(0),
    }
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
