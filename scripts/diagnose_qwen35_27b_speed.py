#!/usr/bin/env python3
"""Bounded speed-only diagnostic for the frozen first local-judge example."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_local_qwen_judge_comparison import PAIRED_PROMPT, SYSTEM_PROMPT  # noqa: E402


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def kernel_status() -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling

    status = {}
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn"):
        function = getattr(modeling, name)
        cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
        implementation = cells.get("implementation")
        status[name] = {
            "accelerated": bool(cells.get("is_new_implementation") and cells["is_new_implementation"].cell_contents),
            "implementation_module": (
                getattr(implementation.cell_contents, "__module__", None) if implementation is not None else None
            ),
        }
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    row = json.loads(args.selection.read_text().splitlines()[0])
    candidate_label = row["ab_candidate_label"]
    clean_label = "B" if candidate_label == "A" else "A"
    a = row["candidate_response"] if candidate_label == "A" else row["clean_response"]
    b = row["clean_response"] if candidate_label == "A" else row["candidate_response"]

    load_started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, device_map={"": 0},
    ).eval()
    load_seconds = time.monotonic() - load_started
    prompt = PAIRED_PROMPT.format(question=row["question"], a=a, b=b)
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180.0)
    generation = {
        **encoded,
        "streamer": streamer,
        "do_sample": False,
        "use_cache": True,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    result: dict = {}

    def generate() -> None:
        try:
            result["tokens"] = model.generate(**generation)
        except BaseException as error:  # propagated after the streaming loop
            result["error"] = repr(error)

    started = time.monotonic()
    worker = threading.Thread(target=generate, daemon=True)
    worker.start()
    chunks, first_token_seconds = [], None
    for chunk in streamer:
        if first_token_seconds is None and chunk:
            first_token_seconds = time.monotonic() - started
        chunks.append(chunk)
    worker.join()
    elapsed = time.monotonic() - started
    if "error" in result:
        raise RuntimeError(result["error"])
    tokens = result["tokens"][:, encoded["input_ids"].shape[-1]:]
    eos_reached = bool(tokenizer.eos_token_id is not None and (tokens == tokenizer.eos_token_id).any().item())
    generated_tokens = int(tokens.shape[-1])
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    metrics = {
        "status": "complete_speed_diagnostic_not_quality_evaluation",
        "canonical_id": row["canonical_id"],
        "clean_label": clean_label,
        "candidate_label": candidate_label,
        "input_tokens": int(encoded["attention_mask"].sum().item()),
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "eos_reached": eos_reached,
        "model_load_seconds": load_seconds,
        "first_nonempty_stream_chunk_seconds": first_token_seconds,
        "generation_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed,
        "enable_thinking": False,
        "use_cache": True,
        "parameter_devices": devices,
        "hf_device_map": getattr(model, "hf_device_map", None),
        "cpu_parameter_count": sum(p.numel() for p in model.parameters() if p.device.type == "cpu"),
        "kernel_status": kernel_status(),
        "raw_output": "".join(chunks),
    }
    atomic_json(args.output, metrics)
    print(json.dumps({key: value for key, value in metrics.items() if key != "raw_output"}, indent=2))


if __name__ == "__main__":
    main()
