from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from dpo_utils import (
    cleanup_generation,
    directory_fingerprint,
    extract_refinement_fields,
    fit_refinement_prompt_to_budget,
    file_sha256,
    question_key,
    read_jsonl,
    row_key,
    write_json,
    write_jsonl,
)


def _model_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFT or DPO adapter inference with the shared q+d prompt.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rows", type=int, default=-1)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gemma 4-bit inference")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.input)
    if args.max_rows >= 0:
        rows = rows[: args.max_rows]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    model.eval()
    device = _model_device(model)

    torch.cuda.reset_peak_memory_stats()
    outputs: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for index, row in enumerate(tqdm(rows, desc=f"Inference: {args.method}")):
        question, initial_draft, _ = extract_refinement_fields(row)
        messages, model_question, model_draft, prompt_truncated = fit_refinement_prompt_to_budget(
            tokenizer, question, initial_draft, args.max_source_len
        )
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] > args.max_source_len:
            raise RuntimeError("Prompt budget enforcement failed")
        generation_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if args.temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**generation_kwargs)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0
        new_tokens = generated[0, input_ids.shape[1] :]
        raw_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        response = cleanup_generation(raw_response)

        record = dict(row)
        record.update(
            {
                "source_row_id": row_key(row, index),
                "question_id": question_key(row),
                "method": args.method,
                "refiner_response": response,
                "response": response,
                "refiner_response_raw": raw_response,
                "input_tokens": int(input_ids.shape[1]),
                "prompt_truncated_for_budget": prompt_truncated,
                "model_question": model_question,
                "model_initial_draft": model_draft,
                "output_tokens": int(new_tokens.shape[0]),
                "latency_ms": latency_ms,
                "adapter_dir": args.adapter_dir,
                "base_model": args.base_model,
                "decoding": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                },
            }
        )
        outputs.append(record)

    elapsed_seconds = time.perf_counter() - total_start
    write_jsonl(outputs, args.output)
    manifest = {
        "input": args.input,
        "input_sha256": file_sha256(args.input),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
        "method": args.method,
        "rows": len(outputs),
        "base_model": args.base_model,
        "adapter": directory_fingerprint(args.adapter_dir),
        "prompt_contract": "fixed instruction + question + initial draft only",
        "decoding": {
            "max_source_len": args.max_source_len,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "seed": args.seed,
        },
        "elapsed_seconds": elapsed_seconds,
        "mean_latency_ms": sum(item["latency_ms"] for item in outputs) / max(1, len(outputs)),
        "prompts_truncated_for_budget": sum(bool(item["prompt_truncated_for_budget"]) for item in outputs),
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }
    write_json(manifest, str(Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
