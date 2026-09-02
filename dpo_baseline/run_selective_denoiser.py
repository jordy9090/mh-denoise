from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from dpo_utils import (
    DIMS,
    cleanup_generation,
    clean_text,
    directory_fingerprint,
    extract_refinement_fields,
    file_sha256,
    question_key,
    read_jsonl,
    repeated_ngram_rate,
    row_key,
    specificity_proxy,
    write_json,
    write_jsonl,
)


def split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean_text(text)) if part.strip()]
    return parts or ([clean_text(text)] if clean_text(text) else [])


def router_text(question: str, response: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Unsafe response:\n{response.strip()}\n\n"
        "Task: identify all violated mental-health response quality dimensions."
    )


def risk_text(question: str, span: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Candidate span:\n{span.strip()}\n\n"
        "Task: predict which counseling quality dimensions this span may violate."
    )


def predict_router(model: Any, tokenizer: Any, question: str, response: str, device: str) -> List[float]:
    import torch

    encoded = tokenizer(
        router_text(question, response),
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    with torch.inference_mode():
        values = torch.sigmoid(model(**encoded).logits.float())[0].detach().cpu().tolist()
    if len(values) != len(DIMS):
        raise ValueError(f"Router returned {len(values)} labels; expected {len(DIMS)}")
    return [float(value) for value in values]


def score_spans(
    model: Any,
    tokenizer: Any,
    question: str,
    spans: Sequence[str],
    device: str,
) -> List[List[float]]:
    import torch

    if not spans:
        return []
    encoded = tokenizer(
        [risk_text(question, span) for span in spans],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=384,
    ).to(device)
    with torch.inference_mode():
        values = torch.sigmoid(model(**encoded).logits.float()).detach().cpu().tolist()
    for row in values:
        if len(row) != len(DIMS):
            raise ValueError(f"Span scorer returned {len(row)} labels; expected {len(DIMS)}")
    return [[float(value) for value in row] for row in values]


def aggregate_risk(g: Sequence[float], span_scores: Sequence[Sequence[float]]) -> Tuple[float, float]:
    if not span_scores:
        return max(g, default=0.0), float(g[DIMS.index("medical_advice")]) if g else 0.0
    med_index = DIMS.index("medical_advice")
    global_risk = max(
        float(g[index]) * float(span[index])
        for span in span_scores
        for index in range(len(DIMS))
    )
    medical_risk = max(float(g[med_index]) * float(span[med_index]) for span in span_scores)
    return global_risk, medical_risk


def mask_anchor(
    anchor: str,
    g: Sequence[float],
    span_scores: Sequence[Sequence[float]],
    *,
    t: int,
    total_steps: int,
    rho: float,
    lambda_mask: float,
    risk_threshold: float,
    mask_threshold: float,
    mask_token: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    spans = split_sentences(anchor)
    beta = t / float(total_steps)
    output: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for span, scores in zip(spans, span_scores):
        risk = max(float(g[index]) * float(scores[index]) for index in range(len(DIMS)))
        pi = max(0.0, min(1.0, rho + lambda_mask * beta * risk))
        p_mask = beta * pi
        masked = p_mask >= mask_threshold or (risk >= risk_threshold and t >= 2)
        output.append(mask_token if masked else span)
        metadata.append(
            {
                "span": span,
                "risk": risk,
                "p_mask": p_mask,
                "state": "MASK" if masked else "KEEP",
            }
        )
    if not output:
        return mask_token, []
    return " ".join(output), metadata


def build_denoiser_prompt(
    question: str,
    original_unsafe: str,
    masked_anchor: str,
    g: Sequence[float],
    *,
    t: int,
) -> str:
    aspect_text = ", ".join(f"{DIMS[index]}={float(g[index]):.2f}" for index in range(len(DIMS)))
    return (
        "Rewrite the unsafe counseling response into a safe response.\n"
        "Only write the final response.\n"
        "Do not copy blaming, diagnostic, toxic, or unsupported wording from the unsafe response.\n\n"
        f"Aspect scores:\n{aspect_text}\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Unsafe response to fix:\n{original_unsafe.strip()}\n\n"
        f"Draft to revise:\n{masked_anchor.strip()}\n\n"
        f"Corruption: sft_output_mask, t={t}\n\n"
        "Safe response:\n"
    )


def quality_checks(
    anchor: str,
    candidate: str,
    *,
    min_words: int,
    min_length_ratio: float,
    max_length_ratio: float,
    max_repeated_ngram_rate: float,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    words = candidate.split()
    anchor_words = max(1, len(anchor.split()))
    ratio = len(words) / anchor_words
    lower = candidate.lower()
    if not candidate:
        reasons.append("empty")
    if len(words) < min_words:
        reasons.append("too_short")
    if ratio < min_length_ratio:
        reasons.append("length_ratio_too_small")
    if ratio > max_length_ratio:
        reasons.append("length_ratio_too_large")
    if "<mask>" in lower or "[mask]" in lower:
        reasons.append("mask_leak")
    if any(marker in lower for marker in ("safe response:", "unsafe response:", "aspect scores:")):
        reasons.append("prompt_leak")
    if repeated_ngram_rate(candidate, n=4) > max_repeated_ngram_rate:
        reasons.append("degenerate_repetition")
    return not reasons, reasons


def _model_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the existing residual-risk denoiser after a DPO refiner without modifying legacy scripts."
    )
    parser.add_argument("--input", required=True, help="DPO inference JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--denoiser_adapter_dir", required=True)
    parser.add_argument("--router_dir", required=True)
    parser.add_argument("--risk_scorer_dir", required=True)
    parser.add_argument("--anchor_field", default="refiner_response")
    parser.add_argument("--method", default="DPO+selective_denoiser")
    parser.add_argument("--classifier_device", default="cuda")
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument("--t", type=int, default=4)
    parser.add_argument("--rho", type=float, default=0.15)
    parser.add_argument("--lambda_mask", type=float, default=0.75)
    parser.add_argument("--mask_risk_threshold", type=float, default=0.35)
    parser.add_argument("--mask_probability_threshold", type=float, default=0.35)
    parser.add_argument("--gate_global_threshold", type=float, default=0.50)
    parser.add_argument("--gate_medical_threshold", type=float, default=0.35)
    parser.add_argument("--risk_tolerance", type=float, default=0.0)
    parser.add_argument("--medical_risk_tolerance", type=float, default=0.0)
    parser.add_argument("--specificity_ratio", type=float, default=0.85)
    parser.add_argument("--min_words", type=int, default=20)
    parser.add_argument("--min_length_ratio", type=float, default=0.60)
    parser.add_argument("--max_length_ratio", type=float, default=1.80)
    parser.add_argument("--max_repeated_ngram_rate", type=float, default=0.20)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--max_rows", type=int, default=-1)
    args = parser.parse_args()

    if args.T <= 0 or not 0 <= args.t <= args.T:
        raise ValueError("Require 0 <= t <= T and T > 0")

    import torch
    from peft import PeftModel
    from tqdm import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Gemma 4-bit inference")
    rows = read_jsonl(args.input)
    if args.max_rows >= 0:
        rows = rows[: args.max_rows]

    classifier_device = args.classifier_device
    if classifier_device == "cuda" and not torch.cuda.is_available():
        classifier_device = "cpu"
    router_tokenizer = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir).to(classifier_device).eval()
    risk_tokenizer = AutoTokenizer.from_pretrained(args.risk_scorer_dir)
    risk_model = AutoModelForSequenceClassification.from_pretrained(args.risk_scorer_dir).to(classifier_device).eval()

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
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    denoiser = PeftModel.from_pretrained(base, args.denoiser_adapter_dir, is_trainable=False)
    denoiser.eval()
    denoiser_device = _model_device(denoiser)

    outputs: List[Dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()

    for index, row in enumerate(tqdm(rows, desc=args.method)):
        question, original_unsafe, _ = extract_refinement_fields(row)
        anchor = clean_text(row.get(args.anchor_field))
        if not anchor:
            raise ValueError(f"Missing anchor field {args.anchor_field!r} in row {row_key(row, index)}")

        gate_start = time.perf_counter()
        anchor_g = predict_router(router, router_tokenizer, question, anchor, classifier_device)
        anchor_spans = split_sentences(anchor)
        anchor_span_scores = score_spans(
            risk_model,
            risk_tokenizer,
            question,
            anchor_spans,
            classifier_device,
        )
        anchor_global, anchor_medical = aggregate_risk(anchor_g, anchor_span_scores)
        anchor_quality_ok, anchor_quality_failures = quality_checks(
            anchor,
            anchor,
            min_words=args.min_words,
            min_length_ratio=0.0,
            max_length_ratio=10.0,
            max_repeated_ngram_rate=args.max_repeated_ngram_rate,
        )
        invoked = (
            anchor_global >= args.gate_global_threshold
            or anchor_medical >= args.gate_medical_threshold
            or not anchor_quality_ok
        )
        gate_latency_ms = (time.perf_counter() - gate_start) * 1000.0

        candidate = ""
        candidate_raw = ""
        masked_anchor = ""
        mask_metadata: List[Dict[str, Any]] = []
        candidate_global = None
        candidate_medical = None
        candidate_specificity = None
        anchor_specificity = specificity_proxy(question, anchor)
        accepted = False
        rejection_reasons: List[str] = []
        denoiser_latency_ms = 0.0

        if invoked:
            masked_anchor, mask_metadata = mask_anchor(
                anchor,
                anchor_g,
                anchor_span_scores,
                t=args.t,
                total_steps=args.T,
                rho=args.rho,
                lambda_mask=args.lambda_mask,
                risk_threshold=args.mask_risk_threshold,
                mask_threshold=args.mask_probability_threshold,
                mask_token="<MASK>",
            )
            prompt = build_denoiser_prompt(
                question,
                original_unsafe,
                masked_anchor,
                anchor_g,
                t=args.t,
            )
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_len,
            ).to(denoiser_device)
            torch.cuda.synchronize()
            generation_start = time.perf_counter()
            with torch.inference_mode():
                generated = denoiser.generate(
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
            denoiser_latency_ms = (time.perf_counter() - generation_start) * 1000.0
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            candidate_raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            candidate = cleanup_generation(candidate_raw)

            candidate_g = predict_router(router, router_tokenizer, question, candidate, classifier_device)
            candidate_spans = split_sentences(candidate)
            candidate_span_scores = score_spans(
                risk_model,
                risk_tokenizer,
                question,
                candidate_spans,
                classifier_device,
            )
            candidate_global, candidate_medical = aggregate_risk(candidate_g, candidate_span_scores)
            candidate_specificity = specificity_proxy(question, candidate)
            quality_ok, quality_failures = quality_checks(
                anchor,
                candidate,
                min_words=args.min_words,
                min_length_ratio=args.min_length_ratio,
                max_length_ratio=args.max_length_ratio,
                max_repeated_ngram_rate=args.max_repeated_ngram_rate,
            )
            if candidate_global > anchor_global + args.risk_tolerance:
                rejection_reasons.append("global_risk_increased")
            if candidate_medical > anchor_medical + args.medical_risk_tolerance:
                rejection_reasons.append("medical_risk_increased")
            required_specificity = args.specificity_ratio * anchor_specificity
            if candidate_specificity < required_specificity:
                rejection_reasons.append("specificity_proxy_decreased")
            if not quality_ok:
                rejection_reasons.extend(quality_failures)
            accepted = not rejection_reasons

        final_response = candidate if invoked and accepted else anchor
        anchor_output_tokens = len(tokenizer(anchor, add_special_tokens=False)["input_ids"])
        candidate_output_tokens = (
            len(tokenizer(candidate, add_special_tokens=False)["input_ids"]) if candidate else 0
        )
        final_output_tokens = len(tokenizer(final_response, add_special_tokens=False)["input_ids"])
        output = dict(row)
        output.update(
            {
                "method": args.method,
                "question_id": question_key(row),
                "source_row_id": row_key(row, index),
                "anchor_method": row.get("method"),
                "anchor_response": anchor,
                "denoiser_invoked": invoked,
                "denoiser_accepted": accepted,
                "denoiser_candidate": candidate,
                "denoiser_candidate_raw": candidate_raw,
                "final_response": final_response,
                "response": final_response,
                "anchor_output_tokens": anchor_output_tokens,
                "denoiser_candidate_tokens": candidate_output_tokens,
                "output_tokens": final_output_tokens,
                "anchor_risk": anchor_global,
                "anchor_medical_risk": anchor_medical,
                "candidate_risk": candidate_global,
                "candidate_medical_risk": candidate_medical,
                "anchor_specificity_proxy": anchor_specificity,
                "candidate_specificity_proxy": candidate_specificity,
                "anchor_quality_failures": anchor_quality_failures,
                "acceptance_rejection_reasons": sorted(set(rejection_reasons)),
                "masked_anchor": masked_anchor,
                "masked_anchor_spans": mask_metadata,
                "gate_latency_ms": gate_latency_ms,
                "denoiser_latency_ms": denoiser_latency_ms,
                "latency_ms": float(row.get("latency_ms", 0.0) or 0.0) + gate_latency_ms + denoiser_latency_ms,
                "selective_thresholds": {
                    "gate_global_threshold": args.gate_global_threshold,
                    "gate_medical_threshold": args.gate_medical_threshold,
                    "risk_tolerance": args.risk_tolerance,
                    "medical_risk_tolerance": args.medical_risk_tolerance,
                    "specificity_ratio": args.specificity_ratio,
                    "mask_risk_threshold": args.mask_risk_threshold,
                    "mask_probability_threshold": args.mask_probability_threshold,
                },
            }
        )
        outputs.append(output)

    elapsed_seconds = time.perf_counter() - total_start
    write_jsonl(outputs, args.output)
    invoked_count = sum(bool(row["denoiser_invoked"]) for row in outputs)
    accepted_count = sum(bool(row["denoiser_accepted"]) for row in outputs)
    manifest = {
        "input": args.input,
        "input_sha256": file_sha256(args.input),
        "output": args.output,
        "output_sha256": file_sha256(args.output),
        "method": args.method,
        "rows": len(outputs),
        "invoked": invoked_count,
        "invocation_rate": invoked_count / max(1, len(outputs)),
        "accepted": accepted_count,
        "acceptance_rate_all": accepted_count / max(1, len(outputs)),
        "acceptance_rate_invoked": accepted_count / max(1, invoked_count),
        "base_model": args.base_model,
        "denoiser_adapter": directory_fingerprint(args.denoiser_adapter_dir),
        "router_dir": args.router_dir,
        "risk_scorer_dir": args.risk_scorer_dir,
        "thresholds": outputs[0]["selective_thresholds"] if outputs else {},
        "elapsed_seconds": elapsed_seconds,
        "mean_end_to_end_latency_ms": sum(float(row["latency_ms"]) for row in outputs) / max(1, len(outputs)),
        "peak_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "implementation_note": (
            "The wrapper preserves the original unsafe response as context and uses the DPO response as the masked "
            "residual-repair draft. Gate and acceptance thresholds must be selected on exp295 validation and frozen "
            "before test inference."
        ),
    }
    write_json(manifest, str(Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
