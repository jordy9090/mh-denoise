from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from dpo_utils import (
    DIMS,
    clean_text,
    directory_fingerprint,
    extract_refinement_fields,
    file_sha256,
    fit_refinement_prompt_to_budget,
    group_by_question,
    lexical_similarity,
    minmax,
    normalize_text,
    question_key,
    read_jsonl,
    row_key,
    stable_seed,
    truncate_completion_to_budget,
    write_json,
    write_jsonl,
)


def _router_text(question: str, response: str) -> str:
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Unsafe response:\n{response.strip()}\n\n"
        "Task: identify all violated mental-health response quality dimensions."
    )


def _model_device(model: Any):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _chat_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]], add_generation_prompt: bool) -> List[int]:
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def score_completion_logps(
    model: Any,
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    completions: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
) -> List[float]:
    import torch

    prompt_ids = _chat_ids(tokenizer, prompt, add_generation_prompt=True)
    examples: List[Dict[str, Any]] = []
    for completion in completions:
        assistant = [{"role": "assistant", "content": clean_text(completion)}]
        full_ids = _chat_ids(tokenizer, list(prompt) + assistant, add_generation_prompt=False)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            completion_ids = tokenizer(
                clean_text(completion) + (tokenizer.eos_token or ""),
                add_special_tokens=False,
            )["input_ids"]
            full_ids = prompt_ids + list(completion_ids)
        completion_start = min(len(prompt_ids), len(full_ids))
        full_ids = full_ids[:max_length]
        completion_end = len(full_ids)
        if completion_end <= completion_start:
            examples.append({"ids": full_ids, "start": completion_start, "valid": False})
        else:
            examples.append({"ids": full_ids, "start": completion_start, "valid": True})

    device = _model_device(model)
    scores: List[float] = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    for offset in range(0, len(examples), batch_size):
        chunk = examples[offset : offset + batch_size]
        max_len = max(len(item["ids"]) for item in chunk)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        completion_mask: List[List[int]] = []
        for item in chunk:
            ids = list(item["ids"])
            padding = max_len - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            mask = [0] * len(ids)
            if item["valid"]:
                for index in range(item["start"], len(ids)):
                    mask[index] = 1
            completion_mask.append(mask + [0] * padding)

        batch_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        batch_attn = torch.tensor(attention_mask, dtype=torch.long, device=device)
        batch_mask = torch.tensor(completion_mask, dtype=torch.float32, device=device)

        with torch.inference_mode():
            outputs = model(input_ids=batch_ids, attention_mask=batch_attn, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            labels = batch_ids[:, 1:]
            mask = batch_mask[:, 1:]
            selected = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            log_norm = torch.logsumexp(logits, dim=-1)
            token_logps = (selected - log_norm) * mask
            denominators = mask.sum(dim=1)
            per_example = token_logps.sum(dim=1) / denominators.clamp_min(1.0)
            per_example = torch.where(
                denominators > 0,
                per_example,
                torch.full_like(per_example, float("-inf")),
            )
        scores.extend(float(value) for value in per_example.detach().cpu().tolist())
        del batch_ids, batch_attn, batch_mask, outputs, logits, labels, mask, selected, log_norm, token_logps

    return scores


def score_router_risk(
    router: Any,
    tokenizer: Any,
    question: str,
    responses: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> List[Dict[str, Any]]:
    import torch

    results: List[Dict[str, Any]] = []
    for offset in range(0, len(responses), batch_size):
        chunk = responses[offset : offset + batch_size]
        encoded = tokenizer(
            [_router_text(question, response) for response in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(router(**encoded).logits.float()).detach().cpu().tolist()
        for values in probs:
            if len(values) != len(DIMS):
                raise ValueError(f"Router returned {len(values)} labels; expected {len(DIMS)}")
            results.append(
                {
                    "router_risk": max(float(value) for value in values),
                    "router_probs": {DIMS[index]: float(value) for index, value in enumerate(values)},
                }
            )
    return results


def sample_policy_completions(
    model: Any,
    tokenizer: Any,
    prompt: Sequence[Mapping[str, str]],
    *,
    count: int,
    base_seed: int,
    row_id: str,
    max_source_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> List[str]:
    import torch

    if count <= 0:
        return []
    device = _model_device(model)
    input_ids = _chat_ids(tokenizer, prompt, add_generation_prompt=True)
    input_ids = input_ids[:max_source_len]
    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(tensor)
    outputs: List[str] = []

    for index in range(count):
        seed = stable_seed(base_seed, row_id, "policy_sample", index)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=tensor,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        continuation = generated[0, tensor.shape[1] :]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        if text:
            outputs.append(text)
    return outputs


def build_candidate_pool(
    source_row: Mapping[str, Any],
    question_rows: Sequence[Mapping[str, Any]],
    policy_samples: Sequence[str],
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for index, row in enumerate(question_rows):
        _, unsafe, _ = extract_refinement_fields(row)
        norm = normalize_text(unsafe)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        pool.append(
            {
                "text": unsafe,
                "origin": "same_question_automated_corruption",
                "origin_row_id": row_key(row, index),
                "weak_dimension": clean_text(row.get("target_dimension")) or None,
                "automated_flaw_evidence": "generator_assigned_violation_dimension",
            }
        )

    for index, text in enumerate(policy_samples):
        norm = normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        pool.append(
            {
                "text": clean_text(text),
                "origin": "frozen_sft_policy_sample",
                "origin_row_id": None,
                "weak_dimension": None,
                "automated_flaw_evidence": "router_risk_margin",
                "policy_sample_index": index,
            }
        )
    return pool


def select_hard_candidate(
    pool: Sequence[Mapping[str, Any]],
    *,
    chosen: str,
    policy_logps: Sequence[float],
    router_scores: Sequence[Mapping[str, Any]],
    chosen_router_risk: float,
    min_policy_risk_margin: float,
    policy_weight: float,
    similarity_weight: float,
    risk_weight: float,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(pool) != len(policy_logps) or len(pool) != len(router_scores):
        raise ValueError("Candidate score arrays must have identical lengths")
    finite_logps = [value if math.isfinite(value) else -1e9 for value in policy_logps]
    normalized_logps = minmax(finite_logps)
    scored: List[Dict[str, Any]] = []

    for index, candidate in enumerate(pool):
        text = clean_text(candidate.get("text"))
        risk = float(router_scores[index]["router_risk"])
        risk_margin = risk - chosen_router_risk
        origin = clean_text(candidate.get("origin"))
        weak_corruption = origin == "same_question_automated_corruption"
        eligible = weak_corruption or risk_margin >= min_policy_risk_margin
        similarity = lexical_similarity(chosen, text)
        hard_score = (
            policy_weight * normalized_logps[index]
            + similarity_weight * similarity
            + risk_weight * risk
        )
        record = dict(candidate)
        record.update(
            {
                "policy_logp_per_token": float(policy_logps[index]),
                "policy_logp_normalized_within_pool": float(normalized_logps[index]),
                "router_risk": risk,
                "router_risk_margin_over_chosen": risk_margin,
                "router_probs": dict(router_scores[index]["router_probs"]),
                "chosen_similarity": similarity,
                "hard_score": hard_score,
                "eligible": eligible,
            }
        )
        if normalize_text(text) == normalize_text(chosen):
            record["eligible"] = False
            record["ineligible_reason"] = "matches_chosen"
        elif not eligible:
            record["ineligible_reason"] = "policy_sample_risk_margin_below_threshold"
        scored.append(record)

    eligible = [item for item in scored if item.get("eligible")]
    eligible.sort(
        key=lambda item: (
            float(item["hard_score"]),
            float(item["router_risk"]),
            float(item["policy_logp_per_token"]),
        ),
        reverse=True,
    )
    if not eligible:
        raise ValueError(
            "No eligible hard negative is available. Increase the automated corruption pool "
            "or relax the policy-sample risk margin."
        )
    return eligible[0], scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one hard DPO negative from K automated candidates for each exp295 row."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base_model", default="google/gemma-4-E4B-it")
    parser.add_argument("--sft_adapter_dir", required=True)
    parser.add_argument("--router_dir", required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--policy_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max_source_len", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--max_target_len", type=int, default=160)
    parser.add_argument("--max_length", type=int, default=672)
    parser.add_argument("--score_batch_size", type=int, default=2)
    parser.add_argument("--router_batch_size", type=int, default=16)
    parser.add_argument("--router_device", default="cuda")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--min_policy_risk_margin", type=float, default=0.05)
    parser.add_argument("--policy_weight", type=float, default=0.45)
    parser.add_argument("--similarity_weight", type=float, default=0.35)
    parser.add_argument("--risk_weight", type=float, default=0.20)
    parser.add_argument("--expected_rows", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    weight_sum = args.policy_weight + args.similarity_weight + args.risk_weight
    if not math.isclose(weight_sum, 1.0, abs_tol=1e-6):
        raise ValueError(f"Hardness weights must sum to 1.0; received {weight_sum}")

    rows = read_jsonl(args.input)
    if args.expected_rows >= 0 and len(rows) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} rows; found {len(rows)}")
    grouped = group_by_question(rows)
    row_index = {row_key(row, index): index for index, row in enumerate(rows)}

    completed: Dict[str, Dict[str, Any]] = {}
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        for record in read_jsonl(output_path):
            completed[clean_text(record.get("source_row_id"))] = record
        print(f"Resuming with {len(completed)} completed rows")

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
        raise RuntimeError("Hard-pair mining requires a CUDA GPU for the Gemma policy")
    if args.router_device == "cuda" and not torch.cuda.is_available():
        args.router_device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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
    policy = PeftModel.from_pretrained(base_model, args.sft_adapter_dir, is_trainable=False)
    policy.eval()

    router_tokenizer = AutoTokenizer.from_pretrained(args.router_dir)
    router = AutoModelForSequenceClassification.from_pretrained(args.router_dir).to(args.router_device).eval()

    output_rows: List[Dict[str, Any]] = list(completed.values())
    start_time = time.perf_counter()

    for index, source_row in enumerate(tqdm(rows, desc="Mining hard negatives")):
        rid = row_key(source_row, index)
        if rid in completed:
            continue
        question, initial_draft, chosen = extract_refinement_fields(source_row)
        prompt, model_question, model_draft, prompt_truncated = fit_refinement_prompt_to_budget(
            tokenizer, question, initial_draft, args.max_source_len
        )
        policy_samples = sample_policy_completions(
            policy,
            tokenizer,
            prompt,
            count=args.policy_samples,
            base_seed=args.seed,
            row_id=rid,
            max_source_len=args.max_source_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        full_pool = build_candidate_pool(source_row, grouped[question_key(source_row)], policy_samples)
        if len(full_pool) < args.k:
            raise ValueError(f"Candidate pool for {rid} has {len(full_pool)} items; required K={args.k}")

        # K denotes the number of automated rejected candidates examined for this prompt.
        # The original input draft is always retained as one candidate; K-1 alternatives
        # are sampled deterministically from the remaining same-question corruptions.
        source_norm = normalize_text(initial_draft)
        source_candidates = [item for item in full_pool if normalize_text(item.get("text")) == source_norm]
        if not source_candidates:
            raise ValueError(f"Original unsafe draft is missing from the candidate pool for {rid}")
        alternatives = [item for item in full_pool if normalize_text(item.get("text")) != source_norm]
        candidate_rng = random.Random(stable_seed(args.seed, rid, "candidate_budget"))
        candidate_rng.shuffle(alternatives)
        pool = [source_candidates[0], *alternatives[: args.k - 1]]
        if len(pool) != args.k:
            raise ValueError(f"Could not form an exact K={args.k} candidate budget for {rid}")

        chosen_for_scoring, chosen_truncated = truncate_completion_to_budget(
            tokenizer, prompt, chosen, args.max_target_len
        )
        budgeted_pool: List[Dict[str, Any]] = []
        for item in pool:
            raw_text = clean_text(item["text"])
            fitted_text, text_truncated = truncate_completion_to_budget(
                tokenizer, prompt, raw_text, args.max_target_len
            )
            candidate = dict(item)
            candidate["text"] = fitted_text
            candidate["completion_truncated_for_budget"] = text_truncated
            budgeted_pool.append(candidate)
        pool = budgeted_pool
        candidate_texts = [clean_text(item["text"]) for item in pool]
        policy_logps = score_completion_logps(
            policy,
            tokenizer,
            prompt,
            candidate_texts,
            max_length=args.max_length,
            batch_size=args.score_batch_size,
        )
        router_values = score_router_risk(
            router,
            router_tokenizer,
            model_question,
            [chosen_for_scoring] + candidate_texts,
            batch_size=args.router_batch_size,
            device=args.router_device,
        )
        chosen_risk = float(router_values[0]["router_risk"])
        selected, scored_candidates = select_hard_candidate(
            pool,
            chosen=chosen_for_scoring,
            policy_logps=policy_logps,
            router_scores=router_values[1:],
            chosen_router_risk=chosen_risk,
            min_policy_risk_margin=args.min_policy_risk_margin,
            policy_weight=args.policy_weight,
            similarity_weight=args.similarity_weight,
            risk_weight=args.risk_weight,
        )

        record = {
            "source_row_id": rid,
            "question_id": question_key(source_row),
            "question": question,
            "initial_draft": initial_draft,
            "model_question": model_question,
            "model_initial_draft": model_draft,
            "prompt_truncated_for_budget": prompt_truncated,
            "chosen": chosen,
            "model_chosen": chosen_for_scoring,
            "chosen_truncated_for_budget": chosen_truncated,
            "chosen_router_risk": chosen_risk,
            "candidate_pool_size": len(full_pool),
            "candidate_budget_k": args.k,
            "candidate_budget": scored_candidates,
            "hard_rejected": selected,
            "hard_rejected_matches_input": normalize_text(selected.get("text")) == normalize_text(pool[0].get("text")),
            "selection_version": "exp295_hard_k4_v1",
            "selection_seed": args.seed,
            "policy_samples": args.policy_samples,
            "hardness_weights": {
                "policy_logp": args.policy_weight,
                "chosen_similarity": args.similarity_weight,
                "router_risk": args.risk_weight,
            },
            "counselor_annotation_used": False,
        }
        output_rows.append(record)
        output_rows.sort(key=lambda item: row_index.get(clean_text(item.get("source_row_id")), 10**12))
        write_jsonl(output_rows, args.output)

    elapsed_seconds = time.perf_counter() - start_time
    manifest = {
        "source_path": args.input,
        "source_sha256": file_sha256(args.input),
        "output_path": args.output,
        "output_sha256": file_sha256(args.output),
        "source_rows": len(rows),
        "source_questions": len(grouped),
        "rows_mined": len(output_rows),
        "candidate_budget_k": args.k,
        "preference_pairs": len(output_rows),
        "hard_rejected_matches_input": sum(bool(row.get("hard_rejected_matches_input")) for row in output_rows),
        "base_model": args.base_model,
        "sft_adapter": directory_fingerprint(args.sft_adapter_dir),
        "router_dir": args.router_dir,
        "policy_samples": args.policy_samples,
        "max_target_len": args.max_target_len,
        "selection_seed": args.seed,
        "elapsed_seconds_current_invocation": elapsed_seconds,
        "counselor_annotation_used": False,
        "selection_rule": "select one hard rejected response from K automated candidates using frozen-SFT likelihood + chosen similarity + automated router risk",
    }
    manifest_path = str(output_path.with_suffix(output_path.suffix + ".manifest.json"))
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
