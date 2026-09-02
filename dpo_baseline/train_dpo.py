from __future__ import annotations

import argparse
import importlib.metadata
import copy
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from dpo_utils import (
    build_completion_message,
    build_prompt_messages,
    clean_text,
    directory_fingerprint,
    file_sha256,
    fit_refinement_prompt_to_budget,
    load_yaml,
    read_jsonl,
    normalize_text,
    truncate_completion_to_budget,
    safe_slug,
    validate_pair_record,
    write_json,
)


def _cfg(config: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {section!r} must be a mapping")
    return value.get(key, default)


def _apply_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(config))
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("data", {})
    if args.sft_adapter_dir:
        config["model"]["sft_adapter_dir"] = args.sft_adapter_dir
    if args.train_file:
        config["data"]["train_pairs"] = args.train_file
    if args.valid_file:
        config["data"]["valid_pairs"] = args.valid_file
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.beta is not None:
        config["training"]["beta"] = args.beta
    if args.seed is not None:
        config["training"]["seed"] = args.seed
        config["training"]["data_seed"] = args.seed
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    return config


def _audit_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: int,
    expected_questions: int,
) -> Dict[str, Any]:
    for row in rows:
        validate_pair_record(row)
    if expected_pairs >= 0 and len(rows) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} preference pairs; found {len(rows)}")
    question_ids = {
        str((row.get("metadata") or {}).get("question_id", "")).strip()
        for row in rows
    }
    question_ids.discard("")
    if expected_questions >= 0 and len(question_ids) != expected_questions:
        raise ValueError(f"Expected {expected_questions} unique questions; found {len(question_ids)}")
    regimes = sorted(
        {
            str((row.get("metadata") or {}).get("pair_regime", "unknown"))
            for row in rows
        }
    )
    return {
        "pairs": len(rows),
        "questions": len(question_ids),
        "pair_regimes": regimes,
    }


def _token_length_audit(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_source_len: int,
    max_target_len: int,
    max_length: int,
) -> Dict[str, Any]:
    prompt_lengths: List[int] = []
    chosen_lengths: List[int] = []
    rejected_lengths: List[int] = []
    source_over = 0
    chosen_over = 0
    rejected_over = 0
    fully_truncated = 0

    for row in rows:
        prompt = row["prompt"]
        prompt_ids = tokenizer.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        chosen_full = tokenizer.apply_chat_template(
            prompt + row["chosen"],
            tokenize=True,
            add_generation_prompt=False,
        )
        rejected_full = tokenizer.apply_chat_template(
            prompt + row["rejected"],
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(chosen_full, "tolist"):
            chosen_full = chosen_full.tolist()
        if hasattr(rejected_full, "tolist"):
            rejected_full = rejected_full.tolist()
        prompt_len = len(prompt_ids)
        chosen_len = max(0, len(chosen_full) - prompt_len)
        rejected_len = max(0, len(rejected_full) - prompt_len)
        prompt_lengths.append(prompt_len)
        chosen_lengths.append(chosen_len)
        rejected_lengths.append(rejected_len)
        source_over += int(prompt_len > max_source_len)
        chosen_over += int(chosen_len > max_target_len)
        rejected_over += int(rejected_len > max_target_len)
        fully_truncated += int(prompt_len >= max_length)

    def stats(values: Sequence[int]) -> Dict[str, float]:
        ordered = sorted(values)
        if not ordered:
            return {"min": 0, "mean": 0.0, "p95": 0, "max": 0}
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        return {
            "min": int(ordered[0]),
            "mean": float(sum(ordered) / len(ordered)),
            "p95": int(ordered[p95_index]),
            "max": int(ordered[-1]),
        }

    if fully_truncated:
        raise ValueError(
            f"{fully_truncated} prompts are at least max_length={max_length}; DPO would lose every completion token"
        )
    return {
        "prompt_tokens": stats(prompt_lengths),
        "chosen_tokens": stats(chosen_lengths),
        "rejected_tokens": stats(rejected_lengths),
        "prompt_over_source_budget": source_over,
        "chosen_over_target_budget": chosen_over,
        "rejected_over_target_budget": rejected_over,
        "fully_truncated_pairs": fully_truncated,
    }


def _enforce_pair_token_budgets(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_source_len: int,
    max_target_len: int,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    prepared: List[Dict[str, Any]] = []
    stats = {
        "pairs": len(rows),
        "prompt_truncated": 0,
        "chosen_truncated": 0,
        "rejected_truncated": 0,
    }
    for row in rows:
        record = copy.deepcopy(dict(row))
        metadata = record.get("metadata") or {}
        question = clean_text(metadata.get("question"))
        initial_draft = clean_text(metadata.get("initial_draft"))
        if not question or not initial_draft:
            raise ValueError(f"Missing question/initial_draft metadata for pair {record.get('pair_id')!r}")
        prompt, model_question, model_draft, prompt_changed = fit_refinement_prompt_to_budget(
            tokenizer, question, initial_draft, max_source_len
        )
        stats["prompt_truncated"] += int(prompt_changed)
        chosen_text = clean_text(record["chosen"][0]["content"])
        rejected_text = clean_text(record["rejected"][0]["content"])
        fitted_chosen, chosen_truncated = truncate_completion_to_budget(
            tokenizer, prompt, chosen_text, max_target_len
        )
        fitted_rejected, rejected_truncated = truncate_completion_to_budget(
            tokenizer, prompt, rejected_text, max_target_len
        )
        stats["chosen_truncated"] += int(chosen_truncated)
        stats["rejected_truncated"] += int(rejected_truncated)
        if normalize_text(fitted_chosen) == normalize_text(fitted_rejected):
            raise ValueError(f"Token truncation collapsed chosen and rejected for pair {record.get('pair_id')!r}")
        record["prompt"] = prompt
        record["chosen"] = build_completion_message(fitted_chosen)
        record["rejected"] = build_completion_message(fitted_rejected)
        record["metadata"]["model_question"] = model_question
        record["metadata"]["model_initial_draft"] = model_draft
        record["metadata"]["prompt_truncated_for_budget"] = prompt_changed
        prepared.append(record)
    return prepared, stats


def _assert_adapter_contract(model: Any, config: Mapping[str, Any]) -> Dict[str, Any]:
    expected_r_raw = _cfg(config, "model", "expected_lora_r", None)
    expected_alpha_raw = _cfg(config, "model", "expected_lora_alpha", None)
    expected_modules_raw = _cfg(config, "model", "expected_target_module_tokens", None)
    expected_r = int(expected_r_raw) if expected_r_raw is not None else None
    expected_alpha = int(expected_alpha_raw) if expected_alpha_raw is not None else None
    expected_modules = set(str(value) for value in expected_modules_raw) if expected_modules_raw else set()

    peft_config = model.peft_config.get("default")
    if peft_config is None:
        raise ValueError("The SFT checkpoint did not load as the default PEFT adapter")

    observed_r = int(getattr(peft_config, "r", -1))
    observed_alpha = int(getattr(peft_config, "lora_alpha", -1))
    raw_targets = getattr(peft_config, "target_modules", None)
    if isinstance(raw_targets, str):
        target_repr = raw_targets
    elif raw_targets is None:
        target_repr = ""
    else:
        target_repr = ",".join(sorted(str(value) for value in raw_targets))

    problems: List[str] = []
    if expected_r is not None and observed_r != expected_r:
        problems.append(f"LoRA r={observed_r}, expected {expected_r}")
    if expected_alpha is not None and observed_alpha != expected_alpha:
        problems.append(f"LoRA alpha={observed_alpha}, expected {expected_alpha}")
    missing_modules = sorted(token for token in expected_modules if token not in target_repr)
    if missing_modules:
        problems.append(f"target_modules missing tokens {missing_modules}; observed={target_repr!r}")
    if problems:
        raise ValueError("SFT adapter contract mismatch: " + "; ".join(problems))

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if not trainable_names:
        raise ValueError("The loaded SFT adapter has no trainable policy parameters")
    non_adapter_trainables = [
        name
        for name in trainable_names
        if ".default." not in name and "modules_to_save.default" not in name
    ]
    if non_adapter_trainables:
        raise ValueError(
            "Unexpected trainable parameters outside the SFT policy adapter: "
            + ", ".join(non_adapter_trainables[:10])
        )
    return {
        "lora_r": observed_r,
        "lora_alpha": observed_alpha,
        "lora_dropout": float(getattr(peft_config, "lora_dropout", 0.0)),
        "target_modules": target_repr,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100.0 * trainable / max(1, total),
        "trainable_parameter_tensors": len(trainable_names),
        "contract_values_enforced": {
            "r": expected_r,
            "alpha": expected_alpha,
            "target_module_tokens": sorted(expected_modules),
        },
    }


def _assert_frozen_reference_adapter(model: Any) -> Dict[str, Any]:
    import torch

    if "ref" not in model.peft_config:
        raise RuntimeError(
            "TRL did not create a frozen `ref` adapter from the initial SFT adapter. "
            "Use trl>=1.9.2 and peft>=0.20.0; do not continue with a base-model reference."
        )
    pairs = 0
    mismatches: List[str] = []
    trainable_reference = 0
    parameters = dict(model.named_parameters())
    for name, parameter in parameters.items():
        if ".ref." in name and parameter.requires_grad:
            trainable_reference += parameter.numel()
        if ".default." not in name:
            continue
        ref_name = name.replace(".default.", ".ref.")
        ref_parameter = parameters.get(ref_name)
        if ref_parameter is None:
            continue
        pairs += 1
        if not torch.equal(parameter.detach().cpu(), ref_parameter.detach().cpu()):
            mismatches.append(name)
            if len(mismatches) >= 5:
                break
    if pairs == 0:
        raise RuntimeError("No matching policy/reference adapter tensors were found")
    if mismatches:
        raise RuntimeError(f"Reference adapter differs from the SFT policy snapshot before training: {mismatches}")
    if trainable_reference:
        raise RuntimeError(f"Frozen reference adapter has {trainable_reference} trainable parameters")
    return {
        "reference_adapter": "ref",
        "matched_tensor_pairs": pairs,
        "reference_trainable_parameters": trainable_reference,
        "snapshot_equal_before_training": True,
    }


def _package_versions(names: Sequence[str]) -> Dict[str, str | None]:
    versions: Dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT-initialized QLoRA DPO for exp295 response refinement.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft_adapter_dir", default=None)
    parser.add_argument("--train_file", default=None)
    parser.add_argument("--valid_file", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    config = _apply_overrides(load_yaml(args.config), args)
    train_file = str(_cfg(config, "data", "train_pairs", ""))
    valid_file = str(_cfg(config, "data", "valid_pairs", ""))
    sft_adapter_dir = str(_cfg(config, "model", "sft_adapter_dir", "") or "")
    output_dir = str(_cfg(config, "training", "output_dir", "") or "")
    if not train_file or not valid_file:
        raise ValueError("data.train_pairs and data.valid_pairs are required")
    if not sft_adapter_dir:
        raise ValueError("Provide the exact existing SFT checkpoint using --sft_adapter_dir or model.sft_adapter_dir")
    if not output_dir:
        raise ValueError("training.output_dir is required")

    train_rows = read_jsonl(train_file)
    valid_rows = read_jsonl(valid_file)
    train_audit = _audit_pair_rows(
        train_rows,
        expected_pairs=int(_cfg(config, "data", "expected_train_pairs", -1)),
        expected_questions=int(_cfg(config, "data", "expected_train_questions", -1)),
    )
    valid_audit = _audit_pair_rows(
        valid_rows,
        expected_pairs=int(_cfg(config, "data", "expected_valid_pairs", -1)),
        expected_questions=int(_cfg(config, "data", "expected_valid_questions", -1)),
    )
    train_qids = {(row.get("metadata") or {}).get("question_id") for row in train_rows}
    valid_qids = {(row.get("metadata") or {}).get("question_id") for row in valid_rows}
    overlap = {value for value in train_qids & valid_qids if value}
    if overlap:
        raise ValueError(f"Train/validation question leakage: {len(overlap)} questions")

    beta = float(_cfg(config, "training", "beta", 0.1))
    seed = int(_cfg(config, "training", "seed", 42))
    max_steps = int(_cfg(config, "training", "max_steps", 234))
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive to keep minimal and hard-K budgets matched")

    preliminary_manifest = {
        "config_path": args.config,
        "config": config,
        "train_file": train_file,
        "train_sha256": file_sha256(train_file),
        "valid_file": valid_file,
        "valid_sha256": file_sha256(valid_file),
        "train_audit": train_audit,
        "valid_audit": valid_audit,
        "train_valid_question_overlap": 0,
        "sft_adapter": directory_fingerprint(sft_adapter_dir),
        "beta": beta,
        "seed": seed,
        "max_steps": max_steps,
        "reference_requirement": "frozen copy of the initial SFT adapter",
        "counselor_annotation_used": False,
    }
    if args.dry_run:
        print(json.dumps(preliminary_manifest, ensure_ascii=False, indent=2))
        return

    import torch
    from datasets import Dataset
    from packaging.version import Version
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    versions = _package_versions(("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes"))
    if versions["trl"] is None or Version(versions["trl"]) < Version("1.9.2"):
        raise RuntimeError(f"trl>=1.9.2 is required; found {versions['trl']}")
    if versions["peft"] is None or Version(versions["peft"]) < Version("0.20.0"):
        raise RuntimeError(f"peft>=0.20.0 is required; found {versions['peft']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 4-bit Gemma DPO run")

    base_model_name = str(_cfg(config, "model", "base_model", "google/gemma-4-E4B-it"))
    trust_remote_code = bool(_cfg(config, "model", "trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_source_len = int(_cfg(config, "training", "max_source_len", 512))
    max_target_len = int(_cfg(config, "training", "max_target_len", 160))
    max_length = int(_cfg(config, "training", "max_length", max_source_len + max_target_len))
    if max_length != max_source_len + max_target_len:
        raise ValueError("training.max_length must equal max_source_len + max_target_len for the controlled run")
    train_rows, train_truncation = _enforce_pair_token_budgets(
        train_rows, tokenizer, max_source_len=max_source_len, max_target_len=max_target_len
    )
    valid_rows, valid_truncation = _enforce_pair_token_budgets(
        valid_rows, tokenizer, max_source_len=max_source_len, max_target_len=max_target_len
    )
    token_audit = {
        "train": {
            **_token_length_audit(
                train_rows,
                tokenizer,
                max_source_len=max_source_len,
                max_target_len=max_target_len,
                max_length=max_length,
            ),
            "truncation": train_truncation,
        },
        "valid": {
            **_token_length_audit(
                valid_rows,
                tokenizer,
                max_source_len=max_source_len,
                max_target_len=max_target_len,
                max_length=max_length,
            ),
            "truncation": valid_truncation,
        },
    }

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        trust_remote_code=trust_remote_code,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=bool(_cfg(config, "training", "gradient_checkpointing", True)),
    )
    model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_dir,
        adapter_name="default",
        is_trainable=True,
    )
    adapter_audit = _assert_adapter_contract(model, config)
    model.print_trainable_parameters()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_name = str(_cfg(config, "experiment", "name", output.name))
    run_name = f"{safe_slug(run_name)}-beta{beta:g}-seed{seed}"
    raw_loss_type = _cfg(config, "training", "loss_type", ["sigmoid"])
    loss_types = [raw_loss_type] if isinstance(raw_loss_type, str) else list(raw_loss_type)

    training_args = DPOConfig(
        output_dir=str(output),
        run_name=run_name,
        beta=beta,
        loss_type=[str(value) for value in loss_types],
        max_length=max_length,
        truncation_mode="keep_start",
        precompute_ref_log_probs=bool(_cfg(config, "training", "precompute_ref_log_probs", True)),
        precompute_ref_batch_size=int(_cfg(config, "training", "precompute_ref_batch_size", 1)),
        per_device_train_batch_size=int(_cfg(config, "training", "per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(_cfg(config, "training", "per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(_cfg(config, "training", "gradient_accumulation_steps", 16)),
        max_steps=max_steps,
        num_train_epochs=float(_cfg(config, "training", "num_train_epochs", 3.0)),
        learning_rate=float(_cfg(config, "training", "learning_rate", 1e-5)),
        lr_scheduler_type=str(_cfg(config, "training", "lr_scheduler_type", "linear")),
        warmup_ratio=float(_cfg(config, "training", "warmup_ratio", 0.06)),
        weight_decay=float(_cfg(config, "training", "weight_decay", 0.01)),
        max_grad_norm=float(_cfg(config, "training", "max_grad_norm", 1.0)),
        optim=str(_cfg(config, "training", "optim", "adamw_torch")),
        bf16=True,
        fp16=False,
        tf32=bool(_cfg(config, "training", "tf32", True)),
        gradient_checkpointing=bool(_cfg(config, "training", "gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=bool(_cfg(config, "training", "disable_dropout", True)),
        eval_strategy="steps",
        eval_steps=int(_cfg(config, "training", "eval_steps", 25)),
        save_strategy="steps",
        save_steps=int(_cfg(config, "training", "save_steps", 100)),
        save_total_limit=int(_cfg(config, "training", "save_total_limit", 2)),
        logging_strategy="steps",
        logging_steps=int(_cfg(config, "training", "logging_steps", 5)),
        logging_first_step=True,
        report_to=[],
        seed=seed,
        data_seed=int(_cfg(config, "training", "data_seed", seed)),
        full_determinism=bool(_cfg(config, "training", "full_determinism", False)),
        dataloader_num_workers=int(_cfg(config, "training", "dataloader_num_workers", 2)),
        remove_unused_columns=True,
        include_num_input_tokens_seen=True,
        load_best_model_at_end=False,
    )

    train_dataset = Dataset.from_list(list(train_rows))
    valid_dataset = Dataset.from_list(list(valid_rows))

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )
    reference_audit = _assert_frozen_reference_adapter(model)
    if trainer.ref_model is not None:
        raise RuntimeError("A standalone reference model was allocated; expected the frozen SFT reference adapter path")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed_seconds = time.perf_counter() - start
    final_dir = output / "final"
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.save_pretrained(final_dir, selected_adapters=["default"], safe_serialization=True)
    tokenizer.save_pretrained(final_dir)

    peak_memory_gib = torch.cuda.max_memory_allocated() / (1024**3)
    reserved_memory_gib = torch.cuda.max_memory_reserved() / (1024**3)
    gpu = torch.cuda.get_device_properties(local_rank)
    manifest = {
        **preliminary_manifest,
        "status": "completed",
        "versions": versions,
        "python": platform.python_version(),
        "token_length_audit": token_audit,
        "adapter_audit": adapter_audit,
        "reference_audit": reference_audit,
        "output_dir": str(output),
        "final_adapter_dir": str(final_dir),
        "training_metrics": dict(train_result.metrics),
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_allocated_gib": peak_memory_gib,
        "peak_memory_reserved_gib": reserved_memory_gib,
        "gpu": {
            "name": gpu.name,
            "total_memory_gib": gpu.total_memory / (1024**3),
            "compute_capability": f"{gpu.major}.{gpu.minor}",
        },
        "reference_semantics": (
            "The `ref` PEFT adapter is an exact frozen copy of the existing SFT adapter before the first DPO update. "
            "Reference log-probabilities are precomputed before policy optimization."
        ),
    }
    write_json(manifest, output / "run_manifest.json")
    write_json(config, output / "resolved_config.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
