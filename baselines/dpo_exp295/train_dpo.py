#!/usr/bin/env python3
"""Train DPO from the existing exp295 SFT LoRA checkpoint.

The trainable policy and the frozen reference start from identical SFT adapter
weights. With TRL 1.9.2, passing a pretrained PeftModel and ref_model=None
creates a frozen ``ref`` adapter copied from ``default`` inside one quantized
backbone. This script verifies that invariant before the first update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when a run would violate the controlled DPO setup."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ConfigurationError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigurationError(f"{path}: expected a YAML object")
    for section in ("experiment", "objective", "training", "quantization"):
        if not isinstance(config.get(section), dict):
            raise ConfigurationError(f"{path}: missing mapping section {section!r}")
    return config


def beta_slug(beta: float) -> str:
    return f"{beta:.6g}".replace("-", "m").replace(".", "p")


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    exp = config["experiment"]
    obj = config["objective"]
    train = config["training"]
    if args.beta is not None:
        obj["beta"] = args.beta
    if args.seed is not None:
        train["seed"] = args.seed
        train["data_seed"] = args.seed
    for key in ("base_model", "sft_adapter", "train_file", "valid_file"):
        value = getattr(args, key)
        if value is not None:
            exp[key] = value
    if args.output_dir is not None:
        exp["output_dir"] = args.output_dir
    if args.max_steps is not None:
        train["max_steps"] = args.max_steps
    if args.max_length is not None:
        train["max_length"] = args.max_length
    return config


def resolve_output_dir(config: Mapping[str, Any]) -> Path:
    exp = config["experiment"]
    if exp.get("output_dir"):
        return Path(str(exp["output_dir"]))
    root = Path(str(exp["output_root"]))
    beta = float(config["objective"]["beta"])
    seed = int(config["training"]["seed"])
    variant = str(exp["variant"])
    return root / variant / f"beta_{beta_slug(beta)}" / f"seed_{seed}"


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def prompt_completion_lengths(tokenizer: Any, row: Mapping[str, Any]) -> tuple[int, int, int]:
    prompt = row.get("prompt")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(prompt, list) or not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ConfigurationError("preference rows must contain conversational prompt/chosen/rejected lists")

    prompt_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
    )
    chosen_ids = tokenizer.apply_chat_template(
        [*prompt, *chosen],
        tokenize=True,
        add_generation_prompt=False,
    )
    rejected_ids = tokenizer.apply_chat_template(
        [*prompt, *rejected],
        tokenize=True,
        add_generation_prompt=False,
    )
    return len(prompt_ids), len(chosen_ids), len(rejected_ids)


def audit_lengths(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
    allow_truncation: bool,
    tolerance: float,
    split: str,
) -> dict[str, Any]:
    if max_length <= 0:
        raise ConfigurationError("max_length must be positive")
    if not 0.0 <= tolerance <= 1.0:
        raise ConfigurationError("truncation_tolerance must be in [0, 1]")

    over = 0
    prompt_over = 0
    lengths: list[int] = []
    longest: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        prompt_len, chosen_len, rejected_len = prompt_completion_lengths(tokenizer, row)
        sequence_len = max(chosen_len, rejected_len)
        lengths.append(sequence_len)
        if prompt_len >= max_length:
            prompt_over += 1
        if sequence_len > max_length:
            over += 1
            longest.append((sequence_len, str(row.get("id", index))))

    fraction = over / max(1, len(rows))
    longest.sort(reverse=True)
    stats = {
        "split": split,
        "rows": len(rows),
        "max_length": max_length,
        "observed_max": max(lengths) if lengths else 0,
        "observed_mean": sum(lengths) / max(1, len(lengths)),
        "overlength_rows": over,
        "overlength_fraction": fraction,
        "prompt_consumes_full_budget": prompt_over,
        "longest_examples": [{"length": length, "id": row_id} for length, row_id in longest[:10]],
    }
    if prompt_over:
        raise ConfigurationError(
            f"{split}: {prompt_over} prompts already consume max_length={max_length}; completion loss would be invalid"
        )
    if over and not allow_truncation:
        raise ConfigurationError(
            f"{split}: {over}/{len(rows)} sequences exceed max_length={max_length}. "
            "Increase max_length or explicitly set allow_truncation=true after inspecting the audit."
        )
    if fraction > tolerance:
        raise ConfigurationError(
            f"{split}: truncation fraction {fraction:.4f} exceeds tolerance {tolerance:.4f}"
        )
    return stats


def adapter_tensor_pairs(model: Any) -> list[tuple[str, Any, Any]]:
    named = dict(model.named_parameters())
    pairs: list[tuple[str, Any, Any]] = []
    for name, parameter in named.items():
        if ".default." not in name:
            continue
        ref_name = name.replace(".default.", ".ref.")
        ref_parameter = named.get(ref_name)
        if ref_parameter is None:
            raise ConfigurationError(f"reference adapter is missing parameter {ref_name}")
        pairs.append((name, parameter, ref_parameter))
    if not pairs:
        raise ConfigurationError("no trainable '.default.' PEFT adapter parameters were found")
    return pairs


def assert_reference_copy(model: Any) -> dict[str, Any]:
    import torch

    peft_config = getattr(model, "peft_config", {})
    if "default" not in peft_config:
        raise ConfigurationError("policy adapter must be loaded under PEFT adapter name 'default'")
    if "ref" not in peft_config:
        raise ConfigurationError(
            "TRL did not create the frozen 'ref' adapter. Pin trl==1.9.2 and pass ref_model=None."
        )

    pairs = adapter_tensor_pairs(model)
    max_abs_diff = 0.0
    trainable_policy = 0
    trainable_reference = 0
    policy_params = 0
    reference_params = 0
    with torch.no_grad():
        for _name, policy, reference in pairs:
            policy_params += policy.numel()
            reference_params += reference.numel()
            trainable_policy += policy.numel() if policy.requires_grad else 0
            trainable_reference += reference.numel() if reference.requires_grad else 0
            diff = (policy.detach().float().cpu() - reference.detach().float().cpu()).abs().max().item()
            max_abs_diff = max(max_abs_diff, float(diff))
            reference.requires_grad_(False)

    if max_abs_diff != 0.0:
        raise ConfigurationError(f"policy/reference adapter initialization differs; max_abs_diff={max_abs_diff}")
    if trainable_policy == 0:
        raise ConfigurationError("SFT policy adapter has no trainable parameters")
    if trainable_reference != 0:
        raise ConfigurationError("reference adapter unexpectedly has trainable parameters")
    return {
        "adapter_parameter_pairs": len(pairs),
        "policy_adapter_parameters": policy_params,
        "reference_adapter_parameters": reference_params,
        "trainable_policy_parameters": trainable_policy,
        "trainable_reference_parameters": trainable_reference,
        "max_abs_initialization_difference": max_abs_diff,
    }


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    names = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"]
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base_model")
    parser.add_argument("--sft_adapter")
    parser.add_argument("--train_file")
    parser.add_argument("--valid_file")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate config/data/token lengths without loading model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    exp = config["experiment"]
    obj = config["objective"]
    train = config["training"]
    quant = config["quantization"]

    beta = float(obj["beta"])
    if beta <= 0:
        raise ConfigurationError("DPO beta must be positive")
    if str(obj.get("loss_type", "sigmoid")) != "sigmoid":
        raise ConfigurationError("the controlled baseline is locked to standard sigmoid DPO")

    train_path = Path(str(exp["train_file"]))
    valid_path = Path(str(exp["valid_file"]))
    sft_adapter = Path(str(exp["sft_adapter"]))
    require_path(train_path, "training preference file")
    require_path(valid_path, "validation preference file")
    if not args.validate_only:
        require_path(sft_adapter, "SFT adapter checkpoint")
        require_path(sft_adapter / "adapter_config.json", "SFT adapter_config.json")

    output_dir = resolve_output_dir(config)
    if args.validate_only:
        output_dir = output_dir / "validation_only"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is non-empty: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(train["seed"])
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # Heavy imports stay inside main so data/config tests run without GPU packages.
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(train.get("tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(train.get("tf32", True))

    base_model_id = str(exp["base_model"])
    tokenizer_source = str(sft_adapter) if (sft_adapter / "tokenizer_config.json").exists() else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    expected_variant = str(exp["variant"])
    for split_name, rows in (("train", train_rows), ("valid", valid_rows)):
        bad = [row.get("pair_mode") for row in rows if str(row.get("pair_mode")) != expected_variant]
        if bad:
            raise ConfigurationError(
                f"{split_name}: preference pair_mode does not match config variant={expected_variant!r}; "
                f"first mismatch={bad[0]!r}"
            )

    max_length = int(train["max_length"])
    length_audit = {
        "train": audit_lengths(
            tokenizer,
            train_rows,
            max_length=max_length,
            allow_truncation=bool(train.get("allow_truncation", False)),
            tolerance=float(train.get("truncation_tolerance", 0.0)),
            split="train",
        ),
        "valid": audit_lengths(
            tokenizer,
            valid_rows,
            max_length=max_length,
            allow_truncation=bool(train.get("allow_truncation", False)),
            tolerance=float(train.get("truncation_tolerance", 0.0)),
            split="valid",
        ),
    }
    write_json(output_dir / "length_audit.json", length_audit)
    write_json(output_dir / "resolved_config.json", config)

    if args.validate_only:
        print(json.dumps({"output_dir": str(output_dir), "length_audit": length_audit}, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured QLoRA DPO run")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    dtype_name = str(quant.get("compute_dtype", "bfloat16")).lower()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    if dtype_name not in dtype_map:
        raise ConfigurationError(f"unsupported quantization compute_dtype={dtype_name!r}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(quant.get("quant_type", "nf4")),
        bnb_4bit_compute_dtype=dtype_map[dtype_name],
        bnb_4bit_use_double_quant=bool(quant.get("double_quant", True)),
    )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        torch_dtype=dtype_map[dtype_name],
        device_map={"": 0},
        trust_remote_code=True,
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
    )
    policy = PeftModel.from_pretrained(
        base,
        str(sft_adapter),
        adapter_name="default",
        is_trainable=True,
    )
    policy.set_adapter("default")

    train_dataset = Dataset.from_list(
        [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in train_rows]
    )
    valid_dataset = Dataset.from_list(
        [{"prompt": row["prompt"], "chosen": row["chosen"], "rejected": row["rejected"]} for row in valid_rows]
    )

    dpo_args = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        run_name=f"{exp['name']}_b{beta_slug(beta)}_s{seed}",
        beta=beta,
        loss_type=[str(obj.get("loss_type", "sigmoid"))],
        label_smoothing=float(obj.get("label_smoothing", 0.0)),
        max_length=max_length,
        truncation_mode="keep_start",
        precompute_ref_log_probs=bool(train.get("precompute_ref_log_probs", True)),
        precompute_ref_batch_size=int(train.get("precompute_ref_batch_size", 1)),
        per_device_train_batch_size=int(train["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        max_steps=int(train["max_steps"]),
        learning_rate=float(train["learning_rate"]),
        warmup_ratio=float(train["warmup_ratio"]),
        lr_scheduler_type=str(train["lr_scheduler_type"]),
        weight_decay=float(train["weight_decay"]),
        max_grad_norm=float(train["max_grad_norm"]),
        logging_steps=int(train["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(train["eval_steps"]),
        save_strategy="steps",
        save_steps=int(train["save_steps"]),
        save_total_limit=int(train["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="rewards/accuracies",
        greater_is_better=True,
        bf16=bool(train.get("bf16", True)),
        fp16=False,
        tf32=bool(train.get("tf32", True)),
        gradient_checkpointing=bool(train.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataset_num_proc=int(train.get("dataset_num_proc", 1)),
        seed=seed,
        data_seed=int(train.get("data_seed", seed)),
        disable_dropout=True,
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )
    reference_audit = assert_reference_copy(trainer.model)
    write_json(output_dir / "reference_audit.json", reference_audit)

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.set_adapter("default")
    unwrapped.save_pretrained(
        final_dir,
        selected_adapters=["default"],
        safe_serialization=True,
    )
    tokenizer.save_pretrained(final_dir)

    elapsed = time.perf_counter() - start_time
    max_allocated = int(torch.cuda.max_memory_allocated())
    max_reserved = int(torch.cuda.max_memory_reserved())
    manifest = {
        "status": "trained",
        "experiment": exp,
        "objective": obj,
        "training": train,
        "quantization": quant,
        "input": {
            "train_file": str(train_path),
            "train_sha256": sha256_file(train_path),
            "train_rows": len(train_rows),
            "valid_file": str(valid_path),
            "valid_sha256": sha256_file(valid_path),
            "valid_rows": len(valid_rows),
            "sft_adapter": str(sft_adapter),
        },
        "reference_initialization": reference_audit,
        "length_audit": length_audit,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "runtime_seconds": elapsed,
        "peak_cuda_allocated_bytes": max_allocated,
        "peak_cuda_reserved_bytes": max_reserved,
        "peak_cuda_allocated_gib": max_allocated / (1024**3),
        "peak_cuda_reserved_gib": max_reserved / (1024**3),
        "hardware": {
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "versions": package_versions(),
        "final_adapter": str(final_dir),
    }
    write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
