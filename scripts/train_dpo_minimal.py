#!/usr/bin/env python3
"""Train DPO-minimal from the verified exp295 SFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence

from build_dpo_minimal_pairs import (
    DATASET_NAME,
    DATASET_PROVENANCE,
    DataContractError,
    EXPECTED_ROWS,
    read_jsonl,
    sha256_file,
)


REQUIRED_PAIR_FIELDS = {"id", "question_group_id", "prompt", "chosen", "rejected", "audit_metadata"}
MODEL_COLUMNS = ("prompt", "chosen", "rejected")
SUPPORTED_TRL_VERSION = "1.4.0"
FULLPAPER_DATASET_NAME = "fullpaper_corrected_dev57_v1"
FULLPAPER_PROVENANCE = "dev120-corrected-v2 TRAIN-origin development data"


def package_versions() -> Dict[str, str]:
    names = ("torch", "transformers", "peft", "trl", "datasets", "accelerate", "bitsandbytes")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def git_revision(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def validate_pair_rows(rows: Sequence[Mapping[str, Any]], split: str) -> Dict[str, Any]:
    if split not in ("train", "valid"):
        raise DataContractError(f"DPO-minimal accepts train/valid only, not {split!r}")
    if len(rows) != EXPECTED_ROWS[split]:
        raise DataContractError(f"{split}: expected {EXPECTED_ROWS[split]} pairs, found {len(rows)}")

    row_ids = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_PAIR_FIELDS - set(row)
        if missing:
            raise DataContractError(f"{split}[{index}] missing pair fields: {sorted(missing)}")
        if row["id"] in row_ids:
            raise DataContractError(f"{split}: duplicate pair id {row['id']!r}")
        row_ids.add(row["id"])
        for field in MODEL_COLUMNS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise DataContractError(f"{split}[{index}] empty/non-string {field}")
        if row["chosen"].strip().casefold() == row["rejected"].strip().casefold():
            raise DataContractError(f"{split}[{index}] chosen equals rejected")
        metadata = row["audit_metadata"]
        if not isinstance(metadata, dict):
            raise DataContractError(f"{split}[{index}] audit_metadata must be an object")
        if metadata.get("dataset") != DATASET_NAME or metadata.get("provenance") != DATASET_PROVENANCE:
            raise DataContractError(f"{split}[{index}] wrong dataset provenance")

    return {"rows": len(rows), "unique_ids": len(row_ids)}


def validate_fullpaper_pair_rows(rows: Sequence[Mapping[str, Any]], role: str) -> Dict[str, Any]:
    """Validate the dynamic full-paper contract without exp295 count assumptions."""
    if role not in ("train", "development_eval_train_origin", "valid"):
        raise DataContractError(f"Unsupported full-paper role: {role!r}")
    if not rows:
        raise DataContractError(f"Full-paper {role} file is empty")
    row_ids = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_PAIR_FIELDS - set(row)
        if missing:
            raise DataContractError(f"{role}[{index}] missing pair fields: {sorted(missing)}")
        row_id = str(row["id"])
        if row_id in row_ids:
            raise DataContractError(f"{role}: duplicate pair id {row_id!r}")
        row_ids.add(row_id)
        for field in MODEL_COLUMNS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise DataContractError(f"{role}[{index}] empty/non-string {field}")
        if row["chosen"] == row["rejected"]:
            raise DataContractError(f"{role}[{index}] chosen equals rejected")
        metadata = row["audit_metadata"]
        if not isinstance(metadata, dict):
            raise DataContractError(f"{role}[{index}] audit_metadata must be an object")
        frozen_dev_contract = (
            metadata.get("dataset") == FULLPAPER_DATASET_NAME
            and metadata.get("provenance") == FULLPAPER_PROVENANCE
        )
        versioned_contract = (
            metadata.get("contract_version") == "fullpaper-dpo-v1"
            and str(metadata.get("dataset", "")).startswith("fullpaper_")
            and bool(metadata.get("provenance"))
        )
        if not (frozen_dev_contract or versioned_contract):
            raise DataContractError(f"{role}[{index}] wrong or unversioned full-paper provenance")
        expected_question_group = row_id if frozen_dev_contract else metadata.get("question_normalized_sha256")
        if not expected_question_group or row.get("question_group_id") != expected_question_group:
            raise DataContractError(f"{role}[{index}] question_group_id does not preserve the declared question group")
        expected_original_split = "valid" if role == "valid" else "train"
        if metadata.get("original_split") != expected_original_split or metadata.get("development_role") != role:
            raise DataContractError(f"{role}[{index}] does not match its declared split/role")
        hashes = {
            "chosen_sha256": hashlib.sha256(row["chosen"].encode()).hexdigest(),
            "rejected_sha256": hashlib.sha256(row["rejected"].encode()).hexdigest(),
        }
        if any(metadata.get(name) != value for name, value in hashes.items()):
            raise DataContractError(f"{role}[{index}] chosen/rejected text hash mismatch")
    return {
        "rows": len(rows),
        "unique_ids": len(row_ids),
        "role": role,
        "original_split": "valid" if role == "valid" else "train",
        "not_final_valid_or_test": role != "valid",
    }


def model_only_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Return only columns that TRL may tokenize or pass to the model."""
    return [{name: str(row[name]) for name in MODEL_COLUMNS} for row in rows]


def validate_checkpoint_contract(
    adapter_dir: Path | str,
    expected_base_model: str | None = None,
    require_exp295_provenance: bool = True,
) -> Dict[str, Any]:
    adapter_dir = Path(adapter_dir)
    required = ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json")
    missing = [name for name in required if not (adapter_dir / name).is_file()]
    if missing:
        raise DataContractError(f"Missing SFT checkpoint files in {adapter_dir}: {missing}")

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = adapter_config.get("base_model_name_or_path")
    if not base_model:
        raise DataContractError("SFT adapter_config.json has no base_model_name_or_path")
    if expected_base_model is not None and base_model != expected_base_model:
        raise DataContractError(f"Base model mismatch: adapter={base_model!r}, CLI={expected_base_model!r}")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise DataContractError("SFT checkpoint is not a causal-LM LoRA adapter")

    train_args_path = adapter_dir.parent / "train_args.json"
    if not train_args_path.is_file():
        raise DataContractError(f"Missing SFT provenance file: {train_args_path}")
    train_args = json.loads(train_args_path.read_text(encoding="utf-8"))
    if train_args.get("prompt_style") != "sft_plain":
        raise DataContractError(f"Expected sft_plain checkpoint, found {train_args.get('prompt_style')!r}")
    if require_exp295_provenance:
        if train_args.get("train_file") != "data/splits_exp295/train_mdlm.jsonl":
            raise DataContractError("SFT checkpoint was not recorded against the exp295 legacy train split")
        if train_args.get("valid_file") != "data/splits_exp295/valid_mdlm.jsonl":
            raise DataContractError("SFT checkpoint was not recorded against the exp295 legacy valid split")

    return {
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_sha256": sha256_file(adapter_dir / "adapter_model.safetensors"),
        "base_model": base_model,
        "peft_type": adapter_config["peft_type"],
        "task_type": adapter_config["task_type"],
        "lora_r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "lora_dropout": adapter_config.get("lora_dropout"),
        "target_module_count": len(adapter_config.get("target_modules", [])),
        "sft_prompt_style": train_args["prompt_style"],
        "sft_train_file": train_args["train_file"],
        "sft_valid_file": train_args["valid_file"],
    }


@dataclass
class DPOPreferenceCollator:
    """Right-pad preferences while preserving separate prompt/completion budgets."""

    pad_token_id: int
    eos_token_id: int | None
    max_prompt_length: int = 768
    max_completion_length: int = 512

    def __post_init__(self) -> None:
        if self.max_prompt_length <= 0 or self.max_completion_length <= 0:
            raise ValueError("Prompt and completion limits must be positive")

    def _truncate_completion(self, ids: Sequence[int]) -> List[int]:
        ids = list(ids)
        if len(ids) <= self.max_completion_length:
            return ids
        truncated = ids[: self.max_completion_length]
        if self.eos_token_id is not None and ids[-1] == self.eos_token_id:
            truncated[-1] = self.eos_token_id
        return truncated

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_sequences: List[torch.Tensor] = []
        rejected_sequences: List[torch.Tensor] = []
        chosen_completion_masks: List[torch.Tensor] = []
        rejected_completion_masks: List[torch.Tensor] = []

        for example in examples:
            prompt = list(example["prompt_ids"])[: self.max_prompt_length]
            chosen = self._truncate_completion(example["chosen_ids"])
            rejected = self._truncate_completion(example["rejected_ids"])
            chosen_sequences.append(torch.tensor(prompt + chosen, dtype=torch.long))
            rejected_sequences.append(torch.tensor(prompt + rejected, dtype=torch.long))
            chosen_completion_masks.append(
                torch.tensor([0] * len(prompt) + [1] * len(chosen), dtype=torch.long)
            )
            rejected_completion_masks.append(
                torch.tensor([0] * len(prompt) + [1] * len(rejected), dtype=torch.long)
            )

        sequences = chosen_sequences + rejected_sequences
        completion_masks = chosen_completion_masks + rejected_completion_masks
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        completion_mask = pad_sequence(completion_masks, batch_first=True, padding_value=0)
        attention_mask = (input_ids != self.pad_token_id).long()
        # Gemma's pad token can equal EOS. Derive attention from true lengths so
        # an EOS completion token is never mistaken for right padding.
        for row_index, sequence in enumerate(sequences):
            attention_mask[row_index, : len(sequence)] = 1
            attention_mask[row_index, len(sequence) :] = 0

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }
        if examples and "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor(
                [float(example["ref_chosen_logps"]) for example in examples]
            )
            output["ref_rejected_logps"] = torch.tensor(
                [float(example["ref_rejected_logps"]) for example in examples]
            )
        return output


def audit_adapter_state(model: torch.nn.Module) -> Dict[str, Any]:
    parameters = dict(model.named_parameters())
    policy_names = sorted(name for name in parameters if ".default." in name)
    reference_names = sorted(name for name in parameters if ".ref." in name)
    if not policy_names or not reference_names:
        raise RuntimeError("Expected both default policy and ref adapters")

    unmatched = []
    max_abs_diff = 0.0
    compared = 0
    for policy_name in policy_names:
        reference_name = policy_name.replace(".default.", ".ref.")
        reference = parameters.get(reference_name)
        if reference is None:
            unmatched.append(policy_name)
            continue
        policy = parameters[policy_name]
        if policy.shape != reference.shape:
            raise RuntimeError(f"Adapter shape mismatch: {policy_name} vs {reference_name}")
        difference = (policy.detach().float() - reference.detach().float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(difference))
        compared += 1
    if unmatched or compared != len(reference_names):
        raise RuntimeError(
            f"Policy/reference tensor structure mismatch: unmatched={unmatched[:5]}, "
            f"policy={len(policy_names)}, ref={len(reference_names)}, compared={compared}"
        )

    reference_trainable = [name for name in reference_names if parameters[name].requires_grad]
    policy_trainable = [name for name in policy_names if parameters[name].requires_grad]
    unexpected_trainable = [
        name for name, parameter in parameters.items() if parameter.requires_grad and ".default." not in name
    ]
    if max_abs_diff != 0.0:
        raise RuntimeError(f"Policy/reference initialization differs: max_abs_diff={max_abs_diff}")
    if reference_trainable:
        raise RuntimeError(f"Reference adapter has trainable tensors: {reference_trainable[:5]}")
    if not policy_trainable:
        raise RuntimeError("Policy adapter has no trainable tensors")
    if unexpected_trainable:
        raise RuntimeError(f"Non-policy tensors are trainable: {unexpected_trainable[:5]}")

    return {
        "policy_tensor_count": len(policy_names),
        "reference_tensor_count": len(reference_names),
        "compared_tensor_count": compared,
        "max_abs_diff": max_abs_diff,
        "reference_trainable_tensor_count": len(reference_trainable),
        "policy_trainable_tensor_count": len(policy_trainable),
        "unexpected_trainable_tensor_count": len(unexpected_trainable),
        "policy_trainable_parameter_count": sum(parameters[name].numel() for name in policy_trainable),
    }


def cast_policy_adapter_dtype(model: torch.nn.Module, dtype: torch.dtype) -> Dict[str, Any]:
    """Cast only the trainable default adapter before TRL snapshots `ref`.

    TRL 1.4 snapshots the reference and then casts trainable QLoRA parameters
    to BF16. Casting first keeps the policy/reference numeric initialization
    identical when reference log-probabilities are precomputed in __init__.
    """
    cast_names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and ".default." in name:
            parameter.data = parameter.data.to(dtype)
            cast_names.append(name)
    if not cast_names:
        raise RuntimeError("No trainable default-adapter tensors found to cast")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".default." not in name
    ]
    if unexpected:
        raise RuntimeError(f"Non-policy tensors were trainable before reference snapshot: {unexpected[:5]}")
    return {
        "dtype": str(dtype),
        "tensor_count": len(cast_names),
        "parameter_count": sum(dict(model.named_parameters())[name].numel() for name in cast_names),
    }


def adapter_fingerprint(model: torch.nn.Module, adapter_name: str) -> str:
    marker = f".{adapter_name}."
    selected = sorted((name, parameter) for name, parameter in model.named_parameters() if marker in name)
    if not selected:
        raise RuntimeError(f"No tensors found for adapter {adapter_name!r}")
    digest = hashlib.sha256()
    for name, parameter in selected:
        digest.update(name.encode("utf-8"))
        raw = parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--sft_adapter_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=234)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--max_prompt_length", type=int, default=768)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--precompute_ref_batch_size", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data_contract",
        choices=("exp295", "fullpaper"),
        default="exp295",
        help="Default preserves the fixed exp295 contract; fullpaper accepts versioned dynamic rows.",
    )
    parser.add_argument(
        "--fullpaper_valid_role",
        choices=("development_eval_train_origin", "valid"),
        default="development_eval_train_origin",
        help="Use TRAIN-origin holdback only for smoke checks; use valid only with a separately constructed final VALID pair file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After validating the complete files, exercise one optimizer step on 2 train/2 valid pairs",
    )
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_file)
    valid_path = Path(args.valid_file)
    train_rows = read_jsonl(train_path)
    valid_rows = read_jsonl(valid_path)
    if args.data_contract == "exp295":
        pair_audit = {
            "train": validate_pair_rows(train_rows, "train"),
            "valid": validate_pair_rows(valid_rows, "valid"),
        }
        dataset_name = DATASET_NAME
        dataset_provenance = DATASET_PROVENANCE
        result_label = "exp295 legacy test"
    else:
        pair_audit = {
            "train": validate_fullpaper_pair_rows(train_rows, "train"),
            "valid": validate_fullpaper_pair_rows(valid_rows, args.fullpaper_valid_role),
        }
        overlap_sets = {
            "canonical_id": ({row["id"] for row in train_rows}, {row["id"] for row in valid_rows}),
            "question_normalized_sha256": (
                {row["audit_metadata"].get("question_normalized_sha256") for row in train_rows},
                {row["audit_metadata"].get("question_normalized_sha256") for row in valid_rows},
            ),
            "duplicate_cluster_id": (
                {row["audit_metadata"].get("duplicate_cluster_id") for row in train_rows},
                {row["audit_metadata"].get("duplicate_cluster_id") for row in valid_rows},
            ),
            "source_group_id": (
                {row["audit_metadata"].get("source_group_id") for row in train_rows},
                {row["audit_metadata"].get("source_group_id") for row in valid_rows},
            ),
        }
        overlap = {
            field: sorted((left - {None}) & (right - {None}))
            for field, (left, right) in overlap_sets.items()
        }
        if any(overlap.values()):
            raise DataContractError(f"Full-paper train/evaluation group overlap: {overlap}")
        dataset_name = FULLPAPER_DATASET_NAME
        dataset_provenance = FULLPAPER_PROVENANCE
        result_label = (
            "TRAIN-origin development smoke; not paper VALID/TEST"
            if args.fullpaper_valid_role == "development_eval_train_origin"
            else "paper VALID supplied under versioned full-paper contract"
        )
    checkpoint = validate_checkpoint_contract(
        args.sft_adapter_dir,
        args.base_model,
        require_exp295_provenance=args.data_contract == "exp295",
    )
    versions = package_versions()
    if versions["trl"] != SUPPORTED_TRL_VERSION:
        raise RuntimeError(
            f"This implementation is verified for TRL {SUPPORTED_TRL_VERSION}; found {versions['trl']}"
        )
    if args.smoke and args.data_contract == "exp295" and args.max_steps != 1:
        raise DataContractError("--smoke requires --max_steps 1")

    repo = Path(__file__).resolve().parents[1]
    static_manifest = {
        "dataset": dataset_name,
        "provenance": dataset_provenance,
        "data_contract": args.data_contract,
        "test_result_label": result_label,
        "pair_audit": pair_audit,
        "pair_files": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
            "valid": {"path": str(valid_path.resolve()), "sha256": sha256_file(valid_path)},
        },
        "checkpoint": checkpoint,
        "packages": versions,
        "git_revision": git_revision(repo),
        "training": {
            "beta": args.beta,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "loss_type": "sigmoid",
            "precompute_ref_log_probs": True,
            "bf16": True,
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "seed": args.seed,
            "smoke": args.smoke,
            "optimizer_train_rows": 2 if args.smoke else len(train_rows),
            "optimizer_valid_rows": 2 if args.smoke else len(valid_rows),
        },
    }
    if args.validate_only:
        print(json.dumps(static_manifest, indent=2, ensure_ascii=False))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO training; use --validate_only for CPU validation")

    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from fullpaper_backbone_utils import load_fullpaper_tokenizer, load_text_generation_model
    from trl import DPOConfig, DPOTrainer

    base_model_name = args.base_model or checkpoint["base_model"]
    tokenizer = load_fullpaper_tokenizer(args.sft_adapter_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = load_text_generation_model(
        base_model_name,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base,
        args.sft_adapter_dir,
        adapter_name="default",
        is_trainable=True,
        autocast_adapter_dtype=False,
    )
    model.config.use_cache = False
    policy_dtype_audit = cast_policy_adapter_dtype(model, torch.bfloat16)

    optimization_train_rows = train_rows[:2] if args.smoke else train_rows
    optimization_valid_rows = valid_rows[:2] if args.smoke else valid_rows
    train_dataset = Dataset.from_list(model_only_rows(optimization_train_rows))
    valid_dataset = Dataset.from_list(model_only_rows(optimization_valid_rows))
    collator = DPOPreferenceCollator(
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
    )
    dpo_args = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type="sigmoid",
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_prompt_length + args.max_completion_length,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args.precompute_ref_batch_size,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    started = time.monotonic()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    reference_audit = audit_adapter_state(trainer.model)
    reference_fingerprint_before = adapter_fingerprint(trainer.model, "ref")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = dict(static_manifest)
    preflight["policy_adapter_cast_before_reference_snapshot"] = policy_dtype_audit
    preflight["reference_snapshot"] = reference_audit
    preflight["reference_snapshot"]["sha256"] = reference_fingerprint_before
    (output_dir / "preflight_manifest.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    reference_fingerprint_after = adapter_fingerprint(trainer.model, "ref")
    if reference_fingerprint_after != reference_fingerprint_before:
        raise RuntimeError("Frozen reference adapter changed during training")
    final_manifest = dict(preflight)
    final_manifest["runtime"] = {
        "wall_clock_seconds": time.monotonic() - started,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(),
        "reference_sha256_after": reference_fingerprint_after,
        "reference_unchanged": True,
    }
    (output_dir / "training_manifest.json").write_text(
        json.dumps(final_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
