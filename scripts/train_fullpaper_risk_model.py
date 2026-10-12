#!/usr/bin/env python3
"""Train the current six-axis full-paper router or exact-span scorer."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fullpaper_risk_contract import (
    AXES,
    AXIS_TO_ID,
    ID_TO_AXIS,
    LEGACY_AXIS_MAP,
    ROUTER_LABEL_SEMANTICS,
    SCORER_LABEL_SEMANTICS,
    VERSION as RISK_CONTRACT_VERSION,
    require_fullpaper_axis_order,
    router_input_text,
    scorer_input_text,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_train_valid_provenance(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    require_full_provenance: bool,
) -> dict[str, Any]:
    fields = ("canonical_id", "question_normalized_sha256", "duplicate_cluster_id", "source_group_id")
    if require_full_provenance:
        missing = [
            (split, index, field)
            for split, rows in (("train", train_rows), ("valid", valid_rows))
            for index, row in enumerate(rows)
            for field in fields
            if not row.get(field)
        ]
        if missing:
            raise RuntimeError(f"Main-training rows lack required provenance: {missing[:5]}")
        if any(row.get("split") != "train" for row in train_rows):
            raise RuntimeError("Main router/scorer TRAIN file contains a non-TRAIN row")
        if any(row.get("split") != "valid" for row in valid_rows):
            raise RuntimeError("Main router/scorer VALID file contains a non-VALID row")
    overlap: dict[str, list[str]] = {}
    for field in fields:
        left = {row.get(field) for row in train_rows} - {None}
        right = {row.get(field) for row in valid_rows} - {None}
        overlap[field] = sorted(left & right)
    if any(overlap.values()):
        raise RuntimeError(f"Router/scorer TRAIN/VALID provenance overlap: {overlap}")
    return {"fields": list(fields), "overlap": overlap, "all_zero": True}


def make_text(component: str, row: dict[str, Any]) -> str:
    if component == "router":
        return router_input_text(row["question"], row["candidate_response"])
    return scorer_input_text(row["question"], row["span"])


class RiskDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, component: str, max_length: int):
        self.rows = []
        for row in rows:
            values = [row["labels"].get(axis) for axis in AXES]
            mask = [value is not None for value in values]
            if component == "scorer" and not any(mask):
                continue
            self.rows.append((row, values, mask))
        self.tokenizer = tokenizer
        self.component = component
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row, values, mask = self.rows[index]
        encoded = self.tokenizer(
            make_text(self.component, row), truncation=True, max_length=self.max_length
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": [0.0 if value is None else float(value) for value in values],
            "label_mask": [float(value) for value in mask],
        }


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        ids, attention = [], []
        for row in rows:
            padding = width - len(row["input_ids"])
            ids.append(row["input_ids"] + [self.pad_token_id] * padding)
            attention.append(row["attention_mask"] + [0] * padding)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor([row["labels"] for row in rows], dtype=torch.float),
            "label_mask": torch.tensor([row["label_mask"] for row in rows], dtype=torch.float),
        }


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_cell = F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
    return (per_cell * mask).sum() / mask.sum().clamp_min(1.0)


def initialize_risk_model(model_ref: str, initialization: str):
    if initialization == "pretrained_base":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_ref,
            local_files_only=True,
            num_labels=len(AXES),
            id2label=ID_TO_AXIS,
            label2id=AXIS_TO_ID,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        detail = "general pretrained backbone; newly initialized six-axis classification head"
    elif initialization == "legacy_checkpoint_smoke":
        model = AutoModelForSequenceClassification.from_pretrained(model_ref, local_files_only=True)
        if model.config.num_labels != len(AXES):
            raise RuntimeError(f"Expected six output labels in legacy smoke checkpoint; found {model.config.num_labels}")
        model.config.id2label = ID_TO_AXIS
        model.config.label2id = AXIS_TO_ID
        detail = "legacy classification checkpoint continuation; smoke/debug only, not main-training initialization"
    else:
        raise ValueError(f"Unsupported initialization mode: {initialization}")
    model.config.problem_type = "multi_label_classification"
    require_fullpaper_axis_order(model.config)
    return model, detail


@torch.no_grad()
def evaluate(model, loader, device) -> float | None:
    model.eval()
    losses = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        mask = batch.pop("label_mask").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(masked_loss(model(**inputs).logits, labels, mask).item()))
    model.train()
    return sum(losses) / len(losses) if losses else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("router", "scorer"), required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True, help="Base model or compatible prior checkpoint")
    parser.add_argument(
        "--initialization",
        choices=("pretrained_base", "legacy_checkpoint_smoke"),
        default="pretrained_base",
        help="Main training must use pretrained_base, which creates a fresh six-axis classification head. Legacy continuation is smoke-only.",
    )
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--validate-initialization-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_path, valid_path = Path(args.train_file), Path(args.valid_file)
    train_rows, valid_rows = read_jsonl(train_path), read_jsonl(valid_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model, initialization_detail = initialize_risk_model(args.model, args.initialization)
    if args.validate_initialization_only:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "initialization_validated_no_training",
            "component": args.component,
            "risk_contract_version": RISK_CONTRACT_VERSION,
            "initialization_mode": args.initialization,
            "initialization_detail": initialization_detail,
            "initial_model": str(Path(args.model).resolve()),
            "axis_order": list(AXES),
            "id2label": {str(index): axis for index, axis in ID_TO_AXIS.items()},
            "seed": args.seed,
            "optimizer_steps": 0,
        }
        (output / "initialization_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    provenance_audit = audit_train_valid_provenance(
        train_rows,
        valid_rows,
        require_full_provenance=args.initialization == "pretrained_base",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = RiskDataset(train_rows, tokenizer, args.component, args.max_length)
    valid_data = RiskDataset(valid_rows, tokenizer, args.component, args.max_length)
    if not train_data or not valid_data:
        raise RuntimeError("No explicitly supervised train or development rows")
    generator = torch.Generator().manual_seed(args.seed)
    collator = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collator,
        generator=generator, num_workers=0,
    )
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses, gradient_norms = [], []
    iterator = iter(train_loader)
    model.train()
    for step in range(1, args.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        labels = batch.pop("labels").to(device)
        mask = batch.pop("label_mask").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(**inputs).logits, labels, mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite {args.component} loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise RuntimeError(f"Invalid {args.component} gradient norm at step {step}: {gradient_norm}")
        optimizer.step()
        losses.append(float(loss.item()))
        gradient_norms.append(float(gradient_norm))
        print(json.dumps({"component": args.component, "step": step, "loss": losses[-1], "gradient_norm": gradient_norms[-1]}), flush=True)

    development_loss = evaluate(model, valid_loader, device)
    output = Path(args.output_dir)
    final = output / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    (final / "dims.json").write_text(json.dumps(list(AXES), indent=2) + "\n", encoding="utf-8")
    reloaded = AutoModelForSequenceClassification.from_pretrained(final, local_files_only=True)
    require_fullpaper_axis_order(reloaded.config)

    label_counts = {
        name: {
            axis: sum(row["labels"].get(axis) == value for row in train_rows)
            for axis in AXES
        }
        for name, value in (("positive", 1), ("negative", 0))
    }
    label_counts["unknown"] = {
        axis: sum(row["labels"].get(axis) is None for row in train_rows) for axis in AXES
    }
    manifest = {
        "status": "complete_training" if args.initialization == "pretrained_base" else "complete_training_smoke",
        "component": args.component,
        "axis_order": list(AXES),
        "risk_contract_version": RISK_CONTRACT_VERSION,
        "current_to_legacy_axis_map": LEGACY_AXIS_MAP,
        "label_semantics": ROUTER_LABEL_SEMANTICS if args.component == "router" else SCORER_LABEL_SEMANTICS,
        "initialization_mode": args.initialization,
        "initialization_detail": initialization_detail,
        "initial_model": str(Path(args.model).resolve()),
        "train_file": {"path": str(train_path.resolve()), "sha256": sha256(train_path), "source_rows": len(train_rows), "used_rows": len(train_data)},
        "development_file": {
            "path": str(valid_path.resolve()),
            "sha256": sha256(valid_path),
            "source_rows": len(valid_rows),
            "used_rows": len(valid_data),
            "policy": (
                "canonical VALID; provenance-disjoint from TRAIN"
                if args.initialization == "pretrained_base"
                else "TRAIN-origin smoke only; not paper VALID/TEST"
            ),
        },
        "train_valid_provenance_audit": provenance_audit,
        "max_steps": args.max_steps,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "development_loss": development_loss,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "label_counts_train_source_rows": label_counts,
        "runtime_seconds": time.monotonic() - started,
        "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "saved": str(final.resolve()),
        "reload_axis_mapping_ok": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
