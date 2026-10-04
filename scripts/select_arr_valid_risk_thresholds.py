#!/usr/bin/env python3
"""Select ARR component risk thresholds from frozen VALID only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fullpaper_risk_contract import AXES, require_fullpaper_axis_order
from selective_risk_refinement_utils import read_jsonl, score_candidate


VERSION = "arr-valid-paired-threshold-selection-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_classifier(path: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
    require_fullpaper_axis_order(model.config)
    model.to(device).eval()
    return tokenizer, model


def metrics_at_threshold(scores: list[float], targets: list[int], threshold: float) -> dict[str, float | int]:
    predictions = [int(score > threshold) for score in scores]
    tp = sum(prediction == 1 and target == 1 for prediction, target in zip(predictions, targets))
    tn = sum(prediction == 0 and target == 0 for prediction, target in zip(predictions, targets))
    fp = sum(prediction == 1 and target == 0 for prediction, target in zip(predictions, targets))
    fn = sum(prediction == 0 and target == 1 for prediction, target in zip(predictions, targets))
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tpr
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "balanced_accuracy": (tpr + tnr) / 2,
        "f1": f1,
    }


def select_threshold(scores: list[float], targets: list[int], anchor: float = 0.35) -> dict[str, float | int]:
    if set(targets) != {0, 1}:
        raise RuntimeError("VALID threshold selection requires both positive and negative targets")
    unique = sorted(set(float(score) for score in scores))
    candidates = [unique[0] - 1e-12]
    candidates.extend((left + right) / 2 for left, right in zip(unique, unique[1:]))
    candidates.append(unique[-1] + 1e-12)
    evaluated = [metrics_at_threshold(scores, targets, threshold) for threshold in candidates]
    return max(
        evaluated,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - anchor),
            -float(row["threshold"]),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-valid", required=True)
    parser.add_argument("--router-valid", required=True)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--scorer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--anchor", type=float, default=0.35)
    parser.add_argument("--router-max-len", type=int, default=512)
    parser.add_argument("--risk-max-len", type=int, default=384)
    args = parser.parse_args()

    sft_path, router_path = Path(args.sft_valid), Path(args.router_valid)
    sft_rows, router_rows = read_jsonl(sft_path), read_jsonl(router_path)
    sft_by_id = {str(row["id"]): row for row in sft_rows}
    if len(sft_by_id) != len(sft_rows) or {str(row["canonical_id"]) for row in router_rows} != set(sft_by_id):
        raise RuntimeError("Frozen VALID SFT/Router membership mismatch")
    if any(
        not any(row["label_mask"].get(axis) and row["labels"].get(axis) == 1 for axis in AXES)
        for row in router_rows
    ):
        raise RuntimeError("Every selected candidate VALID target must have an observed positive Router axis")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router_tok, router = load_classifier(Path(args.router_dir), device)
    scorer_tok, scorer = load_classifier(Path(args.scorer_dir), device)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for router_row in router_rows:
            pair_id = str(router_row["canonical_id"])
            source = sft_by_id[pair_id]
            examples = (
                ("candidate", str(router_row["candidate_response"]), 1),
                ("reference", str(source["safe_response"]), 0),
            )
            for side, response, target in examples:
                scored = score_candidate(
                    str(source["question"]), response,
                    router, router_tok, scorer, scorer_tok, device,
                    router_max_len=args.router_max_len,
                    risk_max_len=args.risk_max_len,
                    component_mode="full",
                )
                scorer_max = max(
                    (float(value) for vector in scored["risk_vecs"] for value in vector),
                    default=0.0,
                )
                records.append({
                    "canonical_id": pair_id,
                    "side": side,
                    "target": target,
                    "scores": {
                        "full": float(scored["risk_score"]),
                        "without_router": scorer_max,
                        "without_scorer": max((float(value) for value in scored["g"]), default=0.0),
                    },
                })

    targets = [int(record["target"]) for record in records]
    selections = {
        mode: select_threshold([float(record["scores"][mode]) for record in records], targets, args.anchor)
        for mode in ("full", "without_router", "without_scorer")
    }
    payload = {
        "status": "selected_from_frozen_valid_only",
        "version": VERSION,
        "selection_rule": (
            "maximize paired VALID balanced accuracy, then F1, then proximity to inherited 0.35; "
            "classification uses risk_score > threshold"
        ),
        "target_contract": {
            "candidate": "positive only when frozen VALID Router has at least one active observed defect axis",
            "reference": "negative because every frozen pair passed the scoped reference suitability review",
            "clinical_expert_adjudication": False,
        },
        "test_accessed": False,
        "valid_pairs": len(router_rows),
        "calibration_examples": len(records),
        "class_counts": {"positive": sum(targets), "negative": len(targets) - sum(targets)},
        "thresholds": selections,
        "variant_mapping": {
            "mask_on": "full", "mask_off": "full",
            "without_router": "without_router", "without_scorer": "without_scorer",
        },
        "inputs": {
            "sft_valid": {"path": str(sft_path.resolve()), "sha256": sha256(sft_path)},
            "router_valid": {"path": str(router_path.resolve()), "sha256": sha256(router_path)},
            "router_dir": str(Path(args.router_dir).resolve()),
            "scorer_dir": str(Path(args.scorer_dir).resolve()),
        },
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
