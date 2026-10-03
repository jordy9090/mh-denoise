#!/usr/bin/env python3
"""Run the frozen one-seed four-backbone SFT/DPO/Proposed experiment sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from fullpaper_risk_contract import AXES, PROVISIONAL_RISK_THRESHOLD


SEED = 20260910
RISK_BASE = "/home/user/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
BACKBONES = [
    {
        "name": "gemma",
        "repo": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "path": "/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2",
        "targets": "q_proj,k_proj,v_proj,o_proj",
    },
    {
        "name": "qwen",
        "repo": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "path": "/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "targets": "q_proj,k_proj,v_proj,o_proj",
    },
    {
        "name": "phi",
        "repo": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "path": "/home/user/.cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083",
        "targets": "qkv_proj,o_proj",
    },
    {
        "name": "ministral",
        "repo": "mistralai/Ministral-3-3B-Instruct-2512",
        "revision": "b35d4dfe56c142746f54dbd64f579faab2744308",
        "path": "/home/user/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/b35d4dfe56c142746f54dbd64f579faab2744308",
        "targets": "q_proj,k_proj,v_proj,o_proj",
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def run_step(name: str, command: list[str], marker: Path, logs: Path, env: dict[str, str]) -> None:
    if marker.exists():
        print(json.dumps({"step": name, "status": "reused", "marker": str(marker)}), flush=True)
        return
    log_path = logs / f"{name}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Step {name} failed with exit code {code}; see {log_path}")
    if not marker.exists():
        raise RuntimeError(f"Step {name} exited successfully without marker {marker}")
    print(json.dumps({"step": name, "status": "complete", "elapsed_seconds": time.monotonic() - started}), flush=True)


def main() -> None:
    run_started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    data = Path(args.data_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    frozen_manifest = data / "frozen_manifest.json"
    if not frozen_manifest.exists() or json.loads(frozen_manifest.read_text())["status"] != "frozen_complete":
        raise RuntimeError("A complete frozen data manifest is required")
    required = {
        name: data / name for name in (
            "sft_train.jsonl", "sft_valid.jsonl", "dpo_train.jsonl", "dpo_valid.jsonl",
            "router_train.jsonl", "router_valid.jsonl",
            "scorer_train_verified_spans.jsonl", "scorer_valid_verified_spans.jsonl",
        )
    }
    if any(not path.exists() for path in required.values()):
        raise RuntimeError("Frozen training directory is incomplete")
    counts = {name: count_rows(path) for name, path in required.items()}
    if min(counts.values()) < 1:
        raise RuntimeError(f"A required training/VALID contract is empty: {counts}")
    if any(not Path(model["path"]).is_dir() for model in BACKBONES) or not Path(RISK_BASE).is_dir():
        raise RuntimeError("One or more pinned local model snapshots are unavailable")

    router_steps = 3 * math.ceil(counts["router_train.jsonl"] / 8)
    scorer_steps = 3 * math.ceil(counts["scorer_train_verified_spans.jsonl"] / 8)
    dpo_steps = 3 * math.ceil(counts["dpo_train.jsonl"] / 16)
    config = {
        "status": "frozen_before_training",
        "seed": SEED,
        "data_dir": str(data),
        "data_manifest_sha256": sha256(frozen_manifest),
        "data_files": {name: {"path": str(path), "sha256": sha256(path), "rows": counts[name]} for name, path in required.items()},
        "axis_order": list(AXES),
        "risk_model_initialization": {
            "repo": "google-bert/bert-base-uncased",
            "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
            "path": RISK_BASE,
            "head": "new randomly initialized six-axis classification head",
        },
        "backbones": BACKBONES,
        "representative_first": "gemma",
        "checkpoint_selection": "use the final optimizer state for every backbone/method; no per-backbone cherry-picking",
        "settings": {
            "sft": {"epochs": 3, "batch": 1, "gradient_accumulation": 16, "lr": 5e-5, "max_source": 512, "max_target": 160, "lora_r": 8},
            "dpo": {"steps": dpo_steps, "epochs_equivalent": 3, "batch": 1, "gradient_accumulation": 16, "lr": 1e-6, "beta": 0.1, "reference": "frozen copy of same SFT adapter"},
            "router": {"steps": router_steps, "epochs_equivalent": 3, "batch": 8, "lr": 1e-5},
            "scorer": {"steps": scorer_steps, "epochs_equivalent": 3, "batch": 8, "lr": 1e-5, "exact_evidence_only": True},
            "proposed": {"epochs": 1, "batch": 1, "gradient_accumulation": 8, "lr": 5e-6, "risk_threshold": PROVISIONAL_RISK_THRESHOLD, "threshold_status": "fixed inherited setting for this first run; no tuning sweep"},
            "generation": {"max_new_tokens": 160, "temperature": 0.0, "repetition_penalty": 1.15, "no_repeat_ngram_size": 4, "enable_thinking": False},
        },
        "test_evaluation": "not_run",
        "external_paid_evaluation": "not_run",
    }
    config_path = output / "experiment_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise RuntimeError("Existing experiment configuration differs; refusing mixed restart")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": "scripts:/mnt/ssd00/user-qwen35-transformers-kernels",
    }
    py = sys.executable
    router_out, scorer_out = output / "router", output / "scorer"
    run_step("router_train", [
        py, "scripts/train_fullpaper_risk_model.py", "--component", "router",
        "--train-file", str(required["router_train.jsonl"]), "--valid-file", str(required["router_valid.jsonl"]),
        "--model", RISK_BASE, "--initialization", "pretrained_base", "--output-dir", str(router_out),
        "--max-steps", str(router_steps), "--batch-size", "8", "--learning-rate", "1e-5", "--seed", str(SEED),
    ], router_out / "training_manifest.json", logs, env)
    run_step("scorer_train", [
        py, "scripts/train_fullpaper_risk_model.py", "--component", "scorer",
        "--train-file", str(required["scorer_train_verified_spans.jsonl"]), "--valid-file", str(required["scorer_valid_verified_spans.jsonl"]),
        "--model", RISK_BASE, "--initialization", "pretrained_base", "--output-dir", str(scorer_out),
        "--max-steps", str(scorer_steps), "--batch-size", "8", "--learning-rate", "1e-5", "--seed", str(SEED),
    ], scorer_out / "training_manifest.json", logs, env)

    for backbone in BACKBONES:
        name, model = backbone["name"], backbone["path"]
        root = output / name
        sft, dpo, proposed = root / "sft", root / "dpo", root / "proposed"
        generations = root / "generation"
        generations.mkdir(parents=True, exist_ok=True)
        rendered_train_dpo = root / "dpo_train_backbone_rendered.jsonl"
        rendered_valid_dpo = root / "dpo_valid_backbone_rendered.jsonl"
        for split, sft_source, dpo_source, rendered_dpo in (
            ("train", required["sft_train.jsonl"], required["dpo_train.jsonl"], rendered_train_dpo),
            ("valid", required["sft_valid.jsonl"], required["dpo_valid.jsonl"], rendered_valid_dpo),
        ):
            run_step(f"{name}_render_{split}_dpo", [
                py, "scripts/render_fullpaper_dpo_for_backbone.py",
                "--sft-file", str(sft_source), "--dpo-file", str(dpo_source),
                "--tokenizer", model, "--repo", backbone["repo"], "--revision", backbone["revision"],
                "--output", str(rendered_dpo),
            ], rendered_dpo.with_suffix(".manifest.json"), logs, env)
        sft_final = sft / "final"
        run_step(f"{name}_sft_train", [
            py, "scripts/train_professor_peft_refiner_textonly.py",
            "--train_file", str(required["sft_train.jsonl"]), "--valid_file", str(required["sft_valid.jsonl"]),
            "--output_dir", str(sft), "--model", model,
            "--max_source_len", "512", "--max_target_len", "160", "--batch_size", "1", "--eval_batch_size", "1",
            "--grad_accum", "16", "--epochs", "3", "--max_steps", "-1", "--lr", "5e-5", "--warmup_ratio", "0.03",
            "--logging_steps", "5", "--eval_steps", "25", "--save_steps", "100", "--num_workers", "0",
            "--target_modules", backbone["targets"], "--lora_r", "8", "--lora_alpha", "16", "--lora_dropout", "0.05",
            "--prompt_style", "sft_plain",
        ], sft / "training_manifest.json", logs, env)
        sft_train_outputs = generations / "sft_train_outputs.jsonl"
        sft_valid_outputs = generations / "sft_valid_outputs.jsonl"
        for split, input_path, generated in (
            ("train", required["sft_train.jsonl"], sft_train_outputs),
            ("valid", required["sft_valid.jsonl"], sft_valid_outputs),
        ):
            run_step(f"{name}_sft_{split}_generation", [
                py, "scripts/build_sft_outputs_for_risk_tuning.py", "--base_model", model,
                "--adapter_dir", str(sft_final), "--input", str(input_path), "--output", str(generated),
                "--max_source_len", "512", "--max_new_tokens", "160", "--temperature", "0.0",
                "--repetition_penalty", "1.15", "--no_repeat_ngram_size", "4", "--sft_prompt_style", "sft_plain",
            ], generated.with_suffix(".manifest.json"), logs, env)

        run_step(f"{name}_dpo_train", [
            py, "scripts/train_dpo_minimal.py", "--data_contract", "fullpaper", "--fullpaper_valid_role", "valid",
            "--train_file", str(rendered_train_dpo), "--valid_file", str(rendered_valid_dpo),
            "--sft_adapter_dir", str(sft_final), "--base_model", model, "--output_dir", str(dpo),
            "--max_steps", str(dpo_steps), "--beta", "0.1", "--learning_rate", "1e-6",
            "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1", "--gradient_accumulation_steps", "16",
            "--max_prompt_length", "768", "--max_completion_length", "512", "--precompute_ref_batch_size", "1",
            "--logging_steps", "5", "--eval_steps", "25", "--save_steps", "100", "--seed", str(SEED),
        ], dpo / "training_manifest.json", logs, env)
        dpo_valid_outputs = generations / "dpo_valid_outputs.jsonl"
        run_step(f"{name}_dpo_valid_generation", [
            py, "scripts/build_sft_outputs_for_risk_tuning.py", "--base_model", model,
            "--adapter_dir", str(dpo / "final"), "--input", str(required["sft_valid.jsonl"]), "--output", str(dpo_valid_outputs),
            "--max_source_len", "512", "--max_new_tokens", "160", "--temperature", "0.0",
            "--repetition_penalty", "1.15", "--no_repeat_ngram_size", "4", "--sft_prompt_style", "sft_plain",
        ], dpo_valid_outputs.with_suffix(".manifest.json"), logs, env)

        run_step(f"{name}_proposed_train", [
            py, "scripts/train_gemma_risk_tune_from_sft.py", "--base_model", model,
            "--init_adapter_dir", str(sft_final), "--train_file", str(sft_train_outputs), "--valid_file", str(sft_valid_outputs),
            "--output_dir", str(proposed), "--router_dir", str(router_out / "final"), "--risk_scorer_dir", str(scorer_out / "final"),
            "--risk_contract", "fullpaper_v1", "--zt_strategy", "staged_risk", "--learning_rate", "5e-6",
            "--epochs", "1", "--batch_size", "1", "--eval_batch_size", "1", "--grad_accum", "8",
            "--max_source_len", "512", "--max_target_len", "160", "--risk_threshold", str(PROVISIONAL_RISK_THRESHOLD),
            "--mask_threshold", str(PROVISIONAL_RISK_THRESHOLD), "--eval_every", "25", "--save_every", "100",
            "--num_workers", "0", "--enable_gradient_checkpointing",
        ], proposed / "training_manifest.json", logs, env)
        proposed_valid_outputs = generations / "proposed_valid_outputs.jsonl"
        run_step(f"{name}_proposed_valid_generation", [
            py, "scripts/run_gemma_selective_risk_refinement.py", "--base_model", model,
            "--sft_adapter_dir", str(sft_final), "--risk_adapter_dir", str(proposed / "final"),
            "--router_dir", str(router_out / "final"), "--risk_scorer_dir", str(scorer_out / "final"),
            "--risk_contract", "fullpaper_v1", "--input", str(sft_valid_outputs), "--output", str(proposed_valid_outputs),
            "--reuse_sft_response", "--sft_response_field", "sft_response", "--zt_strategy", "staged_risk",
            "--risk_threshold", str(PROVISIONAL_RISK_THRESHOLD), "--gate_risk_threshold", str(PROVISIONAL_RISK_THRESHOLD),
            "--mask_threshold", str(PROVISIONAL_RISK_THRESHOLD), "--max_source_len", "512", "--max_new_tokens", "160",
            "--temperature", "0.0", "--repetition_penalty", "1.15", "--no_repeat_ngram_size", "4",
        ], proposed_valid_outputs.with_suffix(".manifest.json"), logs, env)
        comparison = root / "valid_output_comparison.jsonl"
        run_step(f"{name}_valid_compare", [
            py, "scripts/compare_fullpaper_valid_outputs.py", "--sft", str(sft_valid_outputs),
            "--dpo", str(dpo_valid_outputs), "--proposed", str(proposed_valid_outputs), "--output", str(comparison),
        ], comparison.with_suffix(".manifest.json"), logs, env)

    rows = []
    for backbone in BACKBONES:
        root = output / backbone["name"]
        sft_manifest = json.loads((root / "sft/training_manifest.json").read_text())
        dpo_manifest = json.loads((root / "dpo/training_manifest.json").read_text())
        proposed_manifest = json.loads((root / "proposed/training_manifest.json").read_text())
        inference_manifest = json.loads((root / "generation/proposed_valid_outputs.manifest.json").read_text())
        rows.append({
            "backbone": backbone["name"],
            "sft_optimizer_steps": sft_manifest["optimizer_steps"],
            "dpo_optimizer_steps": dpo_manifest["training"]["max_steps"],
            "dpo_reference_unchanged": dpo_manifest["runtime"]["reference_unchanged"],
            "proposed_optimizer_steps": proposed_manifest["optimizer_steps"],
            "valid_rows": inference_manifest["output"]["rows"],
            "proposed_denoiser_called": inference_manifest["denoiser_called"],
            "proposed_denoiser_accepted": inference_manifest["denoiser_accepted"],
            "sft_runtime_seconds": sft_manifest["runtime_seconds"],
            "dpo_runtime_seconds": dpo_manifest["runtime"]["wall_clock_seconds"],
            "proposed_runtime_seconds": proposed_manifest["runtime_seconds"],
            "note": "structural/training status only; not a counseling-quality ranking",
        })
    result_payload = {
        "status": "complete_structural_results_no_test_or_external_quality_eval",
        "rows": rows,
    }
    (output / "results_table.json").write_text(json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Full-paper first-run status table",
        "",
        "These are execution and routing outcomes, not counseling-quality rankings.",
        "",
        "| Backbone | SFT steps | DPO steps | Frozen ref | Proposed steps | VALID rows | Denoiser called | Denoiser accepted |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['backbone']} | {row['sft_optimizer_steps']} | {row['dpo_optimizer_steps']} | "
            f"{row['dpo_reference_unchanged']} | {row['proposed_optimizer_steps']} | {row['valid_rows']} | "
            f"{row['proposed_denoiser_called']} | {row['proposed_denoiser_accepted']} |"
        )
    (output / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "status": "complete_four_backbone_one_seed",
        "config_sha256": sha256(config_path),
        "backbone_status": {model["name"]: "complete" for model in BACKBONES},
        "test_evaluation": "not_run",
        "external_paid_evaluation": "not_run",
        "total_runtime_seconds": time.monotonic() - run_started,
        "results_table": str((output / "results_table.json").resolve()),
        "completed_at_unix": time.time(),
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
