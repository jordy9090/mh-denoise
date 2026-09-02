from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from dpo_utils import load_yaml, safe_slug, write_json


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _parse_floats(value: str | None) -> List[float]:
    if value is None:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_ints(value: str | None) -> List[int]:
    if value is None:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _command_text(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_runs(
    config: Mapping[str, Any],
    *,
    stage: str,
    beta_override: Sequence[float],
    seed_override: Sequence[int],
    selected_beta: float | None,
) -> List[Tuple[float, int]]:
    experiment = _section(config, "experiment")
    if stage == "beta":
        betas = list(beta_override) or [float(value) for value in experiment.get("beta_sweep", [0.03, 0.1, 0.3])]
        seeds = list(seed_override) or [int(experiment.get("screening_seed", 42))]
        if len(seeds) != 1:
            raise ValueError("Beta screening uses one fixed seed; provide exactly one seed")
    else:
        beta = selected_beta
        if beta is None:
            configured = experiment.get("selected_beta")
            if configured is not None:
                beta = float(configured)
        if beta is None:
            raise ValueError("Seed stage requires --selected_beta or experiment.selected_beta")
        betas = [float(beta)]
        seeds = list(seed_override) or [int(value) for value in experiment.get("final_seeds", [42, 43, 44])]
    return [(beta, seed) for beta in betas for seed in seeds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or execute validation beta and final seed sweeps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("beta", "seed"), required=True)
    parser.add_argument("--sft_adapter_dir", required=True)
    parser.add_argument("--betas", default=None, help="Comma-separated beta values")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds")
    parser.add_argument("--selected_beta", type=float, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    experiment = _section(config, "experiment")
    data = _section(config, "data")
    training = _section(config, "training")
    inference = _section(config, "inference")
    runs = build_runs(
        config,
        stage=args.stage,
        beta_override=_parse_floats(args.betas),
        seed_override=_parse_ints(args.seeds),
        selected_beta=args.selected_beta,
    )

    script_dir = Path(__file__).resolve().parent
    sweep_root = Path(str(training.get("sweep_root", "outputs/dpo_exp295")))
    pair_regime = safe_slug(experiment.get("pair_regime", "dpo"))
    valid_source = str(data.get("source_valid", ""))
    if not valid_source:
        raise ValueError("data.source_valid is required for validation inference")

    manifest_runs: List[Dict[str, Any]] = []
    for beta, seed in runs:
        run_dir = sweep_root / pair_regime / args.stage / f"beta_{beta:g}" / f"seed_{seed}"
        train_command = [
            args.python,
            str(script_dir / "train_dpo.py"),
            "--config",
            args.config,
            "--sft_adapter_dir",
            args.sft_adapter_dir,
            "--beta",
            str(beta),
            "--seed",
            str(seed),
            "--output_dir",
            str(run_dir),
        ]
        if args.resume_from_checkpoint:
            train_command.extend(["--resume_from_checkpoint", args.resume_from_checkpoint])

        valid_output = run_dir / "valid_inference.jsonl"
        inference_command = [
            args.python,
            str(script_dir / "run_inference.py"),
            "--input",
            valid_source,
            "--output",
            str(valid_output),
            "--base_model",
            str(_section(config, "model").get("base_model", "google/gemma-4-E4B-it")),
            "--adapter_dir",
            str(run_dir / "final"),
            "--method",
            f"DPO-{pair_regime}-beta{beta:g}-seed{seed}",
            "--max_source_len",
            str(inference.get("max_source_len", 512)),
            "--max_new_tokens",
            str(inference.get("max_new_tokens", 120)),
            "--temperature",
            str(inference.get("temperature", 0.0)),
            "--repetition_penalty",
            str(inference.get("repetition_penalty", 1.15)),
            "--no_repeat_ngram_size",
            str(inference.get("no_repeat_ngram_size", 4)),
            "--seed",
            str(seed),
        ]

        record: Dict[str, Any] = {
            "stage": args.stage,
            "pair_regime": pair_regime,
            "beta": beta,
            "seed": seed,
            "run_dir": str(run_dir),
            "train_command": train_command,
            "train_command_text": _command_text(train_command),
            "valid_inference_command": inference_command,
            "valid_inference_command_text": _command_text(inference_command),
            "status": "planned",
        }
        print("\n# TRAIN")
        print(record["train_command_text"])
        print("# VALID INFERENCE")
        print(record["valid_inference_command_text"])

        if args.execute:
            run_dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            subprocess.run(train_command, check=True)
            subprocess.run(inference_command, check=True)
            record["elapsed_seconds"] = time.perf_counter() - start
            record["status"] = "completed"
        manifest_runs.append(record)

    manifest = {
        "config": args.config,
        "stage": args.stage,
        "sft_adapter_dir": args.sft_adapter_dir,
        "execute": args.execute,
        "runs": manifest_runs,
        "selection_protocol": (
            "Use validation task metrics to select beta. Keep the test split sealed. "
            "After beta selection, run seeds 42/43/44 with the fixed pair dataset and fixed decoding."
        ),
    }
    manifest_path = sweep_root / pair_regime / f"{args.stage}_sweep_manifest.json"
    write_json(manifest, manifest_path)
    print(f"\nSaved sweep manifest: {manifest_path}")


if __name__ == "__main__":
    main()
