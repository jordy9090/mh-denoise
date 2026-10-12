#!/usr/bin/env python3
"""Verify pinned offline model snapshots and finalize the frozen length contract."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from huggingface_hub import list_repo_files, snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_snapshot(repo: str, revision: str, cache_dir: Path) -> dict[str, Any]:
    snapshot = Path(snapshot_download(
        repo_id=repo, revision=revision, cache_dir=cache_dir, local_files_only=True,
    )).resolve()
    if snapshot.name != revision:
        raise RuntimeError(f"Snapshot path does not match requested revision: {snapshot}")
    remote_files = sorted(list_repo_files(repo, revision=revision))
    missing = [name for name in remote_files if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete snapshot {repo}@{revision}: {missing[:10]}")
    empty = [name for name in remote_files if (snapshot / name).stat().st_size == 0]
    if empty:
        raise RuntimeError(f"Zero-byte snapshot files {repo}@{revision}: {empty[:10]}")

    config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer_files = [
        name for name in remote_files
        if Path(name).name in {"tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.txt"}
    ]
    if not tokenizer_files:
        raise RuntimeError(f"No tokenizer assets found in {repo}@{revision}")

    index_files = sorted(snapshot.glob("*.safetensors.index.json"))
    declared_shards = set()
    for index in index_files:
        payload = json.loads(index.read_text(encoding="utf-8"))
        declared_shards.update(payload.get("weight_map", {}).values())
    for shard in declared_shards:
        if not (snapshot / shard).is_file():
            raise RuntimeError(f"Missing declared weight shard: {snapshot / shard}")

    weight_files = sorted({
        path
        for pattern in ("*.safetensors", "*.bin", "*.h5", "*.msgpack")
        for path in snapshot.glob(pattern)
    })
    if not weight_files:
        raise RuntimeError(f"No model weight files found in {repo}@{revision}")
    safetensor_key_counts = {}
    for path in weight_files:
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as handle:
                safetensor_key_counts[path.name] = len(handle.keys())

    text_config = getattr(config, "text_config", config)
    context_limit = getattr(text_config, "max_position_embeddings", None)
    return {
        "repo": repo,
        "requested_revision": revision,
        "snapshot": str(snapshot),
        "snapshot_directory_matches_revision": True,
        "remote_file_count": len(remote_files),
        "missing_remote_files": 0,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_files": tokenizer_files,
        "config_sha256": sha256(snapshot / "config.json"),
        "context_limit": int(context_limit) if context_limit is not None else None,
        "weight_files": [
            {
                "name": str(path.relative_to(snapshot)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in weight_files
        ],
        "declared_safetensor_shards": sorted(declared_shards),
        "safetensor_key_counts": safetensor_key_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-repo", default="google/gemma-4-E4B-it")
    parser.add_argument("--gemma-revision", required=True)
    parser.add_argument("--bert-repo", default="google-bert/bert-base-uncased")
    parser.add_argument("--bert-revision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--length-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = Path(args.cache_dir).resolve()
    gemma = verify_snapshot(args.gemma_repo, args.gemma_revision, cache)
    bert = verify_snapshot(args.bert_repo, args.bert_revision, cache)
    length_path = Path(args.length_audit).resolve()
    length_audit = json.loads(length_path.read_text(encoding="utf-8"))
    selected = length_audit["selected_complete_lengths_before_context_check"]
    generation_budget = max(
        selected["generation_max_new_tokens"],
        selected["sft_and_denoiser_target_len"],
        selected["dpo_completion_len"],
    )
    context_limit = gemma["context_limit"]
    checks = {
        "sft_train_sequence": selected["sft_source_len"] + selected["sft_and_denoiser_target_len"],
        "sft_generation_sequence": selected["sft_source_len"] + generation_budget,
        "dpo_sequence": selected["dpo_prompt_len"] + selected["dpo_completion_len"],
    }
    if context_limit is None or any(value > context_limit for value in checks.values()):
        raise RuntimeError(f"Frozen complete sequence budget exceeds Gemma context: {checks} / {context_limit}")
    payload = {
        "status": "verified_complete_pinned_assets_and_length_contract",
        "verified_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True
        ).strip(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("huggingface-hub", "safetensors", "transformers")
        },
        "gemma": gemma,
        "bert": bert,
        "length_audit": {"path": str(length_path), "sha256": sha256(length_path)},
        "selected_lengths": {**selected, "generation_max_new_tokens": generation_budget},
        "context_checks": checks,
        "all_context_checks_within_limit": True,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace verification manifest: {output}")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
