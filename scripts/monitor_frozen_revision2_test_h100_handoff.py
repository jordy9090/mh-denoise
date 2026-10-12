#!/usr/bin/env python3
"""Publish changed A100 TEST accepted snapshots to the H100 handoff directory."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path("/home/user/hsoh/mh-denoise")
RUN_DIR = REPO / "data/fullpaper_acl_pipeline/frozen_revision2_test_corruption_v1_20261008_a100"
SELECTION_DIR = REPO / "data/fullpaper_acl_pipeline/frozen_revision2_test_selection_v1_20261008"
STATE_DIR = RUN_DIR / "live_handoff_monitor"
REMOTE = "user@10.201.31.127"
REMOTE_DIR = "/home/user/hsoh/revision2_test_live_handoff_20261008"
REMOTE_PORT = "22"
TARGET_ACCEPTED = 100
BASELINE_ACCEPTED = 60
BASELINE_ACCEPTED_IDS_SHA256 = "19caa3a982cac5bb557e7eb286078d2529490e62642cc3137bc7b4e1b28949ff"
READY_SIGNAL = "READY_FOR_H100_INFERENCE_AND_JUDGE.json"


def now() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def remote_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def selection_order() -> dict[str, int]:
    payload = read_json(SELECTION_DIR / "test_generation_selection.json")
    rows = payload["rows"] if isinstance(payload, dict) else payload
    return {row["canonical_id"]: index for index, row in enumerate(rows)}


def latest_remote_accepted() -> int:
    try:
        result = subprocess.run(
            [
                "ssh",
                "-p",
                REMOTE_PORT,
                "-o",
                "BatchMode=yes",
                REMOTE,
                f"jq -r '.accepted_count // 0' {remote_quote(REMOTE_DIR + '/manifest.json')} 2>/dev/null || echo 0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip().splitlines()[-1])
    except Exception:
        return 0


def build_snapshot() -> tuple[str, dict[str, Any], dict[str, str]]:
    checkpoint = RUN_DIR / "results_checkpoint.part0.jsonl"
    run_contract = RUN_DIR / "run_contract.json"
    worker_status = RUN_DIR / "worker_status.part0.json"
    pause_state = RUN_DIR / "pause_after_accepted.json"
    production_log = RUN_DIR / "production.log"
    frozen_selection = SELECTION_DIR / "test_generation_selection.json"
    frozen_manifest = SELECTION_DIR / "manifest.json"
    rows = read_jsonl(checkpoint)
    order = selection_order()
    accepted_rows = sorted(
        [row for row in rows if row.get("status") == "accepted"],
        key=lambda row: order[row["canonical_id"]],
    )
    accepted_ids = [row["canonical_id"] for row in accepted_rows]
    accepted_ids_text = "".join(f"{item}\n" for item in accepted_ids)
    accepted_rows_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in accepted_rows)
    checkpoint_hash = sha256_file(checkpoint)
    snapshot_id = f"snapshot_accepted{len(accepted_ids):03d}_rows{len(rows):03d}_{dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y%m%dT%H%M%S%z')}"
    hashes = {
        "results_checkpoint.part0.jsonl": checkpoint_hash,
        "run_contract.json": sha256_file(run_contract),
        "worker_status.part0.json": sha256_file(worker_status) if worker_status.exists() else None,
        "pause_after_accepted.json": sha256_file(pause_state) if pause_state.exists() else None,
        "production.log": sha256_file(production_log),
        "frozen_selection.json": sha256_file(frozen_selection),
        "frozen_selection_manifest.json": sha256_file(frozen_manifest),
        "accepted_ids.txt": sha256_text(accepted_ids_text),
        "accepted_rows.jsonl": sha256_text(accepted_rows_text),
    }
    snapshot = {
        "snapshot_id": snapshot_id,
        "created_at": now(),
        "source_run_dir": str(RUN_DIR),
        "accepted_count": len(accepted_ids),
        "checkpoint_rows": len(rows),
        "checkpoint_sha256": checkpoint_hash,
        "accepted_ids_frozen_selection_order": accepted_ids,
        "source_file_hashes": hashes,
        "run_contract": read_json(run_contract),
        "frozen_selection_contract": read_json(frozen_manifest),
    }
    return snapshot_id, snapshot, {
        "accepted_ids.txt": accepted_ids_text,
        "accepted_rows.jsonl": accepted_rows_text,
    }


def stage(
    snapshot_id: str,
    snapshot: dict[str, Any],
    payloads: dict[str, str],
    *,
    ready: bool,
) -> tuple[Path, dict[str, Any]]:
    snapshots = STATE_DIR / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    copy_map = {
        "results_checkpoint.part0.jsonl": RUN_DIR / "results_checkpoint.part0.jsonl",
        "run_contract.json": RUN_DIR / "run_contract.json",
        "worker_status.part0.json": RUN_DIR / "worker_status.part0.json",
        "pause_after_accepted.json": RUN_DIR / "pause_after_accepted.json",
        "production.log": RUN_DIR / "production.log",
        "frozen_selection.json": SELECTION_DIR / "test_generation_selection.json",
        "frozen_selection_manifest.json": SELECTION_DIR / "manifest.json",
    }
    published: dict[str, str] = {}
    for suffix, source in copy_map.items():
        if not source.exists():
            continue
        target = snapshots / f"{snapshot_id}.{suffix}"
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, tmp)
        os.replace(tmp, target)
        published[target.name] = sha256_file(target)
    for suffix, text in payloads.items():
        target = snapshots / f"{snapshot_id}.{suffix}"
        write_text(target, text)
        published[target.name] = sha256_file(target)
    snapshot_file = snapshots / f"{snapshot_id}.snapshot.json"
    write_json(snapshot_file, {**snapshot, "published_files": published})
    published[snapshot_file.name] = sha256_file(snapshot_file)
    manifest = {
        "version": "a100-to-h100-live-handoff-v2",
        "status": "ready_for_h100_inference_and_judge" if ready else "collecting_to_target",
        "updated_at": now(),
        "target_accepted": TARGET_ACCEPTED,
        "current_snapshot_id": snapshot_id,
        "current_snapshot_file": f"snapshots/{snapshot_file.name}",
        "accepted_count": snapshot["accepted_count"],
        "checkpoint_rows": snapshot["checkpoint_rows"],
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "accepted_ids_frozen_selection_order": snapshot["accepted_ids_frozen_selection_order"],
        "published_files": published,
        "ready_signal_file": READY_SIGNAL,
    }
    write_json(STATE_DIR / "manifest.json", manifest)
    return snapshots, manifest


def publish(snapshots: Path, manifest: dict[str, Any]) -> None:
    run([
        "ssh",
        "-p",
        REMOTE_PORT,
        "-o",
        "BatchMode=yes",
        REMOTE,
        f"mkdir -p {remote_quote(REMOTE_DIR + '/snapshots')} {remote_quote(REMOTE_DIR + '/logs')}",
    ])
    run([
        "rsync",
        "-a",
        "-e",
        f"ssh -p {REMOTE_PORT} -o BatchMode=yes",
        str(snapshots) + "/",
        f"{REMOTE}:{REMOTE_DIR}/snapshots/",
    ])
    tmp = STATE_DIR / f"manifest.json.tmp.{os.getpid()}"
    write_json(tmp, manifest)
    remote_tmp = f"{REMOTE_DIR}/manifest.json.tmp.{os.getpid()}"
    run(["scp", "-P", REMOTE_PORT, "-o", "BatchMode=yes", str(tmp), f"{REMOTE}:{remote_tmp}"])
    run([
        "ssh",
        "-p",
        REMOTE_PORT,
        "-o",
        "BatchMode=yes",
        REMOTE,
        f"mv -f {remote_quote(remote_tmp)} {remote_quote(REMOTE_DIR + '/manifest.json')}",
    ])


def ready_payload(snapshot_id: str, snapshot: dict[str, Any]) -> dict[str, Any] | None:
    accepted_count = int(snapshot["accepted_count"])
    if accepted_count < TARGET_ACCEPTED:
        return None
    if accepted_count != TARGET_ACCEPTED:
        raise RuntimeError(
            f"Accepted count passed the exact target: {accepted_count} != {TARGET_ACCEPTED}"
        )

    pause_path = RUN_DIR / "pause_after_accepted.json"
    if not pause_path.exists():
        return None
    pause = read_json(pause_path)
    if not (
        pause.get("status") == "paused_after_atomic_checkpoint"
        and int(pause.get("pause_after_accepted") or 0) == TARGET_ACCEPTED
        and int(pause.get("accepted_terminal_rows") or 0) == TARGET_ACCEPTED
        and pause.get("checkpoint_atomic_write_complete") is True
        and pause.get("safe_row_boundary") is True
        and pause.get("worker_checkpoint_sha256") == snapshot["checkpoint_sha256"]
    ):
        return None

    accepted_ids = list(snapshot["accepted_ids_frozen_selection_order"])
    if len(accepted_ids) != TARGET_ACCEPTED or len(set(accepted_ids)) != TARGET_ACCEPTED:
        raise RuntimeError("The target snapshot does not contain 100 unique accepted IDs")
    baseline_ids_text = "".join(f"{item}\n" for item in accepted_ids[:BASELINE_ACCEPTED])
    if sha256_text(baseline_ids_text) != BASELINE_ACCEPTED_IDS_SHA256:
        raise RuntimeError("The frozen TEST60 prefix changed while extending to TEST100")
    additional_ids = accepted_ids[BASELINE_ACCEPTED:]
    additional_ids_text = "".join(f"{item}\n" for item in additional_ids)

    prefix = f"snapshots/{snapshot_id}"
    return {
        "version": "a100-test100-to-h100-action-request-v1",
        "status": "ready_for_h100_inference_and_judge",
        "ready_at": now(),
        "owner": "H100 app",
        "a100_must_not_start_h100_inference": True,
        "handoff_dir": REMOTE_DIR,
        "target": {
            "split": "test",
            "accepted_count": TARGET_ACCEPTED,
            "checkpoint_rows": snapshot["checkpoint_rows"],
            "snapshot_id": snapshot_id,
            "snapshot_file": f"{prefix}.snapshot.json",
            "checkpoint_file": f"{prefix}.results_checkpoint.part0.jsonl",
            "checkpoint_sha256": snapshot["checkpoint_sha256"],
            "accepted_rows_file": f"{prefix}.accepted_rows.jsonl",
            "accepted_rows_sha256": snapshot["source_file_hashes"]["accepted_rows.jsonl"],
            "accepted_ids_file": f"{prefix}.accepted_ids.txt",
            "accepted_ids_sha256": snapshot["source_file_hashes"]["accepted_ids.txt"],
            "run_contract_file": f"{prefix}.run_contract.json",
            "run_contract_sha256": snapshot["source_file_hashes"]["run_contract.json"],
            "frozen_selection_file": f"{prefix}.frozen_selection.json",
            "frozen_selection_sha256": snapshot["source_file_hashes"]["frozen_selection.json"],
        },
        "frozen_prefix": {
            "accepted_count": BASELINE_ACCEPTED,
            "accepted_ids_sha256": BASELINE_ACCEPTED_IDS_SHA256,
            "reuse_only_after_exact_id_and_hash_verification": True,
            "completed_final_selection": (
                "/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix/"
                "test60_final_outputs.jsonl"
            ),
            "completed_pointwise_evaluation": (
                "/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1/"
                "completion_summary.json"
            ),
        },
        "extension": {
            "accepted_count": len(additional_ids),
            "ordered_ids": additional_ids,
            "ordered_ids_sha256": sha256_text(additional_ids_text),
        },
        "requested_h100_stages": [
            "Verify the atomic TEST100 snapshot, every published SHA-256, and the frozen TEST60 prefix.",
            "Build a frozen ordered TEST100 export; reuse exact completed TEST60 outputs and run only the new 40 where hashes match.",
            "Run SFT, DPO, and Proposed inference with the same frozen TEST60 model, decoding, risk gate, and extraction contracts.",
            "Run the same two-order GPT-4.1 selector contract and freeze a separate TEST100 final-selection artifact.",
            "Run the same corrected GPT-4.1 pointwise rubric judge over TEST100 with exact request/result reuse where valid.",
            "Write TEST100 completion, coverage, quality, W/T/L, risk summaries, manifests, and hashes without modifying TEST60 artifacts.",
        ],
        "reference_contracts": {
            "test60_selection_contract": (
                "/home/user/hsoh/test60_rubric_aligned_final_selection_20261009_v2_test40_fix/"
                "run_contract.json"
            ),
            "test60_pointwise_contract": (
                "/home/user/hsoh/test60_corrected_gpt41_pointwise_publication_20261009_v1/"
                "run_contract.json"
            ),
            "selector_contract": "/home/user/hsoh/rubric_aligned_gpt41_rejudge_20261009_v1/run_contract.json",
        },
        "constraints": {
            "preserve_existing_test60": True,
            "separate_test100_output": True,
            "no_training": True,
            "no_rubric_change": True,
            "no_generation_or_judge_contract_change": True,
        },
    }


def publish_ready_signal(payload: dict[str, Any]) -> None:
    local_path = STATE_DIR / READY_SIGNAL
    write_json(local_path, payload)
    remote_tmp = f"{REMOTE_DIR}/{READY_SIGNAL}.tmp.{os.getpid()}"
    run(["scp", "-P", REMOTE_PORT, "-o", "BatchMode=yes", str(local_path), f"{REMOTE}:{remote_tmp}"])
    run([
        "ssh",
        "-p",
        REMOTE_PORT,
        "-o",
        "BatchMode=yes",
        REMOTE,
        f"mv -f {remote_quote(remote_tmp)} {remote_quote(REMOTE_DIR + '/' + READY_SIGNAL)}",
    ])


def monitor(interval_seconds: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATE_DIR / "handoff.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last_key: tuple[int, str, str] | None = None
        while True:
            try:
                snapshot_id, snapshot, payloads = build_snapshot()
                pause_hash = str(snapshot["source_file_hashes"].get("pause_after_accepted.json") or "")
                key = (
                    int(snapshot["accepted_count"]),
                    str(snapshot["checkpoint_sha256"]),
                    pause_hash,
                )
                if key != last_key and snapshot["accepted_count"] >= latest_remote_accepted():
                    action = ready_payload(snapshot_id, snapshot)
                    snapshots, manifest = stage(
                        snapshot_id,
                        snapshot,
                        payloads,
                        ready=action is not None,
                    )
                    publish(snapshots, manifest)
                    log(
                        f"published {snapshot_id} accepted={snapshot['accepted_count']} "
                        f"rows={snapshot['checkpoint_rows']}"
                    )
                    last_key = key
                    if action is not None:
                        publish_ready_signal(action)
                        log(f"published {READY_SIGNAL}; H100 action requested")
                        return
            except Exception as exc:
                log(f"monitor_error {type(exc).__name__}: {exc}")
            time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=60)
    args = parser.parse_args()
    monitor(args.interval_seconds)


if __name__ == "__main__":
    main()
