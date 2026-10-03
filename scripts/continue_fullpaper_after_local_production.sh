#!/usr/bin/env bash
set -euo pipefail

REPO=/home/user/mh-denoise
PRODUCTION_PID=3248718
PRODUCTION_DIR="$REPO/data/fullpaper_acl_pipeline/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
CORRECTED_DIR="$REPO/data/fullpaper_acl_pipeline/development_training_shard_120_corrected_v2"
MERGED_DIR="$REPO/data/fullpaper_acl_pipeline/fullpaper_frozen_raw_qwen35_27b_v3_20260910"
TRAINING_DIR="$REPO/data/fullpaper_acl_pipeline/fullpaper_training_qwen35_27b_v3_frozen_20260910"
MAIN_OUT="$REPO/outputs/fullpaper_main_qwen35_27b_v3_seed20260910"
STATE="$REPO/data/fullpaper_acl_pipeline/fullpaper_main_continuation_state.json"
PY=/home/user/anaconda3/envs/mh-denoise/bin/python

cd "$REPO"
while kill -0 "$PRODUCTION_PID" 2>/dev/null; do
  sleep 30
done

PRODUCTION_DIR="$PRODUCTION_DIR" STATE="$STATE" "$PY" - <<'PY'
import json, os, time
from pathlib import Path
production = Path(os.environ["PRODUCTION_DIR"])
state = Path(os.environ["STATE"])
manifest_path = production / "production_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
payload = {
    "production_status": manifest.get("status", "missing_manifest"),
    "production_processed_inputs": manifest.get("processed_inputs"),
    "checked_at_unix": time.time(),
}
state.write_text(json.dumps(payload, indent=2) + "\n")
if manifest.get("status") != "complete" or manifest.get("processed_inputs") != 1200:
    raise SystemExit("Production did not reach the exact 1,200-input terminal state; training remains blocked")
PY

PYTHONPATH=scripts TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "$PY" scripts/freeze_fullpaper_main_data.py \
  --production-dir "$PRODUCTION_DIR" \
  --corrected-dir "$CORRECTED_DIR" \
  --merged-raw-dir "$MERGED_DIR" \
  --training-dir "$TRAINING_DIR" \
  --canonical-file data/fullpaper_acl_pipeline/canonical_clean_qa.jsonl \
  --tokenizer-dir /home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --expected-inputs 1200

PYTHONPATH=scripts:/mnt/ssd00/user-qwen35-transformers-kernels \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
"$PY" scripts/run_fullpaper_main_experiments.py \
  --data-dir "$TRAINING_DIR" \
  --output-dir "$MAIN_OUT" \
  --gpu 0
