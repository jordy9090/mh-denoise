#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/ssd00/hsoh/base_residual_train_20261011_v1
CODE=/home/user/hsoh/mh-denoise-base-residual-train-20261011/scripts
PY=/home/user/anaconda3/envs/mh-denoise/bin/python
export PYTHONUNBUFFERED=1
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
exec 9>"$ROOT/launch.lock"
flock -n 9
if [[ -n "${1:-}" ]]; then
    tail --pid="$1" -f /dev/null
fi
"$PY" "$CODE/run_base_residual_generation.py" train
"$PY" "$CODE/run_base_residual_experiment.py" pipeline
