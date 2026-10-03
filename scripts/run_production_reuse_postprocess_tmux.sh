#!/usr/bin/env bash
set -euo pipefail

repo_dir=/home/user/hsoh/mh-denoise
base_dir="$repo_dir/data/fullpaper_acl_pipeline"
production_dir="$base_dir/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910"
rejudge_dir="$base_dir/production_accepted652_reuse_rejudge_full_v1_20261001"
interim_dir="$base_dir/production_accepted652_reuse_export_pre_regen_v1_20261001"
regen_dir="$base_dir/production_accepted652_candidate_regen_v1_20261001"
final_dir="$base_dir/production_accepted652_reuse_export_final_v1_20261001"
post_log="$rejudge_dir/postprocess.log"
prior_ledger="$base_dir/production_run_train1000_valid200_local_qwen35_27b_v3r1_20260910_provisional_corrected_v2_20260926/review_ledger.jsonl"
codex_ledger="$base_dir/codex_clean_candidate_review_batch20_v2_20260926/cumulative_ai_review_ledger.jsonl"
tokenizer_dir="$repo_dir/outputs/models/fullpaper_dev57_span_scorer_smoke2/final"
generator_dir=/home/user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
judge_dir=/mnt/ssd00/user-qwen35-27b-hf/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654

cd "$repo_dir"
while [[ ! -f "$rejudge_dir/manifest.json" ]] || \
      [[ "$(jq -r '.status // ""' "$rejudge_dir/manifest.json" 2>/dev/null || true)" != complete ]]; do
  date --iso-8601=seconds >> "$post_log"
  echo waiting_for_completed_full_rejudge >> "$post_log"
  sleep 60
done

PYTHONPATH=scripts:. .venv-fullpaper/bin/python scripts/build_production_reuse_export.py \
  --production-dir "$production_dir" \
  --prior-review-ledger "$prior_ledger" \
  --codex-review-ledger "$codex_ledger" \
  --local-rejudge-checkpoint "$rejudge_dir/rejudged_checkpoint.jsonl" \
  --output-dir "$interim_dir" \
  --tokenizer-dir "$tokenizer_dir" \
  --max-length 512 >> "$post_log" 2>&1

PYTHONPATH=scripts:. .venv-fullpaper/bin/python scripts/prepare_production_candidate_regeneration.py \
  --production-dir "$production_dir" \
  --reuse-export-dir "$interim_dir" \
  --output-dir "$regen_dir" >> "$post_log" 2>&1

regen_count=$(jq '.rows | length' "$regen_dir/selection.json")
if (( regen_count > 0 )); then
  while true; do
    gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    if [[ "$gpu_used_mib" =~ ^[0-9]+$ ]] && (( gpu_used_mib <= 11000 )); then
      break
    fi
    date --iso-8601=seconds >> "$post_log"
    echo "waiting_for_70000_MiB_free_before_regeneration gpu_used_mib=$gpu_used_mib" >> "$post_log"
    sleep 60
  done
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=scripts:. \
    .venv-fullpaper/bin/python -u scripts/run_fullpaper_corruption_production.py \
    --selection-file "$regen_dir/selection.json" \
    --max-inputs "$regen_count" \
    --splits train,valid \
    --known-holds-file "$base_dir/development_training_shard_120_corrected_v2/held_out.jsonl" \
    --output-dir "$regen_dir" \
    --worker-index 0 \
    --worker-count 1 \
    --selected-gpu 0 \
    --required-free-vram-mib 70000 \
    --max-input-tokens 4096 \
    --generator-model-dir "$generator_dir" \
    --budget-ledger "$regen_dir/no_paid_api_budget.sqlite3" \
    --max-api-requests 0 \
    --max-api-usd 0 \
    --input-usd-per-million-tokens 0 \
    --output-usd-per-million-tokens 0 \
    --budget-max-input-tokens-per-request 65536 \
    --judge-backend local-qwen35-27b \
    --local-judge-model-dir "$judge_dir" \
    --local-judge-max-new-tokens 3200 \
    --local-judge-timeout-seconds 300 \
    --automatic-clean-preflight >> "$post_log" 2>&1
fi

final_args=(
  --production-dir "$production_dir"
  --prior-review-ledger "$prior_ledger"
  --codex-review-ledger "$codex_ledger"
  --local-rejudge-checkpoint "$rejudge_dir/rejudged_checkpoint.jsonl"
  --output-dir "$final_dir"
  --tokenizer-dir "$tokenizer_dir"
  --max-length 512
)
if (( regen_count > 0 )); then
  final_args+=(--replacement-production-dir "$regen_dir")
fi
PYTHONPATH=scripts:. .venv-fullpaper/bin/python scripts/build_production_reuse_export.py \
  "${final_args[@]}" >> "$post_log" 2>&1

PYTHONPATH=scripts:. .venv-fullpaper/bin/python scripts/summarize_production_reuse_final.py \
  --rejudge-dir "$rejudge_dir" \
  --regeneration-dir "$regen_dir" \
  --final-export-dir "$final_dir" >> "$post_log" 2>&1

date --iso-8601=seconds >> "$post_log"
echo postprocess_complete >> "$post_log"
