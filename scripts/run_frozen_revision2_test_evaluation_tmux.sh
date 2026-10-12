#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
run_dir="$repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
generation_dir="$run_dir/gemma/generation_1536_test_v1_20261008"
evaluation_dir="$run_dir/evaluation_test_v1_20261008"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"
pipeline_log="$evaluation_dir/evaluation.log"
wait_log="$evaluation_dir/wait.log"

mkdir -p "$evaluation_dir"
cd "$repo_dir"

while [[ ! -f "$evaluation_dir/exact_response_prepare_manifest.json" ]] || \
      [[ "$(jq -r '.status // ""' "$evaluation_dir/exact_response_prepare_manifest.json" 2>/dev/null || true)" != \
          "complete_exact_response_pre_deduplication_no_api_calls" ]]; do
  printf '%s waiting_for_inference_and_exact_response_preparation\n' \
    "$(date --iso-8601=seconds)" >> "$wait_log"
  sleep 60
done

env PYTHONPATH=scripts:. "$python_bin" scripts/run_refinement_llm_judge_budgeted.py \
  --input "$evaluation_dir/judge_representatives.jsonl" \
  --output "$evaluation_dir/judged_representatives.jsonl" \
  --ledger "$evaluation_dir/gpt41mini_budget.sqlite3" \
  --manifest "$evaluation_dir/judge_cost_estimate.json" \
  --model gpt-4.1-mini --seed 20260910 --max_tokens 500 --max_retries 3 \
  --rubric_style json --input-usd-per-million 0.40 --output-usd-per-million 1.60 \
  --hard-max-usd 7.00 \
  --estimate-only >> "$pipeline_log" 2>&1

printf '%s paid_judge_cost_estimate_written_before_dispatch\n' \
  "$(date --iso-8601=seconds)" >> "$pipeline_log"

set -a
# shellcheck disable=SC1091
source "$repo_dir/.env"
set +a
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is unavailable after loading the existing local .env" >> "$pipeline_log"
  exit 81
fi

env PYTHONPATH=scripts:. "$python_bin" scripts/run_refinement_llm_judge_budgeted.py \
  --input "$evaluation_dir/judge_representatives.jsonl" \
  --output "$evaluation_dir/judged_representatives.jsonl" \
  --ledger "$evaluation_dir/gpt41mini_budget.sqlite3" \
  --manifest "$evaluation_dir/judge_manifest.json" \
  --model gpt-4.1-mini --seed 20260910 --max_tokens 500 --max_retries 3 \
  --rubric_style json --input-usd-per-million 0.40 --output-usd-per-million 1.60 \
  --hard-max-usd 7.00 \
  >> "$pipeline_log" 2>&1

env PYTHONPATH=scripts:. "$python_bin" scripts/exact_response_shared_judge.py expand \
  --members "$evaluation_dir/all_method_rows_with_sharing.jsonl" \
  --judged-representatives "$evaluation_dir/judged_representatives.jsonl" \
  --output "$evaluation_dir/judged_exact_response_shared.jsonl" \
  --manifest "$evaluation_dir/exact_response_expand_manifest.json" \
  >> "$pipeline_log" 2>&1

env PYTHONPATH=scripts:. "$python_bin" scripts/aggregate_refinement_judge_scores.py \
  --input "$evaluation_dir/judged_exact_response_shared.jsonl" \
  --output_csv "$evaluation_dir/method_scores_generic.csv" --group_by system \
  >> "$pipeline_log" 2>&1

env PYTHONPATH=scripts:. "$python_bin" scripts/summarize_frozen_revision2_test_evaluation.py \
  --sft "$generation_dir/sft_test_outputs.jsonl" \
  --dpo "$generation_dir/dpo_test_outputs.jsonl" \
  --proposed "$generation_dir/proposed_mask_off_test_outputs.jsonl" \
  --judged "$evaluation_dir/judged_exact_response_shared.jsonl" \
  --output-dir "$evaluation_dir" >> "$pipeline_log" 2>&1

printf '%s frozen_revision2_test_evaluation_complete\n' \
  "$(date --iso-8601=seconds)" >> "$pipeline_log"
