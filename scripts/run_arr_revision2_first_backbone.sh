#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
data_dir="$repo_dir/data/fullpaper_acl_pipeline/production_accepted652_reuse_export_reviewed_v2_20261004"
run_dir="$repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
gemma_dir="/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
risk_base="/home/user/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"
seed=20260910

cd "$repo_dir"
missing_assets=0
for required in "$data_dir/frozen_manifest.json" "$gemma_dir/config.json" "$risk_base/config.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required offline asset: $required" >&2
    missing_assets=1
  fi
done
if [[ "$missing_assets" -ne 0 ]]; then exit 66; fi

active_gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_gpu_pids" ]]; then
  echo "GPU is not exclusive; refusing to share with active PIDs: $active_gpu_pids" >&2
  exit 75
fi

mkdir -p "$run_dir/logs"
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="scripts:/mnt/ssd00/user-qwen35-transformers-kernels"

run_step() {
  local name="$1"
  local marker="$2"
  shift 2
  if [[ -f "$marker" ]]; then
    echo "reuse completed step: $name ($marker)"
    return
  fi
  printf '%q ' "$@" >> "$run_dir/commands.log"
  printf '\n' >> "$run_dir/commands.log"
  "$@" 2>&1 | tee "$run_dir/logs/$name.log"
}

train_pairs="$(wc -l < "$data_dir/router_train.jsonl")"
scorer_rows="$(wc -l < "$data_dir/scorer_train_verified_spans.jsonl")"
router_steps="$((3 * ((train_pairs + 7) / 8)))"
scorer_steps="$((3 * ((scorer_rows + 7) / 8)))"
dpo_steps="$((3 * ((train_pairs + 15) / 16)))"

router_dir="$run_dir/router"
scorer_dir="$run_dir/scorer"
sft_dir="$run_dir/gemma/sft"
generation_dir="$run_dir/gemma/generation"
mkdir -p "$generation_dir"

run_step router_train "$router_dir/training_manifest.json" \
  "$python_bin" scripts/train_fullpaper_risk_model.py \
  --component router --train-file "$data_dir/router_train.jsonl" --valid-file "$data_dir/router_valid.jsonl" \
  --model "$risk_base" --initialization pretrained_base --output-dir "$router_dir" \
  --max-steps "$router_steps" --batch-size 8 --learning-rate 1e-5 --seed "$seed"

run_step scorer_train "$scorer_dir/training_manifest.json" \
  "$python_bin" scripts/train_fullpaper_risk_model.py \
  --component scorer --train-file "$data_dir/scorer_train_verified_spans.jsonl" \
  --valid-file "$data_dir/scorer_valid_verified_spans.jsonl" --model "$risk_base" \
  --initialization pretrained_base --output-dir "$scorer_dir" --max-steps "$scorer_steps" \
  --batch-size 8 --learning-rate 1e-5 --seed "$seed"

run_step gemma_sft_train "$sft_dir/training_manifest.json" \
  "$python_bin" scripts/train_professor_peft_refiner_textonly.py \
  --train_file "$data_dir/sft_train.jsonl" --valid_file "$data_dir/sft_valid.jsonl" \
  --output_dir "$sft_dir" --model "$gemma_dir" --max_source_len 512 --max_target_len 160 \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 --epochs 3 --max_steps -1 --lr 5e-5 \
  --warmup_ratio 0.03 --logging_steps 1 --eval_steps 25 --save_steps 100 --num_workers 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --prompt_style sft_plain --seed "$seed"

for split in train valid; do
  run_step "gemma_sft_${split}_generation" "$generation_dir/sft_${split}_outputs.manifest.json" \
    "$python_bin" scripts/build_sft_outputs_for_risk_tuning.py --base_model "$gemma_dir" \
    --adapter_dir "$sft_dir/final" --input "$data_dir/sft_${split}.jsonl" \
    --output "$generation_dir/sft_${split}_outputs.jsonl" --max_source_len 512 --max_new_tokens 512 \
    --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 --sft_prompt_style sft_plain
done

for mode in mask_on mask_off; do
  strategy="staged_risk"
  if [[ "$mode" == "mask_off" ]]; then strategy="no_mask"; fi
  proposed_dir="$run_dir/gemma/proposed_${mode}"
  run_step "gemma_${mode}_enrichment" "$proposed_dir/enrichment_manifest.json" \
    "$python_bin" scripts/train_gemma_risk_tune_from_sft.py --base_model "$gemma_dir" \
    --init_adapter_dir "$sft_dir/final" --train_file "$generation_dir/sft_train_outputs.jsonl" \
    --valid_file "$generation_dir/sft_valid_outputs.jsonl" --output_dir "$proposed_dir" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --zt_strategy "$strategy" --risk_threshold 0.35 --mask_threshold 0.35 --timestep 3 \
    --lambda_y 0 --risk_oversample_threshold 0.35 --risk_oversample_factor 2 --seed "$seed" --enrich_only
  for split in train valid; do
    audit_dir="$proposed_dir/prompt_audit_${split}"
    run_step "gemma_${mode}_prompt_audit_${split}" "$audit_dir/manifest.json" \
      "$python_bin" scripts/audit_selective_refinement_prompts.py \
      --input "$proposed_dir/risk_tune_${split}_enriched.jsonl" --output-dir "$audit_dir" \
      --tokenizer "$sft_dir/final" --limits 512 1280 2048 4096 8192
  done
done

context_limit="$($python_bin - "$gemma_dir" <<'PY'
import sys
from transformers import AutoConfig
config = AutoConfig.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
text = getattr(config, "text_config", config)
print(int(getattr(text, "max_position_embeddings")))
PY
)"
budget_file="$run_dir/gemma/complete_prompt_budget.json"
if [[ ! -f "$budget_file" ]]; then
  "$python_bin" scripts/select_complete_prompt_budget.py \
    --audit-manifest "$run_dir/gemma/proposed_mask_on/prompt_audit_train/manifest.json" \
    --audit-manifest "$run_dir/gemma/proposed_mask_on/prompt_audit_valid/manifest.json" \
    --audit-manifest "$run_dir/gemma/proposed_mask_off/prompt_audit_train/manifest.json" \
    --audit-manifest "$run_dir/gemma/proposed_mask_off/prompt_audit_valid/manifest.json" \
    --model-context-limit "$context_limit" --generation-budget 512 --output "$budget_file"
fi
max_source_len="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_max_source_len"])' "$budget_file")"

rendered_train="$run_dir/gemma/dpo_train_rendered.jsonl"
rendered_valid="$run_dir/gemma/dpo_valid_rendered.jsonl"
for split in train valid; do
  rendered_var="rendered_${split}"
  rendered="${!rendered_var}"
  run_step "gemma_render_${split}_dpo" "${rendered%.jsonl}.manifest.json" \
    "$python_bin" scripts/render_fullpaper_dpo_for_backbone.py --sft-file "$data_dir/sft_${split}.jsonl" \
    --dpo-file "$data_dir/dpo_${split}.jsonl" --tokenizer "$gemma_dir" --repo google/gemma-4-E4B-it \
    --revision ee0ef6023621cff504d758262d4e04895a5af4a2 --output "$rendered"
done

dpo_dir="$run_dir/gemma/dpo"
run_step gemma_dpo_train "$dpo_dir/training_manifest.json" \
  "$python_bin" scripts/train_dpo_minimal.py --data_contract fullpaper --fullpaper_valid_role valid \
  --train_file "$rendered_train" --valid_file "$rendered_valid" --sft_adapter_dir "$sft_dir/final" \
  --base_model "$gemma_dir" --output_dir "$dpo_dir" --max_steps "$dpo_steps" --beta 0.1 \
  --learning_rate 1e-6 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 --max_prompt_length 768 --max_completion_length 512 \
  --precompute_ref_batch_size 1 --logging_steps 1 --eval_steps 25 --save_steps 100 --seed "$seed"

run_step gemma_dpo_valid_generation "$generation_dir/dpo_valid_outputs.manifest.json" \
  "$python_bin" scripts/build_sft_outputs_for_risk_tuning.py --base_model "$gemma_dir" \
  --adapter_dir "$dpo_dir/final" --input "$data_dir/sft_valid.jsonl" \
  --output "$generation_dir/dpo_valid_outputs.jsonl" --max_source_len 512 --max_new_tokens 512 \
  --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 --sft_prompt_style sft_plain

for mode in mask_on mask_off; do
  strategy="staged_risk"
  if [[ "$mode" == "mask_off" ]]; then strategy="no_mask"; fi
  proposed_dir="$run_dir/gemma/proposed_${mode}"
  run_step "gemma_${mode}_train" "$proposed_dir/training_manifest.json" \
    "$python_bin" scripts/train_gemma_risk_tune_from_sft.py --base_model "$gemma_dir" \
    --init_adapter_dir "$sft_dir/final" --train_file "$proposed_dir/risk_tune_train_enriched.jsonl" \
    --valid_file "$proposed_dir/risk_tune_valid_enriched.jsonl" --output_dir "$proposed_dir" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --zt_strategy "$strategy" --inputs_pre_enriched --learning_rate 5e-6 --epochs 1 \
    --batch_size 1 --eval_batch_size 1 --grad_accum 8 --max_source_len "$max_source_len" \
    --max_target_len 160 --lambda_y 0 --risk_oversample_threshold 0.35 --risk_oversample_factor 2 \
    --risk_threshold 0.35 --mask_threshold 0.35 --eval_every 25 --save_every 100 \
    --num_workers 0 --enable_gradient_checkpointing --seed "$seed"
  run_step "gemma_${mode}_valid_generation" "$generation_dir/proposed_${mode}_valid_outputs.manifest.json" \
    "$python_bin" scripts/run_gemma_selective_risk_refinement.py --base_model "$gemma_dir" \
    --sft_adapter_dir "$sft_dir/final" --risk_adapter_dir "$proposed_dir/final" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --input "$generation_dir/sft_valid_outputs.jsonl" --output "$generation_dir/proposed_${mode}_valid_outputs.jsonl" \
    --reuse_sft_response --sft_response_field sft_response --zt_strategy "$strategy" \
    --risk_threshold 0.35 --gate_risk_threshold 0.35 --mask_threshold 0.35 \
    --max_source_len "$max_source_len" --max_new_tokens 512 --temperature 0.0 \
    --repetition_penalty 1.15 --no_repeat_ngram_size 4
done

echo "ARR Revision 2 first-backbone run complete"
