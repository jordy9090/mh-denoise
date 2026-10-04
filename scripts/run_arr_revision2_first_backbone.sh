#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/user/hsoh/mh-denoise"
data_dir="$repo_dir/data/fullpaper_acl_pipeline/production_accepted652_reuse_export_reviewed_v2_20261004"
length_audit="$repo_dir/data/fullpaper_acl_pipeline/arr_first_run_20261004/frozen_sequence_length_audit.json"
asset_verification="$repo_dir/data/fullpaper_acl_pipeline/arr_first_run_20261004/model_assets_verified.json"
run_dir="$repo_dir/outputs/fullpaper_acl/arr_revision2_gemma_first_seed20260910"
gemma_dir="/home/user/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
risk_base="/home/user/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
python_bin="/home/user/anaconda3/envs/mh-denoise/bin/python"
seed=20260910

cd "$repo_dir"
missing_assets=0
for required in "$data_dir/frozen_manifest.json" "$length_audit" "$asset_verification" "$gemma_dir/config.json" "$risk_base/config.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required offline asset: $required" >&2
    missing_assets=1
  fi
done
if [[ "$missing_assets" -ne 0 ]]; then exit 66; fi
"$python_bin" - "$asset_verification" "$gemma_dir" "$risk_base" <<'PY'
import json
import sys
from pathlib import Path

payload = json.load(open(sys.argv[1], encoding="utf-8"))
expected = "verified_complete_pinned_assets_and_length_contract"
if payload.get("status") != expected or not payload.get("all_context_checks_within_limit"):
    raise SystemExit(f"unverified model/length contract: {sys.argv[1]}")
expected_assets = (
    ("gemma", "ee0ef6023621cff504d758262d4e04895a5af4a2", sys.argv[2]),
    ("bert", "86b5e0934494bd15c9632b12f734a8a67f723594", sys.argv[3]),
)
for name, revision, path in expected_assets:
    record = payload.get(name) or {}
    if record.get("requested_revision") != revision:
        raise SystemExit(f"wrong verified {name} revision")
    if Path(record.get("snapshot", "")).resolve() != Path(path).resolve():
        raise SystemExit(f"wrong verified {name} snapshot path")
PY
"$python_bin" - "$data_dir" "$length_audit" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

data_dir = Path(sys.argv[1])
length_audit = json.load(open(sys.argv[2], encoding="utf-8"))
manifest = json.load(open(data_dir / "frozen_manifest.json", encoding="utf-8"))
if manifest.get("status") != "frozen_complete":
    raise SystemExit("frozen data manifest is not complete")
if manifest.get("splits", {}).get("train", {}).get("pairs") != 276:
    raise SystemExit("frozen TRAIN membership changed")
if manifest.get("splits", {}).get("valid", {}).get("pairs") != 43:
    raise SystemExit("frozen VALID membership changed")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

for name, expected in manifest["files"].items():
    path = data_dir / name
    if not path.is_file():
        raise SystemExit(f"missing frozen file: {path}")
    rows = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    if rows != expected["rows"] or sha256(path) != expected["sha256"]:
        raise SystemExit(f"frozen file changed: {path}")
for name, expected in length_audit["inputs"].items():
    if sha256(data_dir / name) != expected["sha256"]:
        raise SystemExit(f"length audit input changed: {name}")
PY

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

require_generation_complete() {
  local manifest="$1"
  local kind="$2"
  "$python_bin" - "$manifest" "$kind" <<'PY'
import json
import sys

path, kind = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
completion = payload["generation_completion"]
if kind == "single":
    truncated = int(completion["source_input_truncated"])
    limited = int(completion["length_limit_reached"])
else:
    truncated = int(completion["denoiser_source_input_truncated"])
    limited = int(completion["denoiser_length_limit_reached"])
if truncated:
    raise SystemExit(f"source truncation detected in {path}: {truncated}")
if limited:
    raise SystemExit(
        f"generation budget reached in {path}: {limited}; stop before downstream comparison "
        "and rerun every compared method with one larger common budget"
    )
PY
}

train_pairs="$(wc -l < "$data_dir/router_train.jsonl")"
scorer_rows="$(wc -l < "$data_dir/scorer_train_verified_spans.jsonl")"
router_steps="$((3 * ((train_pairs + 7) / 8)))"
scorer_steps="$((3 * ((scorer_rows + 7) / 8)))"
dpo_steps="$((3 * ((train_pairs + 15) / 16)))"
sft_source_len="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_complete_lengths_before_context_check"]["sft_source_len"])' "$length_audit")"
target_len="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_complete_lengths_before_context_check"]["sft_and_denoiser_target_len"])' "$length_audit")"
dpo_prompt_len="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_complete_lengths_before_context_check"]["dpo_prompt_len"])' "$length_audit")"
dpo_completion_len="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_complete_lengths_before_context_check"]["dpo_completion_len"])' "$length_audit")"
generation_budget="$($python_bin -c 'import json,sys; x=json.load(open(sys.argv[1]))["selected_complete_lengths_before_context_check"]; print(max(x["generation_max_new_tokens"],x["sft_and_denoiser_target_len"],x["dpo_completion_len"]))' "$length_audit")"

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

threshold_file="$run_dir/valid_risk_thresholds.json"
run_step valid_risk_threshold_selection "$threshold_file" \
  "$python_bin" scripts/select_arr_valid_risk_thresholds.py \
  --sft-valid "$data_dir/sft_valid.jsonl" --router-valid "$data_dir/router_valid.jsonl" \
  --router-dir "$router_dir/final" --scorer-dir "$scorer_dir/final" \
  --output "$threshold_file" --anchor 0.35

full_threshold="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["thresholds"]["full"]["threshold"])' "$threshold_file")"
without_router_threshold="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["thresholds"]["without_router"]["threshold"])' "$threshold_file")"
without_scorer_threshold="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1]))["thresholds"]["without_scorer"]["threshold"])' "$threshold_file")"

run_step gemma_sft_train "$sft_dir/training_manifest.json" \
  "$python_bin" scripts/train_professor_peft_refiner_textonly.py \
  --train_file "$data_dir/sft_train.jsonl" --valid_file "$data_dir/sft_valid.jsonl" \
  --output_dir "$sft_dir" --model "$gemma_dir" --max_source_len "$sft_source_len" --max_target_len "$target_len" \
  --batch_size 1 --eval_batch_size 1 --grad_accum 16 --epochs 3 --max_steps -1 --lr 5e-5 \
  --warmup_ratio 0.03 --logging_steps 1 --eval_steps 25 --save_steps 100 --num_workers 0 \
  --target_modules q_proj,k_proj,v_proj,o_proj --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
  --prompt_style sft_plain --seed "$seed"

for split in train valid; do
  run_step "gemma_sft_${split}_generation" "$generation_dir/sft_${split}_outputs.manifest.json" \
    "$python_bin" scripts/build_sft_outputs_for_risk_tuning.py --base_model "$gemma_dir" \
    --adapter_dir "$sft_dir/final" --input "$data_dir/sft_${split}.jsonl" \
    --output "$generation_dir/sft_${split}_outputs.jsonl" --max_source_len "$sft_source_len" --max_new_tokens "$generation_budget" \
    --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 --sft_prompt_style sft_plain
  require_generation_complete "$generation_dir/sft_${split}_outputs.manifest.json" single
done

variants=(mask_on mask_off without_router without_scorer)
for mode in "${variants[@]}"; do
  strategy="staged_risk"
  component_mode="full"
  threshold="$full_threshold"
  if [[ "$mode" == "mask_off" ]]; then strategy="no_mask"; fi
  if [[ "$mode" == "without_router" ]]; then component_mode="without_router"; threshold="$without_router_threshold"; fi
  if [[ "$mode" == "without_scorer" ]]; then strategy="no_mask"; component_mode="without_scorer"; threshold="$without_scorer_threshold"; fi
  proposed_dir="$run_dir/gemma/proposed_${mode}"
  run_step "gemma_${mode}_enrichment" "$proposed_dir/enrichment_manifest.json" \
    "$python_bin" scripts/train_gemma_risk_tune_from_sft.py --base_model "$gemma_dir" \
    --init_adapter_dir "$sft_dir/final" --train_file "$generation_dir/sft_train_outputs.jsonl" \
    --valid_file "$generation_dir/sft_valid_outputs.jsonl" --output_dir "$proposed_dir" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --component_mode "$component_mode" \
    --zt_strategy "$strategy" --risk_threshold "$threshold" --mask_threshold "$threshold" --timestep 3 \
    --lambda_y 0 --risk_oversample_threshold "$threshold" --risk_oversample_factor 2 --seed "$seed" --enrich_only
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
if (( sft_source_len + target_len > context_limit )); then
  echo "SFT full source+target exceeds model context" >&2
  exit 67
fi
if (( dpo_prompt_len + dpo_completion_len > context_limit )); then
  echo "DPO full prompt+completion exceeds model context" >&2
  exit 67
fi
budget_file="$run_dir/gemma/complete_prompt_budget.json"
if [[ ! -f "$budget_file" ]]; then
  budget_args=()
  for mode in "${variants[@]}"; do
    budget_args+=(--audit-manifest "$run_dir/gemma/proposed_${mode}/prompt_audit_train/manifest.json")
    budget_args+=(--audit-manifest "$run_dir/gemma/proposed_${mode}/prompt_audit_valid/manifest.json")
  done
  "$python_bin" scripts/select_complete_prompt_budget.py "${budget_args[@]}" \
    --model-context-limit "$context_limit" --generation-budget "$generation_budget" --output "$budget_file"
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
  --gradient_accumulation_steps 16 --max_prompt_length "$dpo_prompt_len" --max_completion_length "$dpo_completion_len" \
  --precompute_ref_batch_size 1 --logging_steps 1 --eval_steps 25 --save_steps 100 --seed "$seed"

run_step gemma_dpo_valid_generation "$generation_dir/dpo_valid_outputs.manifest.json" \
  "$python_bin" scripts/build_sft_outputs_for_risk_tuning.py --base_model "$gemma_dir" \
  --adapter_dir "$dpo_dir/final" --input "$data_dir/sft_valid.jsonl" \
  --output "$generation_dir/dpo_valid_outputs.jsonl" --max_source_len "$sft_source_len" --max_new_tokens "$generation_budget" \
  --temperature 0.0 --repetition_penalty 1.15 --no_repeat_ngram_size 4 --sft_prompt_style sft_plain
require_generation_complete "$generation_dir/dpo_valid_outputs.manifest.json" single

for mode in "${variants[@]}"; do
  strategy="staged_risk"
  component_mode="full"
  threshold="$full_threshold"
  if [[ "$mode" == "mask_off" ]]; then strategy="no_mask"; fi
  if [[ "$mode" == "without_router" ]]; then component_mode="without_router"; threshold="$without_router_threshold"; fi
  if [[ "$mode" == "without_scorer" ]]; then strategy="no_mask"; component_mode="without_scorer"; threshold="$without_scorer_threshold"; fi
  proposed_dir="$run_dir/gemma/proposed_${mode}"
  run_step "gemma_${mode}_train" "$proposed_dir/training_manifest.json" \
    "$python_bin" scripts/train_gemma_risk_tune_from_sft.py --base_model "$gemma_dir" \
    --init_adapter_dir "$sft_dir/final" --train_file "$proposed_dir/risk_tune_train_enriched.jsonl" \
    --valid_file "$proposed_dir/risk_tune_valid_enriched.jsonl" --output_dir "$proposed_dir" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --component_mode "$component_mode" \
    --zt_strategy "$strategy" --inputs_pre_enriched --learning_rate 5e-6 --epochs 1 \
    --batch_size 1 --eval_batch_size 1 --grad_accum 8 --max_source_len "$max_source_len" \
    --max_target_len "$target_len" --lambda_y 0 --risk_oversample_threshold "$threshold" --risk_oversample_factor 2 \
    --risk_threshold "$threshold" --mask_threshold "$threshold" --eval_every 25 --save_every 100 \
    --num_workers 0 --enable_gradient_checkpointing --seed "$seed"
  run_step "gemma_${mode}_valid_generation" "$generation_dir/proposed_${mode}_valid_outputs.manifest.json" \
    "$python_bin" scripts/run_gemma_selective_risk_refinement.py --base_model "$gemma_dir" \
    --sft_adapter_dir "$sft_dir/final" --risk_adapter_dir "$proposed_dir/final" \
    --router_dir "$router_dir/final" --risk_scorer_dir "$scorer_dir/final" --risk_contract fullpaper_v1 \
    --component_mode "$component_mode" \
    --input "$generation_dir/sft_valid_outputs.jsonl" --output "$generation_dir/proposed_${mode}_valid_outputs.jsonl" \
    --reuse_sft_response --sft_response_field sft_response --zt_strategy "$strategy" \
    --risk_threshold "$threshold" --gate_risk_threshold "$threshold" --mask_threshold "$threshold" \
    --max_source_len "$max_source_len" --max_new_tokens "$generation_budget" --temperature 0.0 \
    --repetition_penalty 1.15 --no_repeat_ngram_size 4
  require_generation_complete "$generation_dir/proposed_${mode}_valid_outputs.manifest.json" denoiser
done

echo "ARR Revision 2 first-backbone run complete"
