#!/usr/bin/env bash
set -euo pipefail

cd ~/mh-denoise

BASE_MODEL=google/gemma-4-E4B-it
VALID_FILE=data/splits_exp295/valid_mdlm.jsonl

SFT_ADAPTER=outputs/models/gemma4_peft_sft_plain_exp295/final

RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/best
if [ ! -d "$RISK_TUNED_ADAPTER" ]; then
  RISK_TUNED_ADAPTER=outputs/models/gemma4_selective_sft_plain_risk_tuned_exp295_len256_lr5e6_lambda03_clean/final
fi

ROUTER_DIR=outputs/models/aspect_router_exp295_multilabel/final

RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/best
if [ ! -d "$RISK_SCORER_DIR" ]; then
  RISK_SCORER_DIR=outputs/models/span_risk_multilabel_v1/final
fi

SFT_VALID=outputs/refinement/sft_plain_exp295_valid_outputs_len256.jsonl

echo "============================================================"
echo "[PATH CHECK]"
echo "VALID_FILE=$VALID_FILE"
echo "SFT_ADAPTER=$SFT_ADAPTER"
echo "RISK_TUNED_ADAPTER=$RISK_TUNED_ADAPTER"
echo "ROUTER_DIR=$ROUTER_DIR"
echo "RISK_SCORER_DIR=$RISK_SCORER_DIR"
echo "SFT_VALID=$SFT_VALID"
echo "============================================================"

[ -f "$VALID_FILE" ] || { echo "[missing] $VALID_FILE"; exit 1; }
[ -d "$SFT_ADAPTER" ] || { echo "[missing] $SFT_ADAPTER"; exit 1; }
[ -d "$RISK_TUNED_ADAPTER" ] || { echo "[missing] $RISK_TUNED_ADAPTER"; exit 1; }
[ -d "$ROUTER_DIR" ] || { echo "[missing] $ROUTER_DIR"; exit 1; }
[ -d "$RISK_SCORER_DIR" ] || { echo "[missing] $RISK_SCORER_DIR"; exit 1; }

# Clean old sweep outputs so the proxy CSV does not mix old runs.
rm -f outputs/refinement/selective_valid_sweep_*.jsonl
rm -f outputs/analysis/selective_valid_sweep_len256_proxy.csv

if [ ! -f "$SFT_VALID" ]; then
  echo "============================================================"
  echo "[BUILD] SFT valid outputs len256"
  echo "============================================================"

  python scripts/build_sft_outputs_for_risk_tuning.py \
    --base_model "$BASE_MODEL" \
    --adapter_dir "$SFT_ADAPTER" \
    --sft_prompt_style sft_plain \
    --input "$VALID_FILE" \
    --output "$SFT_VALID" \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --repetition_penalty 1.15 \
    --no_repeat_ngram_size 4
else
  echo "[skip] existing SFT valid outputs: $SFT_VALID"
fi

for TH in 0.005 0.01 0.02; do
  for TSTEP in 1 2; do
    for SPEC in 0.60 0.65; do
      TAG="th${TH/./p}_t${TSTEP}_spec${SPEC/./p}"
      RAW_OUT="outputs/refinement/selective_valid_sweep_${TAG}.jsonl"
      FILT_OUT="outputs/refinement/selective_valid_sweep_${TAG}_trunc_filtered.jsonl"

      echo
      echo "============================================================"
      echo "[RUN] $TAG"
      echo "============================================================"

      python scripts/run_gemma_selective_risk_refinement.py \
        --base_model "$BASE_MODEL" \
        --sft_adapter_dir "$SFT_ADAPTER" \
        --risk_adapter_dir "$RISK_TUNED_ADAPTER" \
        --sft_prompt_style sft_plain \
        --router_dir "$ROUTER_DIR" \
        --risk_scorer_dir "$RISK_SCORER_DIR" \
        --input "$SFT_VALID" \
        --output "$RAW_OUT" \
        --reuse_sft_response \
        --sft_response_field sft_response \
        --zt_strategy staged_risk \
        --timestep "$TSTEP" \
        --risk_threshold 0.35 \
        --gate_strategy aspect_only \
        --gate_focus_aspect medical_advice \
        --gate_focus_threshold "$TH" \
        --min_risk_delta 0.0 \
        --min_focus_risk_delta 0.0 \
        --specificity_min_ratio "$SPEC" \
        --max_new_tokens 256 \
        --temperature 0.0 \
        --repetition_penalty 1.15 \
        --no_repeat_ngram_size 4

      python - "$RAW_OUT" "$FILT_OUT" "$TAG" <<'PY'
import json, re, sys

inp, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]

end_punct = (".", "!", "?", ")", "]", "}", '"', "'")
bad_last = set("""
and or but because so if when while although though since with without for to of in on at by from into onto about
this that these those a an the your my our their his her its as than then just simply really also
be become becoming better lis unde outside crying Good good
""".split())

def last_word(text):
    toks = re.findall(r"[A-Za-z]+", str(text or "").strip())
    return toks[-1] if toks else ""

def likely_truncated(text):
    t = str(text or "").strip()
    if not t:
        return True
    if not t.endswith(end_punct):
        return True
    if last_word(t) in bad_last:
        return True
    return False

rows = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
kept = 0
fallback = 0

with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        rr = dict(r)
        trunc = bool(rr.get("accepted_denoiser")) and likely_truncated(rr.get("denoiser_response", ""))
        rr["sweep_tag"] = tag
        rr["denoiser_likely_truncated"] = trunc
        rr["accepted_denoiser_raw"] = bool(rr.get("accepted_denoiser"))

        if trunc:
            rr["final_response"] = rr.get("sft_response", "")
            rr["accepted_denoiser"] = False
            rr["truncation_fallback"] = True
            fallback += 1
        else:
            rr["truncation_fallback"] = False
            if rr.get("accepted_denoiser"):
                kept += 1

        f.write(json.dumps(rr, ensure_ascii=False) + "\n")

print(
    "tag", tag,
    "n", len(rows),
    "called", sum(bool(r.get("used_denoiser")) for r in rows),
    "accepted_raw", sum(bool(r.get("accepted_denoiser")) for r in rows),
    "accepted_after_filter", kept,
    "trunc_fallback", fallback
)
PY

    done
  done
done

echo
echo "============================================================"
echo "[SUMMARY] Build proxy CSV"
echo "============================================================"

python - <<'PY'
import csv, glob, json, statistics, sys

sys.path.insert(0, "scripts")
from selective_risk_refinement_utils import count_bad_safety_patterns, count_generic_phrases

files = sorted(glob.glob("outputs/refinement/selective_valid_sweep_*_trunc_filtered.jsonl"))
rows_out = []

def mean(xs):
    vals = []
    for x in xs:
        try:
            if x is not None:
                vals.append(float(x))
        except Exception:
            pass
    return statistics.mean(vals) if vals else 0.0

for p in files:
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if not rows:
        continue

    tag = rows[0].get("sweep_tag") or p
    used = [r for r in rows if r.get("used_denoiser")]
    accepted_raw = [r for r in rows if r.get("accepted_denoiser_raw")]
    accepted_final = [r for r in rows if r.get("accepted_denoiser")]

    rec = {
        "tag": tag,
        "file": p,
        "n": len(rows),
        "called": len(used),
        "accepted_raw": len(accepted_raw),
        "accepted_final": len(accepted_final),
        "trunc_fallback": sum(bool(r.get("truncation_fallback")) for r in rows),
        "sft_risk": round(mean(r.get("sft_risk_score") for r in rows), 4),
        "final_risk": round(mean(r.get("final_risk_score") for r in rows), 4),
        "sft_focus": round(mean(r.get("sft_focus_risk_score") for r in rows), 4),
        "final_focus": round(mean(r.get("final_focus_risk_score") for r in rows), 4),
        "spec_ratio_used": round(mean(r.get("specificity_ratio") for r in used), 4) if used else 1.0,
        "bad_before": sum(int(r.get("sft_bad_safety_count") or 0) for r in rows),
        "bad_after": sum(count_bad_safety_patterns(r.get("final_response", "")) for r in rows),
        "generic_before": sum(int(r.get("sft_generic_count") or 0) for r in rows),
        "generic_after": sum(count_generic_phrases(r.get("final_response", "")) for r in rows),
    }
    rows_out.append(rec)

if not rows_out:
    raise RuntimeError("No sweep results found.")

out_csv = "outputs/analysis/selective_valid_sweep_len256_proxy.csv"
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print("saved", out_csv)
print()
print("Top proxy candidates:")
for r in sorted(rows_out, key=lambda x: (x["final_focus"], x["final_risk"], -x["accepted_final"]))[:12]:
    print(r)
PY

echo
echo "============================================================"
echo "[DONE]"
echo "Proxy CSV: outputs/analysis/selective_valid_sweep_len256_proxy.csv"
echo "============================================================"
