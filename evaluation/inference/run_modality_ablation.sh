#!/usr/bin/env bash
# End-to-end Track A modality ablation: G7, G8, Gemini v2 × 6 conditions.
# Conditions: MC, MC4D, FP, SITE, FPSITE, MA
#
# FPSITE = Floorplan + Site image only, no chat text (pure visual interpretation).
#
# Idempotent: skips inference if prediction file already exists.
# Gemini v2 runs locally; G7/G8 run on Modal GPU.
#
# Usage:
#   cd /root/cmu/master_thesis/mscd_demo
#   bash evaluation/inference/run_modality_ablation.sh \
#     2>&1 | tee output/lora6_v2_ap_20260331/modality_ablation_trackA/logs/ablation_complete.log

set -e
cd "$(dirname "$0")/../.."   # mscd_demo root

SLICE_ROOT="../data_curation/datasets/synth_v0.5_ap/train/modality_slices"
PRED_DIR="output/lora6_v2_ap_20260331/modality_ablation_trackA/predictions"
METRICS_ROOT="output/lora6_v2_ap_20260331/modality_ablation_trackA/metrics"
LOG_DIR="output/lora6_v2_ap_20260331/modality_ablation_trackA/logs"
GT_G7="../data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_eval_canonical_m_g7.jsonl"
PLOTS_DIR="docs/plots/phase4_lora6_main"

mkdir -p "$PRED_DIR" "$LOG_DIR" "$PLOTS_DIR"

# Modal adapter paths — use /best for both G7 and G8
G7_ADAPTER="/mscd-lora-v6-g7-position-context/best"
G8_ADAPTER="/mscd-lora-v6-g8-posctx-dim/best"

# Deterministic output tag = _adapter_tag(adapter) + "_MODAB_" + SLICE
# _adapter_tag: strip leading /, replace / with __, sanitize non-alnum → _
G7_BASE_TAG="mscd-lora-v6-g7-position-context__best"
G8_BASE_TAG="mscd-lora-v6-g8-posctx-dim__best"

# ── Helper: run Modal + download ──────────────────────────────────────────────
run_modal() {
  local adapter="$1" base_tag="$2" slice="$3" remote_cases="$4" out="$5"
  local prompt_key="${6:-}"
  local suffix="MODAB_${slice}"
  local vol_file="mscd-lora/eval_constraints_${base_tag}_${suffix}.jsonl"

  if [[ -f "$out" ]]; then
    echo "  [SKIP] Already exists: $out"
    return 0
  fi

  local extra=""
  [[ -n "$prompt_key" ]] && extra="--prompt-key $prompt_key"

  echo "  [Modal] $(basename $adapter) / $slice ..."
  modal run training/eval.py \
    --adapter-dir "$adapter" \
    --cases "$remote_cases" \
    --tag-suffix "$suffix" \
    $extra \
    2>&1 | tee "$LOG_DIR/${base_tag}_${slice}.log"

  echo "  [Download] /$vol_file → $out"
  modal volume get mscd-checkpoints "/$vol_file" "$out"
}

# ── Helper: run Gemini locally ────────────────────────────────────────────────
run_gemini() {
  local slice="$1" cases="$2" out="$3"
  if [[ -f "$out" ]]; then
    echo "  [SKIP] Already exists: $out"
    return 0
  fi
  echo "  [Gemini] $slice ..."
  conda run -n mscd_demo python evaluation/inference/eval_gemini_ap.py \
    --cases "$cases" \
    --output "$out" \
    --sleep-seconds 0.3 \
    2>&1 | tee "$LOG_DIR/gemini_ap_v2_${slice}.log"
}

# ── Helper: score one slice ────────────────────────────────────────────────────
score_slice() {
  local slice="$1"
  local out_dir="$METRICS_ROOT/$slice"
  mkdir -p "$out_dir"
  local pred_args=""
  for model in g7_position_context g8_posctx_dim gemini_ap_v2; do
    local pf="$PRED_DIR/${model}__${slice}__ap_eval.jsonl"
    [[ -f "$pf" ]] && pred_args="$pred_args --pred ${model}=${pf}"
  done
  [[ -z "$pred_args" ]] && { echo "  [SKIP] No predictions for $slice"; return 0; }
  echo "  [Score] $slice ..."
  conda run -n mscd_demo python evaluation/analysis/score_ap_track.py \
    --gt "$GT_G7" \
    $pred_args \
    --out-dir "$out_dir"
}


# ═══════════════════════════════════════════════════════════════════════════════
echo "=== STEP 1: Gemini v2 — all 6 conditions ==="
# All conditions use the standard (non-g7) slice files for Gemini
for slice in MC MC4D FP SITE MA; do
  run_gemini "$slice" \
    "$SLICE_ROOT/lora6_v2_ap_eval_canonical_m_${slice}.jsonl" \
    "$PRED_DIR/gemini_ap_v2__${slice}__ap_eval.jsonl"
done
run_gemini "FPSITE" \
  "$SLICE_ROOT/lora6_v2_ap_eval_canonical_m_FPSITE.jsonl" \
  "$PRED_DIR/gemini_ap_v2__FPSITE__ap_eval.jsonl"

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 2: G7 — all 6 conditions ==="
# G7/G8 use the _g7_ slice files (same label schema with position_context / dims)
for slice in MC MC4D FP SITE MA FPSITE; do
  run_modal "$G7_ADAPTER" "$G7_BASE_TAG" "$slice" \
    "/data/ap_eval_g7_${slice,,}.jsonl" \
    "$PRED_DIR/g7_position_context__${slice}__ap_eval.jsonl"
done

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 3: G8 — all 6 conditions ==="
# G8 MC: reuse the existing main eval output rather than re-running
if [[ ! -f "$PRED_DIR/g8_posctx_dim__MC__ap_eval.jsonl" ]]; then
  cp "output/lora6_v2_ap_20260331/g8_posctx_dim__ap_eval.jsonl" \
     "$PRED_DIR/g8_posctx_dim__MC__ap_eval.jsonl"
  echo "  [Copy] G8 MC from main eval."
fi
# G8 needs explicit prompt key (adapter name doesn't contain 'g7')
for slice in MC4D FP SITE MA FPSITE; do
  run_modal "$G8_ADAPTER" "$G8_BASE_TAG" "$slice" \
    "/data/ap_eval_g7_${slice,,}.jsonl" \
    "$PRED_DIR/g8_posctx_dim__${slice}__ap_eval.jsonl" \
    "lora_system_g7"
done

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 4: Score all 6 conditions ==="
for slice in MC MC4D FP SITE FPSITE MA; do
  score_slice "$slice"
done

# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== STEP 5: Regenerate plot ==="
conda run -n mscd_demo python evaluation/analysis/summarize_ap_modality_ablation.py \
  --out-dir "$PLOTS_DIR"

echo ""
echo "Done. Output: $PLOTS_DIR/fig09_trackA_modality_ablation.png"
