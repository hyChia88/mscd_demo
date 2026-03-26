#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Unified evaluation: 3 LoRA models × 2 conditions on 116 unified cases
#
# Prerequisites:
#   1. Upload adapters to Modal volume:
#      modal volume put mscd-checkpoints models/adapters/v2_lora_qwen      /mscd-unified-eval/v2_lora_qwen
#      modal volume put mscd-checkpoints models/adapters/v5_complex_lora_qwen /mscd-unified-eval/v5_complex_lora_qwen
#      modal volume put mscd-checkpoints models/adapters/v5_lora_qwen_r32  /mscd-unified-eval/v5_lora_qwen_r32
#
#   2. Run this script:
#      bash evaluation/inference/eval_unified.sh
#
#   3. Download results:
#      mkdir -p output/unified
#      modal volume get mscd-checkpoints /mscd-unified-eval/ output/unified/
#
# Output: output/unified/eval_constraints_{tag}.jsonl
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/../.."   # → mscd_demo/

SCRIPT="evaluation/inference/eval_unified.py"

echo "============================================================"
echo "  MSCD Unified Evaluation — 3 models × 2 conditions"
echo "  Test set: 116 cases (66 v05 + 50 v04)"
echo "============================================================"
echo ""

# ── Step 0: Upload adapters if not already on volume ─────────────────────────
echo "[Step 0] Checking adapters on Modal volume..."
echo "  If adapters are missing, run:"
echo "    modal volume put mscd-checkpoints models/adapters/v2_lora_qwen       /mscd-unified-eval/v2_lora_qwen"
echo "    modal volume put mscd-checkpoints models/adapters/v5_complex_lora_qwen /mscd-unified-eval/v5_complex_lora_qwen"
echo "    modal volume put mscd-checkpoints models/adapters/v5_lora_qwen_r32   /mscd-unified-eval/v5_lora_qwen_r32"
echo ""

# ── LoRA2: FP + MC ───────────────────────────────────────────────────────────
echo "[1/6] LoRA2 — FP (floorplan only)"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v2_lora_qwen \
    --tag lora2_FP \
    --modality FP \
    --prompt lora2

echo ""
echo "[2/6] LoRA2 — MC (floorplan + site)"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v2_lora_qwen \
    --tag lora2_MC \
    --modality MC \
    --prompt lora2

# ── LoRA5 r16: FP + MC ──────────────────────────────────────────────────────
echo ""
echo "[3/6] LoRA5-r16 — FP"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v5_complex_lora_qwen \
    --tag lora5r16_FP \
    --modality FP \
    --prompt lora5

echo ""
echo "[4/6] LoRA5-r16 — MC"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v5_complex_lora_qwen \
    --tag lora5r16_MC \
    --modality MC \
    --prompt lora5

# ── LoRA5 r32: FP + MC ──────────────────────────────────────────────────────
echo ""
echo "[5/6] LoRA5-r32 — FP"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v5_lora_qwen_r32 \
    --tag lora5r32_FP \
    --modality FP \
    --prompt lora5

echo ""
echo "[6/6] LoRA5-r32 — MC"
modal run "$SCRIPT" \
    --adapter /mscd-unified-eval/v5_lora_qwen_r32 \
    --tag lora5r32_MC \
    --modality MC \
    --prompt lora5

# ── Gemini: FP + MC (local, no Modal) ────────────────────────────────────────
echo ""
echo "[7/8] Gemini — FP"
conda run -n mscd_demo python evaluation/inference/eval_gemini_baseline.py \
    --cases evaluation/cases/cases_unified_test.jsonl \
    --modality FP \
    --output-dir output/unified

echo ""
echo "[8/8] Gemini — MC"
conda run -n mscd_demo python evaluation/inference/eval_gemini_baseline.py \
    --cases evaluation/cases/cases_unified_test.jsonl \
    --modality MC \
    --output-dir output/unified

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ALL 8 RUNS COMPLETE (6 Modal + 2 Gemini)"
echo "============================================================"
echo ""
echo "Download all results:"
echo "  mkdir -p output/unified"
echo "  modal volume get mscd-checkpoints /mscd-unified-eval/ output/unified/"
echo ""
echo "Expected files in output/unified/:"
echo "  eval_constraints_lora2_FP.jsonl      (Modal)"
echo "  eval_constraints_lora2_MC.jsonl      (Modal)"
echo "  eval_constraints_lora5r16_FP.jsonl   (Modal)"
echo "  eval_constraints_lora5r16_MC.jsonl   (Modal)"
echo "  eval_constraints_lora5r32_FP.jsonl   (Modal)"
echo "  eval_constraints_lora5r32_MC.jsonl   (Modal)"
echo "  eval_constraints_final_FP.jsonl      (Gemini, local)"
echo "  eval_constraints_final_MC.jsonl      (Gemini, local)"
echo ""
echo "Then run retrieval pipeline:"
echo "  python script/run_evaluation.py --profile v2_lora \\"
echo "    --cases evaluation/cases/cases_unified_test.jsonl \\"
echo "    --precomputed output/unified/eval_constraints_lora5r32_FP.jsonl"
