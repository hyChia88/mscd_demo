#!/usr/bin/env bash
# ============================================================================
# MSCD VLM LoRA_3 Training Launcher
#
# Validates prerequisites, runs training on Modal GPU, downloads adapter
# with timestamped name to avoid overwrites.
#
# Usage:
#   ./training/train.sh                          # Default config (3 epochs)
#   ./training/train.sh --epochs 5 --lr 1e-4     # Custom config
#   ./training/train.sh --download-only          # Download adapter only
# ============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_DIR")/data_curation"

# v0.5 training data
TRAIN_DIR="$DATA_ROOT/datasets/synth_v0.5/train"
TRAIN_FILE="$TRAIN_DIR/lora3_train.jsonl"
TEST_FILE="$TRAIN_DIR/lora3_test.jsonl"

# v0.5 images per model (site photos + floorplans)
V05_AP_IMGS="$DATA_ROOT/datasets/synth_v0.5/imgs"
V05_AP_FP="$DATA_ROOT/datasets/synth_v0.5/floorplans"
V05_BH_IMGS="$DATA_ROOT/datasets/synth_v0.5_bh/imgs"
V05_BH_FP="$DATA_ROOT/datasets/synth_v0.5_bh/floorplans"
V05_DXA_IMGS="$DATA_ROOT/datasets/synth_v0.5_dxa/imgs"
V05_DXA_FP="$DATA_ROOT/datasets/synth_v0.5_dxa/floorplans"

# v0.4 images (for enriched records)
AP_IMGS="$DATA_ROOT/datasets/synth_v0.4_ap/cases/imgs"
AP_PLANS="$DATA_ROOT/datasets/synth_v0.4_ap/cases/plans"
BH_IMGS="$DATA_ROOT/datasets/synth_v0.4_bh/cases/imgs"
BH_PLANS="$DATA_ROOT/datasets/synth_v0.4_bh/cases/plans"
DXA_IMGS="$DATA_ROOT/datasets/synth_v0.4_dxa/cases/imgs"
DXA_PLANS="$DATA_ROOT/datasets/synth_v0.4_dxa/cases/plans"

MODAL_VOLUME_PATH="/mscd-lora-v3/final"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
ADAPTER_LOCAL="$PROJECT_DIR/models/adapters/v3_lora_qwen_${RUN_ID}"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── Download-only mode ───────────────────────────────────────────────────────

if [[ "${1:-}" == "--download-only" ]]; then
    info "Downloading trained adapter from Modal volume..."
    info "Saving to: $ADAPTER_LOCAL"
    mkdir -p "$ADAPTER_LOCAL"
    modal volume get --force mscd-checkpoints "$MODAL_VOLUME_PATH" "$ADAPTER_LOCAL"
    ok "Adapter downloaded to: $ADAPTER_LOCAL"
    echo ""
    echo "Run evaluation with:"
    echo "  cd $PROJECT_DIR"
    echo "  python eval/h2_eval.py --adapter $ADAPTER_LOCAL"
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────────────────────

echo "============================================================"
echo "  MSCD VLM LoRA_3 Training — Pre-flight Checks"
echo "============================================================"
echo ""

# 1. Modal CLI
if command -v modal &>/dev/null; then
    ok "Modal CLI installed ($(modal --version 2>/dev/null || echo 'unknown version'))"
else
    fail "Modal CLI not found. Install: pip install modal && modal setup"
fi

# 2. Modal auth
if modal profile current &>/dev/null; then
    ok "Modal authenticated"
else
    fail "Modal not authenticated. Run: modal setup"
fi

# 3. wandb secret
if modal secret list 2>/dev/null | grep -q "wandb-secret"; then
    ok "Modal secret 'wandb-secret' exists"
else
    warn "Modal secret 'wandb-secret' not found."
    echo "       Create it with: modal secret create wandb-secret WANDB_API_KEY=<your-key>"
    echo "       Training will fail without this. Continue anyway? [y/N]"
    read -r ans
    if [[ "${ans}" != "y" && "${ans}" != "Y" ]]; then
        exit 1
    fi
fi

# 4. Training data
if [[ -f "$TRAIN_FILE" ]]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_FILE")
    ok "Training data: $TRAIN_COUNT samples ($TRAIN_FILE)"
else
    fail "Training data not found: $TRAIN_FILE"
fi

if [[ -f "$TEST_FILE" ]]; then
    TEST_COUNT=$(wc -l < "$TEST_FILE")
    ok "Test data:     $TEST_COUNT samples ($TEST_FILE)"
else
    fail "Test data not found: $TEST_FILE"
fi

# 5. v0.5 images per model
V05_IMG_COUNT=0
V05_FP_COUNT=0
for d in "$V05_AP_IMGS" "$V05_BH_IMGS" "$V05_DXA_IMGS"; do
    [[ -d "$d" ]] && V05_IMG_COUNT=$(( V05_IMG_COUNT + $(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) )) || true
done
for d in "$V05_AP_FP" "$V05_BH_FP" "$V05_DXA_FP"; do
    [[ -d "$d" ]] && V05_FP_COUNT=$(( V05_FP_COUNT + $(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) )) || true
done
ok "v0.5 photos:   $V05_IMG_COUNT site images (AP+BH+DXA)"
ok "v0.5 plans:    $V05_FP_COUNT floorplans (AP+BH+DXA)"

# 6. v0.4 images
V04_IMG_COUNT=0
V04_PLAN_COUNT=0
for d in "$AP_IMGS" "$BH_IMGS" "$DXA_IMGS"; do
    [[ -d "$d" ]] && V04_IMG_COUNT=$(( V04_IMG_COUNT + $(find "$d" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l) )) || true
done
for d in "$AP_PLANS" "$BH_PLANS" "$DXA_PLANS"; do
    [[ -d "$d" ]] && V04_PLAN_COUNT=$(( V04_PLAN_COUNT + $(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) )) || true
done
ok "v0.4 photos:   $V04_IMG_COUNT site images (AP+BH+DXA)"
ok "v0.4 plans:    $V04_PLAN_COUNT floorplans (AP+BH+DXA)"

echo ""
echo "============================================================"
echo "  Launching Training on Modal GPU (A100)"
echo "  Run ID: $RUN_ID"
echo "============================================================"
echo ""

# ── Run training ─────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
modal run --detach training/train.py "$@"

# ── Post-training ────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Training launched in detached mode"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  modal app logs mscd-vlm-lora3-train"
echo "  # or WandB: project=mscd-vlm-lora"
echo ""
echo "When complete, download the adapter:"
echo "  ./training/train.sh --download-only"
echo "  # Saves to: $ADAPTER_LOCAL"
echo ""
