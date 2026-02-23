#!/usr/bin/env bash
# ============================================================================
# MSCD VLM LoRA Training Launcher
#
# Validates prerequisites, then runs training on Modal GPU.
#
# Usage:
#   ./training/train.sh                          # Default config
#   ./training/train.sh --epochs 5 --lr 1e-4     # Custom config
#   ./training/train.sh --download-only          # Download adapter after training
# ============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_DIR")/data_curation"

TRAIN_DIR="$DATA_ROOT/datasets/synth_v0.4_merged/train"
AP_IMGS="$DATA_ROOT/datasets/synth_v0.4_ap/cases/imgs"
AP_PLANS="$DATA_ROOT/datasets/synth_v0.4_ap/cases/plans"
BH_IMGS="$DATA_ROOT/datasets/synth_v0.4_bh/cases/imgs"
BH_PLANS="$DATA_ROOT/datasets/synth_v0.4_bh/cases/plans"
DXA_IMGS="$DATA_ROOT/datasets/synth_v0.4_dxa/cases/imgs"
DXA_PLANS="$DATA_ROOT/datasets/synth_v0.4_dxa/cases/plans"

ADAPTER_LOCAL="$PROJECT_DIR/models/adapters/v2_lora_qwen"
MODAL_VOLUME_PATH="/mscd-lora/final"

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
    mkdir -p "$ADAPTER_LOCAL"
    modal volume get --force mscd-checkpoints "$MODAL_VOLUME_PATH" "$ADAPTER_LOCAL"
    ok "Adapter downloaded to: $ADAPTER_LOCAL"
    echo ""
    echo "Run evaluation with:"
    echo "  cd $PROJECT_DIR"
    echo "  python script/run.py --profile v2_lora \\"
    echo "    --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \\"
    echo "    --adapter_path models/adapters/v2_lora_qwen"
    exit 0
fi

# ── Pre-flight checks ───────────────────────────────────────────────────────

echo "============================================================"
echo "  MSCD VLM LoRA Training — Pre-flight Checks"
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
if [[ -f "$TRAIN_DIR/lora_train.jsonl" ]]; then
    TRAIN_COUNT=$(wc -l < "$TRAIN_DIR/lora_train.jsonl")
    ok "Training data: $TRAIN_COUNT samples ($TRAIN_DIR/lora_train.jsonl)"
else
    fail "Training data not found: $TRAIN_DIR/lora_train.jsonl"
fi

if [[ -f "$TRAIN_DIR/lora_test.jsonl" ]]; then
    TEST_COUNT=$(wc -l < "$TRAIN_DIR/lora_test.jsonl")
    ok "Test data:     $TEST_COUNT samples ($TRAIN_DIR/lora_test.jsonl)"
else
    fail "Test data not found: $TRAIN_DIR/lora_test.jsonl"
fi

# 5. Images (AP + BH + DXA) — use find to avoid glob-fail under set -e
IMG_COUNT=0
PLAN_COUNT=0
for d in "$AP_IMGS" "$BH_IMGS" "$DXA_IMGS"; do
    [[ -d "$d" ]] && IMG_COUNT=$(( IMG_COUNT + $(find "$d" -maxdepth 1 \( -name "*.png" -o -name "*.jpg" \) 2>/dev/null | wc -l) )) || true
done
for d in "$AP_PLANS" "$BH_PLANS" "$DXA_PLANS"; do
    [[ -d "$d" ]] && PLAN_COUNT=$(( PLAN_COUNT + $(find "$d" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l) )) || true
done
ok "Site photos:   $IMG_COUNT images (AP+BH+DXA)"
ok "Floorplans:    $PLAN_COUNT patches (AP+BH+DXA)"

echo ""
echo "============================================================"
echo "  Launching Training on Modal GPU (A100)"
echo "============================================================"
echo ""

# ── Run training ─────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
# --detach: job runs on Modal even if the local connection drops (avoids gRPC Deadline exceeded)
modal run --detach training/train.py "$@"

# ── Post-training ────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Training launched in detached mode"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  modal app logs mscd-vlm-lora-train"
echo "  # or watch Wandb: project=mscd-vlm-lora, run=qwen25vl-7b-r16-synth_v04"
echo ""
echo "When complete, download the adapter:"
echo "  ./training/train.sh --download-only"
echo ""
echo "Then evaluate:"
echo "  ./training/eval.sh --step full"
