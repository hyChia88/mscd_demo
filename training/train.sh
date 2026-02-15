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

TRAIN_DIR="$DATA_ROOT/datasets/synth_v0.3/train"
IMGS_DIR="$DATA_ROOT/datasets/synth_v0.3/cases/imgs"
PLANS_DIR="$DATA_ROOT/datasets/synth_v0.3/cases/plans"

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
    modal volume get mscd-checkpoints "$MODAL_VOLUME_PATH" "$ADAPTER_LOCAL"
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

# 5. Images
if [[ -d "$IMGS_DIR" ]]; then
    IMG_COUNT=$(ls "$IMGS_DIR"/*.png 2>/dev/null | wc -l)
    ok "Site photos:   $IMG_COUNT images ($IMGS_DIR)"
else
    warn "Image directory not found: $IMGS_DIR"
fi

if [[ -d "$PLANS_DIR" ]]; then
    PLAN_COUNT=$(ls "$PLANS_DIR"/*.png 2>/dev/null | wc -l)
    ok "Floorplans:    $PLAN_COUNT patches ($PLANS_DIR)"
else
    warn "Plans directory not found: $PLANS_DIR"
fi

echo ""
echo "============================================================"
echo "  Launching Training on Modal GPU (A100)"
echo "============================================================"
echo ""

# ── Run training ─────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
modal run training/train.py "$@"

# ── Post-training ────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Post-Training"
echo "============================================================"
echo ""
echo "To download the trained adapter:"
echo "  ./training/train.sh --download-only"
echo ""
echo "To evaluate:"
echo "  python script/run.py --profile v2_lora \\"
echo "    --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \\"
echo "    --adapter_path models/adapters/v2_lora_qwen"
