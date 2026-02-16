#!/usr/bin/env bash
# ============================================================================
# MSCD LoRA Evaluation Pipeline
#
# End-to-end pipeline for evaluating LoRA-finetuned Qwen2.5-VL on synth_v0.3.
# Orchestrates: Modal GPU extraction -> download -> local retrieval+scoring -> plots.
#
# Usage:
#   ./training/eval.sh                              # Full pipeline (all steps)
#   ./training/eval.sh --step modal                 # Step 1-2: Modal extraction + download
#   ./training/eval.sh --step local                 # Step 3: Local pipeline (needs precomputed)
#   ./training/eval.sh --step plots                 # Step 4: Comparison charts only
#   ./training/eval.sh --adapter final              # Single adapter only
#   ./training/eval.sh --skip-v2-prompt             # Skip V2 prompt baseline (already done)
#   ./training/eval.sh --limit 5                    # Quick test with 5 cases
#
# Prerequisites:
#   - Modal CLI installed and authenticated (pip install modal && modal setup)
#   - Trained LoRA adapters on Modal volume (from training/train.sh)
#   - synth_v0.3 dataset in ../data_curation/datasets/synth_v0.3/
#
# Output:
#   logs/evaluations/
#     eval_constraints_{adapter}.jsonl   # Pre-extracted constraints (from Modal)
#     traces_{timestamp}_v2_lora.jsonl   # Full eval traces
#     summary_{timestamp}_v2_lora.csv    # Summary metrics
#   logs/comparisons/v03_full/
#     *.png                              # Comparison charts
# ============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_DIR")/data_curation"

CASES="$DATA_ROOT/datasets/synth_v0.3/cases_v3_filtered.jsonl"
EVAL_DIR="$PROJECT_DIR/logs/evaluations"
PLOTS_DIR="$PROJECT_DIR/logs/comparisons/v03_full"

# Existing V2 prompt baseline (Exp 3)
V2_PROMPT_TRACES="$EVAL_DIR/traces_20260214_210555_v2_prompt.jsonl"
V2_PROMPT_SUMMARY="$EVAL_DIR/summary_20260214_210555_v2_prompt.csv"

# ── Colors ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()    { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
section() { echo -e "\n${BOLD}════════════════════════════════════════════════════════════${NC}"; echo -e "${BOLD}  $*${NC}"; echo -e "${BOLD}════════════════════════════════════════════════════════════${NC}\n"; }

# ── Parse arguments ──────────────────────────────────────────────────────────

STEP="full"              # full | modal | local | plots
ADAPTERS=("final" "checkpoint-180")
SKIP_V2_PROMPT=false
LIMIT_ARG=""
CONDA_ENV="mscd_demo"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2"; shift 2 ;;
        --adapter)       ADAPTERS=("$2"); shift 2 ;;
        --skip-v2-prompt) SKIP_V2_PROMPT=true; shift ;;
        --limit)         LIMIT_ARG="--limit $2"; shift 2 ;;
        --conda)         CONDA_ENV="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./training/eval.sh [--step modal|local|plots] [--adapter NAME] [--limit N]"
            exit 0 ;;
        *)
            warn "Unknown argument: $1"; shift ;;
    esac
done

# ── Timestamp for this run ───────────────────────────────────────────────────

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$EVAL_DIR/eval_run_${RUN_TS}.log"
mkdir -p "$EVAL_DIR" "$PLOTS_DIR"

# Log everything to file AND stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run timestamp: $RUN_TS"
echo "Log file:      $LOG_FILE"
echo "Step:          $STEP"
echo "Adapters:      ${ADAPTERS[*]}"
echo ""

# ── Pre-flight checks ───────────────────────────────────────────────────────

section "Pre-flight Checks"

# Cases file
if [[ -f "$CASES" ]]; then
    CASE_COUNT=$(wc -l < "$CASES")
    ok "Cases file: $CASE_COUNT cases ($CASES)"
else
    fail "Cases file not found: $CASES"
fi

# Modal CLI (only needed for modal step)
if [[ "$STEP" == "full" || "$STEP" == "modal" ]]; then
    if command -v modal &>/dev/null; then
        ok "Modal CLI installed"
    else
        fail "Modal CLI not found. Install: pip install modal && modal setup"
    fi
fi

# Existing V2 prompt baseline
if [[ -f "$V2_PROMPT_TRACES" ]]; then
    ok "V2 prompt baseline exists: $V2_PROMPT_TRACES"
else
    warn "V2 prompt baseline not found (will need to run separately)"
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Modal GPU Extraction
# ═════════════════════════════════════════════════════════════════════════════

run_modal_extraction() {
    section "Step 1: Modal GPU LoRA Extraction"

    for adapter in "${ADAPTERS[@]}"; do
        local precomputed_file="$EVAL_DIR/eval_constraints_${adapter}.jsonl"

        # Skip if already exists
        if [[ -f "$precomputed_file" ]]; then
            ok "Precomputed constraints exist: $precomputed_file (skipping Modal)"
            continue
        fi

        info "Running Modal extraction for adapter: $adapter"
        cd "$PROJECT_DIR"
        modal run training/eval.py --adapter-dir "/mscd-lora/${adapter}" $LIMIT_ARG

        info "Downloading precomputed constraints..."
        modal volume get mscd-checkpoints \
            "/mscd-lora/eval_constraints_${adapter}.jsonl" \
            "$EVAL_DIR/"

        if [[ -f "$precomputed_file" ]]; then
            local count
            count=$(wc -l < "$precomputed_file")
            ok "Downloaded: $precomputed_file ($count cases)"
        else
            fail "Download failed: $precomputed_file not found"
        fi
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Local Pipeline (Retrieval + Scoring)
# ═════════════════════════════════════════════════════════════════════════════

run_local_pipeline() {
    section "Step 2: Local Pipeline (Retrieval + Scoring)"

    cd "$PROJECT_DIR"

    for adapter in "${ADAPTERS[@]}"; do
        local precomputed_file="$EVAL_DIR/eval_constraints_${adapter}.jsonl"

        if [[ ! -f "$precomputed_file" ]]; then
            warn "Precomputed constraints missing for $adapter — run --step modal first"
            continue
        fi

        info "Running local pipeline with adapter: $adapter"
        conda run -n "$CONDA_ENV" python script/run.py \
            --profile v2_lora \
            --cases "$CASES" \
            --precomputed "$precomputed_file" \
            --output_dir "$EVAL_DIR" \
            $LIMIT_ARG

        ok "Local pipeline complete for $adapter"
    done

    # Run V2 prompt baseline if not done and not skipped
    if [[ "$SKIP_V2_PROMPT" == false && ! -f "$V2_PROMPT_TRACES" ]]; then
        info "Running V2 prompt baseline..."
        conda run -n "$CONDA_ENV" python script/run.py \
            --profile v2_prompt \
            --cases "$CASES" \
            --output_dir "$EVAL_DIR" \
            $LIMIT_ARG
        ok "V2 prompt baseline complete"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: Comparison Charts
# ═════════════════════════════════════════════════════════════════════════════

run_comparison_charts() {
    section "Step 3: Comparison Charts"

    cd "$PROJECT_DIR"

    # Find the latest LoRA traces
    local lora_traces=()
    local lora_summaries=()
    local lora_labels=()

    # Collect the N most recent v2_lora trace files (one per adapter).
    # They are produced sequentially, so the Nth-most-recent corresponds
    # to the Nth adapter (in reverse order: newest = last adapter run).
    local all_lora_traces
    all_lora_traces=$(ls -t "$EVAL_DIR"/traces_*_v2_lora.jsonl 2>/dev/null || true)

    local adapter_count=${#ADAPTERS[@]}
    for i in "${!ADAPTERS[@]}"; do
        local adapter="${ADAPTERS[$i]}"
        # Pick in reverse: last adapter ran = newest file, first = Nth newest
        local pick_idx=$(( adapter_count - 1 - i ))
        local trace_file
        trace_file=$(echo "$all_lora_traces" | head -n "$adapter_count" | tail -n +$(( pick_idx + 1 )) | head -1)

        if [[ -n "$trace_file" ]]; then
            lora_traces+=("$trace_file")
            lora_labels+=("V2 LoRA ($adapter)")
        fi
    done

    # Build plot command
    local plot_args=()

    # Add V2 prompt baseline
    if [[ -f "$V2_PROMPT_TRACES" ]]; then
        plot_args+=(--traces "$V2_PROMPT_TRACES" --label "V2 Prompt")
    fi

    # Add LoRA traces
    for i in "${!lora_traces[@]}"; do
        plot_args+=(--traces "${lora_traces[$i]}" --label "${lora_labels[$i]}")
    done

    if [[ ${#plot_args[@]} -eq 0 ]]; then
        warn "No trace files found — skipping chart generation"
        return
    fi

    info "Generating comparison charts..."
    conda run -n "$CONDA_ENV" python script/compare_results.py \
        "${plot_args[@]}" \
        --cases "$CASES" \
        --plots \
        --output "$PLOTS_DIR" \
        --title "synth_v0.3 Evaluation (${CASE_COUNT:-84} cases)"

    ok "Charts saved to: $PLOTS_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# Execute selected steps
# ═════════════════════════════════════════════════════════════════════════════

case "$STEP" in
    full)
        run_modal_extraction
        run_local_pipeline
        run_comparison_charts
        ;;
    modal)
        run_modal_extraction
        ;;
    local)
        run_local_pipeline
        ;;
    plots)
        run_comparison_charts
        ;;
    *)
        fail "Unknown step: $STEP (use: full, modal, local, plots)"
        ;;
esac

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

section "Pipeline Complete"

echo "Run:      $RUN_TS"
echo "Log:      $LOG_FILE"
echo "Traces:   $EVAL_DIR/traces_*_v2_lora.jsonl"
echo "Charts:   $PLOTS_DIR/"
echo ""
echo "Quick compare:"
echo "  python script/compare_results.py --latest 4"
echo ""
echo "View charts:"
echo "  ls $PLOTS_DIR/*.png"
