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

CASES="$DATA_ROOT/datasets/synth_v0.4_merged/train/test_holdout.jsonl"  # 50 holdout cases (AP=20, BH=20, DXA=10)
EVAL_DIR="$PROJECT_DIR/logs/evaluations/synth_v04"   # new runs land here
PLOTS_DIR="$PROJECT_DIR/logs/comparisons/synth_v0.4_lora_2"

# V2 prompt baseline from Exp 2 (synth_v0.3, 84 cases — for reference only)
V2_PROMPT_TRACES="$PROJECT_DIR/logs/evaluations/synth_v03/traces/traces_20260214_210555_v2_prompt.jsonl"
V2_PROMPT_SUMMARY="$PROJECT_DIR/logs/evaluations/synth_v03/summaries/summary_20260214_210555_v2_prompt.csv"

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

STEP="full"              # full | modal | local | plots | paired-ablation | overall
ADAPTERS=("final" "checkpoint-180")
SKIP_V2_PROMPT=false
LIMIT_ARG=""
CONDA_ENV="mscd_demo"
CONDITION_OVERRIDE=""    # For paired ablation: MA, MB, MC

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2"; shift 2 ;;
        --adapter)       ADAPTERS=("$2"); shift 2 ;;
        --skip-v2-prompt) SKIP_V2_PROMPT=true; shift ;;
        --limit)         LIMIT_ARG="--limit $2"; shift 2 ;;
        --conda)         CONDA_ENV="$2"; shift 2 ;;
        --condition-override) CONDITION_OVERRIDE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./training/eval.sh [--step modal|local|plots|paired-ablation] [--adapter NAME] [--limit N]"
            echo ""
            echo "Steps:"
            echo "  full              Run all steps (modal + local + plots)"
            echo "  modal             Modal GPU extraction only"
            echo "  local             Local pipeline (needs precomputed)"
            echo "  plots             Comparison charts only"
            echo "  paired-ablation   LoRA+Prompt MA/MB/MC ablation + overall comparison"
            echo "  overall           Overall comparison only (LoRA final vs Prompt v2, per-case)"
            echo ""
            echo "Options:"
            echo "  --adapter NAME              Adapter name (default: final + checkpoint-180)"
            echo "  --condition-override COND   Override condition for all cases (MA/MB/MC)"
            echo "  --limit N                   Limit to first N cases"
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
    local cond_override="${1:-}"
    section "Step 1: Modal GPU LoRA Extraction"

    local cond_arg=""
    local cond_suffix=""
    if [[ -n "$cond_override" ]]; then
        cond_arg="--condition-override $cond_override"
        cond_suffix="_${cond_override}"
        info "Condition override: $cond_override"
    fi

    for adapter in "${ADAPTERS[@]}"; do
        local tag="${adapter}${cond_suffix}"
        local precomputed_file="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        # Skip if already exists
        if [[ -f "$precomputed_file" ]]; then
            ok "Precomputed constraints exist: $precomputed_file (skipping Modal)"
            continue
        fi

        info "Running Modal extraction for adapter: $adapter (condition: ${cond_override:-per-case})"
        cd "$PROJECT_DIR"
        modal run training/eval.py --adapter-dir "/mscd-lora/${adapter}" $LIMIT_ARG $cond_arg

        info "Downloading precomputed constraints..."
        modal volume get mscd-checkpoints \
            "/mscd-lora/eval_constraints_${tag}.jsonl" \
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
    local cond_override="${1:-}"
    section "Step 2: Local Pipeline (Retrieval + Scoring)"

    cd "$PROJECT_DIR"

    local cond_suffix=""
    local cond_run_arg=""
    if [[ -n "$cond_override" ]]; then
        cond_suffix="_${cond_override}"
        cond_run_arg="--condition-override $cond_override"
        info "Condition override: $cond_override"
    fi

    for adapter in "${ADAPTERS[@]}"; do
        local tag="${adapter}${cond_suffix}"
        local precomputed_file="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        if [[ ! -f "$precomputed_file" ]]; then
            warn "Precomputed constraints missing for $tag — run --step modal first"
            continue
        fi

        info "Running local pipeline with adapter: $adapter (condition: ${cond_override:-per-case})"
        conda run -n "$CONDA_ENV" python script/run.py \
            --profile v2_lora \
            --cases "$CASES" \
            --precomputed "$precomputed_file" \
            --output_dir "$EVAL_DIR" \
            $LIMIT_ARG $cond_run_arg

        ok "Local pipeline complete for $tag"
    done

    # Run V2 prompt baseline if not done and not skipped
    if [[ "$SKIP_V2_PROMPT" == false && ! -f "$V2_PROMPT_TRACES" && -z "$cond_override" ]]; then
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
        --title "synth_v0.4 Holdout — LoRA v2 (50 cases)"

    ok "Charts saved to: $PLOTS_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# Execute selected steps
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# STEP: Overall Comparison (LoRA final vs Prompt v2, per-case conditions)
# Requires eval_constraints_final.jsonl (no condition override) on the volume.
# Run: modal run training/eval.py --adapter-dir /mscd-lora/final
# ═════════════════════════════════════════════════════════════════════════════

run_overall_comparison() {
    section "Overall Comparison — LoRA final vs Prompt v2 (per-case conditions)"

    cd "$PROJECT_DIR"
    local overall_plot_args=()

    # ── 1. LoRA (per-case conditions) ──────────────────────────────────────
    # Needs eval_constraints_final.jsonl WITHOUT a condition override.
    local overall_precomputed="$EVAL_DIR/eval_constraints_final.jsonl"

    if [[ ! -f "$overall_precomputed" ]]; then
        info "Downloading eval_constraints_final.jsonl from Modal volume..."
        if conda run -n "$CONDA_ENV" modal volume get mscd-checkpoints \
                "/mscd-lora/eval_constraints_final.jsonl" "$EVAL_DIR/" 2>&1; then
            ok "Downloaded eval_constraints_final.jsonl"
        else
            warn "eval_constraints_final.jsonl not yet on volume."
            warn "Run this first, then re-run --step overall:"
            warn "  modal run training/eval.py --adapter-dir /mscd-lora/final"
            overall_precomputed=""
        fi
    fi

    # Validate it is the synth_v0.4 run (50 cases), not an old v0.3 file (84 cases)
    if [[ -f "$overall_precomputed" ]]; then
        local row_count
        row_count=$(wc -l < "$overall_precomputed")
        if [[ "$row_count" -ne 50 ]]; then
            warn "eval_constraints_final.jsonl has $row_count rows (expected 50 for synth_v0.4)."
            warn "Likely a stale synth_v0.3 file. Delete and re-run Modal extraction:"
            warn "  rm $overall_precomputed"
            warn "  modal run training/eval.py --adapter-dir /mscd-lora/final"
            overall_precomputed=""
        else
            ok "eval_constraints_final.jsonl validated: $row_count cases"
        fi
    fi

    if [[ -n "$overall_precomputed" ]]; then
        local existing_lora_overall
        existing_lora_overall=$(ls -t "$EVAL_DIR"/traces_*_v2_lora.jsonl 2>/dev/null \
            | grep -v "_MA\.jsonl\|_MB\.jsonl\|_MC\.jsonl" | head -1)
        if [[ -n "$existing_lora_overall" ]]; then
            ok "LoRA per-case traces exist: $(basename "$existing_lora_overall") (skipping)"
        else
            info "Running v2_lora (per-case conditions)..."
            conda run -n "$CONDA_ENV" python script/run.py \
                --profile v2_lora \
                --cases "$CASES" \
                --precomputed "$overall_precomputed" \
                --output_dir "$EVAL_DIR" \
                $LIMIT_ARG
            existing_lora_overall=$(ls -t "$EVAL_DIR"/traces_*_v2_lora.jsonl 2>/dev/null \
                | grep -v "_MA\.jsonl\|_MB\.jsonl\|_MC\.jsonl" | head -1)
            ok "v2_lora per-case complete"
        fi
        [[ -n "$existing_lora_overall" ]] && \
            overall_plot_args+=(--traces "$existing_lora_overall" --label "V2 LoRA (final)")
    fi

    # ── 2. Prompt v2 (per-case conditions) ─────────────────────────────────
    if [[ "$SKIP_V2_PROMPT" == false ]]; then
        local existing_prompt_overall
        existing_prompt_overall=$(ls -t "$EVAL_DIR"/traces_*_v2_prompt.jsonl 2>/dev/null \
            | grep -v "_MA\.jsonl\|_MB\.jsonl\|_MC\.jsonl" | head -1)
        if [[ -n "$existing_prompt_overall" ]]; then
            ok "Prompt per-case traces exist: $(basename "$existing_prompt_overall") (skipping)"
        else
            info "Running v2_prompt (per-case conditions)..."
            conda run -n "$CONDA_ENV" python script/run.py \
                --profile v2_prompt \
                --cases "$CASES" \
                --output_dir "$EVAL_DIR" \
                $LIMIT_ARG
            existing_prompt_overall=$(ls -t "$EVAL_DIR"/traces_*_v2_prompt.jsonl 2>/dev/null \
                | grep -v "_MA\.jsonl\|_MB\.jsonl\|_MC\.jsonl" | head -1)
            ok "v2_prompt per-case complete"
        fi
        [[ -n "$existing_prompt_overall" ]] && \
            overall_plot_args+=(--traces "$existing_prompt_overall" --label "V2 Prompt")
    fi

    # ── 3. Generate overall metrics chart ──────────────────────────────────
    if [[ ${#overall_plot_args[@]} -ge 4 ]]; then
        info "Generating overall metrics chart..."
        mkdir -p "${PLOTS_DIR}/overall"
        conda run -n "$CONDA_ENV" python script/compare_results.py \
            "${overall_plot_args[@]}" \
            --cases "$CASES" \
            --plots \
            --output "${PLOTS_DIR}/overall" \
            --title "LoRA final vs Prompt v2 — Overall (synth_v0.4, 50 holdout cases)"
        ok "Overall charts saved to: ${PLOTS_DIR}/overall"
    else
        warn "Need both LoRA and Prompt traces for overall chart — skipping"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP: Paired Ablation (LoRA + Prompt, MA/MB/MC forced conditions)
# ═════════════════════════════════════════════════════════════════════════════

run_paired_ablation() {
    section "Paired Modality Ablation (LoRA + Prompt × MA/MB/MC)"

    local CONDITIONS=("MA" "MB" "MC")

    # ── Per-condition runs ──────────────────────────────────────────────────
    for cond in "${CONDITIONS[@]}"; do
        info "━━━ Condition: ${cond} ━━━"

        # LoRA: Modal extraction + local pipeline
        run_modal_extraction "$cond"
        run_local_pipeline "$cond"

        # Prompt v2: local pipeline (Gemini, no Modal needed)
        if [[ "$SKIP_V2_PROMPT" == false ]]; then
            local existing_prompt
            existing_prompt=$(ls -t "$EVAL_DIR"/traces_*_v2_prompt_${cond}.jsonl 2>/dev/null | head -1)
            if [[ -n "$existing_prompt" ]]; then
                ok "Prompt traces exist for $cond: $(basename "$existing_prompt") (skipping)"
            else
                info "Running v2_prompt (condition=$cond)..."
                cd "$PROJECT_DIR"
                conda run -n "$CONDA_ENV" python script/run.py \
                    --profile v2_prompt \
                    --cases "$CASES" \
                    --output_dir "$EVAL_DIR" \
                    --condition-override "$cond" \
                    $LIMIT_ARG
                ok "v2_prompt complete for $cond"
            fi
        fi
    done

    # ── Overall comparison (per-case conditions) ────────────────────────────
    run_overall_comparison

    # ── Paired ablation charts (Charts 10-12) ───────────────────────────────
    section "Paired Ablation Charts (LoRA vs Prompt × MA/MB/MC)"

    cd "$PROJECT_DIR"
    local plot_args=()
    for cond in "${CONDITIONS[@]}"; do
        # LoRA traces for this condition
        local latest_lora
        latest_lora=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest_lora" ]]; then
            plot_args+=(--traces "$latest_lora" --label "LoRA_${cond}")
        fi

        # Prompt traces for this condition
        if [[ "$SKIP_V2_PROMPT" == false ]]; then
            local latest_prompt
            latest_prompt=$(ls -t "$EVAL_DIR"/traces_*_v2_prompt_${cond}.jsonl 2>/dev/null | head -1)
            if [[ -n "$latest_prompt" ]]; then
                plot_args+=(--traces "$latest_prompt" --label "Prompt_${cond}")
            fi
        fi
    done

    if [[ ${#plot_args[@]} -ge 4 ]]; then
        info "Generating paired ablation charts..."
        conda run -n "$CONDA_ENV" python script/compare_results.py \
            "${plot_args[@]}" \
            --cases "$CASES" \
            --plots --paired-ablation \
            --output "$PLOTS_DIR" \
            --title "LoRA v2 vs Prompt v2 — Paired Modality Ablation (synth_v0.4, 50 holdout cases)"
        ok "Paired ablation charts saved to: $PLOTS_DIR"
    else
        warn "Not enough trace files for paired comparison"
    fi
}

case "$STEP" in
    full)
        run_modal_extraction "$CONDITION_OVERRIDE"
        run_local_pipeline "$CONDITION_OVERRIDE"
        run_comparison_charts
        ;;
    modal)
        run_modal_extraction "$CONDITION_OVERRIDE"
        ;;
    local)
        run_local_pipeline "$CONDITION_OVERRIDE"
        ;;
    plots)
        run_comparison_charts
        ;;
    paired-ablation)
        run_paired_ablation
        ;;
    overall)
        run_overall_comparison
        ;;
    *)
        fail "Unknown step: $STEP (use: full, modal, local, plots, paired-ablation, overall)"
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
