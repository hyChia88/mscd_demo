#!/usr/bin/env bash
# ============================================================================
# MSCD LoRA_4 Evaluation Pipeline
#
# End-to-end pipeline for evaluating LoRA_4-finetuned Qwen2.5-VL:
#   Step 1: Modal GPU extraction (LoRA_4 constraints) — MA/MB/MC conditions
#   Step 2: Download precomputed constraints from Modal volume
#   Step 3: Local pipeline (retrieval + scoring via Neo4j)
#   Step 4: Full metrics analysis (GT-in-pool, P0, VLM extraction)
#   Step 5: H2 hard-negative evaluation (213 oracle cases)
#   Step 6: Comparison charts (LoRA_4 vs LoRA_3)
#
# Changes from eval.sh (LoRA_3):
#   - Adapter path: /mscd-lora-v4/final (was /mscd-lora/final)
#   - Uses eval_lora4.py (max_new_tokens=512 for multi-triplet JSON)
#   - Output dir: output/synth_v05_lora4
#   - Adds H2 eval step (213 hard-negative topology cases)
#   - Adds LoRA_3 vs LoRA_4 comparison step
#
# Usage:
#   ./training/eval_lora4.sh                          # Full pipeline
#   ./training/eval_lora4.sh --step modal             # Step 1-2: Modal extraction
#   ./training/eval_lora4.sh --step local             # Step 3-4: Local pipeline + analyze
#   ./training/eval_lora4.sh --step analyze           # Step 4: Full metrics table only
#   ./training/eval_lora4.sh --step h2                # Step 5: H2 eval only
#   ./training/eval_lora4.sh --step compare           # Step 6: Comparison charts
#   ./training/eval_lora4.sh --step quick             # Quick test (3 cases)
#   ./training/eval_lora4.sh --limit 5                # Limit Modal extraction
# ============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_DIR")/data_curation"

# Eval case files
V04_CASES="$DATA_ROOT/datasets/synth_v0.4_merged/train/test_holdout_with_images.jsonl"
V05_CASES="$PROJECT_DIR/eval/cases_v4_test.jsonl"
H2_CASES="$DATA_ROOT/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl"

# Output dirs
EVAL_DIR="$PROJECT_DIR/output/synth_v05_lora4"
PLOTS_DIR="$PROJECT_DIR/plots/comparisons/lora4_vs_lora3"

# LoRA_3 traces (for comparison)
LORA3_EVAL_DIR="$PROJECT_DIR/output/synth_v04"

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

STEP="full"
LIMIT_ARG=""
CONDA_ENV="mscd_demo"
ADAPTER_DIR="/mscd-lora-v4/final"
CASES_SET="v05"              # v04 | v05 | both

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2"; shift 2 ;;
        --limit)         LIMIT_ARG="--limit $2"; shift 2 ;;
        --conda)         CONDA_ENV="$2"; shift 2 ;;
        --adapter-dir)   ADAPTER_DIR="$2"; shift 2 ;;
        --cases-set)     CASES_SET="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./training/eval_lora4.sh [--step STEP] [--limit N]"
            echo ""
            echo "Steps:"
            echo "  full       All steps (modal + local + analyze + h2 + compare)"
            echo "  modal      Modal GPU extraction only (MA/MB/MC)"
            echo "  local      Local pipeline + analyze (needs precomputed)"
            echo "  analyze    Full metrics table only (needs traces)"
            echo "  h2         H2 hard-negative evaluation only"
            echo "  compare    Comparison charts only"
            echo "  quick      Quick test (3 cases, MB condition)"
            echo ""
            echo "Options:"
            echo "  --limit N           Limit to first N cases"
            echo "  --adapter-dir PATH  Modal volume adapter path (default: /mscd-lora-v4/final)"
            echo "  --cases-set SET     v04 | v05 | both (default: v05)"
            exit 0 ;;
        *)
            warn "Unknown argument: $1"; shift ;;
    esac
done

# ── Timestamp for this run ───────────────────────────────────────────────────

RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$EVAL_DIR" "$PLOTS_DIR"
LOG_FILE="$EVAL_DIR/eval_lora4_${RUN_TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run timestamp: $RUN_TS"
echo "Log file:      $LOG_FILE"
echo "Step:          $STEP"
echo "Adapter:       $ADAPTER_DIR"
echo "Cases set:     $CASES_SET"
echo ""

# ── Pre-flight checks ───────────────────────────────────────────────────────

section "Pre-flight Checks"

# v0.5 cases (75 topology cases)
if [[ -f "$V05_CASES" ]]; then
    V05_COUNT=$(wc -l < "$V05_CASES")
    ok "v0.5 cases: $V05_COUNT cases ($V05_CASES)"
else
    fail "v0.5 cases not found: $V05_CASES"
fi

# v0.4 cases (50 holdout)
if [[ -f "$V04_CASES" ]]; then
    V04_COUNT=$(wc -l < "$V04_CASES")
    ok "v0.4 cases: $V04_COUNT cases ($V04_CASES)"
else
    warn "v0.4 cases not found: $V04_CASES"
fi

# H2 cases
if [[ -f "$H2_CASES" ]]; then
    H2_COUNT=$(wc -l < "$H2_CASES")
    ok "H2 cases:  $H2_COUNT cases ($H2_CASES)"
else
    warn "H2 cases not found: $H2_CASES"
fi

# Modal CLI
if [[ "$STEP" == "full" || "$STEP" == "modal" || "$STEP" == "quick" ]]; then
    if command -v modal &>/dev/null; then
        ok "Modal CLI installed"
    else
        fail "Modal CLI not found. Install: pip install modal && modal setup"
    fi
fi

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Modal GPU Extraction (MA/MB/MC conditions)
# ═════════════════════════════════════════════════════════════════════════════

run_modal_extraction() {
    section "Step 1: Modal GPU LoRA_4 Extraction"

    local CONDITIONS=("MB" "MC")  # MA = text-only (no images), MB = images, MC = images+floorplan
    local CASES_FILES=()
    local CASES_LABELS=()

    # Determine which case files to use
    case "$CASES_SET" in
        v05)
            CASES_FILES=("/data/v05_test.jsonl")
            CASES_LABELS=("v05")
            ;;
        v04)
            CASES_FILES=("")  # empty = default (v0.4 holdout)
            CASES_LABELS=("v04")
            ;;
        both)
            CASES_FILES=("/data/v05_test.jsonl" "")
            CASES_LABELS=("v05" "v04")
            ;;
    esac

    for ci in "${!CASES_FILES[@]}"; do
        local cases_file="${CASES_FILES[$ci]}"
        local cases_label="${CASES_LABELS[$ci]}"
        local cases_arg=""
        [[ -n "$cases_file" ]] && cases_arg="--cases $cases_file"

        for cond in "${CONDITIONS[@]}"; do
            # eval_lora4.py saves as eval_constraints_final_{COND}.jsonl
            # (tag = adapter_dir_basename + "_" + condition_override)
            local tag="final_${cond}"
            local dest="$EVAL_DIR/eval_constraints_${tag}.jsonl"

            if [[ -f "$dest" ]]; then
                local count
                count=$(wc -l < "$dest")
                ok "Already exists: $(basename "$dest") ($count cases) — skipping"
                continue
            fi

            info "Running Modal extraction: cases=$cases_label  condition=$cond"
            cd "$PROJECT_DIR"
            modal run training/eval_lora4.py \
                --adapter-dir "$ADAPTER_DIR" \
                --condition-override "$cond" \
                $cases_arg $LIMIT_ARG

            info "Downloading precomputed constraints..."
            modal volume get --force mscd-checkpoints \
                "/mscd-lora-v4/eval_constraints_${tag}.jsonl" \
                "$EVAL_DIR/"

            if [[ -f "$dest" ]]; then
                local count
                count=$(wc -l < "$dest")
                ok "Downloaded: $(basename "$dest") ($count cases)"
            else
                fail "Download failed: $dest"
            fi
        done
    done

    # Also run MA (text-only) for ablation
    info "Running Modal extraction: condition=MA (text-only baseline)"
    local ma_dest="$EVAL_DIR/eval_constraints_final_MA.jsonl"
    if [[ -f "$ma_dest" ]]; then
        ok "Already exists: $(basename "$ma_dest") — skipping"
    else
        cd "$PROJECT_DIR"
        local cases_arg_ma=""
        [[ "$CASES_SET" == "v05" || "$CASES_SET" == "both" ]] && cases_arg_ma="--cases /data/v05_test.jsonl"
        modal run training/eval_lora4.py \
            --adapter-dir "$ADAPTER_DIR" \
            --condition-override "MA" \
            $cases_arg_ma $LIMIT_ARG

        modal volume get --force mscd-checkpoints \
            "/mscd-lora-v4/eval_constraints_final_MA.jsonl" \
            "$EVAL_DIR/"

        if [[ -f "$ma_dest" ]]; then
            local count
            count=$(wc -l < "$ma_dest")
            ok "Downloaded: $(basename "$ma_dest") ($count cases)"
        else
            warn "MA download failed — not critical, continuing"
        fi
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Local Pipeline (Retrieval + Scoring)
# ═════════════════════════════════════════════════════════════════════════════

run_local_pipeline() {
    section "Step 2: Local Pipeline (Retrieval + Scoring)"

    cd "$PROJECT_DIR"

    local CONDITIONS=("MA" "MB" "MC")
    local CASES_FILE="$V05_CASES"
    [[ "$CASES_SET" == "v04" ]] && CASES_FILE="$V04_CASES"

    for cond in "${CONDITIONS[@]}"; do
        local tag="final_${cond}"
        local precomputed_file="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        if [[ ! -f "$precomputed_file" ]]; then
            warn "Precomputed constraints missing for $tag — run --step modal first"
            continue
        fi

        # Check if traces already exist
        local existing_trace
        existing_trace=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$existing_trace" ]]; then
            ok "Traces exist for $cond: $(basename "$existing_trace") — skipping"
            continue
        fi

        info "Running local pipeline: condition=$cond"
        conda run -n "$CONDA_ENV" python -u script/run.py \
            --profile v2_lora \
            --cases "$CASES_FILE" \
            --precomputed "$precomputed_file" \
            --output_dir "$EVAL_DIR" \
            --condition-override "$cond" \
            $LIMIT_ARG

        ok "Local pipeline complete for $cond"
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: H2 Hard-Negative Evaluation
# Uses oracle constraints (ground truth from skeletons) — tests the retrieval
# layer independently of VLM extraction quality.
# ═════════════════════════════════════════════════════════════════════════════

run_h2_eval() {
    section "Step 3: H2 Hard-Negative Evaluation"

    cd "$PROJECT_DIR"

    if [[ ! -f "$H2_CASES" ]]; then
        warn "H2 cases not found: $H2_CASES — skipping"
        return
    fi

    local h2_output="$EVAL_DIR/h2_results_${RUN_TS}.jsonl"
    local h2_plot="$PLOTS_DIR/h2_lora4_${RUN_TS}.png"

    info "Running H2 evaluation ($H2_COUNT cases)..."
    conda run -n "$CONDA_ENV" python -u eval_h2_spatial_triplets/h2_eval.py \
        --h2 "$H2_CASES" \
        --output "$h2_output" \
        --plot "$h2_plot"

    ok "H2 evaluation complete"
    [[ -f "$h2_output" ]] && ok "Results: $h2_output"
    [[ -f "$h2_plot" ]] && ok "Plot: $h2_plot"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Comparison Charts (LoRA_4 vs LoRA_3)
# ═════════════════════════════════════════════════════════════════════════════

run_comparison() {
    section "Step 4: LoRA_4 vs LoRA_3 Comparison"

    cd "$PROJECT_DIR"

    local plot_args=()

    # ── Collect LoRA_4 traces ────────────────────────────────────────────
    for cond in "MB" "MC"; do
        local latest
        latest=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            plot_args+=(--traces "$latest" --label "LoRA_4_${cond}")
        fi
    done

    # ── Collect LoRA_3 traces (for comparison) ───────────────────────────
    if [[ -d "$LORA3_EVAL_DIR" ]]; then
        for cond in "MB" "MC"; do
            local latest
            latest=$(ls -t "$LORA3_EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1)
            if [[ -n "$latest" ]]; then
                plot_args+=(--traces "$latest" --label "LoRA_3_${cond}")
            fi
        done
    else
        warn "LoRA_3 eval dir not found: $LORA3_EVAL_DIR"
    fi

    if [[ ${#plot_args[@]} -lt 4 ]]; then
        warn "Not enough trace files for comparison (need at least 2 series)"
        return
    fi

    info "Generating comparison charts..."
    conda run -n "$CONDA_ENV" python -u script/compare_results.py \
        "${plot_args[@]}" \
        --cases "$V05_CASES" \
        --plots \
        --output "$PLOTS_DIR" \
        --title "LoRA_4 vs LoRA_3 — v0.5 Topology Cases (75 cases)"

    ok "Comparison charts saved to: $PLOTS_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Analyze traces (full metrics table)
# ═════════════════════════════════════════════════════════════════════════════

run_analyze() {
    section "Step 5: Full Metrics Analysis"

    cd "$PROJECT_DIR"

    local gt_labels_arg=""
    local gt_labels_file="$DATA_ROOT/datasets/synth_v0.5/train/lora4_test.jsonl"
    if [[ -f "$gt_labels_file" ]]; then
        gt_labels_arg="--gt-labels $gt_labels_file"
        ok "GT labels: $gt_labels_file"
    else
        warn "GT labels not found: $gt_labels_file — SR accuracy will be skipped"
    fi

    local output_csv="$EVAL_DIR/lora4_metrics_${RUN_TS}.csv"

    info "Running full metrics analysis (MA/MB/MC)..."
    conda run -n "$CONDA_ENV" python -u eval/analyze_traces.py --full \
        --traces-dir "$EVAL_DIR" \
        --precomputed-dir "$EVAL_DIR" \
        --cases "$V05_CASES" \
        $gt_labels_arg \
        --output "$output_csv"

    ok "Full metrics analysis complete"
    [[ -f "$output_csv" ]] && ok "CSV: $output_csv"
}

# ═════════════════════════════════════════════════════════════════════════════
# Quick test (3 cases, MB condition)
# ═════════════════════════════════════════════════════════════════════════════

run_quick_test() {
    section "Quick Test (5 cases, MC condition — verifies SR extraction)"

    cd "$PROJECT_DIR"

    info "Running Modal extraction (5 cases, MC — includes floorplan for SR)..."
    modal run training/eval_lora4.py \
        --adapter-dir "$ADAPTER_DIR" \
        --cases /data/v05_test.jsonl \
        --condition-override MC \
        --limit 5

    info "Downloading results..."
    modal volume get --force mscd-checkpoints \
        "/mscd-lora-v4/eval_constraints_final_MC.jsonl" \
        "$EVAL_DIR/"

    local precomp="$EVAL_DIR/eval_constraints_final_MC.jsonl"
    if [[ -f "$precomp" ]]; then
        ok "Quick test precomputed: $(wc -l < "$precomp") cases"

        info "SR extraction summary:"
        python3 -c "
import json
with open('$precomp') as f:
    results = [json.loads(l) for l in f if l.strip()]
n_sr = sum(1 for r in results if r['constraints'].get('spatial_relations'))
n_hop2 = sum(1 for r in results if len(r['constraints'].get('spatial_relations', [])) >= 2)
print(f'  Total: {len(results)} cases')
print(f'  SR extracted: {n_sr}/{len(results)}')
print(f'  2-hop: {n_hop2}/{len(results)}')
for r in results:
    sr = r['constraints'].get('spatial_relations', [])
    preds = [s.get('predicate','?') for s in sr]
    print(f\"  {r['case_id']}: {r['status']}  SR={preds}  storey={r['constraints'].get('storey_name')}\")
"
    else
        fail "Quick test failed: precomputed file not found"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# Execute selected steps
# ═════════════════════════════════════════════════════════════════════════════

case "$STEP" in
    full)
        run_modal_extraction
        run_local_pipeline
        run_analyze
        run_h2_eval
        run_comparison
        ;;
    modal)
        run_modal_extraction
        ;;
    local)
        run_local_pipeline
        run_analyze
        ;;
    analyze)
        run_analyze
        ;;
    h2)
        run_h2_eval
        ;;
    compare)
        run_comparison
        ;;
    quick)
        run_quick_test
        ;;
    *)
        fail "Unknown step: $STEP (use: full, modal, local, analyze, h2, compare, quick)"
        ;;
esac

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

section "LoRA_4 Evaluation Pipeline Complete"

echo "Run:      $RUN_TS"
echo "Log:      $LOG_FILE"
echo "Traces:   $EVAL_DIR/traces_*_v2_lora_*.jsonl"
echo "Metrics:  $EVAL_DIR/lora4_metrics_${RUN_TS}.csv"
echo "Charts:   $PLOTS_DIR/"
echo ""
echo "Quick commands:"
echo "  # Re-run analysis only"
echo "  ./training/eval_lora4.sh --step analyze"
echo ""
echo "  # Analyze single trace file"
echo "  python eval/analyze_traces.py $EVAL_DIR/traces_*_v2_lora_MC.jsonl"
