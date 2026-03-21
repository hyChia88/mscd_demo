#!/usr/bin/env bash
# ============================================================================
# MSCD LoRA_5 Evaluation Pipeline
#
# End-to-end pipeline for evaluating LoRA_5-finetuned Qwen2.5-VL:
#   Step 1: Modal GPU extraction (LoRA_5 constraints) — MA/MB/MC + FP/SITE
#   Step 2: Download precomputed constraints from Modal volume
#   Step 3: Local pipeline (retrieval + scoring via Neo4j)
#   Step 4: Full metrics analysis (GT-in-pool, P0, VLM extraction)
#   Step 5: H2 hard-negative evaluation (568 oracle cases)
#   Step 6: Comparison charts (LoRA_5 vs LoRA_4 vs LoRA_3)
#   Step 7: Modality ablation analysis (Eval-A: FP, Eval-B: MC, Eval-C: SITE)
#
# Changes from eval_lora4.sh:
#   - Adapter path: /mscd-lora-v5/final
#   - Uses eval_lora5.py (modality ablation support via --modality flag)
#   - Output dir: logs/evaluation_output/synth_v05_lora5
#   - 92 test cases (was 75) with stratified coverage
#   - H2 eval: 568 cases across 5 edge types (was 213)
#   - 3-way comparison: LoRA_5 vs LoRA_4 vs LoRA_3
#   - New: modality ablation step (FP-only, SITE-only)
#
# Usage:
#   ./training/eval_lora5.sh                          # Full pipeline
#   ./training/eval_lora5.sh --step modal             # Step 1-2: Modal extraction
#   ./training/eval_lora5.sh --step local             # Step 3-4: Local pipeline + analyze
#   ./training/eval_lora5.sh --step analyze           # Step 4: Full metrics table only
#   ./training/eval_lora5.sh --step h2                # Step 5: H2 eval only
#   ./training/eval_lora5.sh --step compare           # Step 6: 3-way comparison
#   ./training/eval_lora5.sh --step modality-ablation # Step 7: FP vs MC vs SITE
#   ./training/eval_lora5.sh --step quick             # Quick test (5 cases)
#   ./training/eval_lora5.sh --limit 5                # Limit Modal extraction
# ============================================================================

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="$(dirname "$PROJECT_DIR")/data_curation"

# Eval case files
V05_CASES="$PROJECT_DIR/eval/cases_v5_test.jsonl"
H2_CASES="$DATA_ROOT/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl"

# GT labels (training-format test set — for SR accuracy scoring)
GT_LABELS="$DATA_ROOT/datasets/synth_v0.5/train/lora5_test.jsonl"

# Output dirs
EVAL_DIR="$PROJECT_DIR/logs/evaluation_output/synth_v05_lora5"
PLOTS_DIR="$PROJECT_DIR/logs/comparisons/$(date +%Y%m%d)_plots"

# Prior eval dirs (for comparison)
LORA4_EVAL_DIR="$PROJECT_DIR/logs/evaluation_output/synth_v05_lora4"
LORA3_EVAL_DIR="$PROJECT_DIR/logs/evaluation_output/synth_v04"

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
ADAPTER_DIR="/mscd-lora-v5/final"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)          STEP="$2"; shift 2 ;;
        --limit)         LIMIT_ARG="--limit $2"; shift 2 ;;
        --conda)         CONDA_ENV="$2"; shift 2 ;;
        --adapter-dir)   ADAPTER_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: ./training/eval_lora5.sh [--step STEP] [--limit N]"
            echo ""
            echo "Steps:"
            echo "  full               All steps (modal + local + analyze + h2 + compare + modality-ablation)"
            echo "  modal              Modal GPU extraction (MA/MB/MC + FP/SITE)"
            echo "  local              Local pipeline + analyze (needs precomputed)"
            echo "  analyze            Full metrics table only (needs traces)"
            echo "  h2                 H2 hard-negative evaluation (568 cases)"
            echo "  compare            3-way comparison (LoRA_5 vs LoRA_4 vs LoRA_3)"
            echo "  modality-ablation  Modality ablation (FP vs MC vs SITE vs MA)"
            echo "  strategy-ablation  Retrieval strategy ablation (P0 vs P1 vs P0∩P1 vs P0∪P1)"
            echo "  quick              Quick test (5 cases, MC condition)"
            echo ""
            echo "Options:"
            echo "  --limit N           Limit to first N cases"
            echo "  --adapter-dir PATH  Modal volume adapter path (default: /mscd-lora-v5/final)"
            exit 0 ;;
        *)
            warn "Unknown argument: $1"; shift ;;
    esac
done

# ── Timestamp for this run ───────────────────────────────────────────────────

RUN_TS=$(date +%Y%m%d_%H%M%S)
mkdir -p "$EVAL_DIR" "$PLOTS_DIR"
LOG_FILE="$EVAL_DIR/eval_lora5_${RUN_TS}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run timestamp: $RUN_TS"
echo "Log file:      $LOG_FILE"
echo "Step:          $STEP"
echo "Adapter:       $ADAPTER_DIR"
echo ""

# ── Pre-flight checks ───────────────────────────────────────────────────────

section "Pre-flight Checks"

# v0.5 cases (92 topology cases)
if [[ -f "$V05_CASES" ]]; then
    V05_COUNT=$(wc -l < "$V05_CASES")
    ok "v0.5 cases: $V05_COUNT cases ($V05_CASES)"
else
    warn "v0.5 cases not found: $V05_CASES — will be auto-generated by eval_lora5.py"
    V05_COUNT=0
fi

# GT labels
if [[ -f "$GT_LABELS" ]]; then
    GT_COUNT=$(wc -l < "$GT_LABELS")
    ok "GT labels:  $GT_COUNT records ($GT_LABELS)"
else
    warn "GT labels not found: $GT_LABELS"
fi

# H2 cases
if [[ -f "$H2_CASES" ]]; then
    H2_COUNT=$(wc -l < "$H2_CASES")
    ok "H2 cases:  $H2_COUNT cases ($H2_CASES)"
else
    warn "H2 cases not found: $H2_CASES"
    H2_COUNT=0
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
# STEP 1: Modal GPU Extraction (MA/MB/MC + FP/SITE modality ablation)
# ═════════════════════════════════════════════════════════════════════════════

run_modal_extraction() {
    section "Step 1: Modal GPU LoRA_5 Extraction"

    cd "$PROJECT_DIR"

    # Standard conditions: MB, MC (MA handled separately below)
    # Modality ablation: FP (floorplan-only), SITE (site-photo-only)
    local STANDARD_CONDITIONS=("MB" "MC")
    local MODALITY_MODES=("FP" "SITE")

    # ── Standard conditions (MB, MC) ─────────────────────────────────────
    for cond in "${STANDARD_CONDITIONS[@]}"; do
        local tag="final_${cond}"
        local dest="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        if [[ -f "$dest" ]]; then
            local count
            count=$(wc -l < "$dest")
            ok "Already exists: $(basename "$dest") ($count cases) — skipping"
            continue
        fi

        info "Running Modal extraction: condition=$cond"
        modal run training/eval_lora5.py \
            --adapter-dir "$ADAPTER_DIR" \
            --condition-override "$cond" \
            --cases /data/v05_test.jsonl \
            $LIMIT_ARG

        info "Downloading precomputed constraints..."
        modal volume get --force mscd-checkpoints \
            "/mscd-lora-v5/eval_constraints_${tag}.jsonl" \
            "$EVAL_DIR/"

        if [[ -f "$dest" ]]; then
            local count
            count=$(wc -l < "$dest")
            ok "Downloaded: $(basename "$dest") ($count cases)"
        else
            fail "Download failed: $dest"
        fi
    done

    # ── MA (text-only baseline) ──────────────────────────────────────────
    local ma_dest="$EVAL_DIR/eval_constraints_final_MA.jsonl"
    if [[ -f "$ma_dest" ]]; then
        ok "Already exists: $(basename "$ma_dest") — skipping"
    else
        info "Running Modal extraction: condition=MA (text-only baseline)"
        modal run training/eval_lora5.py \
            --adapter-dir "$ADAPTER_DIR" \
            --condition-override "MA" \
            --cases /data/v05_test.jsonl \
            $LIMIT_ARG

        modal volume get --force mscd-checkpoints \
            "/mscd-lora-v5/eval_constraints_final_MA.jsonl" \
            "$EVAL_DIR/"

        if [[ -f "$ma_dest" ]]; then
            local count
            count=$(wc -l < "$ma_dest")
            ok "Downloaded: $(basename "$ma_dest") ($count cases)"
        else
            warn "MA download failed — not critical, continuing"
        fi
    fi

    # ── Modality ablation conditions (FP-only, SITE-only) ───────────────
    for mod in "${MODALITY_MODES[@]}"; do
        local tag="final_${mod}"
        local dest="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        if [[ -f "$dest" ]]; then
            local count
            count=$(wc -l < "$dest")
            ok "Already exists: $(basename "$dest") ($count cases) — skipping"
            continue
        fi

        info "Running Modal extraction: modality=$mod (ablation)"
        modal run training/eval_lora5.py \
            --adapter-dir "$ADAPTER_DIR" \
            --condition-override "MC" \
            --modality "$mod" \
            --cases /data/v05_test.jsonl \
            $LIMIT_ARG

        modal volume get --force mscd-checkpoints \
            "/mscd-lora-v5/eval_constraints_${tag}.jsonl" \
            "$EVAL_DIR/"

        if [[ -f "$dest" ]]; then
            local count
            count=$(wc -l < "$dest")
            ok "Downloaded: $(basename "$dest") ($count cases)"
        else
            warn "$mod ablation download failed — continuing"
        fi
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Local Pipeline (Retrieval + Scoring)
# ═════════════════════════════════════════════════════════════════════════════

run_local_pipeline() {
    section "Step 2: Local Pipeline (Retrieval + Scoring)"

    cd "$PROJECT_DIR"

    local CONDITIONS=("MA" "MB" "MC" "FP" "SITE")
    local CASES_FILE="$V05_CASES"

    for cond in "${CONDITIONS[@]}"; do
        local tag="final_${cond}"
        local precomputed_file="$EVAL_DIR/eval_constraints_${tag}.jsonl"

        if [[ ! -f "$precomputed_file" ]]; then
            warn "Precomputed constraints missing for $tag — run --step modal first"
            continue
        fi

        # Check if traces already exist
        local existing_trace
        existing_trace=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}_*.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$existing_trace" ]]; then
            ok "Traces exist for $cond: $(basename "$existing_trace") — skipping"
            continue
        fi

        info "Running local pipeline: condition=$cond (p0_strategy=p0_intersect_p1)"
        conda run -n "$CONDA_ENV" python -u script/run.py \
            --profile v2_lora \
            --cases "$CASES_FILE" \
            --precomputed "$precomputed_file" \
            --output_dir "$EVAL_DIR" \
            --condition-override "$cond" \
            --p0-strategy p0_intersect_p1 \
            $LIMIT_ARG

        ok "Local pipeline complete for $cond"
    done
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: H2 Hard-Negative Evaluation (568 cases, 5 edge types)
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
    local h2_plot="$PLOTS_DIR/h2_lora5_${RUN_TS}.png"

    info "Running H2 evaluation ($H2_COUNT cases, 5 edge types)..."
    conda run -n "$CONDA_ENV" python -u eval_h2_spatial_triplets/h2_eval.py \
        --h2 "$H2_CASES" \
        --output "$h2_output" \
        --plot "$h2_plot"

    ok "H2 evaluation complete"
    [[ -f "$h2_output" ]] && ok "Results: $h2_output"
    [[ -f "$h2_plot" ]] && ok "Plot: $h2_plot"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: 3-Way Comparison (LoRA_5 vs LoRA_4 vs LoRA_3)
# ═════════════════════════════════════════════════════════════════════════════

run_comparison() {
    section "Step 4: LoRA_5 vs LoRA_4 vs LoRA_3 Comparison"

    cd "$PROJECT_DIR"

    local plot_args=()

    # ── Collect LoRA_5 traces ────────────────────────────────────────────
    for cond in "MC"; do
        local latest
        latest=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            plot_args+=(--traces "$latest" --label "LoRA_5_${cond}")
        fi
    done

    # ── Collect LoRA_4 traces ────────────────────────────────────────────
    if [[ -d "$LORA4_EVAL_DIR" ]]; then
        local latest
        latest=$(ls -t "$LORA4_EVAL_DIR"/traces_*_v2_lora_MC.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            plot_args+=(--traces "$latest" --label "LoRA_4_MC")
        fi
    else
        warn "LoRA_4 eval dir not found: $LORA4_EVAL_DIR"
    fi

    # ── Collect LoRA_3 traces ────────────────────────────────────────────
    if [[ -d "$LORA3_EVAL_DIR" ]]; then
        local latest
        latest=$(ls -t "$LORA3_EVAL_DIR"/traces_*_v2_lora_MC.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            plot_args+=(--traces "$latest" --label "LoRA_3_MC")
        fi
    else
        warn "LoRA_3 eval dir not found: $LORA3_EVAL_DIR"
    fi

    if [[ ${#plot_args[@]} -lt 4 ]]; then
        warn "Not enough trace files for comparison (need at least 2 series)"
        return
    fi

    info "Generating 3-way comparison charts..."
    conda run -n "$CONDA_ENV" python -u script/compare_results.py \
        "${plot_args[@]}" \
        --cases "$V05_CASES" \
        --plots \
        --output "$PLOTS_DIR" \
        --title "LoRA_5 vs LoRA_4 vs LoRA_3 — v0.5 Topology Cases (92 cases)"

    ok "Comparison charts saved to: $PLOTS_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Analyze traces (full metrics table)
# ═════════════════════════════════════════════════════════════════════════════

run_analyze() {
    section "Step 5: Full Metrics Analysis"

    cd "$PROJECT_DIR"

    local gt_labels_arg=""
    if [[ -f "$GT_LABELS" ]]; then
        gt_labels_arg="--gt-labels $GT_LABELS"
        ok "GT labels: $GT_LABELS"
    else
        warn "GT labels not found: $GT_LABELS — SR accuracy will be skipped"
    fi

    local output_csv="$EVAL_DIR/lora5_metrics_${RUN_TS}.csv"

    info "Running full metrics analysis (MA/MB/MC/FP/SITE)..."
    conda run -n "$CONDA_ENV" python -u eval/analyze_traces.py --full \
        --traces-dir "$EVAL_DIR" \
        --precomputed-dir "$EVAL_DIR" \
        --cases "$V05_CASES" \
        --conditions "MA,MB,MC,FP,SITE" \
        $gt_labels_arg \
        --output "$output_csv"

    ok "Full metrics analysis complete"
    [[ -f "$output_csv" ]] && ok "CSV: $output_csv"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Modality Ablation Analysis (FP vs MC vs SITE vs MA)
# ═════════════════════════════════════════════════════════════════════════════

run_modality_ablation() {
    section "Step 6: Modality Ablation (Eval-A: FP, Eval-B: MC, Eval-C: SITE, MA)"

    cd "$PROJECT_DIR"

    local plot_args=()

    for cond in "FP" "MC" "SITE" "MA"; do
        local latest
        latest=$(ls -t "$EVAL_DIR"/traces_*_v2_lora_${cond}.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest" ]]; then
            local label
            case "$cond" in
                FP)   label="Eval-A (FP only)" ;;
                MC)   label="Eval-B (FP+site)" ;;
                SITE) label="Eval-C (site only)" ;;
                MA)   label="Baseline (text only)" ;;
            esac
            plot_args+=(--traces "$latest" --label "$label")
        else
            warn "No traces for $cond — run --step local first"
        fi
    done

    if [[ ${#plot_args[@]} -lt 4 ]]; then
        warn "Not enough trace files for modality ablation (need at least 2 series)"
        return
    fi

    info "Generating modality ablation charts..."
    conda run -n "$CONDA_ENV" python -u script/compare_results.py \
        "${plot_args[@]}" \
        --cases "$V05_CASES" \
        --plots \
        --output "$PLOTS_DIR" \
        --title "Modality Ablation — LoRA_5 (92 cases)"

    ok "Modality ablation charts saved to: $PLOTS_DIR"
}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Retrieval Strategy Ablation (P0-only vs P1-only vs P0∩P1 vs P0∪P1)
# Uses MC precomputed constraints, varies only the retrieval strategy.
# ═════════════════════════════════════════════════════════════════════════════

run_strategy_ablation() {
    section "Step 7: Retrieval Strategy Ablation (MC condition)"

    cd "$PROJECT_DIR"

    local precomputed_file="$EVAL_DIR/eval_constraints_final_MC.jsonl"
    local CASES_FILE="$V05_CASES"
    local STRAT_DIR="$EVAL_DIR/strategy_ablation"
    mkdir -p "$STRAT_DIR"

    if [[ ! -f "$precomputed_file" ]]; then
        warn "MC precomputed constraints missing — run --step modal first"
        return
    fi

    local STRATEGIES=("p0_only" "p1_only" "p0_intersect_p1" "p0_union_p1")

    for strat in "${STRATEGIES[@]}"; do
        local existing_trace
        existing_trace=$(ls -t "$STRAT_DIR"/traces_*_MC_${strat}.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$existing_trace" ]]; then
            ok "Traces exist for $strat: $(basename "$existing_trace") — skipping"
            continue
        fi

        info "Running retrieval with strategy: $strat"
        conda run -n "$CONDA_ENV" python -u script/run.py \
            --profile v2_lora \
            --cases "$CASES_FILE" \
            --precomputed "$precomputed_file" \
            --output_dir "$STRAT_DIR" \
            --condition-override "MC" \
            --p0-strategy "$strat" \
            $LIMIT_ARG

        ok "Strategy $strat complete"
    done

    # Summarize results
    info "Strategy ablation summary:"
    for strat in "${STRATEGIES[@]}"; do
        local trace
        trace=$(ls -t "$STRAT_DIR"/traces_*_MC_${strat}.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$trace" ]]; then
            python3 -c "
import json
traces = [json.loads(l) for l in open('$trace') if l.strip()]
gt_in = sum(1 for t in traces if t.get('gt_in_pool'))
top1 = sum(1 for t in traces if t.get('top1_correct'))
n = len(traces)
avg_pool = sum(t.get('pool_size',0) for t in traces) / max(n,1)
print(f'  $strat: GT-in-pool={gt_in}/{n} ({100*gt_in/max(n,1):.1f}%) Top-1={top1}/{n} ({100*top1/max(n,1):.1f}%) Avg-pool={avg_pool:.0f}')
"
        fi
    done

    # Generate comparison chart
    local plot_args=()
    for strat in "${STRATEGIES[@]}"; do
        local trace
        trace=$(ls -t "$STRAT_DIR"/traces_*_MC_${strat}.jsonl 2>/dev/null | head -1 || true)
        if [[ -n "$trace" ]]; then
            local label
            case "$strat" in
                p0_only)          label="P0 only" ;;
                p1_only)          label="P1 only (skip P0)" ;;
                p0_intersect_p1)  label="P0 ∩ P1 (defensive)" ;;
                p0_union_p1)      label="P0 ∪ P1 (max recall)" ;;
            esac
            plot_args+=(--traces "$trace" --label "$label")
        fi
    done

    if [[ ${#plot_args[@]} -ge 4 ]]; then
        info "Generating strategy ablation charts..."
        conda run -n "$CONDA_ENV" python -u script/compare_results.py \
            "${plot_args[@]}" \
            --cases "$V05_CASES" \
            --plots \
            --output "$PLOTS_DIR" \
            --title "Retrieval Strategy Ablation — LoRA_5 MC"

        ok "Strategy ablation charts saved to: $PLOTS_DIR"
    fi
}

# ═════════════════════════════════════════════════════════════════════════════
# Quick test (5 cases, MC condition — verifies SR extraction)
# ═════════════════════════════════════════════════════════════════════════════

run_quick_test() {
    section "Quick Test (5 cases, MC condition — verifies SR extraction)"

    cd "$PROJECT_DIR"

    info "Running Modal extraction (5 cases, MC — includes floorplan for SR)..."
    modal run training/eval_lora5.py \
        --adapter-dir "$ADAPTER_DIR" \
        --cases /data/v05_test.jsonl \
        --condition-override MC \
        --limit 5

    info "Downloading results..."
    modal volume get --force mscd-checkpoints \
        "/mscd-lora-v5/eval_constraints_final_MC.jsonl" \
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
        run_modality_ablation
        run_strategy_ablation
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
    modality-ablation)
        run_modality_ablation
        ;;
    strategy-ablation)
        run_strategy_ablation
        ;;
    quick)
        run_quick_test
        ;;
    *)
        fail "Unknown step: $STEP (use: full, modal, local, analyze, h2, compare, modality-ablation, strategy-ablation, quick)"
        ;;
esac

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

section "LoRA_5 Evaluation Pipeline Complete"

echo "Run:      $RUN_TS"
echo "Log:      $LOG_FILE"
echo "Traces:   $EVAL_DIR/traces_*_v2_lora_*.jsonl"
echo "Metrics:  $EVAL_DIR/lora5_metrics_${RUN_TS}.csv"
echo "Charts:   $PLOTS_DIR/"
echo ""
echo "Quick commands:"
echo "  # Re-run analysis only"
echo "  ./training/eval_lora5.sh --step analyze"
echo ""
echo "  # Modality ablation only"
echo "  ./training/eval_lora5.sh --step modality-ablation"
echo ""
echo "  # Analyze single trace file"
echo "  python eval/analyze_traces.py \$EVAL_DIR/traces_*_v2_lora_MC.jsonl"
