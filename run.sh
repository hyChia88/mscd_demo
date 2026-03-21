#!/bin/bash
# =============================================================================
# Unified Evaluation Runner
#
# Subcommands:
#   mcp             Run V1 agent-driven pipeline (single or batch)
#   experiment      Run a named experiment from experiments.yaml
#   vlm-compare     Run the main thesis VLM before/after comparison
#
# Usage:
#   ./run.sh mcp                              # Single V1 run (config.yaml defaults)
#   ./run.sh mcp -e memory                    # Single V1 experiment mode
#   ./run.sh mcp --all                        # Run ALL 4 V1 modes sequentially
#   ./run.sh mcp --all --v2                   # V1 modes + V2 profiles
#   ./run.sh mcp --all --delay 15             # Custom delay between runs
#   ./run.sh experiment <name>                # Run named experiment
#   ./run.sh experiment list                  # List available experiments
#   ./run.sh vlm-compare                      # Full VLM before/after comparison
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Shared helpers ──────────────────────────────────────────────────────────

check_env() {
    if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo "Warning: No virtual environment detected."
        echo "   Recommended: conda activate mscd_demo"
        echo ""
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo "Conda environment: $CONDA_DEFAULT_ENV"
        echo ""
    elif [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual environment: $(basename $VIRTUAL_ENV)"
        echo ""
    fi
}

check_deps() {
    if ! python -c "import fastmcp" 2>/dev/null; then
        echo "Installing MCP dependencies..."
        pip install fastmcp mcp
        echo ""
    fi

    if ! python -c "import langchain_mcp_adapters" 2>/dev/null; then
        echo "Installing langchain-mcp-adapters..."
        pip install langchain-mcp-adapters 2>/dev/null || echo "   Skipping (optional)"
        echo ""
    fi
}

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# ── Subcommand: mcp ────────────────────────────────────────────────────────

cmd_mcp() {
    echo "=================================="
    echo "  BIM Inspection Agent (MCP Mode)"
    echo "=================================="
    echo ""

    check_env
    check_deps

    # Parse arguments: intercept --all, --delay, --v2; pass the rest to main_mcp.py
    local RUN_ALL=false
    local RUN_V2=false
    local DELAY=10
    local PASSTHROUGH_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)   RUN_ALL=true; shift ;;
            --v2)    RUN_V2=true; shift ;;
            --delay) DELAY="$2"; shift 2 ;;
            *)       PASSTHROUGH_ARGS+=("$1"); shift ;;
        esac
    done

    # ── Single run ──────────────────────────────────────────────────────────

    if [ "$RUN_ALL" = false ]; then
        echo "Starting MCP-based agent..."
        echo ""
        python src/main_mcp.py "${PASSTHROUGH_ARGS[@]}"
        echo ""
        echo "Session complete."
        return 0
    fi

    # ── Batch mode (--all) ──────────────────────────────────────────────────

    local V1_EXPERIMENTS=("memory" "neo4j" "memory+clip" "neo4j+clip")
    local V2_PROFILES=("v2_prompt" "v2_memory")
    local V2_CASES="data_curation/datasets/synth_v0.2/cases_v2.jsonl"

    echo "=============================================="
    echo "  RUNNING ALL EXPERIMENTS"
    echo "=============================================="
    echo ""
    echo "  V1 modes:  ${V1_EXPERIMENTS[*]}"
    if [ "$RUN_V2" = true ]; then
        echo "  V2 profiles: ${V2_PROFILES[*]}"
    fi
    echo "  Delay:     ${DELAY}s between runs"
    echo "  Extra args: ${PASSTHROUGH_ARGS[*]:-none}"
    echo ""

    local BATCH_START=$(date +%s)
    local FAILED=()
    local SUCCEEDED=()

    # ── V1 experiments ──────────────────────────────────────────────────────

    for i in "${!V1_EXPERIMENTS[@]}"; do
        local exp="${V1_EXPERIMENTS[$i]}"
        local run_num=$((i + 1))
        local total=${#V1_EXPERIMENTS[@]}

        echo ""
        echo "======================================================"
        echo "  [${run_num}/${total}] V1 experiment: ${exp}"
        echo "======================================================"
        echo ""

        if python src/main_mcp.py --experiment "$exp" "${PASSTHROUGH_ARGS[@]}"; then
            SUCCEEDED+=("v1:${exp}")
            echo "  [${run_num}/${total}] ${exp} -- DONE"
        else
            FAILED+=("v1:${exp}")
            echo "  [${run_num}/${total}] ${exp} -- FAILED (continuing)"
        fi

        if [ "$run_num" -lt "$total" ] || [ "$RUN_V2" = true ]; then
            echo "  Waiting ${DELAY}s before next run..."
            sleep "$DELAY"
        fi
    done

    # ── V2 profiles (optional) ──────────────────────────────────────────────

    if [ "$RUN_V2" = true ]; then
        if [ ! -f "$V2_CASES" ]; then
            local V2_CASES_ABS="${SCRIPT_DIR}/../data_curation/datasets/synth_v0.2/cases_v2.jsonl"
            if [ -f "$V2_CASES_ABS" ]; then
                V2_CASES="$V2_CASES_ABS"
            else
                echo ""
                echo "  WARNING: cases_v2.jsonl not found, skipping V2 profiles"
                RUN_V2=false
            fi
        fi
    fi

    if [ "$RUN_V2" = true ]; then
        for i in "${!V2_PROFILES[@]}"; do
            local profile="${V2_PROFILES[$i]}"
            local run_num=$((i + 1))
            local total=${#V2_PROFILES[@]}

            echo ""
            echo "======================================================"
            echo "  [${run_num}/${total}] V2 profile: ${profile}"
            echo "======================================================"
            echo ""

            if python script/run.py --profile "$profile" --cases "$V2_CASES"; then
                SUCCEEDED+=("v2:${profile}")
                echo "  [${run_num}/${total}] ${profile} -- DONE"
            else
                FAILED+=("v2:${profile}")
                echo "  [${run_num}/${total}] ${profile} -- FAILED (continuing)"
            fi

            if [ "$run_num" -lt "$total" ]; then
                echo "  Waiting ${DELAY}s before next run..."
                sleep "$DELAY"
            fi
        done
    fi

    # ── Summary ─────────────────────────────────────────────────────────────

    local BATCH_END=$(date +%s)
    local BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))
    local BATCH_MINS=$(( BATCH_ELAPSED / 60 ))
    local BATCH_SECS=$(( BATCH_ELAPSED % 60 ))

    echo ""
    echo "=============================================="
    echo "  ALL EXPERIMENTS COMPLETE"
    echo "=============================================="
    echo ""
    echo "  Succeeded: ${#SUCCEEDED[@]}  (${SUCCEEDED[*]:-none})"
    echo "  Failed:    ${#FAILED[@]}  (${FAILED[*]:-none})"
    echo "  Total time: ${BATCH_MINS}m ${BATCH_SECS}s"
    echo ""

    echo "Running comparison..."
    echo ""
    python script/compare_results.py --latest
    echo ""
    echo "Done. Full results in logs/evaluation_output/"
}

# ── Subcommand: experiment ──────────────────────────────────────────────────

cmd_experiment() {
    if [ -z "${1:-}" ]; then
        echo "Usage: ./run.sh experiment <name|list>"
        echo ""
        echo "Available experiments:"
        python script/experiment.py list
        return 1
    fi

    python script/experiment.py run "$@"
}

# ── Subcommand: vlm-compare ────────────────────────────────────────────────

cmd_vlm_compare() {
    echo "======================================================================"
    echo "  Master Thesis Experiment: VLM Integration Impact"
    echo "======================================================================"
    echo ""
    echo "This will:"
    echo "  1. Run baseline evaluation (before VLM fix)"
    echo "  2. Run VLM-enabled evaluation (after VLM fix)"
    echo "  3. Generate comparison plots"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 1
    fi

    echo ""
    echo "======================================================================"
    echo "  Step 1: Running Baseline (without VLM)"
    echo "======================================================================"
    python script/experiment.py run baseline_v2

    echo ""
    echo "======================================================================"
    echo "  Step 2: Running VLM Integration (with VLM)"
    echo "======================================================================"
    python script/experiment.py run vlm_integration

    echo ""
    echo "======================================================================"
    echo "  Step 3: Generating Comparison Plots"
    echo "======================================================================"
    python script/experiment.py compare vlm_impact

    echo ""
    echo "======================================================================"
    echo "  EXPERIMENT COMPLETE"
    echo "======================================================================"
    echo ""
    echo "Results saved to:"
    echo "  - Baseline:   logs/experiments/baseline_v2/"
    echo "  - VLM:        logs/experiments/vlm_integration/"
    echo "  - Comparison:  logs/comparisons/vlm_impact/"
    echo ""
    echo "Key chart: logs/comparisons/vlm_impact/5_vision_impact.png"
}

# ── Subcommand: paired-ablation ────────────────────────────────────────────

cmd_paired_ablation() {
    echo "======================================================================"
    echo "  Paired Modality Ablation (MA / MB / MC)"
    echo "======================================================================"
    echo ""
    echo "Runs the SAME cases under 3 modality conditions to isolate"
    echo "the true effect of visual evidence (text-only vs img+text vs full)."
    echo ""

    check_env

    # Defaults
    local PROFILE="v2_prompt"
    # Use filtered subset: only cases with actual image + floorplan files on disk.
    # This ensures MB/MC conditions have real visual evidence (not silent fallback).
    local CASES="../data_curation/datasets/synth_v0.3/cases_v3_with_images.jsonl"
    local OUTPUT_DIR="logs/ablation"
    local CHART_DIR="logs/comparisons/v03_full"
    local DELAY=5
    local LIMIT_ARGS=()
    local ADAPTER_PATH=""
    local PASSTHROUGH_ARGS=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)       PROFILE="$2"; shift 2 ;;
            --cases)         CASES="$2"; shift 2 ;;
            --output)        OUTPUT_DIR="$2"; shift 2 ;;
            --chart-dir)     CHART_DIR="$2"; shift 2 ;;
            --delay)         DELAY="$2"; shift 2 ;;
            --limit)         LIMIT_ARGS=("--limit" "$2"); shift 2 ;;
            --percent)       LIMIT_ARGS=("--percent" "$2"); shift 2 ;;
            --adapter-path)  ADAPTER_PATH="$2"; shift 2 ;;
            *)               PASSTHROUGH_ARGS+=("$1"); shift ;;
        esac
    done

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$CHART_DIR"

    local CONDITIONS=("MA" "MB" "MC")
    local BATCH_START=$(date +%s)
    local FAILED=()
    local SUCCEEDED=()
    local TRACE_FILES=()

    echo "  Profile:   ${PROFILE}"
    echo "  Cases:     ${CASES}"
    echo "  Output:    ${OUTPUT_DIR}/"
    echo "  Charts:    ${CHART_DIR}/"
    echo "  Limit:     ${LIMIT_ARGS[*]:-all}"
    if [ -n "$ADAPTER_PATH" ]; then
        echo "  Adapter:   ${ADAPTER_PATH}"
    fi
    echo ""

    # ── Run for each modality condition ──────────────────────────────────────

    for i in "${!CONDITIONS[@]}"; do
        local cond="${CONDITIONS[$i]}"
        local run_num=$((i + 1))

        echo ""
        echo "======================================================"
        echo "  [${run_num}/3] ${PROFILE} × ${cond}"
        echo "======================================================"
        echo ""

        local ADAPTER_ARGS=()
        if [ -n "$ADAPTER_PATH" ]; then
            ADAPTER_ARGS=("--adapter_path" "$ADAPTER_PATH")
        fi

        if python script/run.py \
            --profile "$PROFILE" \
            --cases "$CASES" \
            --condition-override "$cond" \
            --output_dir "$OUTPUT_DIR" \
            "${LIMIT_ARGS[@]}" \
            "${ADAPTER_ARGS[@]}" \
            "${PASSTHROUGH_ARGS[@]}"; then
            SUCCEEDED+=("${PROFILE}:${cond}")
            echo "  [${run_num}/3] ${cond} -- DONE"

            # Find the latest trace file for this run
            local latest_trace=$(ls -t "${OUTPUT_DIR}"/traces_*_${PROFILE}_${cond}.jsonl 2>/dev/null | head -1)
            if [ -n "$latest_trace" ]; then
                TRACE_FILES+=("$latest_trace")
            fi
        else
            FAILED+=("${PROFILE}:${cond}")
            echo "  [${run_num}/3] ${cond} -- FAILED (continuing)"
        fi

        if [ "$run_num" -lt 3 ]; then
            echo "  Waiting ${DELAY}s before next run..."
            sleep "$DELAY"
        fi
    done

    # ── Summary ──────────────────────────────────────────────────────────────

    local BATCH_END=$(date +%s)
    local BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))
    local BATCH_MINS=$(( BATCH_ELAPSED / 60 ))
    local BATCH_SECS=$(( BATCH_ELAPSED % 60 ))

    echo ""
    echo "======================================================"
    echo "  PAIRED ABLATION COMPLETE"
    echo "======================================================"
    echo ""
    echo "  Succeeded: ${#SUCCEEDED[@]}  (${SUCCEEDED[*]:-none})"
    echo "  Failed:    ${#FAILED[@]}  (${FAILED[*]:-none})"
    echo "  Total time: ${BATCH_MINS}m ${BATCH_SECS}s"
    echo ""

    # ── Generate paired comparison charts ─────────────────────────────────

    if [ ${#TRACE_FILES[@]} -ge 2 ]; then
        echo "Generating paired ablation charts..."
        echo ""

        local TRACE_ARGS=()
        local LABEL_ARGS=()
        for tf in "${TRACE_FILES[@]}"; do
            TRACE_ARGS+=("--traces" "$tf")
            # Extract condition from filename (e.g., traces_..._MA.jsonl → MA)
            local cond_label=$(echo "$tf" | grep -oP '_(M[ABC])\.jsonl' | tr -d '_.jsonl')
            LABEL_ARGS+=("--label" "${PROFILE}-${cond_label}")
        done

        python script/compare_results.py \
            "${TRACE_ARGS[@]}" "${LABEL_ARGS[@]}" \
            --plots --output "$CHART_DIR" \
            --paired-ablation \
            --cases "$CASES" || echo "  Chart generation failed (non-fatal)"

        echo ""
        echo "Charts saved to: ${CHART_DIR}/"
    else
        echo "  Not enough trace files for comparison (need at least 2)"
    fi

    echo ""
    echo "Trace files:"
    for tf in "${TRACE_FILES[@]}"; do
        echo "  - $tf"
    done
}

# ── Dispatch ────────────────────────────────────────────────────────────────

SUBCOMMAND="${1:-}"
shift 2>/dev/null || true

case "$SUBCOMMAND" in
    mcp)
        cmd_mcp "$@"
        ;;
    experiment|exp)
        cmd_experiment "$@"
        ;;
    vlm-compare|vlm)
        cmd_vlm_compare "$@"
        ;;
    paired-ablation|ablation)
        cmd_paired_ablation "$@"
        ;;
    -h|--help|"")
        echo "Usage: ./run.sh <subcommand> [options]"
        echo ""
        echo "Subcommands:"
        echo "  mcp                  V1 agent-driven pipeline"
        echo "    (no args)            Single run with config.yaml defaults"
        echo "    -e <mode>            Single experiment (memory|neo4j|memory+clip|neo4j+clip)"
        echo "    --all                Run all 4 V1 modes sequentially"
        echo "    --all --v2           Also run V2 profiles after V1"
        echo "    --all --delay N      Delay N seconds between runs (default: 10)"
        echo ""
        echo "  experiment <name>    Run a named experiment from experiments.yaml"
        echo "    list                 List available experiments"
        echo "    <name> [name2 ...]   Run one or more experiments"
        echo ""
        echo "  vlm-compare          Main thesis VLM before/after comparison"
        echo ""
        echo "  paired-ablation      Paired modality ablation (MA/MB/MC)"
        echo "    --profile <name>     Profile to use (default: v2_prompt)"
        echo "    --limit N            Limit to first N cases"
        echo "    --adapter-path PATH  LoRA adapter (for v2_lora profile)"
        echo "    --delay N            Delay between runs (default: 5)"
        echo ""
        echo "V2 pipeline (direct):"
        echo "  python script/run.py --profile v2_prompt --cases <path.jsonl>"
        echo ""
        echo "Training (separate scripts):"
        echo "  ./training/train.sh              LoRA training on Modal GPU"
        echo "  ./training/eval.sh               LoRA evaluation pipeline"
        ;;
    *)
        echo "Unknown subcommand: $SUBCOMMAND"
        echo "Run './run.sh --help' for usage."
        exit 1
        ;;
esac
