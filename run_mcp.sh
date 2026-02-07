#!/bin/bash
# Launcher script for MCP-based BIM inspection agent
#
# Usage:
#   ./run_mcp.sh                              # Single run (config.yaml defaults)
#   ./run_mcp.sh -e memory                    # Single experiment mode
#   ./run_mcp.sh --all                        # Run ALL 4 experiment modes sequentially
#   ./run_mcp.sh --all -d synth              # Run all on synthetic dataset
#   ./run_mcp.sh --all --delay 15            # Run all with 15s delay between runs
#   ./run_mcp.sh --all --v2                  # Run all V1 modes + key V2 profiles

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "  BIM Inspection Agent (MCP Mode)"
echo "=================================="
echo ""

# Check if virtual environment is activated (venv or conda)
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No virtual environment detected."
    echo "   Recommended: conda activate mscd_demo OR source venv/bin/activate"
    echo ""
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Conda environment active: $CONDA_DEFAULT_ENV"
    echo ""
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment active: $(basename $VIRTUAL_ENV)"
    echo ""
fi

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python -c "import fastmcp" 2>/dev/null; then
    echo "Installing MCP dependencies..."
    pip install fastmcp mcp
    echo ""
fi

if ! python -c "import langchain_mcp_adapters" 2>/dev/null; then
    echo "Note: langchain-mcp-adapters not found."
    echo "   This package may not be available yet. Installing alternative if available..."
    pip install langchain-mcp-adapters 2>/dev/null || echo "   Skipping langchain-mcp-adapters (optional)"
    echo ""
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# ─────────────────────────────────────────────────────────────────────────────
# Parse arguments: intercept --all, --delay, --v2; pass the rest to main_mcp.py
# ─────────────────────────────────────────────────────────────────────────────

RUN_ALL=false
RUN_V2=false
DELAY=10
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)
            RUN_ALL=true
            shift
            ;;
        --v2)
            RUN_V2=true
            shift
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Single run mode
# ─────────────────────────────────────────────────────────────────────────────

if [ "$RUN_ALL" = false ]; then
    echo "Starting MCP-based agent..."
    echo ""
    python src/main_mcp.py "${PASSTHROUGH_ARGS[@]}"
    echo ""
    echo "Session complete."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# --all mode: run all experiment modes sequentially
# ─────────────────────────────────────────────────────────────────────────────

V1_EXPERIMENTS=("memory" "neo4j" "memory+clip" "neo4j+clip")
V2_PROFILES=("v2_prompt" "v2_memory")

# Determine V2 cases path (use -d synth passthrough or default)
V2_CASES="data_curation/datasets/synth_v0.2/cases_v2.jsonl"

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

BATCH_START=$(date +%s)
FAILED=()
SUCCEEDED=()

# ── V1 experiments ───────────────────────────────────────────────────────────

for i in "${!V1_EXPERIMENTS[@]}"; do
    exp="${V1_EXPERIMENTS[$i]}"
    run_num=$((i + 1))
    total=${#V1_EXPERIMENTS[@]}

    echo ""
    echo "======================================================"
    echo "  [${run_num}/${total}] V1 experiment: ${exp}"
    echo "======================================================"
    echo ""

    if python src/main_mcp.py --experiment "$exp" "${PASSTHROUGH_ARGS[@]}"; then
        SUCCEEDED+=("v1:${exp}")
        echo ""
        echo "  [${run_num}/${total}] ${exp} -- DONE"
    else
        FAILED+=("v1:${exp}")
        echo ""
        echo "  [${run_num}/${total}] ${exp} -- FAILED (continuing)"
    fi

    # Delay between runs (skip after last)
    if [ "$run_num" -lt "$total" ] || [ "$RUN_V2" = true ]; then
        echo "  Waiting ${DELAY}s before next run..."
        sleep "$DELAY"
    fi
done

# ── V2 profiles (optional) ──────────────────────────────────────────────────

if [ "$RUN_V2" = true ]; then
    # Check if cases file exists
    if [ ! -f "$V2_CASES" ]; then
        # Try absolute path from project root
        V2_CASES_ABS="${SCRIPT_DIR}/../data_curation/datasets/synth_v0.2/cases_v2.jsonl"
        if [ -f "$V2_CASES_ABS" ]; then
            V2_CASES="$V2_CASES_ABS"
        else
            echo ""
            echo "  WARNING: cases_v2.jsonl not found, skipping V2 profiles"
            echo "  Looked in: $V2_CASES and $V2_CASES_ABS"
            RUN_V2=false
        fi
    fi
fi

if [ "$RUN_V2" = true ]; then
    for i in "${!V2_PROFILES[@]}"; do
        profile="${V2_PROFILES[$i]}"
        run_num=$((i + 1))
        total=${#V2_PROFILES[@]}

        echo ""
        echo "======================================================"
        echo "  [${run_num}/${total}] V2 profile: ${profile}"
        echo "======================================================"
        echo ""

        if python script/run.py --profile "$profile" --cases "$V2_CASES"; then
            SUCCEEDED+=("v2:${profile}")
            echo ""
            echo "  [${run_num}/${total}] ${profile} -- DONE"
        else
            FAILED+=("v2:${profile}")
            echo ""
            echo "  [${run_num}/${total}] ${profile} -- FAILED (continuing)"
        fi

        # Delay between runs (skip after last)
        if [ "$run_num" -lt "$total" ]; then
            echo "  Waiting ${DELAY}s before next run..."
            sleep "$DELAY"
        fi
    done
fi

# ── Summary ──────────────────────────────────────────────────────────────────

BATCH_END=$(date +%s)
BATCH_ELAPSED=$(( BATCH_END - BATCH_START ))
BATCH_MINS=$(( BATCH_ELAPSED / 60 ))
BATCH_SECS=$(( BATCH_ELAPSED % 60 ))

echo ""
echo "=============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "=============================================="
echo ""
echo "  Succeeded: ${#SUCCEEDED[@]}  (${SUCCEEDED[*]:-none})"
echo "  Failed:    ${#FAILED[@]}  (${FAILED[*]:-none})"
echo "  Total time: ${BATCH_MINS}m ${BATCH_SECS}s"
echo ""

# ── Compare results ──────────────────────────────────────────────────────────

echo "Running comparison..."
echo ""
python script/compare_results.py --latest
echo ""
echo "Done. Full results in logs/evaluations/"
