#!/bin/bash
# Quick comparison script for before/after VLM evaluation

set -e

echo "=================================================="
echo "  Comparing Old vs New Pipeline Performance"
echo "=================================================="
echo ""

# Check if directories exist
if [ ! -d "logs/evaluations/old_pipeline" ]; then
    echo "⚠️  Old pipeline results not found at logs/evaluations/old_pipeline"
    echo "Run the old pipeline first with:"
    echo "  git stash"
    echo "  python script/run.py --profile v2_prompt --cases ... --output_dir logs/evaluations/old_pipeline"
    echo "  git stash pop"
    exit 1
fi

if [ ! -d "logs/evaluations/new_pipeline" ]; then
    echo "⚠️  New pipeline results not found at logs/evaluations/new_pipeline"
    echo "Run the new pipeline first with:"
    echo "  python script/run.py --profile v2_prompt --cases ... --output_dir logs/evaluations/new_pipeline"
    exit 1
fi

echo "Finding trace files..."
OLD_TRACES=$(ls -t logs/evaluations/old_pipeline/traces_*.jsonl 2>/dev/null | head -1)
NEW_TRACES=$(ls -t logs/evaluations/new_pipeline/traces_*.jsonl 2>/dev/null | head -1)

if [ -z "$OLD_TRACES" ]; then
    echo "ERROR: No traces found in old_pipeline directory"
    exit 1
fi

if [ -z "$NEW_TRACES" ]; then
    echo "ERROR: No traces found in new_pipeline directory"
    exit 1
fi

echo "Old traces: $OLD_TRACES"
echo "New traces: $NEW_TRACES"
echo ""

echo "Generating comparison plots..."
python script/generate_plots.py \
    --traces "$NEW_TRACES" \
    --before "$OLD_TRACES" \
    --output logs/plots/before_after_comparison

echo ""
echo "=================================================="
echo "✓ Comparison plots saved to:"
echo "  logs/plots/before_after_comparison/"
echo "=================================================="
echo ""
echo "Key chart to check:"
echo "  5_vision_impact.png - Shows accuracy improvement"
echo ""
