#!/bin/bash
# Complete VLM Before/After Comparison Workflow
# This is THE script to reproduce your main thesis experiment

set -e

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
    exit 1
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
echo "  ✓ EXPERIMENT COMPLETE!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - Baseline: logs/experiments/baseline_v2/"
echo "  - VLM: logs/experiments/vlm_integration/"
echo "  - Comparison: logs/comparisons/vlm_impact/"
echo ""
echo "Key chart: logs/comparisons/vlm_impact/5_vision_impact.png"
echo ""
