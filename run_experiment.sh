#!/bin/bash
# Quick experiment runner
# Usage: ./run_experiment.sh <experiment_name>

set -e

if [ -z "$1" ]; then
    echo "Usage: ./run_experiment.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    python script/experiment.py list
    exit 1
fi

python script/experiment.py run "$@"
