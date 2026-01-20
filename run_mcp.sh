#!/bin/bash
# Launcher script for MCP-based BIM inspection agent

set -e

echo "=================================="
echo "🏗️  BIM Inspection Agent (MCP Mode)"
echo "=================================="
echo ""

# Check if virtual environment is activated (venv or conda)
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected."
    echo "   Recommended: conda activate mscd_demo OR source venv/bin/activate"
    echo ""
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✅ Conda environment active: $CONDA_DEFAULT_ENV"
    echo ""
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment active: $(basename $VIRTUAL_ENV)"
    echo ""
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
if ! python -c "import fastmcp" 2>/dev/null; then
    echo "📦 Installing MCP dependencies..."
    pip install fastmcp mcp
    echo ""
fi

if ! python -c "import langchain_mcp_adapters" 2>/dev/null; then
    echo "⚠️  Note: langchain-mcp-adapters not found."
    echo "   This package may not be available yet. Installing alternative if available..."
    pip install langchain-mcp-adapters 2>/dev/null || echo "   Skipping langchain-mcp-adapters (optional)"
    echo ""
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run the MCP-based agent
echo "🚀 Starting MCP-based agent..."
echo ""

python src/main_mcp.py "$@"

echo ""
echo "✅ Session complete."
