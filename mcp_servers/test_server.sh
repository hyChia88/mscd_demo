#!/bin/bash
# Test individual MCP servers in isolation

set -e

SERVER=$1

if [ -z "$SERVER" ]; then
    echo "Usage: ./test_server.sh [ifc|visual]"
    echo ""
    echo "Available servers:"
    echo "  ifc     - IFC Query Service"
    echo "  visual  - Visual Matching Service"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

case $SERVER in
    ifc)
        echo "🔧 Testing IFC Query Service..."
        python mcp_servers/ifc_server.py
        ;;
    visual)
        echo "🔧 Testing Visual Matching Service..."
        python mcp_servers/visual_server.py
        ;;
    *)
        echo "❌ Unknown server: $SERVER"
        exit 1
        ;;
esac
