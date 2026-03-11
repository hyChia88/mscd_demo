# MCP Migration Guide

## Overview

This document explains the migration from LangChain-only tool integration to a production-ready **Model Context Protocol (MCP)** architecture.

## What Changed?

### Before: Monolithic Tool Integration

```
┌─────────────────────────────────────┐
│         main.py                     │
│  ┌───────────────────────────┐     │
│  │  LangGraph ReAct Agent    │     │
│  └───────────┬───────────────┘     │
│              │                      │
│  ┌───────────▼───────────────┐     │
│  │  agent_tools.py           │     │
│  │  @tool decorators         │     │
│  └───────────┬───────────────┘     │
│              │                      │
│  ┌───────────▼───────────────┐     │
│  │  IFC Engine + CLIP Model  │     │
│  └───────────────────────────┘     │
└─────────────────────────────────────┘
```

**Issues:**
- Tools tightly coupled to application code
- Cannot reuse tools in other projects
- M×N integration problem (every app needs every tool)
- Difficult to scale or deploy independently

### After: MCP Service Architecture

```
┌─────────────────────────────────────┐
│         main_mcp.py                 │
│  ┌───────────────────────────┐     │
│  │  LangGraph ReAct Agent    │     │
│  └───────────┬───────────────┘     │
│              │                      │
│  ┌───────────▼───────────────┐     │
│  │  MCP Clients              │     │
│  │  (LangChain Adapters)     │     │
│  └──┬─────────┬──────────────┘     │
└─────┼─────────┼─────────────────────┘
      │         │
      │ MCP Protocol (stdio)
      │         │
┌─────▼─────┐ ┌▼──────────────┐
│ IFC       │ │ Visual        │
│ Server    │ │ Server        │
│           │ │               │
│ - list    │ │ - identify    │
│ - query   │ │ - similarity  │
│ - details │ │               │
└───────────┘ └───────────────┘
```

**Benefits:**
- ✅ Tools are decoupled and reusable
- ✅ Standard protocol (works with Claude Desktop, VS Code, etc.)
- ✅ Independent deployment and scaling
- ✅ Better testability and maintainability

## File Structure

### New Files

```
mscd_demo/
├── mcp_servers/
│   ├── __init__.py
│   ├── ifc_server.py          # IFC query MCP server
│   ├── visual_server.py       # Visual matching MCP server
│   └── test_server.sh         # Test individual servers
│
├── src/
│   ├── main_mcp.py            # MCP-based orchestrator (NEW)
│   └── legacy/
│       ├── main.py            # Original orchestrator (BACKUP)
│       └── agent_tools.py     # Original tools (BACKUP)
│
├── config/
│   └── mcp_servers.yaml       # Server configuration
│
├── run_mcp.sh                 # Quick launcher script
└── requirements.txt           # Updated with MCP deps
```

### Updated Files

- `requirements.txt`: Added `fastmcp`, `mcp`, `langchain-mcp-adapters`

## How to Use

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the MCP-based Agent

**Option A: Using the launcher script**
```bash
./run_mcp.sh
```

**Option B: Directly**
```bash
python src/main_mcp.py
```

### 3. Test Individual MCP Servers

```bash
# Test IFC server
cd mcp_servers
./test_server.sh ifc

# Test visual server
./test_server.sh visual
```

## MCP Server Details

### IFC Query Service

**Location:** `mcp_servers/ifc_server.py`

**Exposed Tools:**
- `list_available_spaces()` - Discover rooms/spaces in the model
- `get_elements_by_room(room_name)` - Query elements by location
- `get_element_details(guid)` - Retrieve property sets
- `generate_3d_view(guid)` - Trigger rendering

**Resources:**
- `ifc://model/metadata` - IFC model metadata

### Visual Matching Service

**Location:** `mcp_servers/visual_server.py`

**Exposed Tools:**
- `identify_element_visually(description, guids)` - CLIP-based matching
- `compute_semantic_similarity(text1, text2)` - Semantic comparison

**Resources:**
- `visual://model/info` - CLIP model information

## Integration with Other Tools

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ifc-bim": {
      "command": "python",
      "args": ["/path/to/mscd_demo/mcp_servers/ifc_server.py"]
    },
    "visual-matching": {
      "command": "python",
      "args": ["/path/to/mscd_demo/mcp_servers/visual_server.py"]
    }
  }
}
```

### VS Code with MCP Extension

1. Install MCP extension
2. Add servers to workspace settings:

```json
{
  "mcp.servers": [
    {
      "name": "IFC Query",
      "command": "python mcp_servers/ifc_server.py"
    }
  ]
}
```

## Development Workflow

### Adding New Tools

**1. Add tool to existing server:**

```python
# In mcp_servers/ifc_server.py

@mcp.tool()
def my_new_tool(param: str) -> str:
    """Tool description for LLM"""
    # Implementation
    return result
```

**2. Create new MCP server:**

```python
# mcp_servers/my_server.py

from fastmcp import FastMCP

mcp = FastMCP("My Service")

@mcp.tool()
def my_tool(param: str) -> str:
    """Tool description"""
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**3. Register in config:**

```yaml
# config/mcp_servers.yaml

servers:
  my_service:
    name: "My Service"
    command: "python"
    args:
      - "mcp_servers/my_server.py"
    enabled: true
```

### Testing

**Unit test a server:**
```bash
cd mcp_servers
python ifc_server.py
# Server runs and waits for MCP client connections
```

**Integration test:**
```bash
python src/main_mcp.py
```

## Comparison: Legacy vs MCP

| Aspect | Legacy (`main.py`) | MCP (`main_mcp.py`) |
|--------|-------------------|---------------------|
| Tool definition | `@tool` decorator | `@mcp.tool()` decorator |
| Tool registration | Import in Python | Runtime discovery via MCP |
| Reusability | Single app only | Cross-platform/app |
| Deployment | Monolith | Microservices-ready |
| Testability | Coupled testing | Isolated server tests |
| Ecosystem | LangChain only | MCP ecosystem |

## Rollback

To use the legacy system:

```bash
# The original files are backed up in src/legacy/
python src/legacy/main.py
```

## Troubleshooting

### "MCP dependencies not installed"

```bash
pip install fastmcp mcp langchain-mcp-adapters
```

### "No tools available"

Check that MCP servers are starting correctly:
```bash
python mcp_servers/ifc_server.py
# Should print initialization messages to stderr
```

### Server connection timeout

- Verify Python path is correct in config
- Check server logs for errors
- Ensure IFC model path is accessible

## Production Deployment

For production use:

1. **Containerize MCP servers:**
   ```dockerfile
   FROM python:3.11
   COPY mcp_servers/ /app/
   RUN pip install fastmcp ifcopenshell
   CMD ["python", "/app/ifc_server.py"]
   ```

2. **Use process managers:**
   ```bash
   # supervisord, systemd, or docker-compose
   ```

3. **Add health checks:**
   ```python
   @mcp.resource("health")
   def health_check():
       return "OK"
   ```

4. **Implement logging and monitoring**

## References

- [FastMCP Documentation](https://gofastmcp.com/)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

---

**Migration completed:** January 2026
**Migrated by:** Master Thesis Project
**Status:** Production-ready
