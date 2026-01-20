# MCP Architecture Diagrams

Visual guide to understanding the MCP-based architecture.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Test Scenarios                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              main_mcp.py (Orchestrator)                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │         LangGraph ReAct Agent                      │     │
│  │         (Google Gemini 2.5 Flash)                  │     │
│  │                                                     │     │
│  │  - Analyzes user intent                            │     │
│  │  - Selects appropriate tools                       │     │
│  │  - Executes reasoning loop                         │     │
│  └──────────────────┬─────────────────────────────────┘     │
│                     │                                        │
│  ┌──────────────────▼─────────────────────────────────┐     │
│  │          MCP Client Layer                          │     │
│  │                                                     │     │
│  │  - Discovers tools from MCP servers               │     │
│  │  - Converts MCP tools → LangChain tools           │     │
│  │  - Manages server connections                      │     │
│  └──────────────────┬─────────────────────────────────┘     │
└────────────────────┬┴──────────────────────────────────────┘
                     │
                     │ MCP Protocol (JSON-RPC via stdio)
                     │
        ┌────────────┴──────────────┐
        │                           │
        ▼                           ▼
┌───────────────────┐      ┌────────────────────┐
│  IFC MCP Server   │      │ Visual MCP Server  │
│                   │      │                    │
│  Tools:           │      │ Tools:             │
│  • list_spaces    │      │ • identify_visual  │
│  • get_by_room    │      │ • similarity       │
│  • get_details    │      │                    │
│  • generate_3d    │      │ Resources:         │
│                   │      │ • model_info       │
│  Resources:       │      │                    │
│  • model_metadata │      │ Domain:            │
│                   │      │ • CLIP embeddings  │
│  Domain:          │      │ • Semantic match   │
│  • IFC Engine     │      │                    │
│  • Spatial graph  │      │                    │
└───────────────────┘      └────────────────────┘
```

---

## Data Flow: Query Processing

### Example: "Water pooling on kitchen floor slab"

```
1. User Input
   │
   │ "Water pooling on kitchen floor slab"
   │
   ▼
┌──────────────────────────────────────┐
│  LangGraph Agent (Gemini)           │
│                                      │
│  Reasoning:                          │
│  - Location: "kitchen"               │
│  - Element type: "floor slab"        │
│  - Action needed: Find GUID          │
└──────────────┬───────────────────────┘
               │
               │ Tool Selection
               │
               ▼
┌──────────────────────────────────────┐
│  MCP Client                          │
│  Selected: get_elements_by_room      │
└──────────────┬───────────────────────┘
               │
               │ MCP Request
               │ {"room_name": "kitchen"}
               │
               ▼
┌──────────────────────────────────────┐
│  IFC MCP Server                      │
│                                      │
│  1. Receive request                  │
│  2. Query IFC Engine                 │
│  3. engine.find_elements_in_space()  │
└──────────────┬───────────────────────┘
               │
               │ Results
               │ [
               │   {name: "Slab_05", type: "IfcSlab", guid: "..."},
               │   {name: "Wall_12", type: "IfcWall", guid: "..."}
               │ ]
               │
               ▼
┌──────────────────────────────────────┐
│  LangGraph Agent                     │
│                                      │
│  Reasoning:                          │
│  - Match "floor slab" → Slab_05      │
│  - Need more details? Call details   │
└──────────────┬───────────────────────┘
               │
               │ Optional: get_element_details(guid)
               │
               ▼
┌──────────────────────────────────────┐
│  Final Response                      │
│                                      │
│  "Issue Location:                    │
│   Element: Slab_05                   │
│   Type: IfcSlab                      │
│   GUID: 2O2Fr$t4X7Zf8NOew3FL0I       │
│   Location: Kitchen"                 │
└──────────────────────────────────────┘
```

---

## MCP Protocol Communication

### Server Startup Sequence

```
┌──────────────┐                    ┌──────────────┐
│  MCP Client  │                    │  MCP Server  │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  1. Start server process          │
       │  (python ifc_server.py)           │
       ├──────────────────────────────────▶│
       │                                   │
       │                                   │ Initialize
       │                                   │ - Load IFC model
       │                                   │ - Build spatial index
       │                                   │ - Register tools
       │                                   │
       │  2. Initialize session            │
       │  {"method": "initialize"}         │
       ├──────────────────────────────────▶│
       │                                   │
       │  3. Server info                   │
       │  {capabilities, version}          │
       │◀──────────────────────────────────┤
       │                                   │
       │  4. List tools                    │
       │  {"method": "tools/list"}         │
       ├──────────────────────────────────▶│
       │                                   │
       │  5. Tool definitions              │
       │  [{name, description, params}...] │
       │◀──────────────────────────────────┤
       │                                   │
       │  ✅ Connection ready              │
       │                                   │
```

### Tool Call Sequence

```
┌──────────────┐                    ┌──────────────┐
│  MCP Client  │                    │  MCP Server  │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  1. Call tool                     │
       │  {                                │
       │    "method": "tools/call",        │
       │    "params": {                    │
       │      "name": "get_by_room",       │
       │      "arguments": {               │
       │        "room_name": "Kitchen"     │
       │      }                            │
       │    }                              │
       │  }                                │
       ├──────────────────────────────────▶│
       │                                   │
       │                                   │ Execute
       │                                   │ - Validate input
       │                                   │ - Run tool function
       │                                   │ - Format result
       │                                   │
       │  2. Tool result                   │
       │  {                                │
       │    "result": "[{...}, {...}]",    │
       │    "isError": false               │
       │  }                                │
       │◀──────────────────────────────────┤
       │                                   │
```

---

## Component Responsibilities

### MCP Client (`main_mcp.py`)

**Responsibilities:**
- Manage MCP server lifecycle
- Discover available tools
- Convert MCP tools to LangChain format
- Route tool calls to appropriate servers
- Handle connection errors

**Does NOT:**
- Implement domain logic
- Store IFC data
- Execute actual queries

### IFC MCP Server (`mcp_servers/ifc_server.py`)

**Responsibilities:**
- Load and index IFC model
- Execute spatial queries
- Retrieve element properties
- Expose tools via MCP protocol

**Does NOT:**
- Make decisions about which tools to call
- Interact with LLM directly
- Know about other servers

### Visual MCP Server (`mcp_servers/visual_server.py`)

**Responsibilities:**
- Load CLIP model
- Compute semantic embeddings
- Match descriptions to elements
- Expose tools via MCP protocol

**Does NOT:**
- Access IFC files directly (gets data via params)
- Make decisions about matching strategy
- Interact with LLM directly

---

## Comparison: Before vs After

### Legacy Architecture (Monolith)

```
┌─────────────────────────────────────────┐
│           main.py                       │
│                                         │
│  ┌───────────────────────────────┐     │
│  │  Agent                        │     │
│  └───────┬───────────────────────┘     │
│          │                              │
│  ┌───────▼───────────────────────┐     │
│  │  agent_tools.py               │     │
│  │  (Hardcoded LangChain tools)  │     │
│  │                               │     │
│  │  • get_elements_by_room()     │     │
│  │  • get_element_details()      │     │
│  │  • identify_visually()        │     │
│  └───────┬───────────────────────┘     │
│          │                              │
│  ┌───────▼───────────┐ ┌──────────┐    │
│  │  IFC Engine       │ │  CLIP    │    │
│  └───────────────────┘ └──────────┘    │
│                                         │
└─────────────────────────────────────────┘

❌ Tightly coupled
❌ Single application
❌ Difficult to test
❌ Hard to scale
```

### MCP Architecture (Microservices)

```
┌──────────────────────┐
│   main_mcp.py        │
│   ┌──────────┐       │
│   │  Agent   │       │
│   └────┬─────┘       │
│        │             │
│   ┌────▼─────┐       │
│   │MCP Client│       │
│   └────┬─────┘       │
└────────┼─────────────┘
         │ MCP Protocol
    ┌────┴─────┬───────────────┐
    │          │               │
┌───▼────┐ ┌──▼──────┐ ┌──────▼──┐
│IFC     │ │Visual   │ │Future   │
│Server  │ │Server   │ │Servers  │
└────────┘ └─────────┘ └─────────┘

✅ Loosely coupled
✅ Multi-application
✅ Easy to test
✅ Horizontally scalable
```

---

## Deployment Options

### Development (Current)

```
┌─────────────────────────────────┐
│  Single Machine                 │
│                                 │
│  ┌──────────────┐               │
│  │ main_mcp.py  │               │
│  └──────┬───────┘               │
│         │ stdio                 │
│    ┌────┴─────┬────────┐        │
│    │          │        │        │
│  ┌─▼──┐    ┌─▼──┐  ┌──▼─┐      │
│  │IFC │    │Vis │  │... │      │
│  └────┘    └────┘  └────┘      │
└─────────────────────────────────┘
```

### Production (Future)

```
┌─────────────────┐
│  Orchestrator   │
│  Container      │
│  ┌───────────┐  │
│  │ Agent     │  │
│  │ + Client  │  │
│  └─────┬─────┘  │
└────────┼────────┘
         │ HTTP/WebSocket
    ┌────┴────┬──────────┬────────┐
    │         │          │        │
┌───▼───┐ ┌──▼──┐   ┌───▼──┐ ┌──▼──┐
│IFC    │ │IFC  │   │Visual│ │...  │
│Server │ │Repli│   │Server│ │     │
│Pod 1  │ │ca 2 │   │      │ │     │
└───────┘ └─────┘   └──────┘ └─────┘
    │         │          │
    └────┬────┴──────────┘
         │
    ┌────▼─────┐
    │  Cache   │
    │  Layer   │
    └──────────┘
```

---

## Error Handling Flow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Agent              │
│  Try tool call      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  MCP Client         │────▶│ Connection Error │
│  Send request       │     │ → Retry logic    │
└─────────┬───────────┘     └──────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  MCP Server         │────▶│ Validation Error │
│  Validate params    │     │ → Error response │
└─────────┬───────────┘     └──────────────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────┐
│  Execute Tool       │────▶│ Execution Error  │
│  Run domain logic   │     │ → Fallback logic │
└─────────┬───────────┘     └──────────────────┘
          │
          ▼
     ✅ Success
```

---

## Key Takeaways

### MCP Advantages

1. **Standardization**: Industry-standard protocol
2. **Decoupling**: Clean separation of concerns
3. **Reusability**: Tools work across platforms
4. **Scalability**: Independent deployment
5. **Testability**: Isolated component testing

### When to Use MCP

✅ Multiple tools with distinct domains
✅ Tools need to work across applications
✅ Planning for production deployment
✅ Team collaboration on different services
✅ Long-term maintainability important

### When Legacy is OK

✅ Simple prototype/MVP
✅ Single tool, single application
✅ Time-constrained proof of concept
✅ No plans for reuse

---

**For this thesis project:** MCP demonstrates understanding of modern AI architecture and production-ready system design.
