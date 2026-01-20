# AI-AEC Interpreter Layer

> Bridge unstructured construction site reports to structured BIM models using AI agents.

**Master Thesis MVP** - Computational Design

---

## What is This?

This system translates natural language site reports into specific BIM element references:

**Input:** _"Water pooling on kitchen floor slab"_
**Output:** `Element: Slab_05, Type: IfcSlab, GUID: 2O2Fr$t4X7Zf8NOew3FLOH`

Instead of keyword search, it uses:
- **Spatial Graph Navigation** - Follows IFC topology (Site → Floor → Room → Elements)
- **AI Agent Tool Calling** - LLM selects appropriate query tools
- **Visual Matching** - CLIP embeddings for photo-to-BIM matching

---

## Quick Start

### 1. Install Dependencies
```bash
conda activate mscd_demo  # or your environment
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
```

### 3. Run the Agent

**MCP Architecture (Production):**
```bash
./run_mcp.sh
```

**Legacy Version (Simple):**
```bash
python src/legacy/main.py
```

---

## Architecture Overview

### MCP-based (New - Production Ready)

```
Agent (Gemini) → MCP Client → [IFC Server | Visual Server]
                                     ↓              ↓
                               IFC Engine    CLIP Model
```

**Benefits:**
- ✅ Reusable across Claude Desktop, VS Code, other apps
- ✅ Microservices deployment ready
- ✅ Industry standard protocol

### Legacy (Original - MVP)

```
Agent (Gemini) → LangChain Tools → IFC Engine + CLIP
```

**Benefits:**
- ✅ Simple, single-file setup
- ✅ Good for prototyping

---

## Project Structure

```
mscd_demo/
├── mcp_servers/           # MCP microservices
│   ├── ifc_server.py      # BIM query service
│   └── visual_server.py   # Visual matching service
│
├── src/
│   ├── main_mcp.py        # MCP orchestrator
│   ├── ifc_engine.py      # IFC graph indexing
│   ├── visual_matcher.py  # CLIP embeddings
│   └── legacy/            # Original implementation
│
├── data/
│   └── BasicHouse.ifc     # Sample BIM model
│
├── config/
│   └── mcp_servers.yaml   # Server configuration
│
└── README/                # Detailed documentation
    ├── MCP_MIGRATION_GUIDE.md
    ├── MCP_ARCHITECTURE_DIAGRAM.md
    ├── WORKFLOW_EXAMPLE.md
    └── ...
```

---

## Core Components

### 1. IFC Engine ([src/ifc_engine.py](src/ifc_engine.py))
- Parses IFC files with IfcOpenShell
- Builds spatial topology graph
- Provides semantic query interface

### 2. Agent Tools (MCP or Legacy)
- `list_available_spaces()` - Discover rooms
- `get_elements_by_room(name)` - Query by location
- `get_element_details(guid)` - Property inspection
- `identify_element_visually(desc, guids)` - CLIP matching

### 3. Visual Matcher ([src/visual_matcher.py](src/visual_matcher.py))
- Uses CLIP (clip-vit-base-patch32)
- Text → Embedding → Similarity search
- Matches site descriptions to BIM elements

---

## Usage Examples

### Interactive Chat
```bash
python src/chat_cli.py
```

### Inspect IFC Structure
```bash
python src/inspect_ifc.py
```

### Test Individual MCP Servers
```bash
cd mcp_servers
./test_server.sh ifc     # Test IFC server
./test_server.sh visual  # Test visual server
```

---

## Integration with Other Tools

### Claude Desktop
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ifc-bim": {
      "command": "python",
      "args": ["/full/path/to/mcp_servers/ifc_server.py"]
    }
  }
}
```

### VS Code / Cursor
See [MCP Migration Guide](README/MCP_MIGRATION_GUIDE.md) for setup.

---

## Documentation

### Getting Started
- **[OVERVIEW.md](README/OVERVIEW.md)** - System concept and design
- **[WORKFLOW_EXAMPLE.md](README/WORKFLOW_EXAMPLE.md)** - Step-by-step walkthrough

### MCP Architecture
- **[MCP_MIGRATION_GUIDE.md](README/MCP_MIGRATION_GUIDE.md)** - Full migration guide
- **[MCP_ARCHITECTURE_DIAGRAM.md](README/MCP_ARCHITECTURE_DIAGRAM.md)** - Visual diagrams
- **[MCP_REFACTORING_SUMMARY.md](README/MCP_REFACTORING_SUMMARY.md)** - Technical summary

### Advanced Features
- **[VISUAL_MATCHING_GUIDE.md](README/VISUAL_MATCHING_GUIDE.md)** - CLIP integration
- **[PHOTO_MATCHING_EXAMPLE.md](README/PHOTO_MATCHING_EXAMPLE.md)** - Photo-to-BIM matching
- **[TEST_QUERIES.md](README/TEST_QUERIES.md)** - Example queries

---

## Technical Stack

**AI/LLM:**
- LangChain + LangGraph (Orchestration)
- Google Gemini 2.5 Flash (LLM)
- OpenAI CLIP (Visual matching)

**BIM Processing:**
- IfcOpenShell (IFC parsing)
- Custom spatial graph indexing

**MCP Integration:**
- FastMCP (Server framework)
- MCP Protocol (Tool communication)

---

## Comparison: Legacy vs MCP

| Feature | Legacy | MCP |
|---------|--------|-----|
| Setup | Simple | Requires MCP install |
| Reusability | Single app | Cross-platform |
| Deployment | Monolith | Microservices |
| Tool Discovery | Hardcoded | Dynamic |
| Production Ready | Prototype | Yes |

**Use Legacy if:** Quick prototype, single app, learning the concept
**Use MCP if:** Production deployment, cross-platform tools, scalability

---

## Roadmap

- ✅ Spatial graph-based querying
- ✅ Visual matching with CLIP
- ✅ MCP architecture
- 🚧 CORENET-X compliance checking
- 🚧 BCF issue writing
- 🚧 Blender rendering service

---

## Thesis Context

This MVP demonstrates:
1. **Graph-RAG** for structured BIM data (vs. vector search)
2. **LLM Tool Calling** for precise element retrieval
3. **Multimodal AI** for photo-to-BIM matching
4. **Production Architecture** with MCP microservices

**Key Innovation:** Using IFC's native spatial topology instead of embedding all data.

---

## Troubleshooting

**"MCP dependencies not installed"**
```bash
pip install fastmcp mcp
```

**"IFC file not found"**
```bash
# Check data/BasicHouse.ifc exists
ls data/BasicHouse.ifc
```

**"GOOGLE_API_KEY not set"**
```bash
# Create .env file with your key
echo "GOOGLE_API_KEY=your_key" > .env
```

**Want to use legacy version?**
```bash
python src/legacy/main.py
```

---

## Support

- 📖 **Documentation:** See [README/](README/) folder
- 🐛 **Issues:** Check test output and logs/ directory
- 💡 **Questions:** Review [MCP_MIGRATION_GUIDE.md](README/MCP_MIGRATION_GUIDE.md)

---

## License & Credits

**Thesis Project** - Master of Science, Computational Design
**IFC Processing:** IfcOpenShell
**AI Framework:** LangChain, Google Gemini
**Visual AI:** OpenAI CLIP
**MCP Protocol:** Anthropic

---

**Last Updated:** January 2026
**Status:** Production-ready MCP architecture ✅
