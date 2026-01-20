# Documentation Index

**AI-AEC Interpreter Layer** - Complete Documentation

---

## 📚 Quick Navigation

### 🚀 Getting Started

**[OVERVIEW.md](OVERVIEW.md)**
- System concept and design philosophy
- How the interpreter layer works
- Key architectural decisions

**[WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md)**
- Step-by-step agent execution walkthrough
- Example queries with detailed traces
- Understanding agent reasoning

---

### 🏗️ MCP Architecture (Production)

**[MCP_MIGRATION_GUIDE.md](MCP_MIGRATION_GUIDE.md)** ⭐ Start Here
- Complete migration guide from legacy to MCP
- Installation and setup instructions
- Integration examples (Claude Desktop, VS Code)
- Development workflow
- Troubleshooting

**[MCP_ARCHITECTURE_DIAGRAM.md](MCP_ARCHITECTURE_DIAGRAM.md)**
- Visual architecture diagrams
- Data flow illustrations
- Component responsibilities
- Deployment options

**[MCP_REFACTORING_SUMMARY.md](MCP_REFACTORING_SUMMARY.md)**
- Technical refactoring summary
- File changes overview
- Architecture comparison (before/after)
- Metrics and performance notes

---

### 🎨 Advanced Features

**[VISUAL_MATCHING_GUIDE.md](VISUAL_MATCHING_GUIDE.md)**
- CLIP-based semantic matching
- How visual alignment works
- Text-to-element matching
- Photo-to-BIM matching (future)

**[PHOTO_MATCHING_EXAMPLE.md](PHOTO_MATCHING_EXAMPLE.md)**
- Example: Using site photos to identify BIM elements
- Practical use cases
- Implementation details

**[TEST_QUERIES.md](TEST_QUERIES.md)**
- Example queries and expected outputs
- Testing different scenarios
- Query patterns and best practices

---

## 📖 Documentation by Use Case

### "I want to understand what this project does"
→ Start with **[OVERVIEW.md](OVERVIEW.md)**

### "I want to run the system"
→ See main **[README.md](../README.md)** (Quick Start section)

### "I want to understand MCP architecture"
→ Read **[MCP_ARCHITECTURE_DIAGRAM.md](MCP_ARCHITECTURE_DIAGRAM.md)**

### "I want to migrate to MCP"
→ Follow **[MCP_MIGRATION_GUIDE.md](MCP_MIGRATION_GUIDE.md)**

### "I want to see how the agent thinks"
→ Check **[WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md)**

### "I want to understand visual matching"
→ Read **[VISUAL_MATCHING_GUIDE.md](VISUAL_MATCHING_GUIDE.md)**

### "I need example queries to test"
→ Use **[TEST_QUERIES.md](TEST_QUERIES.md)**

### "I want technical migration details"
→ Review **[MCP_REFACTORING_SUMMARY.md](MCP_REFACTORING_SUMMARY.md)**

---

## 📝 Document Sizes

| Document | Size | Type |
|----------|------|------|
| MCP_ARCHITECTURE_DIAGRAM.md | 19K | Visual Reference |
| VISUAL_MATCHING_GUIDE.md | 14K | Technical Guide |
| MCP_REFACTORING_SUMMARY.md | 12K | Technical Summary |
| OVERVIEW.md | 10K | Conceptual |
| MCP_MIGRATION_GUIDE.md | 8.4K | Practical Guide |
| WORKFLOW_EXAMPLE.md | 8.6K | Tutorial |
| PHOTO_MATCHING_EXAMPLE.md | 3.1K | Example |
| TEST_QUERIES.md | 4.8K | Reference |

---

## 🎯 Learning Path

### Beginner
1. [OVERVIEW.md](OVERVIEW.md) - Understand the concept
2. [../README.md](../README.md) - Quick start guide
3. [WORKFLOW_EXAMPLE.md](WORKFLOW_EXAMPLE.md) - See it in action

### Intermediate
4. [MCP_ARCHITECTURE_DIAGRAM.md](MCP_ARCHITECTURE_DIAGRAM.md) - Architecture patterns
5. [VISUAL_MATCHING_GUIDE.md](VISUAL_MATCHING_GUIDE.md) - Advanced features
6. [TEST_QUERIES.md](TEST_QUERIES.md) - Practice queries

### Advanced
7. [MCP_MIGRATION_GUIDE.md](MCP_MIGRATION_GUIDE.md) - Production setup
8. [MCP_REFACTORING_SUMMARY.md](MCP_REFACTORING_SUMMARY.md) - Technical deep dive

---

## 🔍 Quick Reference

**Architecture:**
- Legacy: `Agent → Tools → IFC/CLIP`
- MCP: `Agent → MCP Client → [IFC Server | Visual Server]`

**Main Commands:**
```bash
./run_mcp.sh                     # Run MCP version
python src/legacy/main.py        # Run legacy version
python src/chat_cli.py           # Interactive mode
cd mcp_servers && ./test_server.sh ifc   # Test server
```

**Key Files:**
- `src/main_mcp.py` - MCP orchestrator
- `mcp_servers/ifc_server.py` - IFC query service
- `mcp_servers/visual_server.py` - Visual matching service
- `src/ifc_engine.py` - Core BIM logic

---

**Last Updated:** January 2026
**Total Documentation:** ~96KB across 8 files
