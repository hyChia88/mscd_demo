# MSCD Demo: 4D-Context BIM Disambiguation System

> Master Thesis MVP - Using 4D construction context to disambiguate BIM element queries

---

## Overview

This system translates natural language site reports into precise BIM element references by leveraging **4D task context** (spatial + temporal).

**Problem:** "Which window?" → 263 candidates (0.38% precision)
**Solution:** "Which window on 6th floor where Module Installation is active?" → 3 candidates (33.33% precision)

**Result:** Correct element found in reduced set. 98.9% search space reduction.

---

## Research Questions Addressed

| RQ | Focus | Implementation |
|----|-------|----------------|
| **RQ1** | Visual + Context Disambiguation | Spatial indexing by storey/room |
| **RQ2** | Compliance Reasoning | Property extraction (Pset_*, SGPset_*) |
| **RQ3** | Abductive Reasoning | 4D task context filtering |

---

## Quick Start

### 1. Install Dependencies
```bash
conda activate mscd_demo
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY
```

### 3. Run MCP Server
```bash
python mcp_servers/ifc_server.py
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (Agent)                      │
│                   (Gemini / Claude / etc.)                  │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
┌─────────────────────────▼───────────────────────────────────┐
│                    ifc_server.py                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              IFCEngine (Singleton)                  │    │
│  │  • Spatial Index (storey → elements)                │    │
│  │  • Property Extraction                              │    │
│  │  • Neo4j Export (optional)                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  MCP Tools:                                                 │
│  • get_element_by_guid()                                    │
│  • get_elements_by_storey()    ← 4D context filtering       │
│  • search_elements_by_type()                                │
│  • list_available_spaces()                                  │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│              AdvancedProject.ifc                            │
│  • 10 Storeys (e.g., "6 - Sixth Floor")                     │
│  • 263 Windows, 126 Doors, 304 Walls                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
mscd_demo/
├── config.yaml              # Centralized configuration
├── mcp_servers/
│   └── ifc_server.py        # MCP server with IFCEngine singleton
│
├── src/
│   └── ifc_engine.py        # Core IFC processing engine
│
├── data/
│   ├── ifc/AdvancedProject/ # BIM model (10 storeys, 263 windows)
│   └── ground_truth/gt_1/   # Evaluation test cases
│
├── script/
│   └── baseline_experiment.py  # Redundancy quantification
│
└── logs/experiments/        # Experiment results
```

---

## Core Components

### 1. IFCEngine ([src/ifc_engine.py](src/ifc_engine.py))

Central data gateway for all IFC operations:

| Method | Purpose |
|--------|---------|
| `find_elements_in_space(storey)` | Get elements on a specific floor |
| `get_element_by_guid(guid)` | Retrieve element by GlobalId |
| `get_element_properties(guid)` | Extract Pset_* property sets |
| `export_to_neo4j()` | Optional graph database export |

**Key Fix:** Spatial index now uses `IfcRelContainedInSpatialStructure` for accurate storey-to-element mapping.

### 2. MCP Server ([mcp_servers/ifc_server.py](mcp_servers/ifc_server.py))

Exposes IFCEngine as MCP tools:

```python
@mcp.tool()
def get_elements_by_storey(storey_name: str) -> str:
    """Find all BIM elements on a specific building storey.
    Example: "6 - Sixth Floor" → returns 3 windows, 12 doors, etc.
    """
```

### 3. Ground Truth ([data/ground_truth/gt_1/gt_1.json](data/ground_truth/gt_1/gt_1.json))

Validated test cases with real GUIDs:

| ID | Scenario | Target Element | Target Storey |
|----|----------|----------------|---------------|
| GT_001 | Crack on wall | `0cRoQU_sD5R8MkkMkeodzx` | Level 1 |
| GT_002 | Broken window | `1KMtYLyv9CyfGv8UjnMBSN` | 1 - First Floor |
| GT_007 | Module installation window | `0Um_J2ClP45uPRcRbJqhxe` | 6 - Sixth Floor |

---

## Baseline Experiment Results

**Scenario:** GT_007 - Find window for module installation on 6th floor

| Condition | Candidates | Precision | Recall |
|-----------|------------|-----------|--------|
| No context (all windows) | 263 | 0.38% | 100% |
| With 4D context (6th floor) | 3 | 33.33% | 100% |

**Key Insight:** Target element included in both cases (Recall=100%), but precision improves 87x with 4D context.

Results stored in: [logs/experiments/baseline_gt007_results.json](logs/experiments/baseline_gt007_results.json)

---

## Configuration

All settings in [config.yaml](config.yaml):

```yaml
ifc:
  model_path: "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"

neo4j:
  uri: "bolt://localhost:7687"
  enabled: false  # Set true to enable graph database

ground_truth:
  file: "data/ground_truth/gt_1/gt_1.json"

llm:
  provider: "google"
  model: "gemini-2.5-flash"
```

---

## Available Storeys in Model

The AdvancedProject.ifc model contains:

| Storey Name | Windows | Doors |
|-------------|---------|-------|
| Level 1 | 0 | 0 |
| 1 - First Floor | 30 | 14 |
| 2 - Second Floor | 30 | 14 |
| ... | ... | ... |
| 6 - Sixth Floor | 3 | 14 |
| ... | ... | ... |
| Roof | 0 | 0 |

---

## Optional: Neo4j Integration

For graph-based reasoning (RQ2/RQ3):

```bash
# Start Neo4j container
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Export IFC to Neo4j
python script/ifc_to_neo4j.py

# Query via browser
open http://localhost:7474
```

---

## Usage Examples

### Run Agent with Test Scenarios
```bash
# Run MCP-based agent with test scenarios
python src/main_mcp.py  # Uses prompts/tests/test_2.yaml

# Run legacy agent
python src/legacy/main.py  # Uses test.yaml
```

Test scenarios are defined in YAML files:
- [prompts/tests/test_2.yaml](prompts/tests/test_2.yaml) - 18 comprehensive scenarios
- [prompts/tests/test_3_min_complete.yaml](prompts/tests/test_3_min_complete.yaml) - Minimal complete flow

### Run Baseline Experiment
```bash
python script/baseline_experiment.py
# Output: logs/experiments/baseline_gt007_results.json
```

### Direct IFCEngine Usage
```python
from src.ifc_engine import IFCEngine

engine = IFCEngine("data/ifc/AdvancedProject/IFC/AdvancedProject.ifc")
windows = engine.find_elements_in_space("6 - sixth floor")
print(f"Found {len(windows)} elements on 6th floor")
```

### Start MCP Server Standalone
```bash
python mcp_servers/ifc_server.py
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| IFC Processing | IfcOpenShell |
| Spatial Indexing | Custom graph (storey → elements) |
| MCP Server | FastMCP |
| LLM Agent | Google Gemini 2.5 Flash |
| Graph Database | Neo4j (optional) |
| Visual Matching | OpenAI CLIP |

---

## Current Progress

- [x] Centralized configuration (config.yaml)
- [x] IFCEngine with accurate spatial indexing
- [x] MCP server with 4D context tools
- [x] Ground truth test cases with real GUIDs
- [x] Baseline experiment (98.9% redundancy reduction)
- [x] Neo4j export support (optional)
- [ ] Full evaluation pipeline
- [ ] CORENET-X compliance checking
- [ ] BCF issue generation

---

## Troubleshooting

**"No elements found for storey"**
- Use exact storey names: `"6 - Sixth Floor"` not `"Level 6"`
- Check available storeys: `engine.list_spaces()`

**"Neo4j connection refused"**
- Ensure Docker container is running
- Wait 30 seconds after container start

**"GUID not found"**
- Verify GUID exists in model: `engine.get_element_by_guid(guid)`

---

## References

- IFC Schema: [buildingSMART IFC4](https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/)
- MCP Protocol: [Model Context Protocol](https://modelcontextprotocol.io/)
- IfcOpenShell: [Documentation](https://ifcopenshell.org/)

---

**Last Updated:** January 2026
**Status:** Baseline experiment complete, MCP architecture validated
