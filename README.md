# MSCD Demo: An “Interpreter Layer” for AEC

> Master Thesis MVP - Cross-Modal Alignment, Schema Mapping, and Compliance

---

## Overview

This demo implements an agentic interpreter layer that orchestrates IFC retrieval (graph), project context docs (e.g., 4D tasks), and multimodal evidence (site images + text) to align messy site reports with IFC elements + schema-aware structured outputs, reducing information loss and enabling BCF handoff.

> Interpreter layer = agentic orchestration over heterogeneous project context (IFC graph + 4D docs + multimodal evidence) with verifiable schema outputs and BCF handoff.

**Problem:** "Which window?" → 263 candidates (0.38% precision)  
**Solution:** "Which window on 6th floor where Module Installation is active?" → 3 candidates (33.33% precision)

**Result:** Correct element found in reduced set. 98.9% search space reduction.

---

## Research Questions Addressed

| RQ | Focus | Implementation | Evaluation |
|----|-------|----------------|------------|
| **RQ1** | Multimodal context: Visual + Text | Spatial indexing by storey/room | Retrieval F1 score |
| **RQ2** | Schema mapping | Property extraction (Pset_*, SGPset_*) | Schema pass rate / fill rate |
| **RQ3** | Abductive Reasoning | Agentic tools | escalation policy & handoff completeness |

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

### 3. Run the Agent

**Interactive Chat Mode (Recommended for Testing):**
```bash
python src/chat_cli.py

# Or pre-load a test scenario:
python src/chat_cli.py --scenario GT_007
```

**Automated Evaluation Mode:**
```bash
python src/main_mcp.py
```

This automatically:
- Spawns the MCP server (`ifc_server.py`) as a subprocess
- Connects to it via MCP protocol
- Runs evaluation against ground truth test cases
- Outputs results to `logs/evaluations/`

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
│  ┌─────────────────────────────────────────────────────┐    │
│  │           VisualAligner (CLIP, lazy-loaded)         │    │
│  │  • Image → Vector embedding                         │    │
│  │  • Text → Vector embedding                          │    │
│  │  • Image ↔ Element matching                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  MCP Tools (Spatial Query):                                 │
│  • get_element_by_guid()                                    │
│  • get_elements_by_storey()    ← 4D context filtering       │
│  • search_elements_by_type()                                │
│  • list_available_spaces()                                  │
│                                                             │
│  MCP Tools (Visual Analysis):                               │
│  • analyze_site_image()        ← Classify photo content     │
│  • match_image_to_elements()   ← Photo → BIM element        │
│  • compare_defect_images()     ← Check same defect          │
│  • match_text_to_elements()    ← Description → BIM element  │
│                                                             │
│  MCP Tools (3D View Generation):                            │
│  • generate_3d_view()          ← GUID → rendered image      │
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
│   ├── main_mcp.py          # MCP-based agent orchestrator (production)
│   ├── chat_cli.py          # Interactive CLI for testing (MCP-based)
│   ├── chat_logger.py       # Conversation logging utility
│   ├── ifc_engine.py        # Core IFC processing engine
│   ├── mcp_langchain_adapter.py  # MCP to LangChain adapter
│   │
│   ├── eval/                # Evaluation Pipeline v2
│   │   ├── __init__.py
│   │   ├── contracts.py     # Pydantic data models (EvalTrace, RQ2Result, etc.)
│   │   ├── metrics.py       # Metric functions (top1_hit, RQ2 metrics, etc.)
│   │   └── runner.py        # run_one_scenario() with RQ2 post-processing
│   │
│   ├── rq2_schema/          # RQ2 Schema-Aware Validation Pipeline
│   │   ├── __init__.py
│   │   ├── extract_final_json.py  # Parse FINAL_JSON from agent output
│   │   ├── schema_registry.py     # Load and manage JSON Schema
│   │   ├── mapping.py             # Deterministic mapping to submission JSON
│   │   ├── validators.py          # JSON Schema + domain validation
│   │   └── pipeline.py            # run_rq2_postprocess() orchestration
│   │
│   ├── visual/              # Visual Analysis Module (CLIP-based)
│   │   ├── __init__.py
│   │   └── aligner.py       # VisualAligner class (singleton, lazy-loaded)
│   │
│   └── handoff/             # P2: BCF Issue Generation & Handoff
│       ├── __init__.py
│       ├── trace.py         # Trace building and GUID extraction
│       ├── bcf_lite.py      # BCF-lite JSON issue output
│       └── bcf_zip.py       # BCFzip 2.1 generation
│
├── schemas/
│   └── corenetx_min/
│       └── v0.schema.json   # CORENET-X-like minimal submission schema
│
├── data/
│   ├── ifc/AdvancedProject/ # BIM model (10 storeys, 263 windows)
│   └── ground_truth/gt_1/   # Evaluation test cases
│
├── script/
│   ├── baseline_experiment.py    # Redundancy quantification
│   ├── eval_pipeline.py          # Unified evaluation pipeline CLI
│   ├── render_worker.py          # Blender headless render script
│   ├── rq2_schema_smoke_test.py  # RQ2 component smoke test
│   ├── test_bcf_generation.py    # BCF handoff sanity test
│   └── test_visual_aligner.py    # Visual aligner test suite
│
├── outputs/                 # P2: Handoff artifacts (generated)
│   ├── traces/<run_id>/     # Per-case trace JSON files
│   ├── issues/<run_id>/     # BCF-lite issue.json files
│   ├── bcf/<run_id>/        # BCFzip 2.1 files
│   └── renders/             # 3D view snapshots from generate_3d_view()
│
└── logs/
    ├── experiments/         # Baseline experiment results
    └── evaluations/         # Eval pipeline outputs (JSONL + CSV)
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

Exposes IFCEngine and VisualAligner as MCP tools:

**Spatial Query Tools:**
```python
@mcp.tool()
def get_elements_by_storey(storey_name: str) -> str:
    """Find all BIM elements on a specific building storey.
    Example: "6 - Sixth Floor" → returns 3 windows, 12 doors, etc.
    """
```

**Visual Analysis Tools:**
| Tool | Purpose | Args |
|------|---------|------|
| `analyze_site_image` | Classify what's visible in a site photo | `image_path` |
| `match_image_to_elements` | Match photo to BIM elements by GUID | `image_path, storey_filter, top_k` |
| `compare_defect_images` | Check if two photos show same defect | `image_path1, image_path2` |
| `match_text_to_elements` | Match verbal description to elements | `description, storey_filter, top_k` |

```python
@mcp.tool()
def match_image_to_elements(image_path: str, storey_filter: str = None, top_k: int = 5) -> str:
    """Match a site photo to BIM elements using CLIP visual AI.
    Combines visual understanding with 4D spatial filtering.
    """
```

**3D View Generation Tools:**
| Tool | Purpose | Args |
|------|---------|------|
| `generate_3d_view` | Generate rendered 3D snapshot of an element | `guid` |

```python
@mcp.tool()
def generate_3d_view(guid: str) -> str:
    """Generate a visual verification snapshot (3D render) of an element.
    Uses Blender headless rendering with Bonsai addon.
    Returns JSON with image_path on success.
    """
```

> **Note:** Requires Blender with Bonsai/BlenderBIM addon. Set `BLENDER_PATH` environment variable if not in PATH.

### 3. Visual Aligner ([src/visual/aligner.py](src/visual/aligner.py))

CLIP-based multimodal embedding for visual grounding:

| Method | Purpose |
|--------|---------|
| `get_image_embedding(path)` | Image → CLIP vector |
| `get_text_embedding(text)` | Text → CLIP vector |
| `match_image_to_descriptions(img, descs, top_k)` | Rank descriptions by image similarity |
| `match_image_to_elements(img, elements, top_k)` | Match photo to BIM elements |
| `compare_images(img1, img2)` | Semantic similarity score (0-1) |
| `find_best_match(query, candidates)` | Legacy-compatible single best match |

**Singleton Pattern:** Model is lazy-loaded on first use to avoid heavy startup cost.

### 4. Ground Truth ([data/ground_truth/gt_1/gt_1.json](data/ground_truth/gt_1/gt_1.json))

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
  model: "gemini-2.5-flash-lite"
  temperature: 0

rq2:
  enabled: true  # Enable RQ2 schema validation post-processing
  schema_path: "schemas/corenetx_min/v0.schema.json"
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

### A/B Experiment: Neo4j vs In-Memory vs CLIP

Run comparative experiments with the same ground truth:

```bash
# Experiment 1: In-memory spatial index (baseline)
python src/main_mcp.py --experiment memory
# Output: logs/evaluations/eval_TIMESTAMP_memory.json

# Experiment 2: Neo4j graph queries
python src/main_mcp.py --experiment neo4j
# Output: logs/evaluations/eval_TIMESTAMP_neo4j.json

# Experiment 3: In-memory + CLIP visual matching
python src/main_mcp.py --experiment memory+clip
# Output: logs/evaluations/eval_TIMESTAMP_memory+clip.json

# Experiment 4: Neo4j + CLIP visual matching
python src/main_mcp.py --experiment neo4j+clip
# Output: logs/evaluations/eval_TIMESTAMP_neo4j+clip.json
```

**Run all experiments in sequence:**
```bash
for mode in memory neo4j memory+clip neo4j+clip; do
  echo "Running experiment: $mode"
  python src/main_mcp.py -e $mode
  sleep 10
done
```

Results include experiment metadata for comparison:
```json
{
  "experiment": {
    "mode": "neo4j+clip",
    "query_mode": "neo4j",
    "visual_enabled": true,
    "description": "Neo4j graph queries + CLIP visual matching"
  },
  "summary": { ... }
}
```

| Experiment Mode | Query Backend | Visual (CLIP) | Use Case |
|-----------------|---------------|---------------|----------|
| `memory` | In-memory spatial index | Disabled | Fast baseline |
| `neo4j` | Neo4j graph database | Disabled | Semantic relationships |
| `memory+clip` | In-memory spatial index | Enabled | Visual grounding |
| `neo4j+clip` | Neo4j graph database | Enabled | Full multimodal |

---

## Usage Examples

### Interactive Chat (Recommended for Testing)
```bash
# Start interactive session
python src/chat_cli.py

# Pre-load a specific test scenario
python src/chat_cli.py --scenario GT_007
```

**Commands in interactive mode:**
| Command | Description |
|---------|-------------|
| `quit` / `q` | Exit the chat |
| `clear` | Clear conversation history |
| `tools` | List available MCP tools |
| `scenarios` | List available test scenarios |
| `load <id>` | Load a scenario (e.g., `load GT_007`) |
| `send` | Send the original query from loaded scenario |

### Run Automated Evaluation
```bash
# Run MCP-based agent with ground truth test cases
python src/main_mcp.py

# A/B Experiment: Multiple Modes
python src/main_mcp.py --experiment memory       # In-memory spatial index
python src/main_mcp.py --experiment neo4j        # Neo4j graph queries
python src/main_mcp.py --experiment memory+clip  # In-memory + CLIP visual
python src/main_mcp.py --experiment neo4j+clip   # Neo4j + CLIP visual
python src/main_mcp.py -e neo4j+clip             # Short form

# Run legacy agent (deprecated, non-MCP)
python src/legacy/main.py
```

**Experiment Mode:** The `--experiment` flag allows A/B testing between:
- `memory`: In-memory spatial index (default, faster)
- `neo4j`: Neo4j graph database queries (enables semantic reasoning)
- `memory+clip`: In-memory + CLIP visual matching (image-to-BIM grounding)
- `neo4j+clip`: Neo4j + CLIP visual matching (full multimodal)

Results are saved with mode suffix: `eval_TIMESTAMP_memory.json`, `eval_TIMESTAMP_neo4j+clip.json`, etc.

Test scenarios are defined in YAML files:
- [prompts/tests/test_2.yaml](prompts/tests/test_2.yaml) - 18 comprehensive scenarios
- [prompts/tests/test_3_min_complete.yaml](prompts/tests/test_3_min_complete.yaml) - Minimal complete flow

### Run Baseline Experiment
```bash
python script/baseline_experiment.py
# Output: logs/experiments/baseline_gt007_results.json
```

### Run Evaluation Pipeline v2
```bash
# Run full evaluation with default config
python script/eval_pipeline.py --config config.yaml

# Run with specific ground truth file
python script/eval_pipeline.py --gt data/ground_truth/gt_1/gt_1.json

# Run only first N scenarios (for testing)
python script/eval_pipeline.py --scenarios 2

# Specify custom output directory
python script/eval_pipeline.py --output logs/experiments

# Outputs:
#   logs/evaluations/traces_YYYYMMDD_HHMMSS.jsonl  (detailed traces)
#   logs/evaluations/summary_YYYYMMDD_HHMMSS.csv   (metrics summary)
```

### Direct IFCEngine Usage
```python
from src.ifc_engine import IFCEngine

engine = IFCEngine("data/ifc/AdvancedProject/IFC/AdvancedProject.ifc")
windows = engine.find_elements_in_space("6 - sixth floor")
print(f"Found {len(windows)} elements on 6th floor")
```

### Start MCP Server Standalone (Advanced)
```bash
# For development/debugging with interactive inspector:
fastmcp dev mcp_servers/ifc_server.py

# Or run standalone (requires separate MCP client to connect):
python mcp_servers/ifc_server.py
```

> **Note:** For normal usage, just run `python src/main_mcp.py` - it spawns the server automatically.

---

## Evaluation Pipeline v2

Structured evaluation framework with data contracts, RQ2 schema validation, and standardized output formats.

### Data Contracts ([src/eval/contracts.py](src/eval/contracts.py))

| Model | Purpose |
|-------|---------|
| `ScenarioInput` | Input scenario parsed from ground truth JSON |
| `ToolStepRecord` | Single tool invocation trace (name, args, result, latency) |
| `InterpreterOutput` | Parsed agent response (GUIDs, candidates, escalation) |
| `EvalTrace` | Complete evaluation record (input + trace + RQ2 result) |
| `MetricsSummary` | Aggregated metrics including RQ2 validation stats |
| `RQ2Result` | RQ2 post-processing result (schema validation, submission) |
| `RQ2Submission` | CORENET-X-like submission structure |
| `RQ2ValidationMetadata` | Schema validation result (passed, fill_rate, errors) |

### Metrics ([src/eval/metrics.py](src/eval/metrics.py))

| Metric | Description |
|--------|-------------|
| `top1_hit(trace)` | First candidate matches ground truth GUID |
| `topk_hit(trace, k)` | Ground truth GUID in top-k candidates |
| `search_space_reduction(trace)` | `1 - (final_pool / initial_pool)` |
| `field_population_rate(trace)` | Fraction of expected fields populated |
| `is_escalation(trace)` | Agent couldn't resolve (needs human) |
| `compute_summary(traces)` | Aggregate all metrics + RQ breakdown + RQ2 validation |

### Output Formats

**JSONL Trace** (`traces_*.jsonl`):
```json
{
  "scenario_id": "GT_004_RQ2",
  "guid_match": true,
  "rq2_result": {
    "schema_id": "corenetx-like-minimal/v0",
    "submission": {
      "validation_metadata": {"passed": true, "required_fill_rate": 1.0}
    }
  }
}
```

**CSV Summary** (`summary_*.csv`):
```
=== OVERALL METRICS ===
Metric,Value
Total Scenarios,6
Top-1 Accuracy,33.33%
...

=== RQ2 SCHEMA VALIDATION ===
Metric,Value
RQ2 Total Scenarios,2
RQ2 Validation Passed,2
RQ2 Validation Pass Rate,100.00%
RQ2 Avg Fill Rate,95.00%
```

---

## RQ2: Schema-Aware Validation Pipeline

The RQ2 pipeline provides deterministic validation of agent outputs against CORENET-X-like submission schemas.

### Architecture

```
Agent Output (with FINAL_JSON tag)
        │
        ▼
┌───────────────────────────────────────────┐
│  extract_final_json.py                    │
│  Parse FINAL_JSON={...} from response     │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  mapping.py (build_submission)            │
│  • Map agent fields → submission schema   │
│  • Call MCP tools for IFC data:           │
│    - get_element_details(guid)            │
│    - list_available_spaces()              │
│  • Deterministic, no LLM calls            │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  validators.py (validate_all)             │
│  • JSON Schema validation (Draft 2020-12) │
│  • Domain checks:                         │
│    - GUID exists in model                 │
│    - Storey name valid                    │
│  • Compute required_fill_rate             │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│  RQ2Result                                │
│  • submission: validated JSON             │
│  • validation_metadata: passed, errors    │
│  • uncertainty: escalation info           │
└───────────────────────────────────────────┘
```

### Agent Output Format

The agent must output a `FINAL_JSON` tag in its response:

```
Based on my analysis, the cracked wall is on Level 1...

FINAL_JSON={"selected_guid": "0cRoQU_sD5R8MkkMkeodzx", "selected_storey_name": "Level 1", "issue_type": "defect", "severity": "medium", "issue_summary": "Crack observed on interior wall"}
```

### Schema Structure ([schemas/corenetx_min/v0.schema.json](schemas/corenetx_min/v0.schema.json))

```json
{
  "submission_id": "demo_abc123",
  "issue": {
    "issue_type": "defect|compliance|safety",
    "severity": "low|medium|high|critical",
    "issue_summary": "Description of the issue"
  },
  "bim_reference": {
    "element_guid": "0cRoQU_sD5R8MkkMkeodzx",
    "ifc_class": "IfcWall",
    "storey_name": "Level 1",
    "property_sets": {}
  },
  "validation_metadata": {
    "schema_id": "corenetx-like-minimal/v0",
    "required_fill_rate": 1.0,
    "passed": true,
    "errors": []
  }
}
```

### Smoke Test

```bash
# Run RQ2 smoke test (no MCP required)
python script/rq2_schema_smoke_test.py

# Expected output:
# ✅ TEST 1 PASSED (basic pass case)
# ✅ TEST 2 PASSED (missing GUID case)
# ✅ TEST 3 PASSED (invalid storey case)
# ✅ TEST 4 PASSED (FINAL_JSON extraction)
```

---

## Visual Analysis Module

The visual analysis module provides CLIP-based multimodal embedding for matching site photos to BIM elements.

### Architecture

```
Site Photo (JPG/PNG)                 User Description (Text)
        │                                     │
        ▼                                     ▼
┌───────────────────────────────────────────────────────────┐
│              VisualAligner (CLIP Model)                   │
│  • openai/clip-vit-base-patch32 (lazy-loaded, singleton)  │
│  • GPU acceleration if available                          │
└───────────────────────────────────────────────────────────┘
        │                                     │
        ▼                                     ▼
   Image Embedding                      Text Embedding
   (1, 512) tensor                      (1, 512) tensor
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
              Cosine Similarity
                       │
                       ▼
              Ranked BIM Elements
              (by visual match score)
```

### MCP Tools

| Tool | Use Case |
|------|----------|
| `analyze_site_image(path)` | "What's in this photo?" → Element classification |
| `match_image_to_elements(path, storey, k)` | Photo → Top-k BIM elements with GUIDs |
| `compare_defect_images(path1, path2)` | "Same defect?" → Similarity score + interpretation |
| `match_text_to_elements(desc, storey, k)` | Description → Top-k BIM elements |

### Usage Example

```python
from src.visual import VisualAligner

aligner = VisualAligner()  # Singleton, lazy CLIP load

# Match site photo to BIM elements
results = aligner.match_image_to_elements(
    image_path="data/ground_truth/gt_1/imgs/crack.jpg",
    elements=[{"guid": "...", "name": "Wall_1", "type": "IfcWall"}, ...],
    top_k=5
)
# Returns: [{"rank": 1, "guid": "...", "score": 0.78}, ...]

# Compare two photos
similarity = aligner.compare_images("photo1.jpg", "photo2.jpg")
# Returns: 0.85 (HIGH - likely same subject)
```

### Test Suite

```bash
# Run visual aligner tests (7 tests)
python script/test_visual_aligner.py

# Expected output:
# ✅ [PASS] Import Check
# ✅ [PASS] Model Loading
# ✅ [PASS] Text Matching
# ✅ [PASS] Top-K Matching
# ✅ [PASS] Image Embedding
# ✅ [PASS] Image-to-Description
# ✅ [PASS] Element Description
```

---

## P2: BCF Issue Generation & Handoff

The BCF handoff pipeline generates interoperable issue files for BIM collaboration tools.

### Architecture

```
Evaluation Result (from main_mcp.py)
        │
        ▼
┌───────────────────────────────────────────┐
│  trace.py (build_trace)                   │
│  • Extract GUID from agent response       │
│  • Combine inputs, tool calls, eval result│
│  • Write trace.json (Single Source of Truth)│
└───────────────────────────────────────────┘
        │
        ├──────────────────┬────────────────┐
        ▼                  ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ BCF-lite JSON   │ │ BCFzip 2.1      │ │ Trace JSON      │
│ (issue.json)    │ │ (.bcfzip)       │ │ (.trace.json)   │
│ • Human-readable│ │ • BIM-tool ready│ │ • Reproducibility│
│ • Evidence refs │ │ • markup.bcf    │ │ • Full context  │
│ • Trace URI     │ │ • viewpoint.bcfv│ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Output Structure

After running `python src/main_mcp.py`, handoff artifacts are generated:

```
outputs/
├── traces/20260127_120000/
│   ├── GT_001.trace.json
│   ├── GT_002.trace.json
│   └── ...
├── issues/20260127_120000/
│   ├── GT_001.issue.json
│   ├── GT_002.issue.json
│   └── ...
└── bcf/20260127_120000/
    ├── GT_001.bcfzip
    ├── GT_002.bcfzip
    └── ...
```

### BCFzip Structure

Each `.bcfzip` file is BCF 2.1 compliant:

```
GT_001.bcfzip
├── bcf.version              # BCF version declaration (2.1)
└── <topic_guid>/
    ├── markup.bcf           # Topic metadata, viewpoint refs
    └── viewpoints/
        ├── viewpoint.bcfv   # Component selection (IfcGuid)
        └── snapshot.png     # Evidence image (if available)
```

### BCF-lite Issue JSON

Lightweight JSON format for downstream systems:

```json
{
  "issue_id": "d2436fb0-f1a6-48ac-8403-a6136b71f64b",
  "case_id": "GT_001",
  "title": "GT_001: Basic Wall:Interior",
  "element_guid": "0cRoQU_sD5R8MkkMkeodzx",
  "guid_source": "regex_from_agent_response",
  "bim_reference": {
    "element_guid": "0cRoQU_sD5R8MkkMkeodzx",
    "ifc_class": "IfcWall",
    "storey_name": "Level 1"
  },
  "schema": {
    "validation": {"passed": true, "fill_rate": 1.0}
  },
  "evidence": ["data/ground_truth/gt_1/imgs/img_gt_001.png"],
  "trace_uri": "outputs/traces/20260127_120000/GT_001.trace.json"
}
```

### Sanity Test

```bash
# Run BCF generation sanity test (no MCP required)
python script/test_bcf_generation.py

# Expected output:
# ✅ TEST 1 PASSED (IFC GUID extraction)
# ✅ TEST 2 PASSED (Trace building)
# ✅ TEST 3 PASSED (Trace JSON writing)
# ✅ TEST 4 PASSED (BCF-lite issue JSON)
# ✅ TEST 5 PASSED (BCFzip generation)
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| IFC Processing | IfcOpenShell |
| Spatial Indexing | Custom graph (storey → elements) |
| MCP Server | FastMCP |
| LLM Agent | Google Gemini 2.5 Flash |
| Data Contracts | Pydantic v2 |
| Schema Validation | jsonschema (Draft 2020-12) |
| BCF Generation | stdlib zipfile + xml.etree (BCF 2.1) |
| Graph Database | Neo4j (optional) |
| Visual Matching | OpenAI CLIP |
| 3D Rendering | Blender + Bonsai addon (headless) |

---

## Current Progress

- [x] Centralized configuration (config.yaml)
- [x] IFCEngine with accurate spatial indexing
- [x] MCP server with 4D context tools
- [x] Ground truth test cases with real GUIDs
- [x] Baseline experiment (98.9% redundancy reduction)
- [x] Neo4j export support (optional)
- [x] **Evaluation Pipeline v2** (JSONL traces + CSV summary)
  - Data contracts (Pydantic models)
  - Metrics computation (top1_hit, topk_hit, search_space_reduction)
  - CLI script with streaming output
- [x] **RQ2: Schema-Aware Validation Pipeline**
  - FINAL_JSON extraction from agent output
  - Deterministic mapping to CORENET-X-like schema
  - JSON Schema validation (Draft 2020-12)
  - Domain validation (GUID exists, storey valid)
  - Integrated into unified eval pipeline
  - RQ2 metrics in summary output
- [x] **P2: BCF Issue Generation & Handoff**
  - Trace builder with GUID extraction fallback chain
  - BCF-lite issue.json for lightweight handoff
  - BCFzip 2.1 with markup.bcf + viewpoint.bcfv
  - Component selection (IfcGuid) in viewpoints
  - Snapshot images from evidence
  - Integrated into main_mcp.py evaluation loop
- [x] **Visual Analysis Module (CLIP-based)**
  - VisualAligner class with lazy-loaded CLIP model
  - Image → Vector and Text → Vector embedding
  - Image-to-element matching with storey filtering
  - Defect image comparison (similarity scoring)
  - Text description to element matching
  - Integrated as MCP tools in ifc_server.py
- [x] **3D View Generation**
  - `generate_3d_view(guid)` MCP tool for visual verification snapshots
  - Blender headless rendering via render_worker.py
  - Automatic camera positioning and element framing
  - GUID-based element lookup using Bonsai API
  - Output to `outputs/renders/` directory

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

**"Blender executable not found"**
- Install Blender with Bonsai/BlenderBIM addon
- Set `BLENDER_PATH` environment variable: `export BLENDER_PATH=/path/to/blender`

**"generate_3d_view returns error"**
- Ensure Bonsai addon is enabled in Blender
- Check IFC file is loaded: `ifc_engine.ifc_path` should be set
- Verify render_worker.py exists: `script/render_worker.py`

---

## References

- IFC Schema: [buildingSMART IFC4](https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/)
- MCP Protocol: [Model Context Protocol](https://modelcontextprotocol.io/)
- IfcOpenShell: [Documentation](https://ifcopenshell.org/)

---

**Last Updated:** February 2026
**Status:** Unified Evaluation Pipeline with RQ2 Schema Validation, BCF Handoff, Visual Analysis, and 3D View Generation complete
