# MSCD Demo: Interpreter Layer for AEC Inspection

> Master Thesis — Cross-Modal Alignment, Schema Mapping, and Compliance

---

## What This Is

An interpreter layer that helps identify BIM elements from messy site inspection reports. It takes chat messages, site photos, floorplans, and 4D project context, then finds the matching IFC element in the building model.

**The problem:** "Which window?" matches 263 candidates (0.38% precision).

**The solution:** Using 4D context (floor, task status, images) narrows it down to 3 candidates (33.33% precision). That's 98.9% search space reduction.

The system supports two pipelines:
- **V1 (agent-driven):** An LLM agent reasons freely and calls tools to find elements.
- **V2 (constraints-driven):** Explicit constraints extraction followed by deterministic query planning. More interpretable, reproducible, and supports controlled experiments.

---

## Quick Start

### 1. Environment Setup

```bash
conda activate mscd_demo
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env — add your GOOGLE_API_KEY
```

### 3. Run

**V1 — Agent-driven evaluation (original):**
```bash
python src/main_mcp.py
```

**V2 — Constraints-driven evaluation (new):**
```bash
python script/run.py --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl
```

**Interactive chat (for testing):**
```bash
python src/chat_cli.py
```

---

## Usage

### Unified Runner (`script/run.py`)

This is the main entry point for all experiments. It uses profiles defined in `profiles.yaml` to control which pipeline, retrieval backend, and features to use.

**Basic usage:**
```bash
python script/run.py --profile <profile_name> --cases <path_to_cases.jsonl>
```

**Arguments:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--profile` | Yes | Profile name from `profiles.yaml` |
| `--cases` | Yes | Path to cases JSONL file |
| `--condition` | No | Filter cases by condition (A1-C3) |
| `--adapter_path` | No | LoRA adapter checkpoint path (for v2 lora mode) |
| `--output_dir` | No | Output directory (default: `logs/evaluations`) |
| `--config` | No | Path to `config.yaml` (default: `config.yaml`) |
| `--profiles` | No | Path to `profiles.yaml` (default: `profiles.yaml`) |

**Examples:**

```bash
# V2 with prompt-only constraints extraction
python script/run.py --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl

# V2 with LoRA extraction (when adapter is trained)
python script/run.py --profile best_v2 \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl \
  --adapter_path models/qwen3-vl-8b-lora/checkpoint-1000

# Run only condition A2 cases
python script/run.py --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl \
  --condition A2

# V1 baseline for comparison
python script/run.py --profile v1_baseline \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl

# Ablation: V2 without CLIP
python script/run.py --profile ablate_no_clip \
  --cases data_curation/datasets/synth_v0.2/cases_v2.jsonl
```

**Output files:**
```
logs/evaluations/
  traces_20260206_153045_v2_prompt.jsonl   # Per-case detailed traces
  summary_20260206_153045_v2_prompt.csv    # Aggregated metrics
```

### Available Profiles

Profiles are defined in `profiles.yaml`. Each profile specifies the full experimental configuration.

| Profile | Pipeline | Constraints | Retrieval | CLIP | Description |
|---------|----------|-------------|-----------|------|-------------|
| `v2_prompt` | v2 | prompt | neo4j | no | V2 with prompt-based extraction |
| `v2_memory` | v2 | prompt | memory | no | V2 with in-memory retrieval (fastest) |
| `best_v2` | v2 | lora | neo4j | yes | V2 with all features enabled |
| `v1_baseline` | v1 | — | memory | no | Original V1 agent pipeline |
| `v1_full` | v1 | — | neo4j | yes | V1 with all features |
| `ablate_no_clip` | v2 | lora | neo4j | no | Ablation: no CLIP reranking |
| `ablate_no_graph` | v2 | lora | memory | yes | Ablation: no graph database |
| `ablate_no_schema` | v2 | lora | neo4j | yes | Ablation: no RQ2 validation |

### Experimental Conditions (A1–C3)

Each case in `cases_v2.jsonl` has a `bench.condition` field. Conditions control which input modalities are available, simulating different real-world scenarios.

| Condition | Chat | Images | Floorplan | 4D Metadata | CLIP Rerank |
|-----------|------|--------|-----------|-------------|-------------|
| **A1** | clear | no | no | yes | no |
| **A2** | blurred | no | no | yes | no |
| **A3** | blurred | no | no | yes (enhanced) | no |
| **B1** | blurred | yes | no | no | no |
| **B2** | blurred | yes | no | no | yes |
| **B3** | clear | yes | no | no | no |
| **C1** | clear | no | yes | no | no |
| **C2** | blurred | yes | yes | no | no |
| **C3** | clear | yes | yes | yes (enhanced) | no |

"Blurred" means specific keywords are replaced (e.g., "window" becomes "opening", "sixth" becomes "upper") to test whether the system can still find the right element without explicit hints.

### V1 Agent-Driven Pipeline

```bash
# Full MCP-based evaluation
python src/main_mcp.py

# With specific experiment mode
python src/main_mcp.py --experiment memory        # In-memory spatial index
python src/main_mcp.py --experiment neo4j         # Neo4j graph queries
python src/main_mcp.py --experiment memory+clip   # In-memory + CLIP
python src/main_mcp.py --experiment neo4j+clip    # Neo4j + CLIP

# Evaluation pipeline with custom options
python script/eval_pipeline.py --config config.yaml
python script/eval_pipeline.py --gt data/ground_truth/gt_1/gt_1.json
python script/eval_pipeline.py --scenarios 2      # Run first N only
```

### Run All Experiments and Compare

Use `run_mcp.sh --all` to run all 4 V1 experiment modes in a row, then automatically compare results.

```bash
./run_mcp.sh --all                        # Run all V1 modes (memory, neo4j, memory+clip, neo4j+clip)
./run_mcp.sh --all -d synth              # Run all on synthetic dataset
./run_mcp.sh --all --v2                  # Also run V2 profiles after V1
./run_mcp.sh --all --delay 15            # 15s delay between runs (default: 10)
```

Failed runs don't stop the batch. At the end you get a summary of which succeeded/failed, total time, and a comparison table.

**Compare results independently:**

```bash
python script/compare_results.py                    # Show all results
python script/compare_results.py --latest           # 4 most recent
python script/compare_results.py --latest 8         # 8 most recent
python script/compare_results.py --v1-only          # V1 results only
python script/compare_results.py --csv out.csv      # Export to CSV
```

Example output:
```
Pipeline              Mode   Cases    Top-1    Top-3    Top-5      P@1   Recall       F1
      v1            memory       6    0.000    0.000    0.000    0.000    0.000    0.000
      v1       memory+clip       6    0.167    0.167    0.167    0.167    0.167    0.167
      v1             neo4j       6    0.000    0.000    0.000    0.000    0.000    0.000
      v1        neo4j+clip       6    0.167    0.167    0.167    0.167    0.167    0.167

  Best Top-1 Accuracy: neo4j+clip (0.167)
  Best F1 Score: neo4j+clip (0.167)
```

### Interactive Chat

```bash
python src/chat_cli.py                    # Start interactive session
python src/chat_cli.py --scenario GT_007  # Pre-load a scenario
```

**Chat commands:**

| Command | Description |
|---------|-------------|
| `quit` / `q` | Exit |
| `clear` | Clear conversation history |
| `tools` | List available MCP tools |
| `scenarios` | List test scenarios |
| `load <id>` | Load a scenario (e.g., `load GT_007`) |
| `send` | Send the loaded scenario's query |

### Running Tests

```bash
conda activate mscd_demo

# V2 smoke tests (no LLM or IFC needed)
python -m pytest test/test_v2_smoke.py -v

# RQ2 schema smoke test
python script/rq2_schema_smoke_test.py

# BCF generation test
python script/test_bcf_generation.py

# Visual aligner test
python script/test_visual_aligner.py
```

---

## Architecture

### V2 Pipeline (Constraints-Driven)

```
Input Case (chat + images + 4D context)
        |
        v
  ConditionMask           Apply A1-C3 modality masking
        |
        v
  ConstraintsExtractor    Extract storey, ifc_class, keywords (prompt or LoRA)
        |
        v
  QueryPlanner            Deterministic template-based query plans
        |
        v
  RetrievalBackend        Execute queries (memory or neo4j, optional CLIP rerank)
        |
        v
  EvalTrace + V2Trace     V1-compatible output + V2 diagnostics
```

### V1 Pipeline (Agent-Driven)

```
Input Case
        |
        v
  MCP Agent (Gemini)      LLM reasons and calls tools freely
        |
        v
  MCP Tools                get_elements_by_storey, match_image_to_elements, etc.
        |
        v
  EvalTrace                Standard evaluation output
```

### Shared Components

Both pipelines share these components:
- **IFCEngine** — IFC model loading, spatial index, property extraction
- **VisualAligner** — CLIP-based image-to-element matching
- **RQ2 Schema Pipeline** — Schema validation of structured outputs
- **BCF Handoff** — BCF 2.1 issue file generation
- **Eval Contracts** — Shared data models (`EvalTrace`, `ScenarioInput`, etc.)

---

## Project Structure

```
mscd_demo/
├── config.yaml                  # IFC path, Neo4j, LLM settings
├── profiles.yaml                # Experiment profiles and conditions
│
├── prompts/                     # Centralized prompt templates
│   ├── constraints_extraction.yaml  # V2 constraints extraction prompts
│   ├── system_prompt.yaml       # V1 agent system prompt
│   └── tool_descriptions.yaml   # MCP tool descriptions
│
├── src/
│   ├── main_mcp.py              # V1 entry point (MCP agent)
│   ├── chat_cli.py              # Interactive chat CLI
│   ├── ifc_engine.py            # Core IFC processing engine
│   ├── pipeline_base.py         # Pipeline abstraction (V1Pipeline, V2Pipeline)
│   ├── mcp_langchain_adapter.py # MCP to LangChain adapter
│   │
│   ├── v2/                      # V2 constraints-driven pipeline
│   │   ├── types.py             # Data models (Constraints, QueryPlan, V2Trace)
│   │   ├── condition_mask.py    # A1-C3 input masking
│   │   ├── constraints_extractor_prompt_only.py  # LLM-based extraction
│   │   ├── constraints_extractor_lora.py         # LoRA-based extraction
│   │   ├── constraints_to_query.py  # Template query planner
│   │   ├── retrieval_backend.py # Memory/Neo4j/CLIP retrieval
│   │   ├── pipeline.py          # V2 pipeline orchestration
│   │   └── metrics_v2.py        # V2 diagnostic metrics
│   │
│   ├── eval/                    # Evaluation framework
│   │   ├── contracts.py         # Pydantic data models
│   │   ├── metrics.py           # Metric functions
│   │   └── runner.py            # V1 scenario runner
│   │
│   ├── rq2_schema/              # RQ2 schema validation
│   │   ├── extract_final_json.py
│   │   ├── schema_registry.py
│   │   ├── mapping.py
│   │   ├── validators.py
│   │   └── pipeline.py
│   │
│   ├── visual/                  # CLIP visual analysis
│   │   └── aligner.py
│   │
│   └── handoff/                 # BCF issue generation
│       ├── trace.py
│       ├── bcf_lite.py
│       └── bcf_zip.py
│
├── mcp_servers/
│   └── ifc_server.py            # MCP server with IFC + visual tools
│
├── schemas/
│   └── corenetx_min/
│       └── v0.schema.json       # CORENET-X minimal submission schema
│
├── data/
│   ├── ifc/AdvancedProject/     # BIM model (10 storeys, 263 windows)
│   └── ground_truth/gt_1/       # Hand-written test cases (6 cases)
│
├── script/
│   ├── run.py                   # Unified evaluation runner (v1 + v2)
│   ├── compare_results.py       # Compare eval results across experiments
│   ├── eval_pipeline.py         # V1 evaluation pipeline
│   ├── baseline_experiment.py   # Redundancy quantification
│   └── ...
│
├── test/
│   └── test_v2_smoke.py         # V2 smoke tests (17 tests)
│
├── logs/evaluations/            # Output traces + summaries
└── outputs/                     # BCF artifacts, renders
```

---

## Configuration

### `config.yaml` — Runtime settings

```yaml
ifc:
  model_path: "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"

neo4j:
  uri: "bolt://localhost:7687"
  enabled: false

ground_truth:
  file: "data/ground_truth/gt_1/gt_1.json"
  image_dir: "data/ground_truth/gt_1/imgs"

llm:
  model: "gemini-2.5-flash"
  temperature: 0

rq2:
  enabled: true
  schema_path: "schemas/corenetx_min/v0.schema.json"
```

### `profiles.yaml` — Experiment configurations

```yaml
profiles:
  v2_prompt:
    pipeline: v2
    constraints_model: prompt
    retrieval: neo4j
    use_clip: false
    rq2_schema: true
    description: "V2 with prompt-based constraints"

  v1_baseline:
    pipeline: v1
    retrieval: memory
    use_clip: false
    rq2_schema: true
    description: "Original V1 agent pipeline"

conditions:
  A1:
    use_images: false
    use_floorplan: false
    chat_blur: false
  B2:
    use_images: true
    use_floorplan: false
    chat_blur: true
    force_clip: true
  # ... (A1-C3 defined)
```

---

## Datasets

There are two datasets. Both pipelines (V1 and V2) read from the same files.

| Dataset | Cases | File | Used by |
|---------|-------|------|---------|
| **gt_1** (hand-written) | 6 | `data/ground_truth/gt_1/gt_1.json` | V1 default |
| **synth_v0.2** (synthetic) | 43 | `../data_curation/datasets/synth_v0.2/cases_v2.jsonl` | V1 (`-d synth`) and V2 (`--cases`) |

The synthetic dataset (`cases_v2.jsonl`) is the main evaluation dataset. It uses a single standardized JSONL format that both V1 and V2 pipelines read directly:

```json
{
  "case_id": "SYNTH_001_SK_001_V2",
  "query_text": "What's the spec for this?",
  "bench": {"group": "B", "condition": "B1"},
  "difficulty_tags": {"tier": "Tier 1", "candidate_density_k": 1},
  "inputs": {
    "chat_history": [{"role": "Site Supervisor", "text": "..."}],
    "images": ["datasets/synth_v0.2/cases/imgs/img_SYNTH_001.png"],
    "floorplan_patch": "datasets/synth_v0.2/cases/plans/plan_SYNTH_001.png",
    "project_context": {"timestamp": "...", "sender_role": "...", "4d_task_status": "..."}
  },
  "ground_truth": {
    "target_guid": "3GzoWuxxn4WO8bCtw8H3Vj",
    "target_storey": "1 - First Floor",
    "target_ifc_class": "IfcWall",
    "target_name": "Basic Wall:MockUp Interior...",
    "rq_category": "RQ1"
  },
  "labels": {
    "constraints": {"storey_name": "1 - First Floor", "ifc_class": "IfcWall"}
  }
}
```

---

## Output Format

### JSONL Traces

Each line is a JSON object with the full evaluation trace for one case:

```json
{
  "scenario_id": "CASE_007",
  "guid_match": true,
  "final_pool_size": 3,
  "initial_pool_size": 1200,
  "total_latency_ms": 2340.5
}
```

### CSV Summary

Three sections in each summary CSV:

1. **Overall Metrics** — top-1 accuracy, top-k accuracy, search space reduction, escalation rate
2. **V2 Diagnostic Metrics** (v2 only) — constraints parse rate, rerank gain, extraction/planning/retrieval latency
3. **Per-Case V2 Detail** (v2 only) — per-case constraints F1, rerank gain

---

## Research Questions

| RQ | Focus | V1 Approach | V2 Approach |
|----|-------|-------------|-------------|
| **RQ1** | Multimodal context | Agent tool calls | Constraints extraction + query planning |
| **RQ2** | Schema mapping | FINAL_JSON extraction | Same (shared pipeline) |
| **RQ3** | Abductive reasoning | Free-form agent | Escalation detection via empty results |

---

## Optional: Neo4j Setup

```bash
# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest

# Export IFC model to Neo4j
python script/ifc_to_neo4j.py

# Browse at http://localhost:7474
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No elements found for storey" | Use exact storey names: `"6 - Sixth Floor"` not `"Level 6"` |
| "Neo4j connection refused" | Check Docker is running, wait 30s after start |
| "GUID not found" | Verify with `engine.get_element_by_guid(guid)` |
| "Profile not found" | Check profile name exists in `profiles.yaml` |
| pytest import errors | Make sure you activated: `conda activate mscd_demo` |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| IFC Processing | IfcOpenShell |
| LLM Agent | Google Gemini 2.5 Flash |
| Visual Matching | OpenAI CLIP |
| Data Models | Pydantic v2 |
| MCP Server | FastMCP |
| Schema Validation | jsonschema (Draft 2020-12) |
| Graph Database | Neo4j (optional) |
| BCF Generation | stdlib zipfile + xml.etree (BCF 2.1) |
| 3D Rendering | Blender + Bonsai addon (headless) |

---

**Last Updated:** February 2026
**Status:** V1 + V2 pipelines operational. V2 supports prompt-only constraints extraction; LoRA extraction pending adapter training.
