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

### Full Experiment Matrix

All runs use the synthetic dataset (43 cases). LLM API is Gemini 2.5 Flash ($0.15/M input, $0.60/M output).

| # | Run | What it tests | Cases | Est. Time | Est. Cost | Command |
|---|-----|--------------|-------|-----------|-----------|---------|
| 1 | V1 memory | Baseline retrieval | 43 | ~4 min | ~$0.10 | `./run_mcp.sh -e memory -d synth` |
| 2 | V1 neo4j | Graph queries | 43 | ~4 min | ~$0.10 | `./run_mcp.sh -e neo4j -d synth` |
| 3 | V1 memory+clip | CLIP reranking | 43 | ~15 min | ~$0.10 | `./run_mcp.sh -e memory+clip -d synth` |
| 4 | V1 neo4j+clip | Graph + CLIP | 43 | ~15 min | ~$0.10 | `./run_mcp.sh -e neo4j+clip -d synth` |
| 5 | V2 v2_prompt | Constraints extraction | 43 | ~2 min | ~$0.02 | `python script/run.py --profile v2_prompt --cases ../data_curation/datasets/synth_v0.2/cases_v2.jsonl` |
| 6 | V2 v2_memory | V2 baseline | 43 | ~1.5 min | ~$0.02 | `python script/run.py --profile v2_memory --cases ../data_curation/datasets/synth_v0.2/cases_v2.jsonl` |
| 7-15 | V2 x 9 conditions | A1-C3 ablations | ~5 each | ~10 min | ~$0.05 | See below |
| | **Total** | | | **~50 min** | **~$0.50** | |

**Run all V1 modes at once:**
```bash
./run_mcp.sh --all -d synth                  # Runs #1-4
./run_mcp.sh --all --v2 -d synth             # Runs #1-6
```

**Run V2 condition ablations (A1-C3):**
```bash
CASES=../data_curation/datasets/synth_v0.2/cases_v2.jsonl
for cond in A1 A2 A3 B1 B2 B3 C1 C2 C3; do
  echo "=== Condition: $cond ==="
  python script/run.py --profile v2_prompt --cases $CASES --condition $cond
done
```

Runs #2 and #4 require Neo4j running (see [Neo4j Setup](#optional-neo4j-setup)). All other runs work without it.

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

### Generating Evaluation Plots

After running evaluations, generate publication-ready visualizations:

```bash
# Auto-generate plots from latest evaluation
conda run -n mscd_demo python script/generate_plots.py --latest

# Compare before/after VLM integration
conda run -n mscd_demo python script/generate_plots.py \
  --traces logs/evaluations/new_pipeline/traces_*.jsonl \
  --before logs/evaluations/old_pipeline/traces_*.jsonl
```

**Output**: Plots saved to `logs/plots/<timestamp>_<profile>/`

**Available charts**:
- Top-1 Accuracy by Condition
- Search Space Reduction (funnel chart — **key metric for thesis**)
- Constraints Parse Rate
- Image Parse Timing (VLM overhead)
- Vision Impact (before/after comparison)
- Per-Case Success Heatmap

**For thesis**: All plots are 300 DPI PNG files ready for LaTeX/Word.

See [logs/plots/PLOTS_README.md](logs/plots/PLOTS_README.md) for detailed usage and examples.

### Experiment Management

For systematic experiment tracking and reproducibility, use the experiment management system:

**Define experiments in `experiments.yaml`:**
```yaml
experiments:
  vlm_integration:
    description: "V2 pipeline with Gemini VLM for image parsing"
    profile: v2_prompt
    conditions: [A1, B1, C1]
    cases: data_curation/datasets/synth_v0.2/cases_v2.jsonl
    output_dir: logs/experiments/vlm_integration
    tags: [main, vlm-enabled]
```

**Run experiments:**
```bash
# List all available experiments
python script/experiment.py list

# Run a single experiment
python script/experiment.py run vlm_integration

# Run multiple experiments and compare
python script/experiment.py run baseline_v2 vlm_integration --compare vlm_impact

# Quick test (5 cases)
python script/experiment.py run quick_test
```

**One-click VLM comparison (main thesis experiment):**
```bash
./run_vlm_comparison.sh
```

This will:
1. Run baseline evaluation (before VLM fix)
2. Run VLM-enabled evaluation (after VLM fix)
3. Generate comparison plots automatically

Results saved to:
- `logs/experiments/baseline_v2/` - Baseline results
- `logs/experiments/vlm_integration/` - VLM results
- `logs/comparisons/vlm_impact/` - Comparison plots

**Experiment metadata tracking:**

Each experiment automatically saves:
- Git commit hash and branch
- Uncommitted changes (diff)
- Conda environment snapshot
- Experiment configuration
- Start/end timestamps

This ensures **full reproducibility** — you can regenerate any result months later.

**Available experiments:**
- `quick_test` - 5 cases for debugging (B1 only)
- `baseline_v2` - V2 without VLM (A1, B1, C1)
- `vlm_integration` - V2 with VLM fix (A1, B1, C1)
- `vlm_full_test` - All 9 conditions (A1-C3)
- `ablation_images_only` - VLM on images only (B1-B3)
- `ablation_floorplan_only` - VLM on floorplan only (C1-C3)

See [experiments.yaml](experiments.yaml) for full configuration.

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
  ImageParserReader       VLM-based image parsing (cached, structured descriptions)
        |
        v
  ConstraintsExtractor    Extract storey, ifc_class, keywords (prompt or LoRA)
        |                 Uses pre-parsed image semantics
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
- **`common/`** — Centralized utilities: config/LLM init, context formatting, GUID extraction, MCP connection helper
- **IFCEngine** — IFC model loading, spatial index, property extraction
- **ImageParserReader** — VLM-based image parsing (Gemini 2.5 Flash), structured descriptions with caching
- **VisualAligner** — CLIP-based image-to-element matching
- **RQ2 Schema Pipeline** — Schema validation of structured outputs
- **BCF Handoff** — BCF 2.1 issue file generation (shared title/description builders in `trace.py`)
- **Eval Contracts** — Shared data models (`EvalTrace`, `ScenarioInput`, etc.)

---

## Project Structure

```
mscd_demo/
├── config.yaml                  # IFC path, Neo4j, LLM settings
├── profiles.yaml                # Experiment profiles and conditions
├── experiments.yaml             # Experiment definitions for reproducibility
│
├── prompts/                     # Centralized prompt templates
│   ├── constraints_extraction.yaml  # V2 constraints extraction prompts
│   ├── image_parsing.yaml       # VLM-based image parsing prompts (site photos & floorplans)
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
│   ├── common/                  # Shared utilities (single source of truth)
│   │   ├── config.py            # Config loading, system prompt, LLM init
│   │   ├── evaluation.py        # Context formatting, evaluation helpers
│   │   ├── guid.py              # IFC GUID extraction (regex)
│   │   ├── response_parser.py   # LangGraph response parsing
│   │   └── mcp.py               # MCP connection helper (async context manager)
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
│   │   ├── runner.py            # V1 scenario runner
│   │   └── visualizations.py    # Plot generators (6 chart types)
│   │
│   ├── rq2_schema/              # RQ2 schema validation
│   │   ├── extract_final_json.py
│   │   ├── schema_registry.py
│   │   ├── mapping.py
│   │   ├── validators.py
│   │   └── pipeline.py
│   │
│   ├── visual/                  # Visual analysis modules
│   │   ├── image_parser.py      # VLM-based image parsing (cached descriptions)
│   │   └── aligner.py           # CLIP-based image-to-element matching
│   │
│   └── handoff/                 # BCF issue generation
│       ├── trace.py             # Trace builder + shared title/description helpers
│       ├── bcf_lite.py          # JSON issue output
│       └── bcf_zip.py           # BCF 2.1 zip generation
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
│   ├── experiment.py            # Experiment management orchestrator
│   ├── generate_plots.py        # Visualization generator
│   ├── compare_results.py       # Compare eval results across experiments
│   ├── baseline_experiment.py   # Redundancy quantification
│   └── ...
│
├── run_experiment.sh            # Quick experiment runner
├── run_vlm_comparison.sh        # One-click VLM before/after comparison
│
├── test/
│   └── test_v2_smoke.py         # V2 smoke tests (17 tests)
│
├── logs/
│   ├── evaluations/             # Ad-hoc evaluation runs (traces + summaries)
│   ├── experiments/             # Organized experiment results with metadata
│   ├── comparisons/             # Comparison plots between experiments
│   └── plots/                   # Generated visualization charts
│
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
| Image Parsing (VLM) | Google Gemini 2.5 Flash (multimodal) |
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
