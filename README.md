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

**V1 — Agent-driven evaluation:**
```bash
./run.sh mcp
```

**V2 — Constraints-driven (Gemini prompt):**
```bash
python script/run.py --profile v2_prompt \
  --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
```

**V2 — Constraints-driven (LoRA_2, synth_v0.4 adapter):**
```bash
# Download the trained adapter first (needs Modal):
./training/train.sh --download-only

# Run with the LoRA adapter:
python script/run.py --profile v2_lora \
  --cases logs/evaluations/synth_v04/eval_constraints_final.jsonl \
  --adapter_path models/adapters/v2_lora_qwen
```

**Train a new LoRA adapter:**
```bash
./training/train.sh              # launch on Modal A100
./training/train.sh --download-only   # download after training
```

**End-to-end LoRA evaluation (synth_v0.4 holdout, 6-condition modality ablation):**
```bash
./training/eval.sh --step modality-6cond --adapter final
```

**Regenerate plots from latest traces (never overwrites existing):**
```bash
./training/eval.sh --step update-plots
./training/eval.sh --step update-plots --experiment modality_6cond_lora   # LoRA only
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
| `--condition` | No | Filter cases by condition (A1-C3 or MA/MB/MC/MA-/MB-/MC-) |
| `--adapter_path` | No | LoRA adapter checkpoint path (for v2 lora mode) |
| `--output_dir` | No | Output directory (default: `logs/evaluations`) |
| `--config` | No | Path to `config.yaml` (default: `config.yaml`) |
| `--profiles` | No | Path to `profiles.yaml` (default: `profiles.yaml`) |

**Examples:**

```bash
# V2 with prompt-only constraints extraction (synth_v0.3)
python script/run.py --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl

# V2 with LoRA extraction (when adapter is trained)
python script/run.py --profile best_v2 \
  --cases data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \
  --adapter_path models/qwen2.5-vl-7b-lora/checkpoint-best

# Run only condition A2 cases
python script/run.py --profile v2_prompt \
  --cases data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl \
  --condition A2

# V1 baseline for comparison
python script/run.py --profile v1_baseline \
  --cases data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl

# Ablation: V2 without CLIP
python script/run.py --profile ablate_no_clip \
  --cases data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
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
| `v2_prompt` | v2 | prompt | neo4j | no | V2 with Gemini prompt-based extraction |
| `v2_lora` | v2 | lora | neo4j | no | **V2 with LoRA_2 adapter (synth_v0.4)** |
| `v2_memory` | v2 | prompt | memory | no | V2 with in-memory retrieval (fastest) |
| `best_v2` | v2 | lora | neo4j | yes | V2 with all features enabled |
| `v1_baseline` | v1 | — | memory | no | Original V1 agent pipeline |
| `v1_full` | v1 | — | neo4j | yes | V1 with all features |
| `ablate_no_clip` | v2 | lora | neo4j | no | Ablation: no CLIP reranking |
| `ablate_no_graph` | v2 | lora | memory | yes | Ablation: no graph database |
| `ablate_no_schema` | v2 | lora | neo4j | yes | Ablation: no RQ2 validation |

### Experimental Conditions

#### LoRA_2 Modality Ablation (MA/MB/MC — synth_v0.4)

The LoRA_2 evaluation uses a 6-condition paired ablation design to isolate the contribution of each visual modality and 4D project context independently.

| Condition | Visual Inputs | 4D Context | Purpose |
|-----------|---------------|------------|---------|
| **MA** | Text only | **ON** | Text + 4D baseline |
| **MB** | Text + Site photos | **ON** | +Photos vs MA |
| **MC** | Text + Photos + Floorplan | **ON** | +Floorplan vs MB |
| **MA-** | Text only | **OFF** | MA without 4D |
| **MB-** | Text + Site photos | **OFF** | MB without 4D |
| **MC-** | Text + Photos + Floorplan | **OFF** | MC without 4D |

Each condition runs 50 holdout cases × 2 profiles (LoRA + Prompt) = 600 traces total.
Comparing MA vs MA-, MB vs MB-, MC vs MC- isolates the pure 4D contribution at each modality level.

#### V2 Prompt Ablation (A1–C3 — synth_v0.3)

The older condition grid (used for V1/V2 prompt baseline experiments on synth_v0.3):

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
# Single V1 run (config.yaml defaults)
./run.sh mcp

# With specific experiment mode
./run.sh mcp -e memory        # In-memory spatial index
./run.sh mcp -e neo4j         # Neo4j graph queries
./run.sh mcp -e memory+clip   # In-memory + CLIP
./run.sh mcp -e neo4j+clip    # Neo4j + CLIP
```

### Run All Experiments and Compare

Use `./run.sh mcp --all` to run all 4 V1 experiment modes in a row, then automatically compare results.

```bash
./run.sh mcp --all                        # Run all V1 modes (memory, neo4j, memory+clip, neo4j+clip)
./run.sh mcp --all -d synth              # Run all on synthetic dataset
./run.sh mcp --all --v2                  # Also run V2 profiles after V1
./run.sh mcp --all --delay 15            # 15s delay between runs (default: 10)
```

Failed runs don't stop the batch. At the end you get a summary of which succeeded/failed, total time, and a comparison table.

### Full Experiment Matrix

All runs use the synthetic dataset (84 cases, synth_v0.3). LLM API is Gemini 2.5 Flash ($0.15/M input, $0.60/M output).

| # | Run | What it tests | Cases | Est. Time | Est. Cost | Command |
|---|-----|--------------|-------|-----------|-----------|---------|
| 1 | V1 memory | Baseline retrieval | 84 | ~8 min | ~$0.20 | `./run.sh mcp -e memory -d synth` |
| 2 | V1 neo4j | Graph queries | 84 | ~8 min | ~$0.20 | `./run.sh mcp -e neo4j -d synth` |
| 3 | V1 memory+clip | CLIP reranking | 84 | ~30 min | ~$0.20 | `./run.sh mcp -e memory+clip -d synth` |
| 4 | V1 neo4j+clip | Graph + CLIP | 84 | ~30 min | ~$0.20 | `./run.sh mcp -e neo4j+clip -d synth` |
| 5 | V2 v2_prompt | Constraints extraction | 84 | ~4 min | ~$0.04 | `python script/run.py --profile v2_prompt --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl` |
| 6 | V2 v2_memory | V2 baseline | 84 | ~3 min | ~$0.04 | `python script/run.py --profile v2_memory --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl` |
| 7-15 | V2 x 9 conditions | A1-C3 ablations | ~9 each | ~20 min | ~$0.10 | See below |
| | **Total** | | | **~100 min** | **~$1.00** | |

**Run all V1 modes at once:**
```bash
./run.sh mcp --all -d synth                  # Runs #1-4
./run.sh mcp --all --v2 -d synth             # Runs #1-6
```

**Run V2 condition ablations (A1-C3):**
```bash
CASES=../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
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

# Run all tests
python -m pytest test/ -v

# Individual test modules
python -m pytest test/test_ifc_engine.py -v       # IFC engine tests
python -m pytest test/test_bcf_generation.py -v    # BCF generation tests
python -m pytest test/test_visual_aligner.py -v    # Visual aligner tests
python -m pytest test/rq2_schema_smoke_test.py -v  # RQ2 schema validation
```

### Generating Evaluation Plots

After running evaluations, generate publication-ready visualizations:

```bash
# Regenerate from latest traces — creates dated, never-overwrites directory
./training/eval.sh --step update-plots

# LoRA only or Prompt only
./training/eval.sh --step update-plots --experiment modality_6cond_lora
./training/eval.sh --step update-plots --experiment modality_6cond_prompt

# Standalone comparison chart (any two JSONL trace files)
python script/compare_results.py \
  --traces logs/evaluations/synth_v04/traces_*_v2_lora_MA.jsonl  --label "LoRA MA" \
  --traces logs/evaluations/synth_v04/traces_*_v2_prompt_MA.jsonl --label "Prompt MA" \
  --cases ../data_curation/datasets/synth_v0.4_merged/train/test_holdout.jsonl \
  --plots --output docs/plots/my_comparison

# Modality-only charts (Charts 9–13) from JSONL files
conda run -n mscd_demo python script/generate_plots.py \
  --modality \
  --traces logs/evaluations/synth_v04/traces_*_v2_lora_M*.jsonl \
           logs/evaluations/synth_v04/traces_*_v2_prompt_M*.jsonl \
  --output docs/plots/my_modality
```

**Output**: Plots saved to `docs/plots/<MMDD>_<experiment>/` (versioned `_v2`, `_v3`… if directory exists)

**Charts 1–8** (comparison, via `compare_results.py`):
1. Overall Top-1 Accuracy (LoRA vs Prompt)
2. Condition-wise comparison (MA/MB/MC/MA-/MB-/MC-)
3. Search Space Reduction (funnel — **key thesis metric**)
4. Efficiency (latency)
5. Accuracy heatmap
6. Accuracy heatmap detail
7. Modality gain by difficulty tier (T1/T2/T3)
8. Difficulty degradation by modality (T1→T2→T3)
9. Candidate density vs accuracy scatter

**Charts 9–13** (modality analysis, via `generate_plots.py --modality`):
- `9_modality_stack_MA_MB_MC.png` — Grouped bars: MA/MB/MC × LoRA/Prompt
- `11_modality_x_building.png` — Heatmap: building (AP/BH/DXA) × modality
- `12_4d_paired_ablation.png` — 4D ON vs OFF per modality level
- `13_modality_dual_profile.png` — Line + bar dual view, all 12 conditions

**For thesis**: All plots are 180–300 DPI PNG, white background, publication-ready.

### Experiment Management

For systematic experiment tracking and reproducibility, use the experiment management system:

**Define experiments in `experiments.yaml`:**
```yaml
experiments:
  vlm_integration:
    description: "V2 pipeline with Gemini VLM for image parsing"
    profile: v2_prompt
    conditions: [A1, B1, C1]
    cases: data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
    output_dir: logs/experiments/vlm_integration
    tags: [main, vlm-enabled]
```

**Run experiments:**
```bash
# List all available experiments
./run.sh experiment list

# Run a single experiment
./run.sh experiment vlm_integration

# Run multiple experiments and compare
./run.sh experiment baseline_v2 vlm_integration --compare vlm_impact

# Quick test (5 cases)
./run.sh experiment quick_test
```

**One-click VLM comparison (main thesis experiment):**
```bash
./run.sh vlm-compare
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

### LoRA Training (LoRA_2 — synth_v0.4)

```bash
# Train on Modal A100 (default: 3 epochs, r=16, lr=2e-4)
./training/train.sh

# Train with custom hyperparams
./training/train.sh --epochs 5 --lr 1e-4 --lora-r 32

# Download the trained adapter locally after training completes
./training/train.sh --download-only
```

Adapter is saved to `models/adapters/v2_lora_qwen/` locally and persisted on the
Modal volume at `/mscd-lora/final` (and `/mscd-lora/checkpoint-180`).

Monitor training progress:
```bash
modal app logs mscd-vlm-lora-train
# or via Wandb: project=mscd-vlm-lora, run=qwen25vl-7b-r16-synth_v04
```

---
### Full Evaluation Pipeline Flow
```bash
# Full 6-condition modality ablation (MA/MB/MC × 4D ON/OFF, LoRA + Prompt):
./training/eval.sh --step modality-6cond --adapter final

# Or step-by-step:
./training/eval.sh --step modal --adapter final  # GPU constraint extraction (Modal A100)
./training/eval.sh --step local --adapter final  # Local retrieval + scoring
./training/eval.sh --step update-plots           # Regenerate docs/plots (never overwrites)

# Paired ablation only (MA/MB/MC LoRA vs Prompt, no 4D-OFF conditions):
./training/eval.sh --step paired-ablation --adapter final

# Standalone comparison (any trace files → charts 1–9):
python script/compare_results.py \
  --traces logs/evaluations/synth_v04/traces_*_v2_lora_MA.jsonl  --label "LoRA MA" \
  --traces logs/evaluations/synth_v04/traces_*_v2_prompt_MA.jsonl --label "Prompt MA" \
  --cases ../data_curation/datasets/synth_v0.4_merged/train/test_holdout.jsonl \
  --plots --output docs/plots/lora2_vs_prompt
```
## Architecture

**Code Organization Note:**
V1 components are at `src/` root level (e.g., `main_mcp.py`, `chat_cli.py`), while V2 components are in `src/v2/` subdirectory. This asymmetry exists for historical reasons (V1 developed first, V2 added later). Shared components (`eval/`, `visual/`, `ifc_engine.py`) work for both pipelines. Both produce compatible `EvalTrace` output, enabling unified visualization and comparison.

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
│   ├── # V1 Components (at root level - historical)
│   ├── main_mcp.py              # [V1] Entry point (MCP agent)
│   ├── chat_cli.py              # [V1] Interactive chat CLI
│   ├── chat_logger.py           # [V1] Conversation logging
│   │
│   ├── # Shared Infrastructure
│   ├── pipeline_base.py         # [SHARED] Pipeline abstraction (V1Pipeline, V2Pipeline)
│   ├── ifc_engine.py            # [SHARED] Core IFC processing engine
│   │
│   ├── common/                  # [SHARED] Utilities (config, GUID extraction, etc.)
│   │   ├── config.py            # Config loading, system prompt, LLM init
│   │   ├── evaluation.py        # Context formatting, evaluation helpers
│   │   ├── guid.py              # IFC GUID extraction (regex)
│   │   ├── response_parser.py   # LangGraph response parsing
│   │   └── mcp.py               # MCP connection helper (async context manager)
│   │
│   ├── v2/                      # [V2] Constraints-driven pipeline (isolated)
│   │   ├── types.py             # Data models (Constraints, QueryPlan, V2Trace)
│   │   ├── condition_mask.py    # A1-C3 input masking
│   │   ├── constraints_extractor_prompt_only.py  # LLM-based extraction
│   │   ├── constraints_extractor_lora.py         # LoRA-based extraction (Qwen2.5-VL)
│   │   ├── constraints_to_query.py  # Template query planner
│   │   ├── retrieval_backend.py # Memory/Neo4j/CLIP retrieval
│   │   ├── pipeline.py          # V2 pipeline orchestration
│   │   └── metrics_v2.py        # V2 diagnostic metrics
│   │
│   ├── eval/                    # [SHARED] Evaluation framework (works for V1 & V2)
│   │   ├── contracts.py         # EvalTrace, ScenarioInput (shared contracts)
│   │   ├── metrics.py           # Metric functions
│   │   ├── runner.py            # V1 scenario runner
│   │   └── visualizations.py    # Plot generators (6 chart types, works for both)
│   │
│   ├── rq2_schema/              # [SHARED] RQ2 schema validation
│   │   ├── extract_final_json.py
│   │   ├── schema_registry.py
│   │   ├── mapping.py
│   │   ├── validators.py
│   │   └── pipeline.py
│   │
│   ├── visual/                  # [SHARED] Visual analysis modules
│   │   ├── image_parser.py      # VLM-based image parsing (cached descriptions)
│   │   └── aligner.py           # CLIP-based image-to-element matching
│   │
│   └── handoff/                 # [SHARED] BCF issue generation
│       ├── trace.py             # Trace builder + shared title/description helpers
│       ├── bcf_lite.py          # JSON issue output
│       └── bcf_zip.py           # BCF 2.1 zip generation
│
├── mcp_servers/
│   ├── ifc_server.py            # MCP server with IFC query tools
│   └── visual_server.py         # MCP server with visual analysis tools
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
│   └── render_worker.py         # Blender rendering subprocess
│
├── run.sh                       # Unified eval launcher (mcp | experiment | vlm-compare)
│
├── test/
│   ├── test_ifc_engine.py       # IFC engine tests
│   ├── test_bcf_generation.py   # BCF generation tests
│   ├── test_visual_aligner.py   # Visual aligner tests
│   └── rq2_schema_smoke_test.py # RQ2 schema validation tests
│
├── logs/
│   ├── evaluations/             # Ad-hoc evaluation runs (traces + summaries)
│   ├── experiments/             # Organized experiment results with metadata
│   ├── comparisons/             # Comparison plots between experiments
│   └── plots/                   # Generated visualization charts
│
├── outputs/                     # BCF artifacts, renders
│
└── legacy/                      # Archived unused/superseded code
    ├── run_mcp.sh                    # Replaced by ./run.sh mcp
    ├── run_experiment.sh             # Replaced by ./run.sh experiment
    ├── run_vlm_comparison.sh         # Replaced by ./run.sh vlm-compare
    ├── src/mcp_langchain_adapter.py  # Replaced by langchain-mcp-adapters package
    ├── test/03_clip_test.py          # Ad-hoc CLIP experiment
    ├── test/ifc_test.py              # Superseded by src/ifc_engine.py
    ├── script/ifc_to_neo4j.py        # One-off Neo4j setup utility
    └── docs/log_20260211.md          # Dated log
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

All pipelines (V1 and V2) read from the same JSONL case format.

| Dataset | Cases | File | Used by |
|---------|-------|------|---------|
| **gt_1** (hand-written) | 6 | `data/ground_truth/gt_1/gt_1.json` | V1 default |
| **synth_v0.2** (synthetic) | 43 | `../data_curation/datasets/synth_v0.2/cases_v2.jsonl` | Legacy |
| **synth_v0.3** (synthetic) | 84 | `../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl` | V1 + V2 prompt baseline |
| **synth_v0.4** (multi-model) | 361 | `../data_curation/datasets/synth_v0.4_*/cases_v3_filtered.jsonl` | **LoRA_2 training + eval** |

### synth_v0.3 (Primary Evaluation Dataset)

84 cases organized into three tiers mapped to research questions:

| Tier | Focus | Cases | RQ | Image Mode | Text Style |
|------|-------|-------|----|------------|------------|
| **T1** (Visual Texture) | Grounding from defect images | ~35% | RQ1 | defect | deictic |
| **T2** (Spatial/4D) | Alignment via floorplan + 4D metadata | ~35% | RQ2 | defect | relative |
| **T3** (Conflict/Negative) | Governance — mismatch or pristine | ~30% | RQ3 | mismatch/pristine | misleading |

Cases are assigned to benchmark groups for controlled experiments:

|   |                  | T1 (Visual Texture) | T2 (Spatial/4D) | T3 (Conflict/Negative) |
|---|------------------|---------------------|-----------------|------------------------|
| A | (text-only)      | A1 = 12             | A2 = 11         | A3 = 8                 |
| B | (img+text)       | B1 = 14             | B2 = 8          | B3 = 6                 |
| C | (full multimodal)| C1 = 10             | C2 = 5          | C3 = 10                |

Each case includes: photoreal site photos (Gemini-generated from IFC wireframes), floorplan patches (matplotlib from IFC geometry), and structured chat context.

**v0.3 case schema (`cases_v3_filtered.jsonl`):**

```json
{
  "case_id": "SYNTH_V3_001_SK_001",
  "query_text": "Check this out.",
  "bench": {"group": "A", "condition": "A1"},
  "difficulty_tags": {
    "tier": "T1",
    "tier_name": "Visual Texture",
    "candidate_density_k": 5,
    "requires_relation": false,
    "conflict_injected": false,
    "image_mode": "defect",
    "text_style": "deictic"
  },
  "inputs": {
    "chat_history": [{"role": "Site Supervisor", "text": "..."}],
    "chat_quality": "clear",
    "images": ["datasets/synth_v0.3/cases/imgs/img_SYNTH_V3_001_SK_001.png"],
    "floorplan_patch": "datasets/synth_v0.3/cases/plans/plan_SYNTH_V3_001_SK_001.png",
    "project_context": {"timestamp": "...", "sender_role": "...", "project_phase": "...", "4d_task_status": "..."}
  },
  "ground_truth": {
    "target_guid": "3GzoWuxxn4WO8bCtw8H3Vj",
    "target_storey": "1 - First Floor",
    "target_ifc_class": "IfcWall",
    "target_name": "Basic Wall:MockUp Interior...",
    "rq_category": "RQ1",
    "expected_output": "defect_found"
  },
  "labels": {
    "constraints": {"storey_name": "1 - First Floor", "ifc_class": "IfcWall", "near_keywords": [], "relations": []}
  }
}
```

### V2 Baseline Results (synth_v0.3)

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 3.57% |
| Top-K Accuracy | 5.95% |
| Search Space Reduction (SSR) | 92.65% |
| Field EM F1 | 21.35% |
| Constraints Parse Rate | 100% |

The low Top-1 reflects the challenge: many cases use vague/deictic text (e.g., "Look at this.") by design, forcing the model to rely on visual grounding rather than keyword matching. This is the baseline the LoRA adapter aims to improve.

### LoRA_2 Evaluation Results (synth_v0.4, 50 holdout cases)

6-condition modality ablation × 2 profiles (300 traces each):

| Profile | Conditions | Top-1 Accuracy |
|---------|------------|---------------|
| **V2 LoRA** (LoRA_2 adapter) | MA/MB/MC/MA-/MB-/MC- (all 6) | **35.3%** |
| **V2 Prompt** (Gemini baseline) | MA/MB/MC/MA-/MB-/MC- (all 6) | 25.7% |

LoRA_2 outperforms the Gemini prompt baseline by **+9.6 pp** on the same 50-case holdout.
Charts: [docs/plots/0224_modality_6cond_v3/](docs/plots/0224_modality_6cond_v3/)

### synth_v0.4 (LoRA_2 Training Dataset)

synth_v0.4 expands the dataset to three IFC building models, giving the LoRA adapter
exposure to different buildings, element vocabularies, and storey naming conventions.

**Three IFC models:**

| Tag | Building | Cases | Holdout | Train cases |
|-----|----------|-------|---------|-------------|
| **AP** | AdvancedProject (10-storey office) | 250 | 20 | 230 |
| **BH** | BasicHouse (2-storey residential) | 31 | 20 | 11 |
| **DXA** | Duplex_A (split-level duplex) | 80 | 10 | 70 |
| | **Total** | **361** | **50** | **311** |

After 3x text augmentation: **933 train samples + 50 test samples.**

Images and floorplan patches are in:
- `../data_curation/datasets/synth_v0.4_ap/cases/imgs/` and `plans/`
- `../data_curation/datasets/synth_v0.4_bh/cases/imgs/` and `plans/`
- `../data_curation/datasets/synth_v0.4_dxa/cases/imgs/` and `plans/`

The 50-case eval holdout lives in:
- `../data_curation/datasets/synth_v0.4_merged/train/test_holdout.jsonl`

### Phase 5: Fine-Grained Constraint Rules

Phase 5 adds three optional constraint fields to help identify elements that can't be
pinpointed by storey + IFC class alone:

| Field | What it captures | Example |
|-------|-----------------|---------|
| `space_name` | Room or space the element is in | `"Kitchen"`, `"Room 601"` |
| `target_name_keyword` | Unique equipment ID or code name | `"AHU-03"`, `"FD-101"` |
| `neighbor_type` | IFC class of an adjacent reference element | `"IfcColumn"`, `"IfcWall"` |

These are set in `labels.constraints` of each case and included in the LoRA training target.
The LoRA model learns to extract them from chat text (e.g., "next to the column" → `neighbor_type: "IfcColumn"`).
Use `null` when the field doesn't apply — the system is conservative by default.

The `Constraints` schema is defined in [`src/v2/types.py`](src/v2/types.py) and the system prompt
in [`training/train.py`](training/train.py) (also mirrored in `prompts/constraints_extraction.yaml`).

**Updated case schema (`test_holdout.jsonl`):**

```json
{
  "case_id": "SYNTH_V3_084_AP_SK_084",
  "labels": {
    "constraints": {
      "storey_name": "Level 1",
      "ifc_class": "IfcDoor",
      "near_keywords": [],
      "relations": [],
      "space_name": null,
      "target_name_keyword": null,
      "neighbor_type": "IfcWall"
    }
  }
}
```

### LoRA_2 Training Data Pipeline

```
synth_v0.4_{ap,bh,dxa}/cases_v3_filtered.jsonl  (361 unique cases across 3 buildings)
    |
    v  data_curation/scripts/synth/6_augment_text.py
    |   (stratified split: hold out 50, augment rest 3x with original/vague/urgent text)
    |
    ├── synth_v0.4_ap/train/augmented.jsonl    (690 AP train samples)
    ├── synth_v0.4_bh/train/augmented.jsonl    (33 BH train samples)
    ├── synth_v0.4_dxa/train/augmented.jsonl   (210 DXA train samples)
    ├── synth_v0.4_*/train/test_holdout.jsonl  (50 holdout: AP=20, BH=20, DXA=10)
    |
    v  data_curation/scripts/synth/7_prepare_lora_data.py
    |   (format each dataset into Qwen2.5-VL ChatML, merge all three)
    |
    ├── synth_v0.4_merged/train/lora_train.jsonl   (933 ChatML samples: AP+BH+DXA)
    └── synth_v0.4_merged/train/lora_test.jsonl    (50 ChatML test samples)
```

**Text augmentation styles** (same images + ground truth, different text):
- **Style A (Original)**: Preserved as-is from the case
- **Style B (Vague/Deictic)**: "Look at this.", "What is wrong here?" — forces image reliance
- **Style C (Urgent/Site Jargon)**: "QA flagged this.", "Need verification ASAP." — simulates real site language

**Regenerate training data (if cases change):**
```bash
cd /root/cmu/master_thesis/data_curation

# Augment each building (run once per dataset)
for tag in ap bh dxa; do
  python scripts/synth/6_augment_text.py \
    --cases datasets/synth_v0.4_${tag}/cases_v3_filtered.jsonl \
    --output datasets/synth_v0.4_${tag}/train/augmented.jsonl \
    --hold-out 20 --seed 42
done

# Format and merge into ChatML for Qwen2.5-VL
python scripts/synth/7_prepare_lora_data.py \
  --train datasets/synth_v0.4_ap/train/augmented.jsonl \
          datasets/synth_v0.4_bh/train/augmented.jsonl \
          datasets/synth_v0.4_dxa/train/augmented.jsonl \
  --test  datasets/synth_v0.4_merged/train/test_holdout.jsonl \
  --output datasets/synth_v0.4_merged/train/ \
  --image-root /root/cmu/master_thesis/data_curation
```

**LoRA_2 hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |
| Adapter | LoRA (r=16, alpha=32) |
| Training samples | 933 (AP=690, BH=33, DXA=210) |
| Test samples | 50 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Effective batch size | 16 (batch=2, grad_accum=8) |
| Max seq length | 2048 |
| Hardware | Modal A100 (40GB) |
| Task | Multimodal constraint extraction (site photo + floorplan + chat → JSON) |

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

# Export IFC model to Neo4j (utility in legacy/)
python legacy/script/ifc_to_neo4j.py

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
| Constraints LoRA | Qwen2.5-VL-7B-Instruct + LoRA (Unsloth) |
| Visual Matching | OpenAI CLIP |
| Data Models | Pydantic v2 |
| MCP Server | FastMCP |
| Schema Validation | jsonschema (Draft 2020-12) |
| Graph Database | Neo4j (optional) |
| BCF Generation | stdlib zipfile + xml.etree (BCF 2.1) |
| 3D Rendering | Blender + Bonsai addon (headless) |
| Photoreal Images | Google Gemini (from IFC wireframes) |

---

**Last Updated:** February 2026
**Status:** V1 + V2 pipelines operational. synth_v0.3 (84 cases): V2 prompt baseline Top-1=3.57%, SSR=92.65%. synth_v0.4 (361 cases, 3 IFC models): LoRA_2 adapter trained (Qwen2.5-VL r=16, 933 samples, 3 epochs) — adapters at `/mscd-lora/final` on Modal and `models/adapters/v2_lora_qwen/` locally. **LoRA_2 vs Prompt eval (50 holdout, 6-condition modality ablation): LoRA 35.3% vs Prompt 25.7% Top-1.** Phase 5 fine-grained constraints (space_name, target_name_keyword, neighbor_type) added. Unused code archived to `legacy/`. Plot generation: `eval.sh --step update-plots` → `docs/plots/<MMDD>_modality_6cond_*/`.
