# MSCD Demo — Technical Introduction

**Multi-modal Site Condition Detection for BIM Element Identification**

> A compound AI system for agentic interpretation of AEC site inspection data,
> bridging unstructured field observations with structured Building Information Models.

---

## 1. Problem Statement

In Architecture, Engineering, and Construction (AEC), site supervisors generate continuous observation streams — photos, voice notes, chat messages — that must eventually link to a specific physical element in the Building Information Model (BIM). The canonical workflow today is manual: a supervisor opens a BIM viewer, scrolls through a 10-storey model, and hand-selects the element they're reporting on.

The computational formulation is a **cross-modal retrieval problem**: given a noisy, multimodal query `q = (text, images, 4D metadata)`, retrieve the correct IFC element `e*` from a candidate pool `E` of size `|E|` up to several thousand.

**Why is this hard?**

A query as simple as `"Check this window"` matches **263 candidates** in a 10-storey residential building, yielding a naive precision of **0.38%**. Adding any single modality of context can collapse the search space dramatically:

| Signal | Pool After Filter | Reduction |
|---|---|---|
| No context (baseline) | 263 | 0% |
| Storey from 4D task status | ~12 | 95.4% |
| Storey + element type | ~3 | 98.9% |
| Storey + type + visual reranking | 1–3 | 99.2%+ |

This project implements and ablates two complete retrieval pipelines to study the contribution of each modality and to compare **agentic** vs **deterministic** architectures for this task.

---

## 2. System Architecture

### 2.1 High-Level Overview

The system is structured as a multi-stage compound AI pipeline with two parallel runtime paths (V1 and V2) that share a common input layer, IFC backend, evaluation framework, and output schema.

```
┌──────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                              │
│  chat_history · site_photos · floorplan_patch · 4D task status   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Condition Mask  │
                    │   (A1 → C3)     │
                    └───────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼──────────┐    ┌──────────▼──────────────┐
    │    V1: Agent Path  │    │    V2: Constraints Path  │
    │  (LangGraph ReAct) │    │  (Deterministic Planner) │
    └─────────┬──────────┘    └──────────┬───────────────┘
              │                          │
              │           ┌──────────────▼──────────────┐
              │           │    Image Parser (VLM)        │
              │           │  Gemini 2.5 Flash multimodal │
              │           │  (always runs; raw→ParsedImage)│
              │           └──────┬───────────────────────┘
              │                  │ image_parse_result (skipped entirely if LoRA)
              │           ┌──────┴──────────────────────────────────┐
              │           │          Constraints Extractor           │
              │           │                                          │
              │           │  ┌─────────────────┐  ┌──────────────┐  │
              │           │  │   Prompt-only   │  │  LoRA(Qwen)  │  │
              │           │  │  (Gemini Flash) │  │  VL-7B-Inst  │  │
              │           │  │                 │  │              │  │
              │           │  │ uses ParsedImage│  │ IGNORES      │  │
              │           │  │ as text context │  │ ParsedImage; │  │
              │           │  │ + fallback hints│  │ reads raw    │  │
              │           │  │                 │  │ images direct│  │
              │           │  └─────────────────┘  └──────────────┘  │
              │           └──────────────┬───────────────────────────┘
              │                          │
              │           ┌──────────────▼──────────────┐
              │           │      Query Planner           │
              │           │   (Priority Rule Templates)  │
              │           └──────────────┬───────────────┘
              │                          │
    ┌─────────▼──────────────────────────▼───────────────┐
    │                  Retrieval Backend                   │
    │     ┌─────────────┐       ┌──────────────────┐      │
    │     │  In-Memory  │       │  Neo4j (optional) │      │
    │     │  Spatial    │       │  IFC Graph DB     │      │
    │     │  Index      │       │  (Cypher Queries) │      │
    │     └─────────────┘       └──────────────────┘      │
    │                   ┌──────────┐                       │
    │                   │  CLIP    │  (optional reranking) │
    │                   │ Reranker │                       │
    │                   └──────────┘                       │
    └─────────────────────────┬───────────────────────────┘
                              │
              ┌───────────────▼──────────────┐
              │     IFC Engine (IfcOpenShell) │
              │   Spatial Index · GUID Lookup │
              └───────────────┬──────────────┘
                              │
    ┌─────────────────────────▼────────────────────────────┐
    │                   Output Layer                        │
    │   EvalTrace · RQ2 Schema Validation · BCF 2.1 ZIP    │
    └──────────────────────────────────────────────────────┘
```

### 2.2 Modality Ablation Grid (Condition Mask A1–C3)

All experiments run under a controlled **condition mask** that simulates degraded real-world inputs. This isolates the contribution of each modality:

| Condition | Chat Quality | Site Photos | Floorplan | 4D Metadata | Purpose |
|---|---|---|---|---|---|
| **A1** | clear | — | — | basic | Text + 4D baseline |
| **A2** | blurred | — | — | basic | OCR robustness |
| **A3** | blurred | — | — | enhanced | Best 4D context |
| **B1** | blurred | ✓ | — | — | Pure visual grounding |
| **B2** | blurred | ✓ | — | — | Visual + CLIP reranking |
| **B3** | clear | ✓ | — | — | Best vision, no 4D |
| **C1** | clear | — | ✓ | — | Spatial layout only |
| **C2** | blurred | ✓ | ✓ | — | Vision + spatial |
| **C3** | clear | ✓ | ✓ | enhanced | Full multimodal (upper bound) |

---

## 3. Pipeline Designs

### 3.1 V1: Agent-Driven Pipeline

**Philosophy**: Treat the problem as a tool-use task. An LLM reasons freely, then calls structured APIs to navigate the IFC model.

**Entry point**: `src/main_mcp.py`

**Runtime stack**:
```
Gemini 2.5 Flash
  └── LangGraph ReAct Agent
        └── MCP Dispatcher (langchain-mcp-adapters)
              ├── IFC MCP Server  (mcp_servers/ifc_server.py)
              │     ├── list_available_spaces()
              │     ├── get_elements_by_storey()
              │     ├── search_elements_by_type()
              │     ├── get_element_details()
              │     └── generate_3d_view()
              └── Visual MCP Server (mcp_servers/visual_server.py)
                    ├── analyze_site_image()
                    ├── match_image_to_elements()
                    └── compare_defect_images()
```

**Control flow**:
1. **System prompt injection**: Instructions enforce a four-phase decision procedure — (1) extract storey from 4D task, (2) keyword extraction from chat, (3) tool calls to narrow candidates, (4) output `FINAL_JSON={selected_guid: ..., ...}`.
2. **Tool calls**: MCP dispatcher routes each tool call to the appropriate server. Tools are stateless HTTP services, decoupled from the agent.
3. **Response parsing**: `src/common/response_parser.py` uses regex to extract the `FINAL_JSON` block from the LLM's free-text output.

**Characteristics**: Non-deterministic, high flexibility, opaque reasoning. Useful as a human-level baseline but difficult to ablate systematically.

---

### 3.2 V2: Constraints-Driven Pipeline

**Philosophy**: Decompose the retrieval problem into explicit stages, each independently testable. Replace LLM reasoning with deterministic query planning.

**Entry point**: `script/run.py --profile v2_*`

**Stage 1 — Image Parser** (`src/visual/image_parser.py`)

Converts raw images into structured semantic records using Gemini 2.5 Flash in multimodal mode. Each image produces a `ParsedImage`:

```python
@dataclass
class ParsedImage:
    element_type: Optional[str]      # "window", "slab", "wall"
    ifc_class_hint: Optional[str]    # "IfcWindow"
    material: Optional[str]          # "concrete", "glass"
    defect_type: Optional[str]       # "crack", "water_damage"
    defect_severity: Optional[str]   # "minor" | "moderate" | "severe"
    location_cues: List[str]         # ["north facade", "near column"]
    spatial_zone: Optional[str]      # (floorplan only) "room 602"
    storey_hint: Optional[str]       # "6 - Sixth Floor"
    description: str
    confidence: float
    parse_latency_ms: float
```

Results are cached in-memory (keyed on image path hash) to avoid redundant VLM calls across repeated runs.

**Stage 2 — Constraints Extractor**

Fuses parsed images, chat text, and 4D metadata into a typed constraint record:

```python
@dataclass
class Constraints:
    storey_name: Optional[str]          # "6 - Sixth Floor" (exact BIM name)
    ifc_class: Optional[str]            # "IfcWindow" (canonical IFC type)
    near_keywords: List[str]            # ["north", "elevator shaft"]
    space_name: Optional[str]           # "Living Room" (room-level, not floor)
    target_name_keyword: Optional[str]  # "Daikin", "AHU-03" (brand/ID)
    neighbor_type: Optional[str]        # "IfcColumn" (topological reference)
    confidence: float
    source: str                         # "prompt" | "lora" | "prompt_failed"
```

Two extraction backends are compared, with a critical difference in how each consumes the Image Parser output:

| Backend | Model | Uses Image Parser output? | Latency | Training Cost |
|---|---|---|---|---|
| **Prompt-only** | Gemini 2.5 Flash | **Yes** — injected as text context + constraint fallback | 1–2 s | None |
| **LoRA** | Qwen2.5-VL-7B-Instruct | **Skipped** — Image Parser does not run; Qwen reads raw images directly via its own vision encoder | 5–10 s | ~192 samples |

**Image Parser output routing** (`pipeline.py:137`, `constraints_extractor_lora.py:154`):

- **Prompt-only path**: Image Parser runs. `image_parse_result.combined_description` is injected into the LLM prompt under a `"VISUAL ANALYSIS (from vision model):"` section. If the LLM JSON parse then fails, `inferred_ifc_class`, `inferred_storey`, and `floorplan.spatial_zone` are used as constraint fallbacks.
- **LoRA path**: Image Parser is **skipped entirely** (`constraints_model == "lora"` guard in `pipeline.py`). Qwen's own vision encoder processes the raw `file://` image paths directly — running Gemini to produce `ParsedImage` first would be a redundant VLM call with no downstream effect.

The prompt-only extractor fails on vague/deictic text ("Right here.", "Check this.") because even with image descriptions converted to text, the LLM cannot reliably map a description like "cracked surface" to a canonical IFC class without training signal. The LoRA extractor learns this mapping end-to-end from raw images.

**Stage 3 — Query Planner** (`src/v2/constraints_to_query.py`)

Translates a `Constraints` record into a `QueryPlan` via a **priority-ordered rule table** — no LLM generation, no randomness:

```
Priority 0: space_name + ifc_class         → ~5 candidates   (most specific)
Priority 1: target_name_keyword            → ~1–3 candidates
Priority 2: neighbor_type + ifc_class      → ~3–8 candidates  (Neo4j only)
Priority 3: storey_name + ifc_class        → ~50 candidates
Priority 4: storey_name only               → ~200 candidates
Priority 5: ifc_class only                 → ~150 candidates
Fallback:   return first 100 elements
```

The planner selects the highest-priority rule whose required fields are non-null in the constraints record.

**Stage 4 — Retrieval Backend** (`src/v2/retrieval_backend.py`)

Executes the query plan against one of two storage backends:

- **Memory**: In-memory `spatial_index` built from `IfcRelContainedInSpatialStructure` at startup. Supports storey-filtered and type-filtered lookups. Latency < 10 ms. Zero setup.
- **Neo4j**: Full IFC topological graph with Cypher query support. Enables neighbor-type queries (`HAS_OPENING`, `FILLS`, `ADJACENT_TO` relations). Latency 50–200 ms. Requires Docker.

Optional **CLIP reranking** (`src/visual/aligner.py`): after retrieval, each candidate element is encoded as a text description, embedded with `openai/clip-vit-base-patch32`, and sorted by cosine similarity to the query image embeddings. Implemented as a singleton (CLIP model loaded once per process).

---

## 4. Data Model

### 4.1 Input Case Schema

```json
{
  "case_id": "SYNTH_V3_001_SK_001",
  "bench": { "group": "A", "condition": "A1" },
  "difficulty_tags": {
    "tier": "T1",
    "candidate_density_k": 5,
    "requires_relation": false,
    "conflict_injected": false,
    "image_mode": "defect",
    "text_style": "deictic"
  },
  "inputs": {
    "chat_history": [{ "role": "Site Supervisor", "text": "..." }],
    "chat_quality": "clear",
    "images": ["datasets/synth_v0.3/cases/imgs/img_001.png"],
    "floorplan_patch": "datasets/synth_v0.3/cases/plans/plan_001.png",
    "project_context": {
      "timestamp": "...",
      "sender_role": "Site Supervisor",
      "project_phase": "Construction",
      "4d_task_status": "Window Installation - 1 - First Floor - IN_PROGRESS"
    }
  },
  "ground_truth": {
    "target_guid": "3GzoWuxxn4WO8bCtw8H3Vj",
    "target_storey": "1 - First Floor",
    "target_ifc_class": "IfcWall",
    "rq_category": "RQ1",
    "expected_output": "defect_found"
  },
  "labels": {
    "constraints": {
      "storey_name": "1 - First Floor",
      "ifc_class": "IfcWall",
      "near_keywords": []
    }
  }
}
```

### 4.2 Evaluation Trace (`EvalTrace`)

Each pipeline run produces a structured trace for offline analysis:

```python
@dataclass
class EvalTrace:
    scenario_id: str
    profile: str
    condition: str

    # Retrieval outcome
    guid_match: bool              # ground-truth GUID in final candidates?
    final_pool_size: int          # candidates returned
    initial_pool_size: int        # full BIM element count (baseline)

    # Diagnostic fields
    constraints: Optional[Constraints]
    query_plan: Optional[QueryPlan]
    candidates: List[CandidateElement]
    ground_truth: GroundTruth

    # Performance
    total_latency_ms: float
    image_parse_latency_ms: float
    retrieval_latency_ms: float

    # RQ2
    rq2_validation: Optional[ValidationResult]

    error: Optional[str]
```

---

## 5. Dataset

### 5.1 BIM Model

Primary model: `AdvancedProject.ifc` — a 10-storey residential building with:
- ~2,000 total elements
- 263 `IfcWindow` instances (the dominant ambiguity class)
- Full spatial hierarchy: `IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey → elements`
- Named storeys: `"1 - First Floor"` through `"10 - Tenth Floor"` + `"00 - Ground Floor"`

### 5.2 Evaluation Datasets

| Dataset | Cases | Status | Notes |
|---|---|---|---|
| **gt_1** | 6 | Active | Hand-written ground truth, high quality |
| **synth_v0.3** | 84 | Primary | Thesis main evaluation, 3 difficulty tiers |
| **synth_v0.4_merged** | 361 unique / 933 train samples | Extended | 3 building models (AP+BH+DXA), LoRA_2 training + eval |

### 5.3 Difficulty Tiers

| Tier | Focus | RQ | Image Mode | Text Style | % of v0.3 |
|---|---|---|---|---|---|
| **T1** (Visual Texture) | Visual defect grounding | RQ1 | defect | deictic | ~35% |
| **T2** (Spatial / 4D) | Floorplan + 4D alignment | RQ2 | defect | relative | ~35% |
| **T3** (Conflict / Negative) | Governance, mismatch detection | RQ3 | mismatch/pristine | misleading | ~30% |

T1 cases deliberately use deictic text ("Right here.") with no element keywords, forcing the system to rely entirely on visual evidence — the hardest condition for the prompt-only extractor.

---

## 6. LoRA Fine-Tuning Pipeline

### 6.1 Training Data Construction

LoRA_2 trains on **synth_v0.4** — three IFC building models providing richer element vocabulary and storey naming diversity than the single-building synth_v0.3.

```
synth_v0.4_{ap,bh,dxa}/cases_v3_filtered.jsonl  (361 unique cases across 3 buildings)
    |
    v  6_augment_text.py  (stratified holdout + 3× text augmentation)
    |
    ├── synth_v0.4_ap/train/augmented.jsonl    (690 AP train samples)
    ├── synth_v0.4_bh/train/augmented.jsonl    ( 33 BH train samples)
    ├── synth_v0.4_dxa/train/augmented.jsonl   (210 DXA train samples)
    └── test_holdout.jsonl                      ( 50 cases: AP=20, BH=20, DXA=10)
    |
    v  7_prepare_lora_data.py  (format into Qwen2.5-VL ChatML, merge all three)
    |
    ├── synth_v0.4_merged/train/lora_train.jsonl  (933 samples total)
    │     ├── 444 MC samples — plan + Gemini site photo (2 images)
    │     └── 489 MA samples — plan only (1 image)
    └── synth_v0.4_merged/train/lora_test.jsonl   (50 test samples)
```

**Text augmentation styles** (same images + ground truth, different chat text):
- **Style A (Original)**: Preserved as-is from the generated case
- **Style B (Vague/Deictic)**: "Look at this.", "What is wrong here?" — forces image reliance
- **Style C (Urgent/Site Jargon)**: "QA flagged this.", "Need verification ASAP." — simulates real site language

**Building models:**

| Tag | Building | Unique cases | Holdout | Train samples (3×) |
|-----|----------|-------------|---------|---------------------|
| AP  | AdvancedProject (10-storey office) | 250 | 20 | 690 |
| BH  | BasicHouse (2-storey residential)  |  31 | 20 |  33 |
| DXA | Duplex_A (split-level duplex)      |  80 | 10 | 210 |
| | **Total** | **361** | **50** | **933** |

### 6.2 Model & Training Config

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |
| Fine-tuning method | LoRA (via Unsloth) |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| Dropout | 0.0 |
| Epochs | 3 (best checkpoint by eval loss) |
| Effective batch size | 16 (2 × 8 gradient accumulation) |
| Training samples | 933 (AP=690, BH=33, DXA=210) |
| Test samples | 50 |
| Hardware | Modal A100 (40 GB) |
| Training platform | Modal (cloud GPU) |
| Input format | ChatML with `file://` image paths (Qwen VL utils resolves from disk) |
| Output | Constraints JSON (storey, ifc_class, space_name, target_name_keyword, neighbor_type, near_keywords) |

### 6.3 Inference Contract

```python
# Input (T1/MC case — site photo + floorplan patch)
messages = [
    {"role": "system", "content": CONSTRAINTS_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": "file:///data/images/ap/imgs/img_001.png"},   # Gemini site photo
        {"type": "image", "image": "file:///data/images/ap/plans/plan_001.png"}, # floorplan patch
        {"type": "text", "text": chat_text + "\n" + context_text}
    ]}
]

# Output (parsed from model response)
{
  "storey_name": "6 - Sixth Floor",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "neighbor_type": null,
  "near_keywords": ["north"],
  "confidence": 0.85
}
```

---

## 7. Tool Infrastructure (MCP Servers)

The V1 pipeline exposes IFC and visual operations as **Model Context Protocol (MCP)** servers. MCP decouples tool implementation from the agent framework, enabling:
- Independent deployment and testing of each tool server
- Compatibility with any MCP-aware LLM client (Claude Desktop, VS Code, custom)
- Parallel development by domain experts without LLM knowledge

```
mcp_servers/
├── ifc_server.py      # IfcOpenShell queries as MCP tools
└── visual_server.py   # CLIP + Gemini VLM as MCP tools
```

Tools are invoked via `langchain-mcp-adapters`, which translates MCP tool schemas into LangChain-compatible tool objects for the ReAct graph.

---

## 8. Schema Validation (RQ2)

Post-retrieval, every identified element is validated against **CORENET-X minimal schema** — Singapore's regulatory BIM submission format. This tests whether AI-extracted information is structurally compatible with downstream regulatory workflows.

```
EvalTrace (GUID + element properties)
    ↓
schema_registry.py     → Load JSON Schema (Draft 2020-12)
mapping.py             → element-type-specific field extraction
validators.py          → jsonschema.validate()
    ↓
ValidationResult {
    valid: bool,
    errors: List[str],
    record: dict       # the generated submission record
}
```

Schema: `schemas/corenetx_min/v0.schema.json`

---

## 9. Output Artifacts

### 9.1 BCF 2.1 Handoff

Identified elements are packaged as **BIM Collaboration Format (BCF) 2.1** files — the ISO-standard issue tracking format readable by Revit, Navisworks, and all major BIM authoring tools. This makes the interpreter layer deployable in existing AEC workflows without toolchain changes.

```
outputs/bcf/
├── issue_001.json        # Machine-readable issue (intermediate)
└── issues.bcfzip         # BCF 2.1 ZIP container (opens in Revit)
```

### 9.2 Evaluation Outputs

```
logs/
├── evaluations/
│   ├── traces_<timestamp>_<profile>.jsonl   # Per-case EvalTrace records
│   └── summary_<timestamp>_<profile>.csv    # Aggregated metrics
└── plots/<timestamp>_<profile>/
    ├── top1_accuracy.png                     # Accuracy by condition (A1–C3)
    ├── search_space_reduction.png            # SSR funnel chart
    ├── constraints_parse_rate.png            # V2 extraction diagnostic
    ├── image_parse_timing.png                # VLM overhead breakdown
    ├── vision_impact.png                     # Before/after CLIP reranking
    └── per_case_heatmap.png                  # condition × case success matrix
```

All plots: 300 DPI PNG, LaTeX-ready.

---

## 10. Evaluation Metrics

| Metric | Definition | Primary Scope |
|---|---|---|
| **Top-1 Accuracy** | Ground-truth GUID is the top-ranked candidate | V1 + V2 |
| **Top-K Accuracy** | Ground-truth GUID appears in top-K candidates (K=5) | V1 + V2 |
| **Search Space Reduction (SSR)** | `(initial_pool - final_pool) / initial_pool` | V2 |
| **Constraints Field EM-F1** | Token-level F1 between predicted and label constraint fields | V2 |
| **Constraints Parse Rate** | Fraction of cases producing valid JSON | V2 |
| **Escalation Rate** | Fraction of cases where system declines to select (RQ3) | V1 |
| **Avg Latency** | End-to-end wall-clock time per case | V1 + V2 |

---

## 11. Configuration & Experiment Management

### 11.1 Profile System (`profiles.yaml`)

A profile fully specifies one experimental configuration:

```yaml
v2_lora_neo4j_clip:
  pipeline: v2
  constraints_model: lora       # or "prompt"
  retrieval: neo4j              # or "memory"
  use_clip: true
  thin_agent: true              # optional LLM post-filter
  rq2_schema: true
  bcf: true
```

Running `python script/run.py --profile v2_lora_neo4j_clip --cases $CASES` is fully reproducible: same profile + same cases → same trace output (modulo LLM stochasticity in prompt-mode).

### 11.2 Experiment Tracking (`experiments.yaml`)

Each named experiment records:
- Profile used
- Dataset path and condition filter
- Git commit hash + uncommitted diff
- Conda environment snapshot
- Start/end timestamps

This supports exact reproduction of any result in `RESULTS.md`.

---

## 12. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| BIM parsing | IfcOpenShell 0.7+ | IFC model loading, spatial graph traversal |
| LLM (agent + extraction) | Google Gemini 2.5 Flash | Reasoning, multimodal understanding, constraints extraction |
| VLM fine-tuning | Qwen2.5-VL-7B-Instruct + Unsloth LoRA | Visual-aware constraints extraction |
| Visual matching | OpenAI CLIP `vit-base-patch32` | Semantic image-to-text cosine reranking |
| Agent framework | LangChain + LangGraph | ReAct agent loop, tool dispatch |
| Tool protocol | FastMCP 0.2+ | Standardized MCP server/client |
| Data validation | Pydantic v2 | Runtime type safety across all data contracts |
| Schema validation | jsonschema (Draft 2020-12) | CORENET-X regulatory compliance checks |
| Graph database | Neo4j (optional) | Topological IFC queries via Cypher |
| BCF packaging | Python stdlib (`zipfile`, `xml.etree`) | BCF 2.1 artifact generation |
| GPU training | Modal (cloud) | Distributed LoRA training |
| Visualization | Matplotlib + Seaborn | Publication-ready evaluation plots |
| 3D rendering | Blender + Bonsai | Headless BIM scene rendering |

---

## 13. Repository Layout

```
mscd_demo/
├── src/
│   ├── main_mcp.py                          # V1 agent entry point
│   ├── ifc_engine.py                        # Shared IFC gateway (spatial index, lookups)
│   ├── v2/
│   │   ├── types.py                         # Constraints, QueryPlan, V2Trace
│   │   ├── condition_mask.py                # A1–C3 input masking
│   │   ├── constraints_extractor_prompt_only.py
│   │   ├── constraints_extractor_lora.py
│   │   ├── constraints_to_query.py          # Deterministic rule-based planner
│   │   ├── retrieval_backend.py             # Memory / Neo4j / CLIP execution
│   │   └── pipeline.py                      # V2 orchestration
│   ├── eval/
│   │   ├── contracts.py                     # EvalTrace, GroundTruth, CandidateElement
│   │   ├── metrics.py                       # Top-1, SSR, F1 aggregation
│   │   ├── runner.py                        # V1 scenario execution loop
│   │   └── visualizations.py               # 6 plot types
│   ├── visual/
│   │   ├── image_parser.py                  # VLM image → ParsedImage
│   │   └── aligner.py                       # CLIP singleton, cosine reranking
│   ├── rq2_schema/
│   │   ├── schema_registry.py
│   │   ├── validators.py
│   │   └── pipeline.py
│   └── handoff/
│       ├── bcf_lite.py
│       └── bcf_zip.py
├── mcp_servers/
│   ├── ifc_server.py                        # MCP server: IFC tools
│   └── visual_server.py                     # MCP server: visual tools
├── training/
│   ├── train.py                             # Modal LoRA training script
│   └── eval.py                              # Post-training evaluation
├── script/
│   ├── run.py                               # Unified evaluation runner
│   ├── generate_plots.py
│   └── compare_results.py
├── prompts/                                 # Centralized YAML prompt templates
├── schemas/corenetx_min/v0.schema.json     # Regulatory compliance schema
├── data/
│   ├── ifc/AdvancedProject/                 # Primary BIM model
│   └── ground_truth/gt_1/                  # Hand-written test cases
├── config.yaml                              # IFC path, LLM config, schema path
├── profiles.yaml                            # Experiment profiles
├── experiments.yaml                         # Reproducible experiment definitions
└── requirements.txt
```

---

## 14. Research Questions

| RQ | Question | Operationalization |
|---|---|---|
| **RQ1** | Can cross-modal signals (images + 4D metadata) disambiguate a text query from hundreds of BIM candidates? | Top-1/Top-K accuracy across conditions A1–C3 |
| **RQ2** | Can AI-extracted constraints be mapped to regulatory submission schemas (CORENET-X) without human reformatting? | JSON schema validation pass rate + field coverage |
| **RQ3** | When should the system escalate (refuse to select) vs. auto-select? | Escalation rate on T3 (conflict/negative) cases |

---

## 15. Key Design Decisions

**V1 vs V2**: The two pipelines are intentionally complementary. V1 (agent) serves as a human-reasoning baseline — how would an expert navigate the BIM given the same tools? V2 (constraints) enables systematic ablation: swap any single stage (extractor, planner, retrieval backend) independently without re-running the full agent.

**MCP for tool integration**: Using the Model Context Protocol rather than hardcoded LangChain tools decouples tool development from agent development. Domain experts (BIM engineers) can build and test MCP servers without LLM knowledge; AI engineers integrate them without BIM knowledge.

**Deterministic query planning**: The priority-rule planner in V2 is intentionally LLM-free. Reproducibility requires that the same constraints always produce the same query — impossible with LLM-generated queries. This also makes failure analysis tractable: if Top-1 fails, the reason is always traceable to a specific rule firing.

**LoRA vs prompt-only extraction**: Prompt-only extraction has zero training cost and runs at LLM-inference speed, making it the right default for conditions with rich text (A1, A3). LoRA becomes necessary for vague/deictic conditions (B1–B3, C2) where element type cannot be inferred from text alone, and must be learned from visual patterns.

**Search Space Reduction as primary metric**: Top-1 accuracy is sensitive to exact ranking; SSR measures whether the system is useful as a **filter** even when it cannot pinpoint the exact element. A system that reduces 263 candidates to 3 enables a supervisor to hand-select in seconds rather than minutes — a practical win even when Top-1 = 0.

---

## 16. Demo UI

The demo is a **Streamlit web application** that visualises evaluation traces from any completed pipeline run. It serves as both a qualitative inspection tool (step through individual cases) and a communication artifact for the thesis.

**Launch:**
```bash
cd mscd_demo
streamlit run demo/app.py
# opens at http://localhost:8501
```

A background HTTP server starts automatically on port 8502 to serve the IFC model file and the compiled JS/WASM assets required by the 3D viewer.

---

### 16.1 Layout

```
┌──────────────────┬────────────────────────────────────────────────────┐
│   SIDEBAR        │   MAIN PANEL                                       │
│                  │                                                     │
│  Run selector    │  ┌─────────────────────────────────────────────┐  │
│  Case selector   │  │  Header: ✅/❌  case_id  ·  pipeline  ·  run │  │
│                  │  └─────────────────────────────────────────────┘  │
│  ─────────────   │                                                     │
│  GUID   ✓/✗     │  ┌──────────────┬──────────────┬─────────────┐    │
│  Name   ✓/✗     │  │ 📋 Context   │ 🔍 Pipeline  │ 🏗️ Result   │    │
│  Storey ✓/✗     │  │              │   Trace       │             │    │
│                  │  └──────────────┴──────────────┴─────────────┘    │
│  Ground Truth    │                                                     │
│  GUID / Name     │                                                     │
│  ⏱ latency       │                                                     │
└──────────────────┴────────────────────────────────────────────────────┘
```

---

### 16.2 Sidebar

| Element | Description |
|---------|-------------|
| **Run selector** | Dropdown over all trace files in `outputs/traces/`, sorted by timestamp. Each entry = one `./run.sh` or `script/run.py` invocation. |
| **Case selector** | Dropdown over all case IDs within the selected run. |
| **Evaluation metrics** | Three `st.metric` tiles: GUID match ✓/✗, Name match ✓/✗, Storey match ✓/✗ — read from the `evaluation` block of the trace. |
| **Ground truth** | GUID code block, element name, storey. |
| **Latency** | Total end-to-end wall-clock time for the case. |

---

### 16.3 Tab 1 — Query Context

Shows the raw input for the selected case:

- **Chat history** — plain-text role: message display (e.g. `Site Supervisor: Check this wall`)
- **Input images** — up to 3 columns; renders both site photos and floorplan patch if present in the trace's `image_paths` list
- **4D Context** — three columns: sender role, project phase, 4D task status (e.g. `Window Installation - Level 6 - IN_PROGRESS`)

---

### 16.4 Tab 2 — Pipeline Trace

Varies by pipeline type.

**V2 pipeline (three bordered sections):**

*Constraints Extraction*
```
Spatial                    Semantic
─────────────────────      ─────────────────────────────────
Storey: 6 - Sixth Floor    IFC class: IfcWindow
Space:  —                  Name keyword: —
                           Neighbor type: IfcColumn

Confidence [██████████░░] 0.82  ·  source: lora
```
All five Phase 5 constraint fields are shown (`storey_name`, `space_name`, `ifc_class`, `target_name_keyword`, `neighbor_type`).

*Query Plans* — one expander per plan, ordered by priority. The first non-null rule is auto-expanded:
```
▼ P1: `storey_name + ifc_class`  →  ~12 candidates
    { "storey_name": "6 - Sixth Floor", "ifc_class": "IfcWindow" }
```

*Retrieval Results* — ranked candidate table (top-10):

| Rank | GT | Name | Type | Storey | CLIP | GUID |
|------|----|----|------|--------|------|------|
| 1 | ✓ | Basic Wall:Generic ... | IfcWindow | 6 - Sixth Floor | 0.812 | 3Gzo... |

Backend label, pool size, and whether CLIP reranking was applied are shown above the table.

*Stage timing:* three `st.metric` tiles — Constraints extraction ms, Query planning ms, Retrieval ms.

**V1 pipeline:**
Collapsible expanders for each agent tool call, showing tool name, arguments, and the first 500 characters of the tool result.

---

### 16.5 Tab 3 — Result

Split 50/50 left-right:

**Left — IFC STEP Text**

Raw STEP entity string for the predicted element (from `ifcopenshell.by_guid()`), then the ground truth element if different. Labelled `Predicted (✓ / ✗)` and `Ground truth`. Result is `@st.cache_data`-cached so reopening the same GUID is instant.

```
Predicted element ✗
#1234 = IFCWINDOW('3GzoWuxx...', ..., 'Basic Wall:Generic ...');

Ground truth element
#5678 = IFCWALL('AbcDef12...', ..., 'Basic Wall:MockUp...');
```

**Right — 3D BIM Viewer**

An `<iframe>` component (520 px height) running Three.js + `@thatopen/components` in the browser:

1. IFC file streams from the background static server with a progress bar (5 % → 82 % download → 100 % geometry parsed)
2. Predicted element highlighted:
   - **Green** (`#22c55e`) — GUID match (correct)
   - **Red** (`#ef4444`) — wrong prediction
3. Ground truth element highlighted in **blue** (`#3b82f6`) when it differs from the prediction
4. Full orbit camera controls (zoom, pan, rotate)
5. Color legend rendered in the iframe HTML

---

### 16.6 Data Flow

```
outputs/traces/<run_id>/<case_id>.json   ← written by script/run.py
        │
        ▼
   loader.load_trace()                   ← parses trace JSON
        │
        ▼
   app.py                                ← Streamlit router
        ├── sidebar.render_metrics()
        ├── tab_context.render()
        ├── tab_pipeline.render()
        └── tab_result.render()
                    │
                    ├── ifc_index.py     ← GUID → Express ID (cached)
                    └── static server    ← serves IFC + viewer.bundle.js
```

---

### 16.7 File Structure

```
demo/
├── app.py                   # Streamlit entry point (page config, tab routing)
├── loader.py                # list_runs(), list_cases(), load_trace()
├── server.py                # Background HTTP server on :8502 (static assets)
├── ifc_index.py             # GUID → Express ID index (ifcopenshell, cached)
│
├── ui/
│   ├── sidebar.py           # Run/case selector + evaluation metrics
│   ├── tab_context.py       # Chat history, input images, 4D context
│   ├── tab_pipeline.py      # Constraints, query plans, retrieval table, timing
│   └── tab_result.py        # IFC STEP text + 3D viewer iframe
│
├── src/
│   └── main.js              # Three.js + @thatopen/components viewer (91 lines)
│
├── static/
│   ├── viewer.bundle.js     # esbuild-compiled bundle (~4.9 MB)
│   └── web-ifc.wasm         # IFC geometry parser WASM (~1.3 MB)
│
├── templates/
│   └── viewer.html          # Iframe template (config injection, progress bar, legend)
│
└── package.json             # esbuild + @thatopen/components + three + web-ifc
```

**Rebuild the JS bundle** (needed only after editing `src/main.js`):
```bash
cd demo
npm run build    # esbuild src/main.js → static/viewer.bundle.js
```

---

*This document describes the system as of the current codebase. For experiment-specific results, see `RESULTS.md`. For component-level API documentation, see `README/OVERVIEW.md`.*
