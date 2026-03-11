# MSCD Demo — Technical Introduction

**Multi-modal Site Condition Detection for BIM Element Identification**

> A neuro-symbolic system for cross-modal IFC element retrieval: a Vision-Language Model
> extracts spatial-topological constraints from site observations, and a deterministic
> graph engine resolves them against an IFC knowledge graph.

---

## 1. Problem Statement

In Architecture, Engineering, and Construction (AEC), site supervisors generate continuous observation streams — photos, voice notes, chat messages — that must eventually link to a specific physical element in the Building Information Model (BIM). The canonical workflow today is manual: a supervisor opens a BIM viewer, scrolls through a 10-storey model, and hand-selects the element they're reporting on.

The computational formulation is a **cross-modal retrieval problem**: given a noisy, multimodal query `q = (text, images, 4D metadata)`, retrieve the correct IFC element `e*` from a candidate pool `E` of size `|E|` up to several thousand.

**Why is this hard?**

A query as simple as `"Check this window"` matches **263 candidates** in a 10-storey residential building, yielding a naive precision of **0.38%**. Adding any single modality of context can collapse the search space dramatically:

| Signal | Pool After Filter | Reduction |
|---|---|---|
| No context (baseline) | 263 | 0% |
| Storey from 4D task status | ~46 | 82.5% |
| Storey + element type | ~46 | 82.5% (identical windows) |
| **Storey + type + spatial relation** | **~3** | **98.9%** |

The **attribute entropy bottleneck**: on floors 2–5, there are 46 identical `IfcWindow` instances per floor — same type, same name, same material. Storey + type filtering alone yields Top-1 accuracy of just 2.2%. Breaking this bottleneck requires **spatial-topological** reasoning: which wall does the window fill? What element is it adjacent to?

This project implements and ablates two generations of retrieval pipelines to study the contribution of each modality, compare **agentic** vs **deterministic** architectures, and quantify the impact of **neuro-symbolic spatial reasoning** on retrieval precision.

---

## 2. System Architecture

### 2.1 High-Level Overview

The system is structured as a multi-stage compound AI pipeline. The current production architecture (V2 + LoRA_3 + Neo4j) implements a **neuro-symbolic** design: a neural front-end (fine-tuned VLM) extracts structured constraints including spatial triplets, and a symbolic back-end (deterministic planner + Neo4j Cypher) executes graph traversal.

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
    ┌─────────▼──────────┐    ┌──────────▼──────────────────────┐
    │    V1: Agent Path  │    │  V2: Neuro-Symbolic Pipeline     │
    │  (LangGraph ReAct) │    │                                  │
    │  [baseline only]   │    │  ┌────────────────────────────┐  │
    └─────────┬──────────┘    │  │  NEURAL LAYER (LoRA_3 VLM) │  │
              │               │  │  Qwen2.5-VL-7B + LoRA      │  │
              │               │  │  → SpatialTriplet extraction│  │
              │               │  │  → Constraints JSON         │  │
              │               │  └────────────┬───────────────┘  │
              │               │               │                   │
              │               │  ┌────────────▼───────────────┐  │
              │               │  │  SYMBOLIC LAYER             │  │
              │               │  │  Query Planner (P0–P8)      │  │
              │               │  │  + Neo4j Cypher Execution   │  │
              │               │  │  → Topological graph walk   │  │
              │               │  └────────────┬───────────────┘  │
              │               └───────────────┤                   │
              │                               │
    ┌─────────▼───────────────────────────────▼───────────────┐
    │                    Retrieval Backend                      │
    │     ┌─────────────┐       ┌──────────────────────┐      │
    │     │  In-Memory  │       │  Neo4j IFC Graph     │      │
    │     │  Spatial    │       │  FILLS / ADJACENT_TO  │      │
    │     │  Index      │       │  / CONTINUOUS edges   │      │
    │     └─────────────┘       └──────────────────────┘      │
    │                   ┌──────────┐                           │
    │                   │  CLIP    │  (optional reranking)     │
    │                   │ Reranker │                           │
    │                   └──────────┘                           │
    └─────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────▼──────────────┐
              │     IFC Engine (IfcOpenShell) │
              │   Spatial Index · GUID Lookup │
              │   Neo4j Export (topology)     │
              └───────────────┬──────────────┘
                              │
    ┌─────────────────────────▼────────────────────────────────┐
    │                   Output Layer                            │
    │   EvalTrace · RQ2 Schema Validation · BCF 2.1 ZIP        │
    └──────────────────────────────────────────────────────────┘
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

**Characteristics**: Non-deterministic, high flexibility, opaque reasoning. Useful as a human-level baseline but difficult to ablate systematically.

---

### 3.2 V2: Constraints-Driven Pipeline (Current Production)

**Philosophy**: Decompose the retrieval problem into explicit stages, each independently testable. Replace LLM reasoning with deterministic query planning. Use a fine-tuned VLM to extract **spatial-topological constraints** that a symbolic engine can execute.

**Entry point**: `script/run.py --profile v2_*`

**Stage 1 — Image Parser** (`src/visual/image_parser.py`)

Converts raw images into structured semantic records using Gemini 2.5 Flash in multimodal mode. Each image produces a `ParsedImage` with element type, material, defect info, location cues, and confidence. Results are cached in-memory. **Skipped when using LoRA extractor** — the VLM reads raw images directly.

**Stage 2 — Constraints Extractor**

Fuses parsed images, chat text, and 4D metadata into a typed constraint record:

```python
@dataclass
class Constraints:
    storey_name: Optional[str]          # "6 - Sixth Floor" (exact BIM name)
    ifc_class: Optional[str]            # "IfcWindow" (canonical IFC type)
    near_keywords: List[str]            # ["north", "elevator shaft"]
    space_name: Optional[str]           # "Living Room" (room-level)
    target_name_keyword: Optional[str]  # "Daikin", "AHU-03" (brand/ID)
    spatial_relations: List[SpatialTriplet]  # topology (LoRA_3)
    confidence: float
    source: str                         # "prompt" | "lora3" | "prompt_failed"
```

**SpatialTriplet** — the key innovation in V2 + LoRA_3:

```python
class SpatialTriplet(BaseModel):
    subject_type: str           # "IfcWindow"
    predicate: Literal[
        "FILLS",                # door/window occupies opening in wall
        "CONTINUOUS",           # element spans multiple storeys
        "ADJACENT_TO",          # same-storey centroid distance < 1.5 m
        "ON_TOP_OF",            # vertical stacking
        "PERPENDICULAR_TO",     # wall orientation ~90°
        "PARALLEL_TO",          # wall orientation ~0°
    ]
    object_type: str            # "IfcWallStandardCase"
    object_material: Optional[str]  # "Concrete" (discriminating)
    confidence: float           # VLM-output quality gate (threshold ~0.7)
```

Three extraction backends have been compared across the project:

| Backend | Model | Spatial Relations | Training Data | Source Tag |
|---|---|---|---|---|
| **Prompt-only** | Gemini 2.5 Flash | None (no spatial extraction) | None | `"prompt"` |
| **LoRA_2** | Qwen2.5-VL-7B + LoRA | None (6-field output) | synth_v0.4 (933 train) | `"lora"` |
| **LoRA_3** | Qwen2.5-VL-7B + LoRA | SpatialTriplet array | synth_v0.5 (1,377 train) | `"lora3"` |

**Stage 3 — Query Planner** (`src/v2/constraints_to_query.py`)

Translates a `Constraints` record into a `QueryPlan` via a **priority-ordered rule table**. The planner selects the highest-priority rule whose required fields are non-null. **Priority 0 (spatial_triplet)** is the key addition in the LoRA_3 pipeline:

| Priority | Strategy | Required Fields | Est. Pool | Description |
|---|---|---|---|---|
| **0a** | `spatial_triplet` | spatial_relations + ifc_class | ~3 | Neo4j edge traversal (FILLS, ADJACENT_TO) |
| **0b** | `continuous_span` | spatial_relations + ifc_class (CONTINUOUS) | ~5 | Property filter: `is_continuous=true` |
| 1 | `space+type` | space_name + ifc_class | ~5 | Named room + type |
| 2 | `name_keyword` | target_name_keyword | ~3 | Equipment ID fuzzy match |
| 3 | `neighbor+type` | neighbor_type + ifc_class | ~8 | Legacy adjacency (pre-spatial) |
| 4 | `storey+type` | storey_name + ifc_class | ~50 | Both storey and type |
| 5 | `storey_only` | storey_name | ~200 | Floor filter only |
| 6 | `type_only` | ifc_class | ~150 | Type across all storeys |
| 7 | `keyword` | near_keywords | ~100 | Text search |
| 8 | `fallback` | (none) | ~100 | Return first 100 elements |

**Confidence gate**: Priority 0 only fires if `max(triplet.confidence for triplet in spatial_relations) >= 0.7`. Below threshold, the planner falls through to Priority 1+, preventing hallucinated spatial relations from producing empty result sets.

**Stage 4 — Retrieval Backend** (`src/v2/retrieval_backend.py`)

Executes the query plan against one of two storage backends:

- **Memory**: In-memory `spatial_index` built from `IfcRelContainedInSpatialStructure` at startup. Supports storey-filtered and type-filtered lookups. Latency < 10 ms. Spatial triplets gracefully degrade to storey+type.
- **Neo4j**: Full IFC topological graph with Cypher query support. Enables spatial_triplet queries via edge traversal (`FILLS`, `ADJACENT_TO`, `CONTINUOUS`). Latency 50–200 ms.

**Neo4j Cypher for Priority 0a (spatial_triplet)**:
```cypher
MATCH (target:IFCElement)-[:FILLS]->(ref:IFCElement)
WHERE (target.ifc_type = $type OR target.ifc_type STARTS WITH $type)
  AND (ref.ifc_type = $ref_type OR ref.ifc_type STARTS WITH $ref_type)
  AND target.storey = $storey
RETURN DISTINCT target
```

**Two-step predicate relaxation**:
1. If 0 results: retry without storey filter (stays Priority 0)
2. If still 0: fall back to storey+type (Priority 4)

Both `fallback_triggered` and `strategy_actually_used` are recorded in `RetrievalResult` for diagnostic tracing.

Optional **CLIP reranking** (`src/visual/aligner.py`): after retrieval, each candidate element is encoded as a text description, embedded with `openai/clip-vit-base-patch32`, and sorted by cosine similarity to the query image embeddings.

---

## 4. Neo4j Topology Layer

### 4.1 IFC Graph Construction

The IFC model is exported to a Neo4j property graph via `src/ifc_engine.py`. Each IFC element becomes a node with properties (`guid`, `name`, `ifc_type`, `storey`, `centroid_x/y/z`, `is_continuous`, `top_constraint`). Three edge types encode spatial-topological relationships:

| Edge Type | Source → Target | Count (AP) | Derivation |
|---|---|---|---|
| **FILLS** | IfcDoor/IfcWindow → IfcWall | 389 | `IfcRelFillsElement` → `IfcRelVoidsElement` chain |
| **ADJACENT_TO** | Any → Any (cross-type) | ~200 | Centroid distance 100mm < d ≤ 1500mm, same storey |
| **CONTINUOUS** | IfcWall → (property) | 150 | Revit `top_constraint` pset, `is_continuous=true` |

### 4.2 FILLS Edge Bug Fix

The original `_create_element_relationships()` created edges to `IfcOpeningElement` nodes — but these don't exist in the graph (only physical elements are exported). Fixed by chaining through the `opening_to_host` dict to create direct `Door/Window → Wall` edges.

### 4.3 Storey Resolution

IFC models use inconsistent storey naming: walls may be `"1 - First Floor"` while windows reference `"Level 1"`. The retrieval backend includes a `_resolve_storey()` helper that fuzzy-matches user-facing storey names to canonical BIM storey names via `engine._resolve_storey_query()`.

**Key data insight**: Multi-story continuous walls have `storey = base_floor` in Neo4j (e.g., "1 - First Floor"), not the upper floors they span to. FILLS Cypher filters by `target.storey` (the window's storey), not `ref.storey` (the wall's base storey).

---

## 5. Data Model

### 5.1 Input Case Schema

```json
{
  "case_id": "SYNTH_V3_001_SK_001",
  "bench": { "group": "A", "condition": "A1" },
  "difficulty_tags": { "tier": "T1", "candidate_density_k": 5, ... },
  "inputs": {
    "chat_history": [{ "role": "Site Supervisor", "text": "..." }],
    "images": ["datasets/.../imgs/img_001.png"],
    "floorplan_patch": "datasets/.../plans/plan_001.png",
    "project_context": { "4d_task_status": "Window Installation - Level 6 - IN_PROGRESS" }
  },
  "ground_truth": {
    "target_guid": "3GzoWuxxn4WO8bCtw8H3Vj",
    "target_storey": "1 - First Floor",
    "target_ifc_class": "IfcWall"
  },
  "labels": {
    "constraints": {
      "storey_name": "1 - First Floor",
      "ifc_class": "IfcWall",
      "spatial_relations": [
        { "predicate": "FILLS", "object_type": "IfcWallStandardCase", "object_material": "Concrete", "confidence": 0.95 }
      ]
    }
  }
}
```

### 5.2 Evaluation Trace (`EvalTrace`)

Each pipeline run produces a structured trace. V2 traces additionally include `constraints`, `query_plan`, `retrieval_result` (with `fallback_triggered` and `strategy_actually_used`), and timing breakdowns.

---

## 6. Datasets

### 6.1 BIM Models

| Model | Tag | Storeys | Elements | Primary Use |
|---|---|---|---|---|
| AdvancedProject | AP | 10 | ~1,233 | Primary eval + Neo4j topology |
| BasicHouse | BH | 2 | ~120 | Cross-building generalization |
| Duplex_A | DXA | Split-level | ~300 | Cross-building generalization |

**AdvancedProject key statistics**:
- 263 `IfcWindow`, 390 `IfcWall`, 381 `IfcWallStandardCase`, 126 `IfcDoor`
- 46 identical windows per floor on floors 2–5 (the attribute entropy bottleneck)
- 17/19 railings without storey_name (aggregated via `IfcRelAggregates`)
- Neo4j: 389 FILLS edges, ~200 ADJACENT_TO edges, 150 CONTINUOUS walls

### 6.2 Evaluation Datasets

| Dataset | Cases | Training | Purpose |
|---|---|---|---|
| **gt_1** | 6 | — | Hand-written ground truth |
| **synth_v0.3** | 84 | — | V1/V2 prompt baseline (3 tiers × A1–C3) |
| **synth_v0.4** | 361 | 933 train / 50 test | LoRA_2 (attribute-only, 3 buildings) |
| **synth_v0.5** | 487 | 1,377 train / 69 test | **LoRA_3 (topology + attributes)** |
| **H2 hard-neg** | 213 | — | Topology retrieval stress test |

### 6.3 Difficulty Tiers (synth_v0.3)

| Tier | Focus | RQ | Image Mode | Text Style | % of v0.3 |
|---|---|---|---|---|---|
| **T1** (Visual Texture) | Visual defect grounding | RQ1 | defect | deictic | ~35% |
| **T2** (Spatial / 4D) | Floorplan + 4D alignment | RQ2 | defect | relative | ~35% |
| **T3** (Conflict / Negative) | Governance, mismatch detection | RQ3 | mismatch/pristine | misleading | ~30% |

---

## 7. LoRA Fine-Tuning Pipeline

### 7.1 LoRA_2 — Attribute-Only Extraction (synth_v0.4)

**Training data**: synth_v0.4 — three IFC building models, 3× text augmentation.

```
synth_v0.4_{ap,bh,dxa}/cases_v3_filtered.jsonl  (361 unique cases)
    → 6_augment_text.py  (stratified holdout + 3× text augmentation)
    → 7_prepare_lora_data.py  (Qwen2.5-VL ChatML format)
    → 933 train samples + 50 test samples
```

**Output schema (6 fields)**:
```json
{
  "storey_name": "6 - Sixth Floor",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "neighbor_type": null,
  "near_keywords": ["north"]
}
```

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |
| LoRA rank / alpha | 16 / 32 |
| Dropout | 0.0 |
| Epochs | 3 |
| Effective batch size | 16 (2 × 8 grad accum) |
| Max seq length | 2,048 |
| Training samples | 933 |
| Hardware | Modal A100 (40 GB) |

**Results (50 holdout, 6-condition modality ablation)**:
- LoRA_2 Top-1: **35.3%** vs Prompt baseline: 25.7% (+9.6 pp)
- Highest priority rule available: storey+type (Priority 4 in current numbering)
- Cannot break attribute entropy bottleneck (46 identical windows/floor)

---

### 7.2 LoRA_3 — Spatial-Topological Extraction (synth_v0.5)

**Motivation**: LoRA_2 hits a ceiling because storey+type alone cannot disambiguate identical elements. LoRA_3 adds `spatial_relations` output — a structured array of `SpatialTriplet` objects that the symbolic layer can execute as graph traversals.

**Training data construction (3-tier labeling strategy)**:

```
Tier 1 — Spatial signal present (topology cases):
    synth_v0.5 topology skeletons (FILLS=28, ADJACENT_TO=34, CONTINUOUS=22)
    + synth_v0.4 SPATIAL_PROXIMITY (33 cases, relabeled)
    → spatial_relations populated with ground-truth triplets

Tier 2 — No spatial signal (attribute-only cases):
    synth_v0.4 attribute-only cases (~900)
    → spatial_relations = []  (teaches model when NOT to extract)

Tier 3 — New topology cases:
    Fresh cases from 3 IFC models with relation crop renders

Target mix: ~60-70% attribute-only + ~30-40% topology
    → Prevents hallucination of spatial relations
```

**Skin generation pipeline**:
- `3a_render_relation_crops.py` — wireframe renders via ifcopenshell.geom + matplotlib 3D (subject=blue, anchor=orange)
- `3b_generate_skin.py` — Gemini Flash text generation + photoreal image synthesis + LLM-as-Judge quality gate
- `3c_assemble.py` — assembles into ChatML format with spatial_relations labels

**Output schema (5 fields + spatial_relations)**:
```json
{
  "storey_name": "6 - Sixth Floor",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {
      "predicate": "FILLS",
      "object_type": "IfcWallStandardCase",
      "object_material": "Concrete",
      "confidence": 0.95
    }
  ]
}
```

| Parameter | LoRA_2 | LoRA_3 | Change |
|---|---|---|---|
| Training samples | 933 | **1,377** | +47.6% |
| Test samples | 50 | **69** | +38% |
| Output fields | 6 (flat) | **5 + spatial_relations[]** | Topology-aware |
| Epochs | 3 | **5** | +2 (more topology data) |
| LoRA dropout | 0.0 | **0.1** | Regularization |
| Max seq length | 2,048 | **4,096** | Multi-image support |
| Eval strategy | Per-epoch | **Per-epoch + spatial accuracy** | New metrics |
| Spatial predicates | — | FILLS, CONTINUOUS, ADJACENT_TO | 3 active |
| Confidence field | Hardcoded 0.85 | **VLM-output (max across relations)** | Dynamic |

**LoRA_3 training metrics**:
- `test_json_rate`: % valid JSON output
- `test_class_accuracy`: ifc_class exact match
- `test_storey_accuracy`: storey_name exact match
- `test_spatial_predicate_acc`: predicate match on topology samples
- `test_spatial_false_positive_rate`: spatial_relations output when GT = []

**Inference contract (LoRA_3)**:
```python
# Input: site photo + floorplan + chat text + 4D metadata
messages = [
    {"role": "system", "content": CONSTRAINTS_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image", "image": site_photo},
        {"type": "image", "image": floorplan_patch},
        {"type": "text", "text": "[4D Status] ...\n[Chat Log] ..."}
    ]}
]

# Output: parsed into Constraints with SpatialTriplet array
{
    "storey_name": "3 - Third Floor",
    "ifc_class": "IfcWindow",
    "space_name": null,
    "target_name_keyword": null,
    "spatial_relations": [
        {"predicate": "FILLS", "object_type": "IfcWall",
         "object_material": "Brick", "confidence": 0.92}
    ]
}
```

---

### 7.3 Comparison: LoRA_2 vs LoRA_3

| Metric | LoRA_2 | LoRA_3 | Delta |
|---|---|---|---|
| Training data | synth_v0.4 (933) | synth_v0.5 (1,377) | +444 topology samples |
| Highest priority available | P4 (storey+type) | **P0 (spatial_triplet)** | +4 priority levels |
| Spatial predicate accuracy | N/A | **93.0%** (40/43) | New capability |
| False positive rate | N/A | **0%** (0/26) | Anti-hallucination |
| Spatial extraction | None | FILLS / ADJACENT_TO / CONTINUOUS | New capability |
| Confidence | Static 0.85 | Dynamic (VLM output) | Quality gate |

The fundamental improvement: LoRA_2 cannot distinguish between 46 identical windows on a single floor. LoRA_3 extracts "this window FILLS a concrete wall" — when the symbolic layer has complete graph edges, Neo4j resolves it to a small candidate pool, breaking the attribute entropy bottleneck.

---

## 8. H2 Hard-Negative Evaluation

### 8.1 Design

The H2 eval set (`eval/h2_eval.py`) stress-tests the topology retrieval path with cases specifically constructed to require spatial reasoning. Cases are drawn from `2b_build_h2_hardneg.py` which samples elements in dense pools where attribute-only retrieval fails.

### 8.2 Full H2 Evaluation (213 cases)

| Predicate | Cases | GT-in-Pool | Fallback | Notes |
|---|---|---|---|---|
| ADJACENT_TO | 66 | 0/66 (0%) | 66/66 | Missing edges in Neo4j |
| FILLS | 84 | 0/84 (0%) | 84/84 | Missing edges in Neo4j |
| CONTINUOUS | 63 | 63/63 (100%) | 63/63 | Property-based, works e2e |
| **Total** | **213** | **63/213 (30%)** | **213/213** | **Graph incompleteness** |

**Root cause**: `neo4j_init.sh` topology enrichment did not load ADJACENT_TO/FILLS edges. CONTINUOUS works because it uses node properties (`is_continuous`, `top_constraint`), not edge traversal. See `README/0224_demo_plan.md` §7.8.3 for full root cause analysis.

### 8.3 P2 Unit Test Results (83 cases, edges manually loaded)

With all graph edges present (389 FILLS, ~200 ADJACENT_TO), the symbolic layer achieves:

| Predicate | Cases | GT-in-Pool | Avg Pool Size | SSR | Attr Baseline |
|---|---|---|---|---|---|
| ADJACENT_TO | 34 | 34/34 (100%) | 32 | 30% | 5.2% |
| CONTINUOUS | 21 | 21/21 (100%) | 74 | 65% | 0.4% |
| FILLS | 28 | 28/28 (100%) | 43 | 0% | N/A |

**Key findings from P2 tests**:
- **100% GT-in-pool (83/83)** when edges are loaded — symbolic layer never drops the ground-truth element
- **0 fallbacks** — Priority 0 fires for all 83 cases without degrading to storey+type
- ADJACENT_TO provides the best pool reduction: from avg 122 (full storey) to 32 candidates
- Attribute baseline avg 3.0% confirms storey+type alone fails for these cases

### 8.4 Projected Results After Graph Fix

Based on P2 unit tests and VLM accuracy (93% spatial predicate):

| Metric | Current | Projected |
|---|---|---|
| GT-in-pool | 63/213 (30%) | **213/213 (100%)** |
| SSR | -74% to 100% | **75–90%** |
| Top-1 (threshold=0.7) | N/A | **~57%** (22.8× over 2.5% attr baseline) |
| Top-1 (threshold=0.5) | N/A | **~93%** (37.2× over baseline) |

**Blocker**: Fix `neo4j_init.sh` edge loading → re-run H2 eval for validated numbers.

### 8.3 Evaluation Harness

```bash
# Run H2 evaluation
conda run -n mscd_demo python eval/h2_eval.py \
  --output eval/results/h2_eval.jsonl \
  --plot eval/results/h2_eval_figure.png

# Output: 3-panel thesis figure
#   Panel 1: GT-in-pool rate per predicate
#   Panel 2: Search Space Reduction per predicate
#   Panel 3: Pool size reduction (initial → final) per predicate
```

---

## 9. Post-Hoc Explainability

### 9.1 Motivation

Understanding *where* the VLM looks when extracting spatial relations is critical for thesis presentation and model trust. Three approaches were considered:

| Approach | Feasibility | Notes |
|---|---|---|
| Attention extraction | **Blocked** | Flash attention (unsloth) does not support `output_attentions` |
| Gradient-based saliency | **Blocked** | 4-bit quantized weights have no gradients |
| **Occlusion-based** | **Used** | Model-agnostic, no architecture requirements |

### 9.2 Occlusion-Based Saliency

**Method**: For each input image, divide into an N×N grid (default 4×4). For each patch:
1. Replace patch with gray (128, 128, 128)
2. Re-run VLM inference on the masked input
3. Measure prediction degradation:
   - Spatial relation disappeared → importance = 1.0
   - Predicate changed → importance = 0.8
   - Confidence drop → importance = `baseline_conf - masked_conf`
4. Normalize per image to [0, 1]

**Implementation**: `training/inference.py :: LoRA3Predictor.explain()`

```python
@modal.method()
def explain(self, image_bytes_list, chat_text="", metadata_text="", grid_size=4) -> dict:
    """Occlusion-based saliency: mask patches, measure prediction change."""
    baseline = self._predict_core(pil_images, chat_text, metadata_text)
    # ... NxN grid sweep per image ...
    return {
        "baseline": baseline,
        "heatmaps": [[[float, ...], ...], ...],  # grid_size x grid_size per image
        "image_sizes": [(w, h), ...],
        "grid_size": int,
        "spatial_focus_tokens": [str, ...],
    }
```

**Cost**: N² × num_images inference calls per explanation (16 for 4×4 grid). Runs on the same Modal A100 endpoint as `predict()`.

### 9.3 Graph Explainability

The demo UI provides three levels of graph visualization via Graphviz DOT rendering:

1. **Query Plan Cascade** — shows which priority rules fire and why, with the winning rule highlighted
2. **1-Hop Subgraph** — Neo4j neighborhood around the retrieved elements (target → edge → reference), showing the spatial relationship that resolved the query
3. **Whole-IFC Graph Snapshot** — bubble plot of all elements in the model, colored by type, with retrieved candidates highlighted

---

## 10. Tool Infrastructure (MCP Servers)

The V1 pipeline exposes IFC and visual operations as **Model Context Protocol (MCP)** servers. MCP decouples tool implementation from the agent framework:

```
mcp_servers/
├── ifc_server.py      # IfcOpenShell queries as MCP tools
└── visual_server.py   # CLIP + Gemini VLM as MCP tools
```

---

## 11. Schema Validation (RQ2)

Post-retrieval, every identified element is validated against **CORENET-X minimal schema** — Singapore's regulatory BIM submission format. This tests whether AI-extracted information is structurally compatible with downstream regulatory workflows.

---

## 12. Live Inference Demo

### 12.1 Architecture

The live inference tab (`demo/ui/tab_inference.py`) provides an interactive end-to-end pipeline demonstration with five stages:

```
Stage 1: Input Assembly          (images + chat + 4D metadata)
    ↓
Stage 2: VLM Constraint Extract  (Modal A100 endpoint → Constraints JSON)
    ↓
Stage 3: Query Planning          (Priority cascade → QueryPlan)
    ↓
Stage 4: Symbolic Retrieval      (Neo4j Cypher → candidate pool)
    ↓
Stage 5: Result & Explainability (3D viewer + saliency + graph viz)
```

### 12.2 Modal Serverless Endpoint

Inference runs on `training/inference.py` deployed as a Modal serverless function:

```python
app = modal.App("mscd-vlm-lora3-inference")

@app.cls(gpu="A100", container_idle_timeout=300)
class LoRA3Predictor:
    @modal.enter()
    def load_model(self):   # loads once, stays warm 5 min
        ...
    @modal.method()
    def predict(self, image_bytes_list, chat_text, metadata_text) -> dict:
        ...
    @modal.method()
    def explain(self, image_bytes_list, chat_text, metadata_text, grid_size=4) -> dict:
        ...
```

### 12.3 Demo Features

| Feature | Description |
|---|---|
| **Pipeline stage visualization** | Each stage shown with timing, inputs/outputs, expand/collapse |
| **Confidence gate indicator** | Shows whether spatial_relations exceed 0.7 threshold for P0 |
| **GT comparison** | Compare predictions against ground-truth from loaded trace |
| **3D BIM viewer** | Three.js iframe with highlighted predicted (green/red) and GT (blue) elements |
| **Occlusion saliency heatmap** | Matplotlib overlay showing which image regions the VLM relies on |
| **Graph explainability** | Graphviz DOT diagrams for plan cascade, 1-hop subgraph, IFC snapshot |
| **Multi-IFC support** | Switch between AP/BH/DXA models at inference time |

---

## 13. Evaluation Metrics

| Metric | Definition | Scope |
|---|---|---|
| **Top-1 Accuracy** | GT GUID is top-ranked candidate | V1 + V2 |
| **Top-K Accuracy** | GT GUID in top-K (K=5) | V1 + V2 |
| **Search Space Reduction (SSR)** | `(initial_pool - final_pool) / initial_pool` | V2 |
| **GT-in-Pool** | GT GUID appears anywhere in candidate pool | V2 (H2 eval) |
| **Constraints Field EM-F1** | Token-level F1 between predicted and label constraints | V2 |
| **Constraints Parse Rate** | Fraction producing valid JSON | V2 |
| **Spatial Predicate Accuracy** | Correct predicate on topology samples | LoRA_3 |
| **Spatial False-Positive Rate** | spatial_relations output when GT = [] | LoRA_3 |
| **Avg Latency** | End-to-end wall-clock time | All |

---

## 14. Configuration & Experiment Management

### 14.1 Profile System (`profiles.yaml`)

A profile fully specifies one experimental configuration:

```yaml
v2_lora_neo4j_clip:
  pipeline: v2
  constraints_model: lora       # or "prompt"
  retrieval: neo4j              # or "memory"
  use_clip: true
```

### 14.2 Experiment Tracking

Each named experiment records: profile used, dataset, git commit hash, conda snapshot, and timestamps for full reproducibility.

---

## 15. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| BIM parsing | IfcOpenShell 0.8+ | IFC model loading, spatial graph traversal |
| Graph database | **Neo4j Community 5.26** | FILLS/ADJACENT_TO/CONTINUOUS topology |
| VLM fine-tuning | **Qwen2.5-VL-7B + Unsloth LoRA** | Spatial-topological constraint extraction |
| GPU inference | **Modal (A100 serverless)** | LoRA_3 predict + explain endpoints |
| LLM (agent + prompt) | Google Gemini 2.5 Flash | V1 agent, prompt-only extraction, skin generation |
| Visual matching | OpenAI CLIP `vit-base-patch32` | Cosine reranking |
| Agent framework | LangChain + LangGraph | V1 ReAct agent |
| Tool protocol | FastMCP 0.2+ | MCP server/client |
| Data validation | Pydantic v2 | Runtime type safety |
| Schema validation | jsonschema (Draft 2020-12) | CORENET-X compliance |
| Demo UI | **Streamlit** | Live inference + trace visualization |
| 3D viewer | **Three.js + @thatopen/components** | In-browser IFC rendering |
| Graph visualization | **Graphviz (DOT)** | Query plan + subgraph diagrams |
| Saliency visualization | **Matplotlib** | Occlusion heatmap overlay |
| BCF packaging | Python stdlib | BCF 2.1 artifact generation |
| Photoreal synthesis | Google Gemini | Wireframe → site photo for training |

---

## 16. Repository Layout

```
mscd_demo/
├── config.yaml                       # IFC path, Neo4j, LLM settings
├── profiles.yaml                     # Experiment profiles
│
├── src/
│   ├── main_mcp.py                   # V1 agent entry point
│   ├── ifc_engine.py                 # IFC gateway + Neo4j export
│   ├── v2/
│   │   ├── types.py                  # Constraints, SpatialTriplet, QueryPlan
│   │   ├── constraints_extractor_lora.py   # LoRA_3 inference
│   │   ├── constraints_extractor_prompt_only.py
│   │   ├── constraints_to_query.py   # Priority 0–8 rule planner
│   │   ├── retrieval_backend.py      # Memory / Neo4j / CLIP execution
│   │   ├── pipeline.py               # V2 orchestration
│   │   └── metrics_v2.py             # V2 diagnostic metrics
│   ├── eval/                         # Evaluation framework (V1 + V2)
│   ├── visual/                       # Image parser + CLIP aligner
│   ├── common/                       # Config, GUID extraction, etc.
│   ├── rq2_schema/                   # CORENET-X validation
│   └── handoff/                      # BCF 2.1 generation
│
├── training/
│   ├── train.py                      # Modal LoRA training (LoRA_2/3)
│   ├── inference.py                  # Modal serverless endpoint (predict + explain)
│   └── eval.py                       # Post-training evaluation
│
├── eval/
│   ├── h2_eval.py                    # H2 hard-negative eval harness
│   └── results/                      # JSONL + plot outputs
│
├── demo/
│   ├── app.py                        # Streamlit entry point
│   ├── ui/
│   │   ├── sidebar.py                # Run/case selector
│   │   ├── tab_context.py            # Input visualization
│   │   ├── tab_pipeline.py           # Pipeline trace
│   │   ├── tab_result.py             # 3D viewer + STEP text
│   │   └── tab_inference.py          # Live inference (5-stage + explainability)
│   ├── static/                       # viewer.bundle.js + web-ifc.wasm
│   └── templates/                    # Iframe HTML templates
│
├── mcp_servers/                      # V1 MCP tool servers
├── script/
│   ├── run.py                        # Unified evaluation runner
│   └── neo4j_init.sh                # Neo4j setup + IFC graph load
│
├── data/ifc/AdvancedProject/         # Primary BIM model
└── schemas/corenetx_min/             # Regulatory schemas
```

---

## 17. Research Questions

| RQ | Question | V1 Approach | V2 + LoRA_3 Approach |
|---|---|---|---|
| **RQ1** | Can cross-modal signals disambiguate? | Agent tool calls | SpatialTriplet extraction + Neo4j traversal |
| **RQ2** | Can constraints map to regulatory schemas? | FINAL_JSON extraction | Same (shared pipeline) |
| **RQ3** | When should the system escalate? | Free-form agent | Confidence gate + empty result detection |

---

## 18. Key Design Decisions

**Neuro-symbolic split**: The VLM handles perception (what spatial relation exists?) while the symbolic layer handles execution (find elements with that relation in the graph). This separation enables: (1) each layer tested independently, (2) VLM hallucinations caught by empty Cypher results, (3) new predicates added without retraining.

**SpatialTriplet as the bridge**: The triplet `(subject_type, predicate, object_type)` is the interface contract between neural and symbolic layers. It's rich enough to express topological relationships but constrained enough for deterministic Cypher generation.

**Confidence gate at 0.7**: Prevents hallucinated spatial relations from producing empty result sets. Below threshold, the planner gracefully degrades to attribute-only filtering (Priority 4+).

**Occlusion over attention**: Model-agnostic saliency was chosen because the production stack (unsloth + 4-bit quantization + flash attention) prevents both attention extraction and gradient-based approaches. The N² cost is acceptable for interactive demo use (16 calls for 4×4 grid on a warm A100).

**Three-tier training data**: The ~60/40 attribute/topology split prevents the model from hallucinating spatial relations on attribute-only inputs. Tier 2 (empty spatial_relations) is critical — without it, the model outputs spatial_relations for every input, triggering false P0 queries.

**Priority 0 predicate relaxation**: Two-step fallback (drop storey → drop spatial entirely) ensures the system never returns 0 candidates. The `fallback_triggered` flag enables post-hoc analysis of when spatial reasoning adds value vs. when attribute filtering suffices.

---

*This document describes the system as of March 2026. For component-level API documentation, see `README.md`. For the demo plan, see `README/0224_demo_plan.md`.*
