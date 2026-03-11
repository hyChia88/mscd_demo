# MSCD: AI Interpreter for Construction Site Data

> Neuro-symbolic system that grounds messy site observations to exact BIM elements.
> VLM extracts spatial constraints from photos and chat → graph engine resolves them against IFC.

---

## Why This Exists

On construction sites, subcontractors snap photos, scribble notes, and hand stacks of unstructured paperwork to managers. Managers spend days manually linking this data back to the BIM. Updates lag. Issues get lost. People leave, and their notes become unreadable.

**This system is the missing Interpreter Layer.** It takes unstructured site input — photos, chat, floorplans — and automatically grounds it to the exact element in the building model. Before we can ask *what happened* or *why*, we must answer: **WHERE IS IT?**

## The Problem

*"Check this window"* matches **263 candidates** in a 10-storey building. Even with storey filtering, **46 identical windows per floor** remain. Attribute-only retrieval gives Top-1 = 2.2%.

## The Solution

**Neuro-symbolic**: a VLM extracts spatial relationships (*"this window FILLS a concrete wall"*) → a graph engine resolves them against the IFC model. VLM achieves **93% spatial predicate accuracy** with **0% false positive rate**. Symbolic layer (when graph edges are loaded) achieves **100% ground-truth retention** on 83 topology test cases.

```
Floorplan + Site Photos + Chat + 4D Metadata
        ↓
  NEURO:  LoRA_3 VLM (Qwen2.5-VL-7B, Modal A100)
        ↓ SpatialTriplet[] + Constraints JSON
  SYMBOLIC: Query Planner(Priority 0–8) → Neo4j Cypher
        ↓ ~3 candidates
  Result: exact GUID + 3D view + explainability
```

---

## Quick Start

### 1. Environment

```bash
conda activate mscd_demo
pip install -r requirements.txt
```

### 2. Neo4j Setup

```bash
# Install + configure + load IFC graph
./script/neo4j_init.sh

# Or manually:
/tmp/neo4j-community-5.26.0/bin/neo4j start
# Browse: http://localhost:7474 (neo4j / password)
```

### 3. API Keys

```bash
cp .env.example .env
# Add GOOGLE_API_KEY (for Gemini prompt extractor + registry LLM)
# Modal token configured via `modal token set`
```

### 4. Run Evaluation

```bash
# V2 with LoRA_3 + Neo4j (production pipeline)
python script/run.py --profile v2_lora \
  --cases ../data_curation/datasets/synth_v0.5/cases.jsonl

# V2 with Gemini prompt extraction (baseline)
python script/run.py --profile v2_prompt \
  --cases ../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl

# H2 hard-negative topology eval
conda run -n mscd_demo python eval/h2_eval.py \
  --output eval/results/h2_eval.jsonl \
  --plot eval/results/h2_eval_figure.png
```

### 5. Live Demo

```bash
cd mscd_demo
streamlit run demo/app.py
# Opens at http://localhost:8501
# Navigate to "Live Inference" tab
```

### 6. Train LoRA_3

```bash
# Launch on Modal A100 (5 epochs, 1377 samples)
modal run training/train.py

# Deploy inference endpoint
modal deploy training/inference.py

# Test inference
modal run training/inference.py --chat "crack near the railing on floor 3"
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    INPUT LAYER                           │
│  site_photos · floorplan · chat_history · 4D_task_status │
└─────────────────────────┬────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │  NEURAL LAYER         │
              │  LoRA_3 VLM           │
              │  (Qwen2.5-VL-7B)      │
              │  Modal A100 endpoint  │
              └───────────┬───────────┘
                          │ Constraints JSON
                          │ + SpatialTriplet[]
              ┌───────────▼───────────┐
              │  SYMBOLIC LAYER       │
              │  Query Planner P0–P8  │
              │  + Neo4j Cypher       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  RETRIEVAL BACKEND    │
              │  Neo4j graph walk     │
              │  (+ optional CLIP)    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │  OUTPUT LAYER         │
              │  GUID · 3D highlight  │
              │  Saliency · Graph viz │
              │  BCF 2.1 · RQ2 schema │
              └───────────────────────┘
```

### Priority Cascade (Query Planner)

| Priority | Strategy | Required Fields | Pool Size |
|---|---|---|---|
| **0a** | `spatial_triplet` | spatial_relations + ifc_class | ~3 |
| **0b** | `continuous_span` | spatial_relations (CONTINUOUS) | ~5 |
| 1 | `space+type` | space_name + ifc_class | ~5 |
| 2 | `name_keyword` | target_name_keyword | ~3 |
| 3 | `neighbor+type` | neighbor_type + ifc_class | ~8 |
| 4 | `storey+type` | storey_name + ifc_class | ~50 |
| 5 | `storey_only` | storey_name | ~200 |
| 6 | `type_only` | ifc_class | ~150 |
| 7 | `keyword` | near_keywords | ~100 |
| 8 | `fallback` | (none) | ~100 |

Priority 0 fires when `max(confidence) >= 0.7` across spatial_relations. Below threshold, falls through to P1+.

### LoRA_3 Output Schema

```json
{
  "storey_name": "3 - Third Floor",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {
      "predicate": "FILLS",
      "object_type": "IfcWallStandardCase",
      "object_material": "Concrete",
      "confidence": 0.92
    }
  ]
}
```

### Neo4j Topology

Three edge types in the IFC graph:

| Edge | Meaning | Count (AP) |
|---|---|---|
| `FILLS` | Door/Window → Wall (via opening) | 389 |
| `ADJACENT_TO` | Cross-type pair within 1.5m | ~200 |
| `CONTINUOUS` | Wall spans multiple storeys | 150 |

---

## Live Inference Demo

The `Live Inference` tab in the Streamlit demo provides interactive end-to-end inference with explainability:

### 5-Stage Pipeline View

1. **Input Assembly** — upload images, enter chat text, select IFC model
2. **VLM Constraint Extraction** — calls Modal endpoint, shows extracted constraints + confidence
3. **Query Planning** — displays priority cascade, highlights winning rule
4. **Symbolic Retrieval** — Neo4j Cypher execution, candidate pool with timing
5. **Result & Explainability** — 3D viewer + graph viz + saliency

### Explainability Features

| Feature | Implementation |
|---|---|
| **Occlusion saliency** | Mask N×N image patches, measure prediction degradation, render heatmap |
| **Query plan cascade** | Graphviz DOT showing which rules fire and why |
| **1-hop subgraph** | Neo4j neighborhood around retrieved elements |
| **Whole-IFC snapshot** | Bubble plot of all elements, candidates highlighted |
| **GT comparison** | Side-by-side with ground truth from loaded trace |
| **3D BIM viewer** | Three.js iframe with highlighted elements (green=correct, red=wrong, blue=GT) |

### Modal Inference Endpoint

```python
# predict — standard inference
import modal
f = modal.Function.from_name("mscd-vlm-lora3-inference", "LoRA3Predictor.predict")
result = f.remote(image_bytes_list=[...], chat_text="...", metadata_text="...")
# → {"raw_output": str, "parsed": dict, "valid_json": bool}

# explain — occlusion saliency
f = modal.Function.from_name("mscd-vlm-lora3-inference", "LoRA3Predictor.explain")
result = f.remote(image_bytes_list=[...], chat_text="...", grid_size=4)
# → {"baseline": {...}, "heatmaps": [[[float]]], "image_sizes": [...], ...}
```

---

## Training

### LoRA_3 Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen2.5-VL-7B-Instruct` (4-bit quantized) |
| LoRA rank / alpha | 16 / 32 |
| Dropout | 0.1 |
| Epochs | 5 |
| Effective batch size | 16 (2 × 8 grad accum) |
| Learning rate | 2e-4 (cosine schedule) |
| Max seq length | 4,096 |
| Training samples | 1,377 (synth_v0.5) |
| Test samples | 69 |
| Hardware | Modal A100 (40 GB) |
| Optimizer | AdamW 8-bit |
| Tracking | Wandb (`mscd-vlm-lora` project) |

### Training Data (synth_v0.5)

3-tier labeling strategy:

| Tier | Description | Samples | spatial_relations |
|---|---|---|---|
| 1 | Topology cases (FILLS, ADJACENT_TO, CONTINUOUS) | ~400 | Populated with GT triplets |
| 2 | Attribute-only (v0.4 cases) | ~900 | Empty [] |
| 3 | New cross-IFC topology | ~77 | Populated |

~60/40 attribute/topology split prevents hallucination of spatial relations.

### Skin Generation Pipeline

```bash
cd /root/cmu/master_thesis/data_curation

# 1. Render wireframes (subject=blue, anchor=orange)
python scripts/synth/3a_render_relation_crops.py

# 2. Generate text + photoreal site photos + LLM-as-Judge
python scripts/synth/3b_generate_skin.py

# 3. Assemble into ChatML training format
python scripts/synth/3c_assemble.py
```

---

## Results

### LoRA_3 VLM Extraction (69 test samples)

| Metric | Score | Notes |
|---|---|---|
| JSON parse rate | **100%** (69/69) | — |
| IFC class accuracy | **98.6%** (68/69) | — |
| Storey accuracy | **87.0%** (60/69) | Weakest field; minimal impact on P0 |
| Spatial predicate accuracy | **93.0%** (40/43) | Target was 60% |
| False positive rate | **0%** (0/26) | No hallucinated spatial relations |

Per-predicate: FILLS 24/24 (100%), CONTINUOUS 3/3 (100%), ADJACENT_TO 13/16 (81%).

### H2 Hard-Negative Eval (213 topology cases)

**Current (Neo4j edges partially loaded):**

| Predicate | Cases | GT-in-Pool | Notes |
|---|---|---|---|
| ADJACENT_TO | 66 | 0/66 (0%) | Missing ADJACENT_TO edges in Neo4j |
| FILLS | 84 | 0/84 (0%) | Missing FILLS edges in Neo4j |
| CONTINUOUS | 63 | 63/63 (100%) | Property-based, works end-to-end |
| **Total** | **213** | **63/213 (30%)** | **Blocked by graph incompleteness** |

Root cause: `neo4j_init.sh` topology enrichment did not load ADJACENT_TO/FILLS edges. See [§7.8.3](README/0224_demo_plan.md) for full analysis.

**P2 unit tests (edges manually loaded, 83 cases):**

| Predicate | Cases | GT-in-Pool | Avg Pool | SSR | Attr Baseline |
|---|---|---|---|---|---|
| ADJACENT_TO | 34 | 34/34 (100%) | 32 | 30% | 5.2% |
| CONTINUOUS | 21 | 21/21 (100%) | 74 | 65% | 0.4% |
| FILLS | 28 | 28/28 (100%) | 43 | 0% | N/A |

**Projected after graph fix**: 213/213 GT-in-pool (100%), 75–90% SSR, Top-1 ~57% at threshold=0.7.

### Comparison: LoRA_2 vs LoRA_3

| Capability | LoRA_2 | LoRA_3 |
|---|---|---|
| Training data | synth_v0.4 (933) | synth_v0.5 (1,377) |
| Output schema | 6 flat fields | 5 fields + spatial_relations[] |
| Max priority | P4 (storey+type) | **P0 (spatial_triplet)** |
| Spatial extraction | None | FILLS / ADJACENT_TO / CONTINUOUS |
| Confidence | Static 0.85 | Dynamic (VLM output, threshold 0.7) |

---

## Project Structure

```
mscd_demo/
├── config.yaml                     # Runtime config (IFC path, Neo4j, LLM)
├── profiles.yaml                   # Experiment profiles
│
├── src/
│   ├── ifc_engine.py               # IFC gateway + Neo4j export
│   ├── v2/
│   │   ├── types.py                # Constraints, SpatialTriplet, QueryPlan
│   │   ├── constraints_extractor_lora.py   # LoRA_3 inference
│   │   ├── constraints_extractor_prompt_only.py
│   │   ├── constraints_to_query.py # Priority 0–8 rule planner
│   │   ├── retrieval_backend.py    # Neo4j / Memory / CLIP
│   │   ├── pipeline.py             # V2 orchestration
│   │   └── metrics_v2.py
│   ├── eval/                       # Evaluation framework
│   ├── visual/                     # Image parser + CLIP aligner
│   ├── common/                     # Config, GUID extraction
│   ├── rq2_schema/                 # CORENET-X validation
│   └── handoff/                    # BCF 2.1 generation
│
├── training/
│   ├── train.py                    # Modal LoRA training
│   ├── inference.py                # Modal endpoint (predict + explain)
│   └── eval.py                     # Post-training evaluation
│
├── eval/
│   ├── h2_eval.py                  # H2 hard-negative eval harness
│   └── results/                    # JSONL + plot outputs
│
├── demo/
│   ├── app.py                      # Streamlit entry point
│   ├── ui/
│   │   ├── sidebar.py              # Run/case selector
│   │   ├── tab_context.py          # Input visualization
│   │   ├── tab_pipeline.py         # Pipeline trace
│   │   ├── tab_result.py           # 3D viewer + STEP text
│   │   └── tab_inference.py        # Live inference (5-stage + explainability)
│   ├── static/                     # viewer.bundle.js + web-ifc.wasm
│   └── templates/                  # Iframe HTML templates
│
├── mcp_servers/                    # V1 MCP tool servers (ifc + visual)
├── script/
│   ├── run.py                      # Unified evaluation runner
│   ├── neo4j_init.sh               # Neo4j setup + IFC graph load
│   ├── generate_plots.py           # Plot generation
│   └── compare_results.py          # Cross-experiment comparison
│
├── data/ifc/AdvancedProject/       # Primary BIM model (10 storeys, 1233 elements)
├── schemas/corenetx_min/           # Regulatory schemas
├── test/                           # Unit tests
└── legacy/                         # Archived superseded code
```

---

## Neo4j Setup

```bash
# Automated setup (recommended)
./script/neo4j_init.sh

# Manual start (if already installed)
/tmp/neo4j-community-5.26.0/bin/neo4j start

# Verify
cypher-shell -u neo4j -p password "MATCH (n) RETURN count(n)"
# Expected: ~1233 nodes

# Browse graph
# http://localhost:7474
```

Config in `config.yaml`:
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  enabled: true
```

---

## Running Tests

```bash
conda activate mscd_demo

# Unit tests
python -m pytest test/ -v

# H2 topology eval
python eval/h2_eval.py --output eval/results/h2_eval.jsonl --plot eval/results/h2_eval_figure.png

# Generate evaluation plots
./training/eval.sh --step update-plots
```

---

## Tech Stack

| Component | Technology |
|---|---|
| IFC Processing | IfcOpenShell 0.8+ |
| Graph Database | Neo4j Community 5.26 |
| VLM (LoRA_3) | Qwen2.5-VL-7B-Instruct + Unsloth LoRA |
| GPU Inference | Modal (A100 serverless) |
| LLM (prompt) | Google Gemini 2.5 Flash |
| Visual Reranking | OpenAI CLIP `vit-base-patch32` |
| Data Models | Pydantic v2 |
| Demo UI | Streamlit |
| 3D Viewer | Three.js + @thatopen/components |
| Graph Viz | Graphviz (DOT) |
| Saliency | Matplotlib (occlusion heatmap) |
| Schema Validation | jsonschema (Draft 2020-12) |
| Agent Framework | LangChain + LangGraph (V1 only) |
| BCF Output | Python stdlib (BCF 2.1) |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named 'common'` | Ensure `src/` is on sys.path (app.py handles this) |
| Neo4j connection refused | Run `./script/neo4j_init.sh` or start manually |
| Modal endpoint cold start | First call takes ~60s (model loading), subsequent calls ~5s |
| "No elements found for storey" | Use exact BIM storey names: `"6 - Sixth Floor"` not `"Level 6"` |
| Priority 0 not firing | Check confidence >= 0.7 and Neo4j is running |
| FILLS returning too many results | Expected — FILLS pool = all windows on storey; reranking disambiguates |
| `IfcWall` not matching | IFC uses `IfcWallStandardCase` — Cypher uses `STARTS WITH` |

---

**Last Updated:** March 2026
**Status:** LoRA_3 pipeline operational. synth_v0.5 (1,377 train / 69 test, 5 epochs). H2 eval: 83/83 GT-in-pool, 0 fallbacks. Modal endpoint: `mscd-vlm-lora3-inference` with predict + explain. Live demo: 5-stage pipeline + occlusion saliency + graph explainability.
