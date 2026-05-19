# MSCD: AI Interpreter for Construction Site Data

MSCD tries to link messy site evidence to the correct IFC/BIM element.
It takes floorplans, site photos, chat messages, and task metadata, turns them into structured constraints, and then searches the IFC model with a rule-based retrieval pipeline.

Full results, ablations, and threats-to-validity are in [`results.md`](results.md).

## Latest Status

Two complementary headline results, both `p0_union_p1`:

**Unified cross-model eval (n=116, AP+BH+DXA, condition=FP)** — last updated 2026-03-26:

| System | GT-in-Pool | Top-10 | Top-1 | Avg Pool |
|--------|-----------|--------|-------|----------|
| **LoRA5-r32** | **53.4%** (62/116) | 20.7% | 4.3% | 73 |
| LoRA5-r16 | 52.6% | 24.1% | 4.3% | 71 |
| Gemini | 50.9% | **25.9%** | 4.3% | 81 |
| LoRA2 | 36.2% | 21.6% | 2.6% | 80 |

**AP held-out deep dive (n=60, LoRA6 G-series on enriched/phase-5 graph)** — G0–G8 last updated 2026-04-08; G9 size-band + rerank extensions added 2026-04-29:

| System | Configuration | GT-Pool | Top-10 | Top-1 | MRR@10 |
|--------|---|--------:|-------:|------:|-------:|
| **G8 PosCtx+Dim** (best) | enriched graph + dim labels | **100%** | **30.0%** | **6.7%** | **0.1104** |
| G7 PosCtx | enriched graph + position_context | 100% | 26.7% | 6.7% | 0.1015 |
| Gemini v2 | zero-shot, v2_lora profile | 91.7% | 18.3% | 1.7% | 0.0557 |
| **G9 (final extended)** | enriched + ResNet size-band + Graph-RAG rerank | 98.3% | 26.7% | 3.3% | 0.0920 |

Main takeaways:

- The symbolic retrieval layer is sound — Oracle achieves 100% GT-in-pool.
- The main bottleneck is VLM extraction, especially `ifc_class`.
- Spatial relations compress the pool but, for LoRA5, don't beat `storey + type` for GT recovery (P0 ⊂ P1).
- LoRA6 / **G8** reaches 100% GT-in-pool on AP — closing the gap to oracle (pool 76 → 9) requires structured `object_subtype`, `distance_mm`, `connection_degree`, `wall_position_index`.
- **G9** is the final extended architecture (G8 + ResNet size-band classifier + Graph-RAG reranker). Useful negative result: it does **not** beat G8 — Top-1 drops 6.7% → 3.3% and MRR drops 0.110 → 0.092. A downstream visual size classifier and a Gemini reranker are not substitutes for emitting those fingerprint fields at constraint time. See [`results.md §G6`](results.md#g6-g9-extended-architecture).
- Asking the user for the element type is still the simplest high-impact UX lever (+23pp GT-in-pool).

Full breakdown: [`results.md`](results.md).

## Current Implementation

```text
floorplan / site photo / chat / 4D metadata
        -> LoRA constraint extraction
        -> structured JSON constraints
        -> query planner (P0-P8)
        -> Neo4j retrieval
        -> candidate list, traces, and demo views
```

### 1. Constraint Extraction

- Main runtime extractor: [`src/neurosym/constraints_extractor_lora.py`](src/neurosym/constraints_extractor_lora.py)
- Prompt-only fallback: [`src/neurosym/constraints_extractor_prompt_only.py`](src/neurosym/constraints_extractor_prompt_only.py)
- Latest training script: [`training/train_lora6.py`](training/train_lora6.py) (G-series; supersedes `train_lora5.py`)
- Base model family: `Qwen2.5-VL-7B + LoRA`
- Main output fields: `storey_name`, `ifc_class`, `space_name`, `target_name_keyword`, `target_width_mm`, `target_height_mm`, `position_context`, `spatial_relations`

### 2. Query Planning

- Main planner: [`src/neurosym/constraints_to_query.py`](src/neurosym/constraints_to_query.py)
- The planner builds fixed priority rules from `P0` to `P8`.
- The best retrieval setting (`p0_union_p1`) is handled in [`src/neurosym/retrieval_backend.py`](src/neurosym/retrieval_backend.py). In plain terms:
  - run spatial retrieval first;
  - also keep the `storey + type` pool as a safety net.

### 3. Retrieval and Pipeline

- Main pipeline: [`src/neurosym/pipeline.py`](src/neurosym/pipeline.py)
- Local evaluation runner: [`script/run.py`](script/run.py)
- Retrieval backend: [`src/neurosym/retrieval_backend.py`](src/neurosym/retrieval_backend.py)
- Graph-RAG reranker: [`src/neurosym/graph_rag_rerank.py`](src/neurosym/graph_rag_rerank.py) (use only on P1-only / coarse pools)
- Neo4j is the main retrieval backend for the current thesis result.

### 4. Evaluation and Demo

- Unified evaluation extraction: [`evaluation/inference/eval_unified.py`](evaluation/inference/eval_unified.py)
- Unified batch script: [`evaluation/inference/eval_unified.sh`](evaluation/inference/eval_unified.sh)
- AP held-out builder: [`evaluation/build_ap_heldout_e2e_cases.py`](evaluation/build_ap_heldout_e2e_cases.py)
- Oracle waterfall: [`evaluation/oracle_waterfall.py`](evaluation/oracle_waterfall.py), [`evaluation/oracle_ap_heldout.py`](evaluation/oracle_ap_heldout.py)
- Plot suites: [`evaluation/analysis/generate_final_plot_suite.py`](evaluation/analysis/generate_final_plot_suite.py), [`evaluation/analysis/generate_phase4_plot_suite.py`](evaluation/analysis/generate_phase4_plot_suite.py)
- Streamlit demo: [`demo/app.py`](demo/app.py)

## Repo Map

| Path | What it is for now |
| --- | --- |
| [`src/neurosym/`](src/neurosym/) | Main neuro-symbolic pipeline (extractor, planner, retrieval, reranker) |
| [`src/common/`](src/common/) | Shared utilities (config, guid, evaluation, trace I/O, MCP) |
| [`src/evaluation_infra/`](src/evaluation_infra/) | Pydantic contracts, metrics, runner, visualisations |
| [`src/handoff/`](src/handoff/) | BCF issue handoff (RQ2 deliverable) |
| [`src/rq2_schema/`](src/rq2_schema/) | RQ2 compliance schema validation |
| [`src/visual/`](src/visual/) | CLIP-based visual aligner + image parser |
| [`mcp_servers/`](mcp_servers/) | IFC and visual MCP services |
| [`training/`](training/) | LoRA training (current: `train_lora6.py`) |
| [`evaluation/`](evaluation/) | Cases, inference, baselines, analysis, oracle |
| [`demo/`](demo/) | Streamlit app for saved traces and live pipeline inspection |
| [`output/`](output/) | Saved outputs from evaluation runs (e.g. `ap_e2e_phase5_g8/`, `lora6_v2_ap_20260331/`) |
| [`plots/`](plots/), [`docs/plots/`](docs/plots/) | Key experiment plots used in the thesis |
| [`results.md`](results.md) | Consolidated results (Track U unified + Track G AP held-out) |

> Earlier README revisions called the main pipeline `src/v2/`. The current location is `src/neurosym/` — same code, renamed.

## Quick Start

### 1. Install

```bash
conda activate mscd_demo
pip install -r requirements.txt
```

### 2. Add API Keys

Create `mscd_demo/.env` with whatever you need:

```bash
GOOGLE_API_KEY=your_key_here
```

If you want to run Modal jobs, make sure Modal is already set up on your machine.

### 3. Start Neo4j

```bash
./script/neo4j_init.sh
```

Neo4j browser (WSL2 note):
1. If `http://localhost:7474` doesn't open in Windows, get the WSL2 IP with `hostname -I | awk '{print $1}'` and open `http://<that-ip>:7474` instead.
2. Username `neo4j`, password `password` (default).
3. Connects to `bolt://localhost:7687`.

### 4. Run the Demo

```bash
streamlit run demo/app.py
```

### 5. Run the Latest Unified Evaluation

```bash
bash evaluation/inference/eval_unified.sh
```

This batch script assumes the LoRA adapters are already uploaded to the Modal volume — see the comments at the top of [`evaluation/inference/eval_unified.sh`](evaluation/inference/eval_unified.sh).

### 6. Replay the Best Local Retrieval Setting

```bash
python script/run.py \
  --profile v2_lora \
  --cases evaluation/cases/cases_unified_test.jsonl \
  --precomputed logs/evaluation_output/unified/eval_constraints_lora5r32_FP.jsonl \
  --p0-strategy p0_union_p1
```

For the AP held-out G-series eval:

```bash
python script/run.py \
  --profile v2_lora \
  --cases evaluation/cases/cases_ap_heldout_e2e.jsonl \
  --precomputed logs/evaluation_output/ap_e2e_phase5_g8/g8_posctx_dim__ap_eval.jsonl \
  --p0-strategy p0_union_p1
```

### 7. Train the Latest LoRA Model

```bash
modal run training/train_lora6.py --epochs 5 --lr 2e-4 --lora-r 32
```

## System Architecture

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OFFLINE: IFC -> Neo4j Graph  (src/ifc_engine.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 IFC File -> Nodes (IFCElement, IFCStorey, IFCSpace)
   .guid  .ifc_type  .storey  .material  .object_type
   .width_mm  .height_mm                          (Fix 3: IfcWindow/Door)
   .wall_position_index  .wall_child_total

 Edges:
   -[:CONTAINS]->                  storey/space containment
   -[:FILLS]->                     Door/Window -> host Wall
   -[:NEXT_TO {wall_guid, pos}]->  consecutive fillers on wall
   -[:CONNECTS_TO {conn_type}]->   wall T/L/X junctions (IFC rel)
   -[:ADJACENT_TO {distance_mm}]-> centroid dist 100-1500mm   (Fix 4)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OFFLINE: Training Labels (../data_curation/scripts/synth/6_assemble_lora6.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 skeleton + skins + element_index + IFC dims
   ->  Constraints JSON label per case:
         storey_name, ifc_class
         target_width_mm, target_height_mm        (Fix NEW)
         position_context (text)                  (Fix 1/2)
         spatial_relations[]:
           predicate, object_type, object_subtype
           object_material, direction
           connection_degree (int)                (Fix 2)
           distance_mm (float)                    (Fix 4)
           confidence: 1.0 (schema) / 0.75 (geometry)  (Fix 3)
   ->  filter_label_for_evidence()
         drops fields not observable at given crop scale
   ->  LoRA fine-tuning (G8, Qwen2.5-VL-7B, r=32)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ONLINE: Inference (per query)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Query: text + site_photo(s) + floorplan
      |
      v
 +------------------------------------------------------+
 |  NEURO LAYER - LoRA VLM (G8)                         |
 |  Output: Constraints JSON                            |
 |    ifc_class, storey_name                            |
 |    target_width_mm, target_height_mm                 |
 |    position_context (text)                           |
 |    spatial_relations[{predicate, object_type,        |
 |      object_subtype, object_material, direction,     |
 |      connection_degree, distance_mm, confidence}]    |
 +------------------------------------------------------+
      |
      v
 +------------------------------------------------------+
 |  QUERY PLANNER - constraints_to_query.py             |
 |                                                      |
 |  fingerprint_level_requested:                        |
 |    "exact_slot"           position_index set         |
 |    "relation_fingerprint" subtype/direction set      |
 |    "topology_only"        predicate + obj_type only  |
 |    "attribute_only"       no spatial_relations       |
 |                                                      |
 |  Priority 0a: spatial_triplet  (FILLS / NEXT_TO /    |
 |               CONNECTS_TO / ADJACENT_TO traversal)   |
 |  Priority 0b: continuous_span  (property filter)     |
 |  Priority 1:  space + type                           |
 |  Priority 2:  name_keyword                           |
 |  Priority 4:  storey + type                          |
 |  Priority 6:  type_only                              |
 |  Priority 8:  fallback                               |
 +------------------------------------------------------+
      |  QueryPlan
      v
 +------------------------------------------------------+
 |  SYMBOLIC LAYER - Neo4j Cypher                       |
 |  (retrieval_backend.py _execute_neo4j)               |
 |                                                      |
 |  Single-hop (1 SR):                                  |
 |    MATCH (target)-[:PRED]->(ref)                     |
 |    WHERE ifc_type, storey_list,                      |
 |          material, width/height_mm +/- 50mm          |
 |                                                      |
 |  Multi-anchor (2+ SR): _execute_multi_anchor()       |
 |    FILLS  -> MATCH promoted (wall-pin)               |
 |      + object_subtype CONTAINS          (Fix 1)      |
 |    NEXT_TO -> MATCH (wall_guid pin)                  |
 |      + direction, object_subtype                     |
 |    CONNECTS_TO -> WHERE EXISTS                       |
 |      + COUNT{(t)-[:CONNECTS_TO]-()} = N (Fix 2)      |
 |    ADJACENT_TO -> WHERE EXISTS                       |
 |      + r.distance_mm +/- 200mm          (Fix 4)      |
 |    target.wall_position_index = K  (exact slot)      |
 |    target.width/height_mm +/- 50mm     (Fix 3)       |
 |                                                      |
 |  Relaxation ladder (on empty):                       |
 |    drop exact_slot -> drop fingerprint               |
 |    -> drop storey -> drop weakest SR                 |
 |    -> fallback storey + type (P4)                    |
 +------------------------------------------------------+
      |  P0 candidate pool
      |
      v  p0_union_p1:  P0  U  storey+type pool
         (P0 elements ranked first, P1-only appended)
      |
      v  _post_filter_by_name_keyword()  (Python, graceful)
      |
      |  Shortlist (top-K candidates, default K=10)
      v
 +------------------------------------------------------+
 |  GRAPH-RAG RERANKER (graph_rag_rerank.py)            |
 |                                                      |
 |  For each candidate -> query Neo4j for context:      |
 |    host wall, NEXT_TO neighbours (left/right),       |
 |    CONNECTS_TO walls, ADJACENT_TO elements,          |
 |    wall slot position (pos / total)                  |
 |                                                      |
 |  Format as text description per candidate letter     |
 |  (A. IfcWindow on Level 1; host: Brick wall;         |
 |   position 3 of 14; left: IfcDoor ...)               |
 |                                                      |
 |  Gemini VLM prompt:                                  |
 |    site_photo + floorplan + query_text               |
 |    + candidate descriptions                          |
 |    -> ranked letter sequence  "C A B D ..."          |
 |                                                      |
 |  Reorder shortlist by Gemini ranking                 |
 +------------------------------------------------------+
      |
      v
 Final ranked pool -> Top-1 answer
```

> **Graph-RAG caveat:** only helpful on P1-only / coarse pools (+8.3pp Top-1). On already topology-filtered pools it *degrades* Top-1 by ~5pp — see [`results.md §G3`](results.md#g3-graph-rag-reranker).

## Useful Files

- Consolidated results: [`results.md`](results.md)
- Unified test cases: [`evaluation/cases/cases_unified_test.jsonl`](evaluation/cases/cases_unified_test.jsonl)
- AP held-out cases: [`evaluation/cases/cases_ap_heldout_e2e.jsonl`](evaluation/cases/cases_ap_heldout_e2e.jsonl)
- Main evaluation runner: [`script/run.py`](script/run.py)
- Current training script: [`training/train_lora6.py`](training/train_lora6.py)
