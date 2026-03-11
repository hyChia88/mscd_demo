> **2026-03-07 → Post-Mid Review Plan**
> Updated: 2026-03-11

---

## Status Snapshot

| Component | Status | Blocker |
|---|---|---|
| LoRA_3 VLM (93% spatial acc, 0% FP) | Done | — |
| Neo4j CONTINUOUS predicate | Works e2e | — |
| Neo4j FILLS / ADJACENT_TO edges | **Broken** | `neo4j_init.sh` edge loading |
| H2 eval (213 cases) | 30% GT-in-pool | Blocked on edges |
| Live demo (Streamlit 5-stage) | Working | — |
| Occlusion saliency explain endpoint | Deployed | — |
| `object_material` in P0 Cypher | **Not wired** | Neo4j nodes lack material property |
| Nested/compound spatial queries | **Not implemented** | Design needed |

---

## P4 — Fix Neo4j Graph Completeness (CRITICAL)

**Problem**: `neo4j_init.sh` does not load ADJACENT_TO or FILLS edges into Neo4j.
CONTINUOUS works because it uses node properties (`is_continuous`, `top_constraint`), not edges.
Result: 0/66 ADJACENT_TO and 0/84 FILLS in H2 eval → 30% overall GT-in-pool.

**Root cause**: See `0224_demo_plan.md` §7.8.3.

### Tasks

```
P4.1  Fix neo4j_init.sh to call _create_element_relationships()     — CRITICAL
      after node creation. Verify: MATCH ()-[r:FILLS]->() RETURN count(r)
      Expected: 389 FILLS, ~200 ADJACENT_TO edges.

P4.2  Add edge-count pre-check to h2_eval.py                        — HIGH
      Fail-fast: if FILLS < 100 or ADJACENT_TO < 50, abort with
      "Graph incomplete — run neo4j_init.sh first."

P4.3  Fix continuous_span storey filter                              — HIGH
      Current pool = 74 (worse than storey+type = 46).
      Root: CONTINUOUS Cypher doesn't filter by base storey.

P4.4  Re-run H2 eval with complete graph                            — Blocked on P4.1
      Expected: 213/213 GT-in-pool, ~75-90% SSR.

P4.5  Generate thesis plots with corrected data                      — Blocked on P4.4
```

---

## P5 — Property-Enriched P0 Queries (HIGH)

**Problem**: P0 spatial_triplet Cypher ignores `object_material` even though the VLM
extracts it and the planner passes it to params. FILLS pool = 43 (all windows on a floor
fill walls) — material could cut this to ~5-10.

**Data reality check** (from `element_index.jsonl`):
- Wall materials: Leather/weathered=~16, Interior Wall A=~100, Concrete=~16, Brick=~16, Generic=380+
- Material IS discriminating — 16 concrete walls vs 46 total windows per floor.
- `psets.Materials and Finishes.Structural Material` exists on every wall element.
- Also available: `is_external`, `Construction.Function`, `Construction.Width`.

### Tasks

```
P5.1  Add `material` property to Neo4j nodes                        — HIGH
      In ifc_engine.py _create_element_nodes():
      Extract from psets["Materials and Finishes"]["Structural Material"]
      → node_props["material"] = material_value

P5.2  Add object_material WHERE clause to P0 Cypher                 — HIGH
      In retrieval_backend.py _execute_neo4j() spatial_triplet branch:
      AND ($object_material = '' OR toLower(ref.material) CONTAINS toLower($object_material))
      Graceful: empty object_material = no filter (backward compatible).

P5.3  Post-filter P0 results with other constraint fields           — MEDIUM
      After P0 Cypher returns candidates, intersect with:
      - space_name (if non-null): filter candidates by room containment
      - target_name_keyword (if non-null): filter by name substring
      - is_external (if extractable): facade vs interior
      Graceful: if post-filter removes all candidates, return unfiltered.

P5.4  Evaluate improvement                                          — Blocked on P4.1 + P5.1-2
      Re-run H2 eval, measure:
      - Pool size reduction: FILLS 43→? ADJACENT_TO 32→?
      - New SSR per predicate
      - Any GT dropped by material filter? (should be 0 if material is correct)
```

**Estimated impact** (FILLS predicate):
```
Current:    Window FILLS IfcWall + storey           → pool 43
+ material: Window FILLS IfcWall(Concrete) + storey → pool ~5-10
+ space:    Window FILLS IfcWall(Concrete) in Kitchen → pool ~1-3
```

---

## P6 — Nested/Compound Spatial Queries (MEDIUM-HIGH)

**Problem**: Current SpatialTriplet is flat — one hop only. But IFC topology is inherently
nested. A window fills a wall, and that wall is adjacent to a column. Using only the first
hop leaves significant discriminating information on the table.

### Analysis: Topology Depth in IFC

```
Level 0 (current):  Window ──[FILLS]──> Wall                     pool: 43
Level 1 (props):    Window ──[FILLS]──> Wall(material=Concrete)   pool: ~5-10
Level 2 (2-hop):    Window ──[FILLS]──> Wall ──[ADJACENT_TO]──> Column   pool: ~1-2
```

The data supports this — `query_adjacent_elements()` in ifc_engine.py already uses
`*1..2` variable-length patterns. The skeleton miner has `RELATIONAL_MULTI_HOP` as a
pattern type. The VLM `spatial_relations` field is already a **list** (supports multiple
triplets), but training data only has single-triplet examples.

### Design: Compound SpatialTriplet

**Option A — Multiple independent triplets (intersect results)**:
```json
{
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWall", "object_material": "Concrete", "confidence": 0.92},
    {"predicate": "ADJACENT_TO", "object_type": "IfcColumn", "confidence": 0.78}
  ]
}
```
Execution: Run P0 for triplet[0], then P0 for triplet[1], return intersection.
Pro: No Cypher change, just set intersection in Python.
Con: Subject of triplet[1] is ambiguous (is it the window or the wall?).

**Option B — Chained triplets (2-hop Cypher)**:
```json
{
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWall", "object_material": "Concrete",
     "chain": {"predicate": "ADJACENT_TO", "object_type": "IfcColumn"}}
  ]
}
```
Execution: Single Cypher with variable-length path.
```cypher
MATCH (target:IFCElement)-[:FILLS]->(ref1:IFCElement)-[:ADJACENT_TO]->(ref2:IFCElement)
WHERE target.ifc_type STARTS WITH 'IfcWindow'
  AND ref1.ifc_type STARTS WITH 'IfcWall'
  AND ref1.material CONTAINS 'Concrete'
  AND ref2.ifc_type STARTS WITH 'IfcColumn'
RETURN DISTINCT target
```
Pro: Most precise, single query.
Con: Requires schema change + new training data.

**Recommended approach**: Start with Option A (intersect), upgrade to B later.

### Tasks

```
P6.1  Mine 2-hop skeleton training data                              — HIGH
      Extend 2_hunt_skeletons.py to emit multi-triplet examples:
      For each FILLS skeleton, check if the host wall has ADJACENT_TO
      neighbors → emit compound skeleton with 2 triplets.
      Target: 50+ compound examples.

P6.2  Add compound triplet training examples to synth_v0.6           — HIGH
      3c_assemble.py: support multi-element spatial_relations.
      Generate text skins referencing both relations:
      "The window in the concrete wall, next to the column"

P6.3  Implement intersection execution in retrieval_backend.py       — MEDIUM
      If len(spatial_relations) > 1:
        pool_0 = execute P0 for triplet[0]
        pool_1 = execute P0 for triplet[1]
        candidates = pool_0 ∩ pool_1 (by guid)
        if empty: fall back to pool_0 only

P6.4  (Future) Implement chained Cypher for Option B                 — LOW
      Only if Option A intersection is insufficient.
```

---

## P7 — Visual Grounding: Scene Graph Visualization (MEDIUM)

**From mid-review feedback**: Need a middle visualization layer showing how the VLM maps
2D image regions to 3D spatial relationships. Currently the pipeline is a black box between
"input image" and "SpatialTriplet JSON output."

### Idea

Add a **Scene Graph (SG) visualization** layer between the VLM output and the Cypher query:

```
Site Photo ──► VLM ──► SpatialTriplet JSON ──► Scene Graph Viz ──► Neo4j Cypher
                                                     │
                                              ┌──────▼──────┐
                                              │  2D overlay: │
                                              │  bbox + edge │
                                              │  labels      │
                                              └─────────────┘
```

This addresses the reviewers' question: "How do we know the VLM actually understands
spatial relationships, not just pattern-matching text?"

### Options

| Approach | Input | Output | Effort |
|---|---|---|---|
| **Bounding box overlay** | VLM extracts bbox coords for subject + object | 2D image with labeled boxes + edge arrow | Medium — needs bbox in VLM output |
| **Scene Graph diagram** | SpatialTriplet JSON | Graphviz node-edge diagram (like 1-hop subgraph) | Low — already have Graphviz in demo |
| **Attention/saliency overlay** | Occlusion heatmap from explain() | Heatmap on image showing spatial focus regions | Done — already implemented |

### Tasks

```
P7.1  Add Scene Graph Graphviz rendering to demo                     — LOW
      Already have graph viz infrastructure. Render SpatialTriplet as:
      [IfcWindow] ──FILLS──> [IfcWall: Concrete]
      Show in the pipeline stage 2 output section.

P7.2  (Stretch) Add bbox extraction to VLM output schema             — HIGH
      Extend LoRA_3 output to include bounding boxes:
      {"predicate": "FILLS", "object_type": "IfcWall",
       "subject_bbox": [x1,y1,x2,y2], "object_bbox": [x1,y1,x2,y2]}
      Requires new training data with bbox annotations.
      Consider: Qwen2.5-VL supports grounding natively — may just need
      prompt engineering, not LoRA changes.

P7.3  Overlay bboxes + edge labels on site photo in demo             — MEDIUM
      Blocked on P7.2. Render PIL image with:
      - Blue box around subject (window)
      - Orange box around reference (wall)
      - Arrow + predicate label between them
```

---

## P8 — synth_v0.5 Dataset Completion (HIGH)

**Current state**: synth_v0.5 has 1,377 train / 69 test samples. But:
- Only 3 IFC models (AP, BH, DXA)
- Topology cases are ~30-40% of total
- No compound/nested triplet examples
- Relation crop renders had black-image bug (some renders are empty)

### Tasks

```
P8.1  Fix 3a_render_relation_crops.py black-image bug                — HIGH
      Some renders produce all-black PNGs. Debug matplotlib 3D
      camera positioning for edge cases (very tall walls, small doors).

P8.2  Re-render all topology skeletons                               — Blocked on P8.1

P8.3  Raise skeleton mining quotas                                   — MEDIUM
      Re-run 2_hunt_skeletons.py with:
      --max-fills 200 --max-continuous 100 --max-adjacent 200
      Target: 500+ topology skeletons (current: 126).

P8.4  Cross-IFC pipeline                                             — MEDIUM
      Run full skeleton mining + skin generation on BasicHouse + Duplex_A.
      Goal: test generalization to different IFC schemas and element vocabularies.

P8.5  Add compound triplet skeletons (P6.1 dependency)               — Blocked on P6.1
```

---

## P9 — 4D Timestamp Integration (LOW)

**From mid-review**: 4D task status provides temporal context ("Window Installation -
Level 6 - IN_PROGRESS" with timestamp). Currently used as text injection into the VLM
prompt but not structured.

### Idea

Parse 4D task status into structured temporal constraints:
```json
{
  "4d_parsed": {
    "activity": "Window Installation",
    "storey": "Level 6",
    "status": "IN_PROGRESS",
    "timestamp": "2026-03-07T14:30:00",
    "schedule_window": ["2026-03-05", "2026-03-10"]
  }
}
```

Use schedule_window to further constrain: if a task is IN_PROGRESS on Level 6, elements
on other floors are deprioritized. This is already partially captured by storey extraction
but making it explicit could help with multi-storey queries.

### Tasks

```
P9.1  Parse 4D task status string into structured fields             — LOW
P9.2  Add temporal constraint to query planner                       — LOW
      If 4D says "Level 6 IN_PROGRESS", boost P4(storey+type) confidence
      for Level 6 matches.
P9.3  Evaluate 4D-aware retrieval on synth_v0.4 MA vs MA- pairs     — LOW
      Already have paired ablation data. Check if explicit 4D parsing
      improves over raw text injection.
```

---

## Priority Order

```
CRITICAL ──► P4.1  Fix neo4j_init.sh edge loading
             P4.2  Edge count pre-check in h2_eval.py
             P4.3  Fix continuous_span storey filter
             P4.4  Re-run H2 eval (validates everything)

HIGH ────► P5.1  Add material to Neo4j nodes
             P5.2  Wire object_material into P0 Cypher
             P8.1  Fix render black-image bug
             P6.1  Mine 2-hop skeletons

MEDIUM ──► P5.3  Post-filter P0 with space_name/keyword
             P6.3  Implement intersection execution
             P8.3  Raise skeleton quotas
             P8.4  Cross-IFC pipeline
             P7.2  Bbox extraction (stretch)

LOW ─────► P7.1  Scene Graph Graphviz in demo
             P9.*  4D timestamp integration
             P6.4  Chained Cypher (future)
```

**Timeline estimate**:
- P4 (graph fix + re-eval): 1-2 days
- P5 (material enrichment): 1 day
- P6 (compound queries): 3-5 days (mining + training + execution)
- P7 (visual grounding): 1-2 days for SG viz, 1 week for bbox
- P8 (dataset): 2-3 days
- P9 (4D): 1 day

---

## Key Metrics to Track

| Metric | Current | After P4 | After P5 | After P6 |
|---|---|---|---|---|
| H2 GT-in-pool | 30% (63/213) | ~100% | ~100% | ~100% |
| FILLS avg pool | N/A (edges missing) | ~43 | **~5-10** | **~1-3** |
| ADJACENT_TO avg pool | N/A | ~32 | ~25 | ~5-10 |
| CONTINUOUS avg pool | 74 | ~46 (storey fix) | ~46 | ~46 |
| Overall SSR | -74% to 100% | 75-90% | **85-95%** | **95%+** |
| Top-1 (t=0.7) | N/A | ~57% | ~70% | **~85%+** |

---

*Adapter checkpoint: `models/adapters/v3_lora_qwen_20260310_5ep/final/` (Modal volume `/mscd-lora/final`)*
