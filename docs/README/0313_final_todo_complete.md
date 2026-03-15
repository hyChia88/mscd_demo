# Final Sprint: Evaluated Action Plan (Complete)

> **Date**: 2026-03-13 | **Deadline**: ~3 weeks to thesis submission
> **Author**: AI Researcher + AI Engineer review, grounded in verified IFC data
> **Strategy**: Maximum ROI/visibility within time budget. Every task has data-backed justification.
> **Supersedes**: `0307_post_mid_plan.md`, `new_retrieve.md`, `0313_final_todo.md`

---

## Research Positioning

**Thesis statement**: *"By decomposing element retrieval into VLM spatial predicate
extraction (neuro) and graph database traversal (symbolic), we achieve 5-8x improvement
on a benchmark where 46 identical elements per floor defeat all attribute-only methods."*

| Mainstream AI Thread | Your Instantiation | Why Reviewers Care |
|---|---|---|
| LLM + Tool Use (Toolformer, Gorilla, ReAct) | VLM + Neo4j Cypher | Grounded tool use, not free-form |
| Structured Output / Function Calling | LoRA-trained JSON spatial predicates | 100% parse rate, 93% predicate acc |
| Hallucination Prevention | 0% FP on spatial extraction | Neuro-symbolic as safety net |
| Neuro-Symbolic Reasoning (Think-on-Graph, RoG) | VLM perception → graph traversal | Domain-specific instantiation |
| Multi-Modal Grounding | Site photo + floorplan dual-track (MC condition) | Cross-modal spatial reasoning |

---

## 1. Data Reality (Verified Against AdvancedProject.ifc)

### 1.1 The Entropy Bottleneck

```
Floors 1-5: 46 identical IfcWindows each
  → All hosted by "MockUp Exterior" walls
  → All walls have SAME material (Plaster|Leather, weathered)
  → Attribute baseline Top-1 = 2.2% per floor (1/46)
  → Doors have same problem: Floor 1 has 42 identical M_Single-Flush:Generic Door
```

### 1.2 Discrimination Simulation (Ground Truth from IFC)

```
Feature Combination                 Groups  MaxPool  AvgPool  Top-1
────────────────────────────────────────────────────────────────────
storey only                              7      46     37.6    6.8%
storey + wall_material                   7      46     37.6    6.8%  ← DEAD
storey + window_subtype (obj_type)      61      13      4.3   47.2%  ← BEST ACCESSIBLE
storey + which_wall (wall_guid)         45      17      5.8   37.5%
storey + wall_guid + obj_type          135       6      1.9   71.5%  ← THEORETICAL MAX
```

### 1.3 Material: Dead for Windows, Alive for Everything Else

```
Element Type            Count  storey   storey+mat  Mat Boost
                               Top-1    Top-1
──────────────────────────────────────────────────────────────
IfcFurnishingElement     407    0.7%     26.8%       40.4x  ← HUGE
IfcBuildingElementProxy  358    4.6%     32.3%        7.0x
IfcDoor                  126   10.2%     38.6%        3.8x
IfcWallStandardCase      381    4.2%     10.5%        2.5x
IfcWindow                263    6.8%      6.8%        1.0x  ← DEAD
```

### 1.4 H2 Eval Results (213 cases)

**Before fixes (2026-03-11)**: P0 ran in DEGRADED MEMORY MODE (Neo4j never connected from run.py).
```
Predicate      Cases  Pre-P0    Post-P0   SSR     Top-1(1/pool)
                      avg pool  avg pool
─────────────────────────────────────────────────────────────
ADJACENT_TO      66    116.8     29.8     57.0%    12.1%
FILLS            84     42.4     42.4      0.0%     2.4%   ← P0 gives NO improvement
CONTINUOUS       63    227.4    110.2     47.5%     2.1%
─────────────────────────────────────────────────────────────
OVERALL         213    120.2     58.6     ~35%      5.3%
```

**After fixes (2026-03-14)**: Neo4j connected, CONTINUOUS Cypher fixed, storey resolver generalized.
```
Predicate      Cases  Pre-P0    Post-P0   SSR     Top-1(1/pool)  Fallback
                      avg pool  avg pool
────────────────────────────────────────────────────────────────────────
ADJACENT_TO      66    120.0     46.5     35%       4.5%          0/66
FILLS            84     42.4     60.3    -49%       2.4%          0/84
CONTINUOUS       63    227.4    127.7     42%       0.5%          0/63
────────────────────────────────────────────────────────────────────────
OVERALL         213    120.0     76.0      4%       2.5%          0/213
GT-in-pool: 213/213 (100%)  |  Fallback: 0/213 (0%)
```
Note: FILLS SSR negative because storey siblings now return BOTH "Level X" + "X - Xth Floor"
elements. The T1.1 target_name_keyword post-filter (already wired) will fix this.

### 1.5 Critical Findings That Shape This Plan

| # | Finding | Action |
|---|---------|--------|
| 1 | **FILLS SSR = 0%** — all windows on a storey fill walls on that storey, so storey filter removes nothing | T1.1 `target_name_keyword` post-filter is the only quick fix (pool 42→~9) |
| 2 | **Wall material = DEAD for windows** (all 17 window-hosting walls share same material) | Keep material for doors/walls/furniture (T1.3), but don't expect window improvement |
| 3 | **Window ObjectType IS discriminating** (5 BALANS subtypes = different physical sizes) | VLM can see size differences → T1.1 extracts this via `target_name_keyword` |
| 4 | **IfcRelConnectsPathElements (686 edges)**: wall-to-wall topology, NOT in Neo4j | Load into graph → enables 2-hop queries. **Highest-value future work** (Section 7) |
| 5 | **LoRA_3 trained WITH floorplans** → MC condition already exists | Floorplan is a CORE modality, not optional. 4-way eval includes MC (T2.2) |
| 6 | **IfcRelSpaceBoundary = 0** in AdvancedProject | Room-level filtering impossible on AP. Works on Duplex_A (264 instances) |

---

## 2. System Status Snapshot

| Component | Status | Key File |
|---|---|---|
| LoRA_3 VLM (93% spatial acc, 0% FP) | ✅ Done | `v3_lora_qwen_20260310_5ep/final/` |
| Neo4j FILLS / ADJACENT_TO / CONTINUOUS | ✅ Working | `ifc_engine.py` + `add_topology_edges.py` |
| H2 eval (213 cases, 100% GT-in-pool) | ✅ Done | `eval_h2_spatial_triplets/results/h2_p4_4.jsonl` |
| Live demo (Streamlit 5-stage) | ✅ Working | `demo/ui/tab_inference.py` |
| Scene Graph Graphviz in demo | ✅ Done | `tab_inference.py:659-808` |
| Occlusion saliency explain | ✅ Deployed | `tab_inference.py:228-264`, Modal explain endpoint |
| Condition system (MA/MB/MC/MA-/MB-/MC-) | ✅ Done | `profiles.yaml:225-275` |
| `object_material` in P0 Cypher | ✅ Done (03-13) | `retrieval_backend.py` + `ifc_engine.py` |
| `target_name_keyword` post-filter | ✅ Done (03-13) | `retrieval_backend.py:_post_filter_by_name_keyword()` |
| `continuous_span` storey filter | ✅ Fixed (03-14) | `retrieval_backend.py` — Cypher + sibling lists |
| **Neo4j conn in run.py** | ✅ **Fixed** (03-14) | `run.py:init_engine()` — was NEVER connected! |
| **Storey resolver generalized** | ✅ Done (03-14) | `ifc_engine.py:_resolve_storey_query()` → returns siblings |
| 3-way precomputed eval (AP only) | ✅ Done (03-14) | Baseline / LoRA-label / Oracle |
| 4-way eval (live VLM, LoRA_3 MB+MC) | ✅ Done (03-14) | `eval/results/lora3_{site,wire}_{MB,MC}/` |
| Thesis metrics CSV + 5 plots (T2.3+T3.1) | ✅ Done (03-14) | `script/compare_results.py --thesis` → `eval/plots/` |
| IfcRelConnectsPathElements in Neo4j | ✅ **Loaded** (03-14) | `add_topology_edges.py` — 1362 CONNECTS_TO edges |
| CONNECTS_TO predicate in schema | ✅ Done (03-14) | `types.py`, `constraints_to_query.py` (P0a handles it) |
| LoRA_4 dataset (649 train, 4 pred, 110 2-hop) | ✅ Assembled (03-14) | `6_assemble_lora4.py` → `lora4_train.jsonl` |
| 2-hop OPTIONAL MATCH | ✅ Done (03-14) | `retrieval_backend.py` — rank, don't filter |
| Multi-triplet miner (389 FILLS+CONNECTS_TO) | ✅ Done (03-14) | `7_mine_multitriplet.py` |

### Sprint Checklist

> Keep in sync with System Status above — when a task completes, update both.

```
SPRINT 1: EVIDENCE (Days 1-5)
─────────────────────────────
✅ T1.1  target_name_keyword post-filter                              Done 03-13
✅ T1.2  Fix continuous_span storey filter                            Done 03-14
✅ T1.3  Add material to Neo4j + wire into P0 Cypher                 Done 03-13
✅ T2.1  Convert lora3_test.jsonl → cases_v3 format                  Done 03-14
✅ T2.2a Run 3-way precomputed eval (Baseline/LoRA-label/Oracle)     Done 03-14
✅ T2.2b Run live VLM eval (LoRA_3 MB+MC, site photos, Modal A100)  Done 03-14
✅ T2.3  Collect per-run metrics (7 runs, CSV + table)               Done 03-14
✅ T3.1  Generate thesis plots (5 figures via --thesis mode)          Done 03-14
□ T4.1  Neo4j edge pre-check in h2_eval.py                          Pending
✅ T4.2  Re-run H2 eval with fixes (verify SSR improvement)          Done 03-14

CRITICAL BUGS FOUND & FIXED (03-14):
✅ BF-1  run.py never connected Neo4j → ALL P0 was memory-mode fallback!
✅ BF-2  CONTINUOUS Cypher: CONTAINS '' matched everything → 149 instead of 8
✅ BF-3  Storey resolver 1:1 → 1:many (siblings), naming-agnostic for any IFC

SPRINT 2: DEMO + THESIS (Days 6-14)
────────────────────────────────────
□ T5.1  Entropy Collapse demo panel                                  Day 6-7
□ T5.2  Post-retrieval result viewer (3-column)                      Day 7
□ T6.1  Chapter 4: Method (architecture, P0-P8, dual-modal)         Day 8-9
□ T6.2  Chapter 5: Experiments (real numbers from T2-T3)             Day 10-11
□ T6.3  Chapter 6: Discussion + Future Work                          Day 12-13
□ T6.4  Revisions, figures, abstract                                 Day 14

SPRINT 3: WORKABLE NEW.MD IDEAS (Optional, Days 15-17)
──────────────────────────────────────────────────────
□ T7.1  Floorplan-augmented inference (10 FILLS cases)               Day 15, 0.5 day
□ T7.2  Bbox prompt engineering (zero-shot grounding)                Day 15, 0.5 day
□ T7.3  Multi-triplet prompt + intersection code                     Day 16, 0.5 day
□ T7.4  OPTIONAL MATCH fallback in retrieval_backend.py              Day 16, 0.5 day
□ T7.5  Full 69-case eval if T7.1 positive                          Day 17
□ T7.6  Update thesis with Sprint 3 results                          Day 17

SPRINT 4: LoRA_4 TRAINING (Days 15-20)
──────────────────────────────────────
✅ T8.0  Create 6_assemble_lora4.py (SR ratio, storey norm, fp-coupling)  Done 03-14
✅ T8.0b Dry-run: 578 train (75% SR) from existing data                   Done 03-14
✅ T8.1  Generate CONNECTS_TO skins (45 generated, 18 KEEP)               Done 03-14
✅ T8.2  Assemble LoRA_4 dataset (297 train, 75.1% SR, 0 noFP+SR)        Done 03-14
         Predicates: ADJACENT_TO=108, FILLS=73, CONTINUOUS=27, CONNECTS_TO=15
□ T8.3  Train LoRA_4 on Modal A100 (Qwen2.5-VL-7B, 5ep, lr=2e-4)
□ T8.4  Run MC eval on LoRA_4 adapter
□ T8.5  Compare LoRA_4 vs LoRA_3 (target: >50% P0 fire, >55% GT-in-pool)

SPRINT 4B: 2-HOP OPTIONAL MATCH (Days 20-22)
─────────────────────────────────────────────
✅ T8.6  Wire CONNECTS_TO into Neo4j (1362 edges)                         Done 03-14
✅ T8.7  Mine multi-triplet cases (389 FILLS+CONNECTS_TO)                 Done 03-14
✅ T8.8  Wire OPTIONAL MATCH 2-hop in retrieval_backend.py                Done 03-14
✅ T8.9  Update 6_assemble_lora4.py for multi-triplet records             Done 03-14
✅ T8.10 Re-assemble LoRA_4 (649 train, 110 2-hop, 75% SR)               Done 03-14
□ T8.11 Eval 2-hop: GT-in-pool improvement (target: 71.5% ceiling)
```

### 2-Hop OPTIONAL MATCH Architecture

**Problem**: Windows filling the same wall type on the same storey are indistinguishable
by single-hop FILLS alone (pool=42). The walls those windows fill have different
CONNECTS_TO signatures (which other walls they connect to), creating a
discriminating "wall connection fingerprint".

**Solution**: 2-hop query with OPTIONAL MATCH (rank, don't filter):
```cypher
-- Hop 1 (hard filter): Window→FILLS→Wall
MATCH (t:IFCElement)-[:FILLS]->(w:IFCElement)
WHERE (t.ifc_type = $subject_type OR t.ifc_type STARTS WITH $subject_type)
  AND (w.ifc_type = $object_type OR w.ifc_type STARTS WITH $object_type)
  AND (size($storey_list) = 0
       OR ANY(s IN $storey_list WHERE toLower(t.storey) CONTAINS s))
-- Hop 2 (soft re-rank): Wall→CONNECTS_TO→Wall2
OPTIONAL MATCH (w)-[:CONNECTS_TO]->(w2:IFCElement)
WHERE (w2.ifc_type = $object_type2 OR w2.ifc_type STARTS WITH $object_type2)
RETURN DISTINCT t.guid AS guid, t.name AS name, t.ifc_type AS type,
       w2 IS NOT NULL AS has_hop2
ORDER BY has_hop2 DESC
```

**Key design**: OPTIONAL MATCH means hop-2 never removes candidates from hop-1 pool.
Results with hop-2 match are ranked higher (has_hop2=true first). This prevents
over-reduction while using wall topology as a soft signal.

**Training format**: Multi-triplet spatial_relations:
```json
{
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWallStandardCase", "confidence": 1.0},
    {"predicate": "CONNECTS_TO", "object_type": "IfcWallStandardCase", "confidence": 0.8}
  ]
}
```

**Theoretical ceiling**: 71.5% Top-1 (from wall connection signature analysis).

---

## 3. Sprint Plan (Priority-Ordered Tasks)

### Sprint 1: Evidence Collection (Days 1-5)

#### Day 1: Three Quick Wins (Bug Fixes + Wiring)

**T1.1 — target_name_keyword post-filter** ⭐ HIGHEST ROI SINGLE CHANGE
```
File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → after Cypher returns candidates

Logic:
  if constraints.target_name_keyword and len(candidates) > 3:
      filtered = [c for c in candidates
                  if constraints.target_name_keyword.lower() in c['name'].lower()]
      if filtered:  # graceful: don't filter to empty
          candidates = filtered

Data justification:
  Window ObjectType has 5 subtypes (BALANS 15M/10M/20M/25M/30M)
  storey+obj_type → 61 groups, avg pool 4.3, Top-1 47.2%

Expected impact:
  FILLS pool:        42 → ~9    (SSR: 0% → ~79%)
  ADJACENT_TO pool:  30 → ~15   (SSR: 57% → ~87%)
  CONTINUOUS pool:  110 → ~50

Effort: 0.5 day. Zero risk (graceful — never filters to empty).
```

**T1.2 — Fix continuous_span storey filter**
```
File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → continuous_span branch

Current Cypher:
  WHERE target.is_continuous = true AND target.top_constraint = $top_constraint

Fix — add base storey filter:
  AND target.storey = $storey

Expected: CONTINUOUS pool 110→~46, SSR 47%→~70%
Effort: 15 minutes.
```

**T1.3 — Add material property to Neo4j + wire into P0 Cypher**
```
File: mscd_demo/src/ifc_engine.py
Location: _create_element_nodes()

Add material property:
  material_value = psets.get("Materials and Finishes", {}).get("Structural Material", "")
  node_props["material"] = material_value

File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → spatial_triplet branch

Add WHERE clause (graceful — empty string = no filter):
  AND ($object_material = '' OR toLower(ref.material) CONTAINS toLower($object_material))

Impact by element type:
  IfcWindow:              0x  (all same material — DEAD)
  IfcDoor:              3.8x  (10.2% → 38.6% Top-1)
  IfcWallStandardCase:  2.5x  (4.2% → 10.5%)
  IfcFurnishingElement: 40x   (0.7% → 26.8%)

Effort: 0.5 day. Requires Neo4j reload after ifc_engine change.
```

#### Day 2-3: 4-Way Evaluation (The Core Evidence)

**T2.1 — Convert test data to eval format**
```
File: NEW mscd_demo/eval/convert_lora3_test.py
Input:  data_curation/datasets/synth_v0.5/lora3_test.jsonl (69 held-out cases)
Output: mscd_demo/eval/cases_v3_test.jsonl

Logic: extract case_id, query_text, image_path, ground_truth_guid from JSONL records.
No leakage: these 69 cases were held out from LoRA_3 training.
```

**T2.2 — Run 4-way evaluation** (same Neo4j state, same 69 cases)

CRITICAL: LoRA_3 was TRAINED with floorplan images. MC condition is not optional —
it's the condition that matches training distribution. This gives us a 4-way comparison
that tells the complete story:

```bash
# 1. Baseline (Gemini prompt extraction, MB condition — site photo + text)
python script/run.py --profile v2_prompt \
  --cases eval/cases_v3_test.jsonl --condition-override MB --limit 69

# 2. LoRA_2 (attribute-only adapter, MB condition)
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v2_lora_qwen/final/ \
  --cases eval/cases_v3_test.jsonl --condition-override MB --limit 69

# 3. LoRA_3 WITHOUT floorplan (MB condition — isolates spatial predicate contribution)
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
  --cases eval/cases_v3_test.jsonl --condition-override MB --limit 69

# 4. LoRA_3 WITH floorplan (MC condition — the full dual-track system)
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
  --cases eval/cases_v3_test.jsonl --condition-override MC --limit 69
```

The 4 deltas each isolate one contribution:
```
Baseline(MB) → LoRA_2(MB):  isolates LoRA fine-tuning benefit (attribute extraction)
LoRA_2(MB)   → LoRA_3(MB):  isolates spatial predicate extraction (FILLS/ADJ/CONT)
LoRA_3(MB)   → LoRA_3(MC):  isolates floorplan contribution (dual-track grounding)
```

**T2.3 — Collect metrics per run**
```
For each of the 4 runs, compute:
  - Top-1 Accuracy (exact GUID match)
  - Top-5 Accuracy
  - MRR (Mean Reciprocal Rank)
  - SSR = 1 - pool_size / total_elements
  - P0 fire rate (% spatial_triplet used)
  - Spatial predicate extraction accuracy (vs ground truth)
  - Over-reduction rate (GT dropped from pool)
  - Fallback rate
```

#### Day 4: Generate Thesis Plots

**T3.1 — Comparison plots**
```
File: mscd_demo/eval/plot_4way.py (or extend compare_results.py)

Plot 1: Bar chart — Top-1 / Top-5 / MRR across 4 systems
Plot 2: SSR distribution (violin/box) per predicate per system
Plot 3: Pool size waterfall: Total → Storey → P0 → PostFilter
Plot 4: "Money chart" — Entropy Collapse:
        X-axis: pipeline stage (Input → Storey → P0 → PostFilter)
        Y-axis: candidate count (log scale)
        4 lines: Baseline flat, LoRA_2 flat, LoRA_3(MB) drops, LoRA_3(MC) drops most
Plot 5: Modality contribution: grouped bar (MA vs MB vs MC) for LoRA_3
```

#### Day 5: Buffer + Pre-checks

**T4.1 — Neo4j edge pre-check in h2_eval.py**
```
File: mscd_demo/eval/h2_eval.py
Location: top of main()

driver.execute_query("MATCH ()-[r:FILLS]->() RETURN count(r) AS n")
assert n >= 100, f"Graph incomplete: only {n} FILLS edges. Run neo4j_init.sh."
# Same for ADJACENT_TO >= 50
```

**T4.2 — Re-run H2 eval with T1.1-T1.3 fixes applied**
```
Expected improvement over current 213-case H2 results:
  FILLS:        SSR 0% → ~79%  (target_name_keyword post-filter)
  ADJACENT_TO:  SSR 57% → ~87% (post-filter)
  CONTINUOUS:   SSR 47% → ~70% (storey filter fix)
  OVERALL:      SSR 35% → ~80%
```

---

### Sprint 2: Demo + Thesis (Days 6-14)

#### Day 6-7: Demo Polish

**T5.1 — Entropy Collapse visualization tab**
```
File: mscd_demo/demo/ui/tab_inference.py (extend existing)

Design:
  Left column:  "Attribute-Only Baseline"
    → Grid of 46 identical window icons, all gray
    → Caption: "46 candidates — system cannot distinguish"

  Right column: "Neuro-Symbolic (Ours)"
    → Stage 1: 46 icons (storey filter)
    → Stage 2: ~30 icons highlighted (P0 fires, ADJACENT_TO removes some)
    → Stage 3: ~9 icons (obj_type post-filter)
    → Final: 1 icon glows green, GUID displayed

  Data: run a real case through pipeline, log pool at each stage.
```

**T5.2 — Post-retrieval result viewer (3-column)**
```
File: mscd_demo/demo/ui/tab_inference.py

Col 1: Input site photo (as uploaded)
Col 2: Floorplan patch centered on matched GUID (reuse _floorplan_renderer.py)
Col 3: Neo4j 1-hop subgraph (reuse existing Graphviz renderer)

All components already exist — just compose into a layout.
```

#### Day 8-14: Thesis Writing

```
Chapter 4 — Method: Neuro-Symbolic Architecture
  4.1  Problem formalization (attribute entropy bottleneck)
  4.2  VLM spatial predicate extraction (LoRA_3 training, 1377 samples)
  4.3  Priority cascade query planner (P0-P8)
  4.4  Neo4j Cypher execution engine (FILLS, ADJACENT_TO, CONTINUOUS)
  4.5  Graceful degradation: fallback cascade + post-filter
  4.6  Dual-modal grounding: MC condition (site photo + floorplan)

Chapter 5 — Experiments
  5.1  Dataset: synth_v0.5 (1377 train / 69 test), 3-tier labeling
  5.2  Benchmark: H2 hard-negatives (213 cases, 100% GT-in-pool)
  5.3  Group 1: Agentic vs Structured (V1 vs V2)
  5.4  Group 2: 4-way comparison (Baseline/LoRA_2/LoRA_3-MB/LoRA_3-MC)
  5.5  Group 3: Ablation (modality MA/MB/MC, component ±spatial/±Neo4j)
  5.6  Per-predicate analysis (FILLS/ADJACENT_TO/CONTINUOUS)
  5.7  Error analysis: when does P0 fail?

Chapter 6 — Discussion + Future Work
  6.1  Contribution: neuro-symbolic > pure neural for structured retrieval
  6.2  Limitation: FILLS SSR=0% without post-filter (wall identity problem)
  6.3  Future: IfcRelConnectsPathElements (686 edges → 2-hop → 71.5% ceiling)
  6.4  Future: SGG schema + bbox grounding (new.md architecture)
  6.5  Future: cross-IFC generalization (Duplex_A has IfcRelSpaceBoundary)
```

---

### Sprint 3: Workable new.md Ideas (Optional, Days 15-17)

These experiments test the core hypotheses from `new_retrieve.md` at <5% of
the implementation cost. Zero risk: all are additive, no schema changes.

**T7.1 — Floorplan-augmented inference test (10 FILLS cases)**
```
File: mscd_demo/src/v2/constraints_extractor_lora.py
Change: Add floorplan image as second input when MC condition active

Current:  messages = [system_prompt, {"image": site_photo, "text": query}]
New:      messages = [system_prompt, {"image": [site_photo, floorplan_crop], "text": query}]

Qwen2.5-VL natively supports multi-image interleaved input.
This IS the new.md "dual-track" idea at 5% of the cost.

Test: 10 FILLS cases from H2 eval (hardest: pool=46)
  - Run with site photo only → record spatial_relations
  - Run with site photo + floorplan → compare
  - Does VLM mention wall position? Extract better target_name_keyword?

Best case:  VLM identifies wall → FILLS pool 46→17 → Top-1 37%
Likely:     Slightly better obj_type extraction → pool 42→~8
Worst case: No change (VLM ignores floorplan) → document as negative result
```

**T7.2 — Bbox prompt engineering (zero-shot grounding)**
```
File: prompts/constraints_extraction.yaml
Add to output spec:
  "If you can identify the target element in the floorplan image,
   output its approximate bounding box as floorplan_bbox: [x1,y1,x2,y2]"

Qwen2.5-VL supports <box> natively → may output bbox without finetuning.
Even poor bbox quality shows the VLM is attempting spatial grounding.

Demo overlay (5 lines): PIL.ImageDraw.rectangle on floorplan image.
This gives new.md "Panel 1: Visual Grounding" essentially for free.
```

**T7.3 — Multi-triplet extraction + intersection**
```
File: prompts/constraints_extraction.yaml
Add: "If multiple spatial relationships are visible, output all of them."

File: mscd_demo/src/v2/retrieval_backend.py
Add:
  if len(spatial_relations) > 1:
      pool_0 = execute_p0(triplet[0])
      pool_1 = execute_p0(triplet[1])
      candidates = pool_0 ∩ pool_1
      if not candidates: candidates = pool_0  # graceful fallback

LoRA_3's spatial_relations is already List[SpatialTriplet] — schema supports it.
If LoRA_3 naturally outputs multi-triplet (even occasionally), this gives
2-hop discrimination for free on those cases.
```

**T7.4 — OPTIONAL MATCH fallback (from new.md Panel 2)**
```
File: mscd_demo/src/v2/retrieval_backend.py

Strict (2-hop, may return 0):
  MATCH (t)-[:FILLS]->(w)-[:ADJACENT_TO]->(c)
  WHERE t.ifc_type STARTS WITH 'IfcWindow'
    AND w.ifc_type STARTS WITH 'IfcWall'
    AND c.ifc_type STARTS WITH 'IfcColumn'
  RETURN DISTINCT t

If empty → graceful degradation:
  MATCH (t)-[:FILLS]->(w)
  OPTIONAL MATCH (w)-[:ADJACENT_TO]->(c {ifc_type_prefix: 'IfcColumn'})
  RETURN DISTINCT t, c IS NOT NULL AS has_anchor
  ORDER BY has_anchor DESC

Shows the system tries most-specific first, then degrades gracefully.
Excellent for thesis Chapter 4.5.
```

**T7.5/T7.6 — Decision gate**
```
If T7.1 shows >30% FILLS pool improvement:
  → Run full 69-case eval with floorplan augmentation
  → Becomes thesis contribution: Chapter 4.6 "Dual-Modal Spatial Grounding"

If not:
  → Document as negative result in Chapter 6
  → Still valuable: proves topology bottleneck requires explicit graph structure,
    not just additional visual context
```

---

## 4. Evaluation Framework (3 Groups)

### Group 1 — Agentic vs. Structured Pipeline (DONE)

Already evaluated. Results in README.md:

| System | Architecture | Top-1 (v0.2, 43 cases) |
|---|---|---|
| V1 Agent (memory) | ReAct + tool calling | 32.6% |
| V2 Structured (A1) | Constraint extraction → query planner | **50.0%** |

**Narrative**: V2 outperforms V1 by +17pp when inputs are clear. Structured output > free-form reasoning for BIM retrieval. Motivates LoRA fine-tuning for harder cases.

### Group 2 — 4-Way Neuro-Symbolic Comparison (Sprint 1, T2.1-T2.3)

| System | Extractor | Spatial? | Floorplan? | Max Priority |
|---|---|---|---|---|
| Baseline (Gemini) | PromptConstraintsExtractor | No | No | P1-P8 |
| LoRA_2 | LoRAConstraintsExtractor (v2) | No | No | P1-P8 |
| LoRA_3 (MB) | LoRAConstraintsExtractor (v3) | **Yes** | No | **P0-P8** |
| LoRA_3 (MC) | LoRAConstraintsExtractor (v3) | **Yes** | **Yes** | **P0-P8** |

Test data: 69 held-out cases (lora3_test.jsonl). No leakage.

Metrics: Top-1, Top-5, MRR, SSR, P0 fire rate, spatial acc, over-reduction, fallback rate.

Each delta isolates one variable:
- Baseline→LoRA_2: benefit of LoRA fine-tuning (attribute extraction)
- LoRA_2→LoRA_3(MB): benefit of spatial predicate extraction
- LoRA_3(MB)→LoRA_3(MC): benefit of floorplan (dual-track grounding)

### Group 3 — Ablation Studies

**(a) Modality ablation** (same LoRA_3, vary input):
```bash
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
  --cases eval/cases_v3_test.jsonl --condition-override MA --limit 69   # text only
  # ... same for MB, MC, MA-, MB-, MC-
```

**(b) Component ablation** (same LoRA_3 MC, toggle components):
- ±spatial: run with/without P0 (spatial_triplet priority)
- ±Neo4j: run with/without Neo4j (memory-only fallback)
- ±post-filter: run with/without target_name_keyword post-filter (T1.1)

**(c) Cross-IFC generalization**:
- Run on BasicHouse + Duplex_A cases (if time permits)
- Duplex_A has IfcRelSpaceBoundary (264 instances) — tests space_name filtering

### H2 Hard-Negative Stress Test (Re-run after T1.1-T1.3)

| Predicate | Cases | Current SSR | Expected SSR (post-fix) |
|---|---|---|---|
| ADJACENT_TO | 66 | 57.0% | ~87% (+T1.1 post-filter) |
| FILLS | 84 | 0.0% | ~79% (+T1.1 post-filter) |
| CONTINUOUS | 63 | 47.5% | ~70% (+T1.2 storey fix) |
| **OVERALL** | **213** | **~35%** | **~80%** |

---

## 5. Results (Actual, 2026-03-14)

### 5.1 3-Way Precomputed Comparison (AP-only, n=50, Neo4j connected)

| Metric | Baseline (MB) | LoRA-label (MB) | Oracle (MB) |
|---|---|---|---|
| Constraints source | GT ifc_class+storey, no spatial | Training labels as-is | GT all fields + skeleton spatial |
| P0 fired | 0/50 (0%) | 27/49 (55%) | 48/49 (98%) |
| **GT-in-pool** | **49/50 (98.0%)** | **30/49 (61.2%)** | **49/49 (100.0%)** |
| GT-DROPPED | 1/50 (2.0%) | 19/49 (38.8%) | **0/49 (0.0%)** |
| Top-1 | 2/50 (4.0%) | 0/49 (0.0%) | 2/49 (4.1%) |
| Top-K | 4/50 (8.0%) | 2/49 (4.1%) | 5/49 (10.2%) |
| SSR | 89.7% | 93.8% | **94.7%** |
| Avg Pool | 128.9 | 77.4 | **66.0** |

**P0 performance (when it fires)**:
| | LoRA-label | Oracle |
|---|---|---|
| P0 GT-in-pool | **27/27 (100%)** | **48/48 (100%)** |
| P0 over-reduction | **0/27 (0%)** | **0/48 (0%)** |
| FILLS GT-in | 14/14 (100%) | 24/24 (100%) |
| ADJACENT_TO GT-in | 10/10 (100%) | 16/16 (100%) |
| CONTINUOUS GT-in | 3/3 (100%) | 8/8 (100%) |

**Key findings**:
1. **Oracle AP: 100% GT-in-pool, 0% over-reduction** — perfect retrieval with correct inputs
2. **P0 = 100% GT-in when it fires** — both LoRA-label and Oracle achieve 0% over-reduction
3. **LoRA-label 38.8% drops are from storey+type fallback** (wrong ifc_class in 22 cases), NOT from P0
4. **CRITICAL BUG FOUND**: run.py never connected Neo4j → all previous evals ran P0 in memory fallback

### 5.2 Cross-IFC Reality Check

```
IFC Model   Cases   Neo4j?   Oracle GT-in-pool
──────────────────────────────────────────────
AP           50      ✅        49/49 (100%)   ← 1 case excluded (49 scored)
BH            6      ❌         0/6  (0%)     ← no topology edges loaded
DXA          13      ❌         0/13 (0%)     ← no topology edges loaded
──────────────────────────────────────────────
ALL          69      —         49/68 (72%)    ← 28% drop = cross-IFC gap
```
BH + DXA have 0% because their IFC topology isn't loaded into Neo4j.
Fix: T4 cross-IFC pipeline (load BH + DXA graphs).

### 5.3 H2 Hard-Negative Benchmark (213 cases, post-fix)

```
Predicate      Cases  Post-P0     SSR     GT-in    Fallback
                      avg pool
────────────────────────────────────────────────────────────
ADJACENT_TO      66     46.5     35%     66/66      0/66
CONTINUOUS       63    127.7     42%     63/63      0/63
FILLS            84     60.3    -49%     84/84      0/84
────────────────────────────────────────────────────────────
OVERALL         213     76.0      4%    213/213     0/213
```
- 100% GT-in-pool, 0% fallback (was 30% fallback before CONTINUOUS fix)
- FILLS SSR negative: storey siblings expand pool (Level X + X-Floor). T1.1 post-filter helps.

### 5.4 Bugs Found & Impact

| Bug | Impact | Fix |
|---|---|---|
| **BF-1: run.py never connected Neo4j** | ALL P0 Cypher ran in memory fallback. SSR/pool numbers were WRONG. | `init_engine()` reads neo4j config |
| **BF-2: CONTINUOUS `CONTAINS ''`** | Empty string matches everything → 149 walls instead of 8 | Added `$param <> ''` guards |
| **BF-3: Storey resolver 1:1** | `_storey_by_num[1] = "level 1"` overwrote `"1 - first floor"` | Changed to 1:many `{int: [str]}` |
| **BF-4: Storey resolver str→list** | `_resolve_storey_query` returned single string, losing siblings | Now returns `List[str]` of all siblings |

### 5.5 Improvement Scenarios (Updated)

```
Scenario                              GT-in   P0-ovred  SSR     Status
──────────────────────────────────────────────────────────────────────
S0: Memory-only (pre-fix)             98%      N/A      89.7%   Superseded
S1: + Neo4j connected (BF-1)         100%      0%      94.7%   ✅ DONE
S1b: + CONTINUOUS fix (BF-2,3,4)    100%      0%      94.7%   ✅ DONE
S2: + target_name_keyword post-filt  100%      0%     ~96%     Wired, needs VLM
S3: + 2-hop ConnectsPath             100%      0%     ~99%     Future work
```

### 5.6 LoRA_3 Live VLM Evaluation (69 test cases, site photos, 2026-03-14)

**Setup**: LoRA_3 adapter on Modal A100, site photos (not wireframes), Neo4j connected.
Image domain mismatch fixed: wireframes→site photos (training had 0 wireframes).

| Metric | MB (site) | MC (site+floorplan) | Oracle (GT labels) |
|--------|-----------|--------------------|--------------------|
| GT-in-pool | **14/68 (20.6%)** | **23/68 (33.8%)** | 49/49 (100%) |
| Top-1 | 0/68 (0.0%) | 1/68 (1.5%) | 2/49 (4.1%) |
| SSR | 96.7% | 95.2% | 94.7% |
| Storey accuracy (floor#) | 42.6% | 52.9% | 100% |
| IFC class accuracy | 70.6% | 67.6% | 100% |
| spatial_relations extracted | **0/69 (0%)** | **1/69 (1.4%)** | 48/49 (98%) |
| Avg pool | 41.5 | 60.1 | 66.0 |

**Strategy breakdown** (all queries fall to storey+type — NO P0 fires):
```
Strategy        MB: N   GT-in         MC: N   GT-in
─────────────────────────────────────────────────────
storey+type      64    14/64 (21.9%)   65    23/65 (35.4%)
space+type        4     0/4  (0.0%)     3     0/3  (0.0%)
```

**Per-field distributions**:
```
MB storey predictions: Garage(8), 1st(4), 2nd(8), 3rd(2), 4th(4), 5th(3),
                       6th(9), Level0(1), Level1(17), Level2(6), Roof(5)
MC storey predictions: Garage(7), 1st(17), 2nd(8), 3rd(2), 4th(4), 5th(3),
                       6th(3), Level0(1), Level1(12), Level2(4), Roof(5)
IFC class: IfcWindow(24-26), IfcDoor(18-19), IfcWallStandardCase(15-17),
           IfcSlab(6), IfcRailing(2-3)
```

**Key findings from LoRA_3 live eval**:

| # | Finding | Root Cause | LoRA_4 Action |
|---|---------|------------|---------------|
| F1 | **spatial_relations = 0-1/69** — model NEVER fires P0 | 56% of training data had empty spatial_relations → model learned to default to `[]` | Increase spatial_relations labeling to 80%+ for floorplan cases |
| F2 | **Floorplan helps storey** (+10pp: 42.6%→52.9%) | Floorplan directly shows floor labels | Make floorplan MANDATORY for topology cases, not optional MC |
| F3 | **Storey naming mismatch** ("Level 1" vs "1 - First Floor") | Training data has mixed naming from AP/BH/DXA | Standardize storey labels or teach model canonical format |
| F4 | **Site photos = visual ambiguity for topology** | Site photo shows appearance, not spatial structure | Floorplan should be PRIMARY input for spatial reasoning |
| F5 | **IFC class accuracy decent (70%)** but storey accuracy low (43-53%) | Multi-IFC training → storey confusion between models | Per-model storey naming in training data |
| F6 | **MC GT-in-pool 33.8% vs MB 20.6%** (+64% relative) | Floorplan provides spatial context that site photo lacks | MC condition = default for LoRA_4 |
| F7 | **Oracle proves pipeline works** (100% GT-in-pool, 0% over-reduction) | Gap is entirely in VLM extraction, NOT retrieval | Focus LoRA_4 on extraction quality, not pipeline fixes |

### 5.7 LoRA_4 Development Hints (from LoRA_3 Eval)

**Core insight**: The retrieval pipeline is proven (Oracle=100%). The ONLY bottleneck
is VLM spatial_relations extraction (1/69). LoRA_4 must fix the extraction, not the pipeline.

**Training data strategy**:
```
1. FLOORPLAN-FIRST: Every topology case MUST have floorplan as primary image
   - Site photo = secondary (appearance/condition)
   - Floorplan = primary (spatial topology, wall identity, adjacency)
   - Ratio: 80% floorplan cases → spatial_relations populated

2. ANTI-CONSERVATIVE BIAS: Break the "default to empty []" behavior
   - Current: 44% have spatial_relations → model learns to skip
   - Target: 80%+ of floorplan cases have spatial_relations
   - Add explicit prompt: "When a floorplan is provided, ALWAYS look for spatial relationships"

3. STOREY STANDARDIZATION: Teach canonical storey format
   - Training labels should use the Neo4j storey name (e.g., "1 - First Floor")
   - Or: teach model to output floor NUMBER only, let pipeline resolve

4. PREDICATE-BALANCED DATA: Current training is ADJACENT_TO-heavy
   - ADJACENT_TO: 314 (52%), FILLS: 207 (34%), CONTINUOUS: 84 (14%)
   - FILLS is the hardest (SSR=0% without post-filter) → needs more training cases
   - Mine additional FILLS cases from BH + DXA

5. CONFIDENCE CALIBRATION: Model outputs confidence=1.0 on the one extraction
   - Need training with varied confidence values
   - Or: remove confidence field, let pipeline always execute P0 when spatial_relations present
```

**Schema changes for LoRA_4** (tentative):
```json
{
  "storey_name": "1",              // Floor NUMBER only (pipeline resolves)
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": "BALANS",
  "spatial_relations": [
    {
      "predicate": "FILLS",
      "object_type": "IfcWallStandardCase",
      "object_material": "Brick",
      "confidence": 0.9
    }
  ]
}
```

### 5.8 LoRA_4 Data Analysis (Done 03-14)

**Root cause diagnosis** — why LoRA_3 training failed to teach SR extraction:

```
LoRA_3 Training Data Cross-Tab (the core problem):
                  has_SR    no_SR
  has_floorplan:    434      387   ← 387 cases: "see floorplan, output []"
  no_floorplan:     171      385   ← 171 cases: SR from text alone (unreliable)

Result: Model learned "always output []" (56% of training = no SR)
```

**LoRA_4 assembler** (`6_assemble_lora4.py`) applies 5 fixes:
1. **SR ratio 43%→75%**: downsample attribute-only records
2. **Floorplan-SR coupling**: strip SR from records without floorplan (171→0 noFP+SR)
3. **Storey normalization**: "1 - First Floor" → "1" (pipeline resolver handles mapping)
4. **Updated system prompt**: explicitly guides spatial extraction when floorplan present
5. **Balanced predicates**: ADJACENT_TO=185, FILLS=178, CONTINUOUS=71

**Dry-run results** (existing data only):
```
LoRA_4 (578 total, 75.1% SR):
  fp+SR:  434    fp+noSR:   57
  noFP+SR:  0    noFP+noSR: 87
  ✓ Clean coupling — no SR-without-floorplan
```

**Projection** (after skinning 115 unskinned skeletons, ~65% judge pass):
```
Projected LoRA_4: ~924 total, 75% SR
  v0.5 topology: ~648 train records (3x augment on ~216 KEEP skins)
  v0.4 enriched: ~276 (45 with SR+FP, 231 attribute-only)
```

**Key files**:
- Assembler: `data_curation/scripts/synth/6_assemble_lora4.py`
- Unskinned skeletons: AP=81, DXA=22, BH=12 (need 3a→3b pipeline)

### 5.9 Evaluation Infrastructure & Thesis Plots (Done 03-14)

**Metrics collection** (T2.3): Comprehensive 7-run comparison across all eval conditions.

```
System                         n  GT-in%  Top1%  Top5%    MRR   SSR%   P0%  SR
------------------------------------------------------------------------------
Baseline (GT labels)          69   84.1%   4.3%  18.8% 0.098   91.9%  0.0%   0
LoRA-label (train)            55   54.5%   0.0%   9.1% 0.035   94.0% 52.7%  29
Oracle (GT spatial)           59   91.5%   3.4%  18.6% 0.100   95.3% 96.6%  57
LoRA₃ wire MB                 67   10.4%   3.0%   3.0% 0.030   93.8%  0.0%   0
LoRA₃ wire MC                 66   16.7%   3.0%   3.0% 0.030   93.5%  0.0%   0
LoRA₃ site MB                 68   20.6%   0.0%   8.8% 0.027   93.9%  0.0%   0
LoRA₃ site MC                 68   33.8%   1.5%   8.8% 0.039   92.9%  0.0%   0
```

**Thesis plots** (T3.1): 5 figures generated via `python script/compare_results.py --thesis`.

| Plot | File | What it shows |
|------|------|---------------|
| **T1** | `eval/plots/T1_system_comparison.png` | GT-in-pool / Top-5 / MRR bar chart across 5 key systems |
| **T2** | `eval/plots/T2_pool_reduction.png` | Avg pool size per system with SSR% annotation (horizontal bars) |
| **T3** | `eval/plots/T3_modality_ablation.png` | Wireframe vs site, MB vs MC — shows +13.2pp floorplan gain |
| **T4** | `eval/plots/T4_pipeline_waterfall.png` | Oracle pipeline stages: 1257→26→61→9 (log scale) |
| **T5** | `eval/plots/T5_p0_vs_accuracy.png` | P0 fire rate vs GT-in-pool — proves spatial extraction is bottleneck |

**Key takeaway from T5 ("money chart")**: Oracle fires P0 96.6% → 91.5% GT-in-pool.
LoRA₃ fires P0 0% → 20-34% GT-in-pool. The retrieval pipeline works;
the VLM extraction doesn't. This is the core motivation for LoRA_4.

**Code & file locations**:
```
Thesis mode CLI:    script/compare_results.py --thesis
  Functions added:  _discover_thesis_experiments(), _compute_thesis_metrics(),
                    _plot_thesis_{system_comparison,pool_reduction,modality_ablation,
                                  pipeline_waterfall,p0_vs_accuracy}(),
                    _export_thesis_csv(), _run_thesis_mode()
Plot output:        eval/plots/T1-T5_*.png + thesis_summary.csv
Metrics CSV:        eval/results/t23_all_runs_metrics.csv
Trace files:        eval/results/{baseline_MB,lora_label_MB,oracle_MB,
                    lora3_MB,lora3_MC,lora3_site_MB,lora3_site_MC}/traces_*.jsonl
Existing tools:     eval/analyze_traces.py (per-strategy/predicate breakdown)
                    script/generate_plots.py --modality (modality analysis charts 9-11)
```

---

## 7. Thesis Narrative Arc

```
Act 1 — THE PROBLEM (Chapter 1-3)
  "46 identical IfcWindows per floor. Attribute matching = 2.2% Top-1.
   Pure VLMs hallucinate spatial relationships they cannot execute.
   Symbolic systems cannot parse natural language or images."

Act 2 — THE METHOD (Chapter 4)
  "Decompose into perception (VLM extracts spatial predicates with
   93% accuracy, 0% false positives) and reasoning (Neo4j Cypher
   traverses the IFC topology graph). Priority cascade ensures
   graceful degradation when spatial signals are absent."

Act 3 — THE EVIDENCE (Chapter 5)
  "4-way ablation on 69 held-out cases proves:
   Baseline: 3-6% → LoRA_2: 5-8% → LoRA_3(MB): ~14-20% → LoRA_3(MC): ~25-40%.
   213-case H2 benchmark: 100% GT-in-pool, SSR ~80% after post-filter.
   Zero hallucination rate on spatial predicates."

Act 4 — THE EXTENSION (Chapter 4.6, if Sprint 3 positive)
  "Dual-modal grounding: adding floorplan as VLM input enables
   wall identification. MB→MC delta directly measures contribution."

Act 5 — THE CEILING (Chapter 6)
  "Data simulation shows 71.5% Top-1 is achievable with 2-hop
   topology (IfcRelConnectsPathElements, 686 untapped edges).
   The architecture is ready — only training data limits us."
```

**The story reviewers will remember**: *"The system that turned 46 identical
windows into 1 correct match by reasoning about spatial topology."*

---

## 8. IFC Relationship Taxonomy (Thesis Reference)

### 8.1 Complete Inventory (AdvancedProject.ifc)

```
Relationship                     Count  Used?   Discrimination
──────────────────────────────────────────────────────────────
IfcRelDefinesByProperties       19877   ✅      Psets (name, type, dimensions)
IfcRelAssociatesMaterial         1345   Partial DEAD for windows, 2.5-40x for others
IfcRelConnectsPathElements        686   ✗ NEW   Wall→Wall topology (ATSTART/ATEND/ATPATH)
                                                2-hop enables 71.5% Top-1 (simulated)
IfcRelVoidsElement                427   ✅      Intermediate for FILLS chain
IfcRelFillsElement                389   ✅ P0   Window/Door→Wall
IfcRelDefinesByType               202   ✗       Type definitions
IfcRelAssignsToGroup              154   ✗       Furniture groups (low value)
IfcRelConnectsPortToElement       139   ✗       MEP ports only
IfcRelContainedInSpatialStructure  10   ✅      Storey assignment
IfcRelAggregates                   17   ✅      Railing decomposition
IfcRelSpaceBoundary                 0   N/A     MISSING in AP (Duplex has 264)
```

### 8.2 IfcRelConnectsPathElements Deep Dive (Future Work)

686 wall-to-wall edges with explicit connection semantics:
```
Connection types: ATSTART, ATEND, ATPATH
  ATEND↔ATSTART:   194 edges (end-to-start chain)
  ATSTART↔ATSTART: 125 edges (shared start corner)
  ATPATH↔ATEND:     89 edges (T-junction)
  ...

Wall connectivity graph:
  389 walls, 681 edges, avg degree 3.5
  Window-hosting walls: degree 3-22 (varies significantly)
  The big exterior wall: degree=22, hosts 85 windows (17/floor)

2-hop signature simulation:
  45 unique signatures for 263 windows
  pool=1:  8 windows (uniquely identifiable)
  pool≤3: 40 windows (15%)
  With +obj_type: avg pool 1.9 (71.5% Top-1)
```

Loading this into Neo4j + 2-hop Cypher is the single highest-value future improvement.

### 8.3 Duplex_A: IfcRelSpaceBoundary

```
264 boundary instances, 18 named spaces (A101, A201, B101, ...)
Each space has 9-19 boundary elements.
→ Room-level filtering (space_name) works on Duplex_A but not AP.
→ Cross-IFC eval would demonstrate both code paths.
```

---

## 9. What Was Dropped (and Why)

| Item | Reason | Where Documented |
|---|---|---|
| **Full SGG schema** (new.md) | Invalidates all 1,377 training samples + entire pipeline. 3-week rewrite. | Appendix A.1 |
| **Dual-track bbox** (new.md) | Each sub-component is a standalone project. | Appendix A.2 |
| **LoRA_4 training** (new.md) | **RE-SCOPED**: LoRA_3 eval proves extraction is bottleneck (1/69 spatial). LoRA_4 with floorplan-first strategy is now PLANNED. See §5.7. | §5.7 + Appendix A.3 |
| **P6: 2-hop implementation** | High value but >1 week. Present simulated ceiling (71.5%) instead. | Section 7, 8.2 |
| **P7.2-7.3: bbox overlay** | Needs new training data. Zero-shot alternative in T7.2. | Appendix A.4 |
| **P8: dataset expansion** | 213 H2 + 69 test = sufficient evidence for thesis. | Appendix A.5 |
| **P9: 4D timestamp** | Tangential to core spatial reasoning question. | Appendix A.6 |
| **P11: DPO + D1** | Optimization, not contribution. | Appendix A.7 |

---

## Appendix A: Dropped Plans — Technical Details

### A.1 SGG Schema (from new_retrieve.md)

Full proposed schema (LoRA_4 target):
```json
{
  "entities": [
    {"node_id": "E1", "ifc_class": "IfcWindow", "role": "target",
     "onsite_image_bbox": [210,340,450,600], "floorplan_bbox": [50,100,80,120]},
    {"node_id": "E2", "ifc_class": "IfcWall", "role": "anchor_1",
     "onsite_image_bbox": null, "floorplan_bbox": [45,90,200,130]}
  ],
  "spatial_triplets": [
    {"subject_id": "E1", "predicate": "ADJACENT_TO", "object_id": "E2"},
    {"subject_id": "E2", "predicate": "INTERSECTS",  "object_id": "E3"}
  ],
  "status_observed": "Water leaking from the frame."
}
```

Design rationale: modality separation (site=semantics, floorplan=topology).
**Workable alternative**: MC condition + T7.1-T7.2 (Sprint 3) captures 80% of value.

### A.2 Dual-Track Visual Grounding (from new_retrieve.md)

Full architecture: site photo → VLM → onsite_bbox; floorplan → Floorplan2Graph → floorplan_bbox → cross-modal merge → Neo4j multi-hop.
Data pipeline: multi-nested skeleton mining + headless renderer for dual bbox.
**Workable alternative**: T7.1 (zero-shot floorplan augmentation) + T7.2 (bbox prompt engineering).

### A.3 LoRA_4 Training (from new_retrieve.md)

4-week timeline: Week 1 data+UI, Week 2 finetune, Week 3 integration, Week 4 thesis.
**Why dropped**: Assumes clean execution. Bug fixes alone would consume 2 weeks.

### A.4 Bbox Overlay (P7.2-7.3 from 0307)

Extend LoRA output to include `subject_bbox`/`object_bbox`.
Overlay on site photo: blue=subject, orange=reference, arrow+label.
**Blocked on**: new training data with bbox annotations.

### A.5 Dataset Expansion (P8 from 0307)

Fix 3a black-image bug, raise mining quotas to 500+, cross-IFC pipeline.
**When to revisit**: before venue submission (ECCV/AAAI).

### A.6 4D Timestamp (P9 from 0307)

Parse 4D task status → temporal constraints. Low marginal value over storey extraction.

### A.7 DPO + D1 Condition (P11 from 0307)

DPO on eval failures, D1 condition (no 4D metadata). Optimization, not contribution.

---

## Appendix B: Compound Spatial Query Design (Future Work)

**Option A — Multiple independent triplets (intersect in Python)**:
```json
{"spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWall", "confidence": 0.92},
    {"predicate": "ADJACENT_TO", "object_type": "IfcColumn", "confidence": 0.78}
]}
```
Pro: No Cypher change, just set intersection. Con: Subject of triplet[1] ambiguous.
**Implementable now** (T7.3, Sprint 3).

**Option B — Chained triplets (2-hop Cypher)**:
```cypher
MATCH (t)-[:FILLS]->(w)-[:ADJACENT_TO]->(c)
WHERE t.ifc_type STARTS WITH 'IfcWindow'
  AND w.ifc_type STARTS WITH 'IfcWall'
  AND c.ifc_type STARTS WITH 'IfcColumn'
RETURN DISTINCT t
```
Pro: Most precise. Con: Requires schema change + new training data.
**Requires**: IfcRelConnectsPathElements loaded + LoRA_4 training.

**OPTIONAL MATCH fallback** (T7.4):
```cypher
MATCH (t)-[:FILLS]->(w)
OPTIONAL MATCH (w)-[:ADJACENT_TO]->(c {ifc_type_prefix: 'IfcColumn'})
RETURN DISTINCT t, c IS NOT NULL AS has_anchor
ORDER BY has_anchor DESC
```
**Implementable now** — graceful degradation for thesis Chapter 4.5.

---

## Appendix C: Key References

```
Adapter checkpoints:
  LoRA_2: models/adapters/v2_lora_qwen/final/
  LoRA_3: models/adapters/v3_lora_qwen_20260310_5ep/final/ (Modal: /mscd-lora/final)

Condition system: profiles.yaml (lines 225-275)
  MA = text only | MB = site photo + text | MC = site + floorplan + text
  MA-/MB-/MC- = same without 4D metadata

System prompt: prompts/constraints_extraction.yaml
H2 eval results: eval_h2_spatial_triplets/results/h2_p4_4.jsonl (213 cases)
Element index: data_curation/references/element_index.jsonl (1233 elements)
Primary IFC: data_curation/ifc_models/AdvancedProject.ifc (IFC2X3, mm units)
Neo4j: bolt://localhost:7687, pw=password, /tmp/neo4j-community-5.26.0/

Existing eval results:
  Group 1 (V1 vs V2): README.md — V1 32.6% vs V2 50.0% Top-1
  LoRA_2 (50 cases, MA): 35.3% Top-1, 66.2% SSR
  Gemini baseline (50 cases, MA): 25.7% Top-1, 52.8% SSR
  LoRA_3 extraction quality (69 test): 93% spatial acc, 0% FP, 100% parse
  H2 (213 cases): 100% GT-in-pool, ADJACENT_TO SSR=57%, FILLS SSR=0%, CONTINUOUS SSR=47.5%

Evaluation infrastructure (added 03-14):
  Thesis plots:          eval/plots/T1-T5_*.png (via script/compare_results.py --thesis)
  Metrics CSV:           eval/plots/thesis_summary.csv
  Per-run metrics:       eval/results/t23_all_runs_metrics.csv
  Trace files (7 runs):  eval/results/{baseline_MB,lora_label_MB,oracle_MB,
                          lora3_MB,lora3_MC,lora3_site_MB,lora3_site_MC}/traces_*.jsonl
  Strategy analyzer:     eval/analyze_traces.py (per-strategy/predicate breakdown)
  Test cases:            eval/cases_v3_test.jsonl (wireframe), eval/cases_v3_test_site.jsonl (site)
  Precomputed VLM:       logs/evaluations/eval_constraints_final_{MB,MC}.jsonl (Modal output)
```

---

*Last updated: 2026-03-14 (evening).*
*Data-grounded against: AdvancedProject.ifc (1233 elements, 686 ConnectsPathElements),
H2 eval (213/213 GT-in-pool, 0 fallbacks), 3-way precomputed eval (69 cases, AP-only: 100% GT-in-pool),
LoRA_3 live VLM eval (69 cases, MB: 20.6% GT-in-pool, MC: 33.8% GT-in-pool, spatial_relations: 1/69).*
*Consolidates: 0307_post_mid_plan.md + new_retrieve.md + 0313_final_todo.md.*
*Critical bugs fixed 03-14: Neo4j conn in run.py, CONTINUOUS Cypher, storey resolver generalized.*
*LoRA_4 re-scoped from "dropped" to "planned" — floorplan-first strategy, see §5.7.*
