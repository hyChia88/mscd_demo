# 0313 Final Sprint: Data-Grounded Action Plan

> **Date**: 2026-03-13
> **Deadline**: ~3 weeks to thesis submission
> **Strategy**: Sharpen focus on the highest-value research question.
> Drop everything that doesn't directly produce thesis evidence or demo.

---

## Status Snapshot (as of 2026-03-13)

| Component | Status | Key File |
|---|---|---|
| LoRA_3 VLM (93% spatial acc, 0% FP) | ✅ Done | `v3_lora_qwen_20260310_5ep/final/` |
| Neo4j FILLS / ADJACENT_TO / CONTINUOUS | ✅ Working | `ifc_engine.py` + `add_topology_edges.py` |
| H2 eval (213 cases) | ✅ **213/213 GT-in-pool**, 0 fallbacks | `h2_p4_4.jsonl` |
| Live demo (Streamlit 5-stage) | ✅ Working | `demo/ui/tab_inference.py` |
| Scene Graph Graphviz in demo | ✅ Done | `tab_inference.py:659-808` |
| Occlusion saliency explain | ✅ Deployed | — |
| `object_material` in P0 Cypher | **Not wired** | Blocked → T1.3 |
| Nested/compound spatial queries | **Not implemented** | Dropped → future work |

---

## 0. Sharpened Research Question

**"Can neuro-symbolic spatial reasoning disambiguate visually identical
building elements where pure VLM / attribute-based methods fail?"**

This is mainstream, high-visibility, career-relevant because it sits at
the intersection of:

| Mainstream AI Thread | Your Instantiation |
|---|---|
| LLM + Tool Use (Toolformer, Gorilla, ReAct) | VLM + Neo4j Cypher |
| Structured Output / Function Calling | LoRA-trained JSON spatial predicates |
| Hallucination Prevention | 0% false positive rate on spatial extraction |
| Neuro-Symbolic Reasoning (Think-on-Graph, RoG) | VLM perception → graph traversal |
| Visual Grounding (Kosmos-2, Qwen2.5-VL) | Spatial predicate grounding in site photos |

**One-sentence thesis contribution**: By decomposing element retrieval into
VLM spatial predicate extraction (neuro) and graph database traversal (symbolic),
we achieve 5x improvement on a benchmark where 46 identical elements per floor
defeat all attribute-only methods.

---

## 1. Data Reality (Verified Against AdvancedProject.ifc)

### 1.1 The Entropy Bottleneck (Ground Truth)

```
Floors 1-5: 46 identical IfcWindows each
  → All hosted by "MockUp Exterior" walls
  → All walls have SAME material (Plaster|Leather, weathered)
  → Attribute baseline Top-1 = 2.2% per floor (1/46)
```

### 1.2 What Actually Discriminates (Simulation Results)

```
Feature Combination                 Groups  MaxPool  AvgPool  Top-1
────────────────────────────────────────────────────────────────────
storey only                              7      46     37.6    6.8%
storey + wall_material                   7      46     37.6    6.8%  ← DEAD
storey + window_subtype (obj_type)      61      13      4.3   47.2%  ← BEST ACCESSIBLE
storey + which_wall (wall_guid)         45      17      5.8   37.5%
storey + wall_guid + obj_type          135       6      1.9   71.5%  ← THEORETICAL MAX
```

### 1.3 Key Findings That Change the Plan

| Finding | Impact |
|---|---|
| **Wall material = DEAD for FILLS (windows)**. All 17 window-hosting walls on floors 1-5 share `Plaster\|Leather, weathered`. All 263 windows share identical material. | Material gives **0x improvement** for window retrieval. |
| **Material ALIVE for other types**. Doors 3.8x, Walls 2.5x, Furniture 40x, Proxy 7x improvement. | **KEEP P5.1-5.2** — material helps non-window elements significantly. |
| **Window ObjectType IS discriminating**. 5 subtypes (BALANS 15M/10M/20M/25M/30M) = different physical sizes. VLM can see size differences. | **ELEVATE P5.3** (target_name_keyword post-filter). Highest ROI single change. |
| **IfcRelConnectsPathElements** (686 edges): wall-to-wall topology. NOT currently in Neo4j. | Load into graph → enables 2-hop. Future work, not sprint. |
| **IfcRelSpaceBoundary = 0** in AdvancedProject. Room-level filtering impossible. | DROP space_name filtering for AP. |
| **Doors have same problem**: Floor 1 has 42 identical `M_Single-Flush:Generic Door`. | Validates the research question generalizes beyond windows. |

#### Material Discrimination by Element Type (Verified)

```
Element Type            Count  storey   storey+mat  storey+mat+type  Mat Boost
                               Top-1    Top-1       Top-1
──────────────────────────────────────────────────────────────────────────────
IfcFurnishingElement     407    0.7%     26.8%       35.7%            40.4x  ←← HUGE
IfcBuildingElementProxy  358    4.6%     32.3%       54.3%             7.0x
IfcDoor                  126   10.2%     38.6%       35.9%             3.8x
IfcWallStandardCase      381    4.2%     10.5%        9.3%             2.5x
IfcSlab                   43   83.7%     91.2%       91.2%             1.1x
IfcWindow                263    6.8%      6.8%       47.2%             1.0x  ← DEAD
```

Key: material is the single best discriminator for furniture/proxy/doors.
For windows, `obj_type` (window subtype/size) is the only effective signal.

### 1.4 Actual H2 Eval Pool Sizes (213 cases, 2026-03-11)

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

### 1.5 Projected Improvement Scenarios

```
Scenario                              FILLS    ADJ_TO   Overall  Effort
                                      Top-1    Top-1    Top-1
──────────────────────────────────────────────────────────────────────
S0: Current (P0 only)                  2.4%    12.1%     5.3%    Done
S1: + obj_type post-filter (P5.3)     ~11%    ~20%     ~14%     1 day    ← DO THIS
S2: + 2-hop ConnectsPath (P6)         ~40%    ~30%     ~30%     1 week   ← Future work
S3: Theoretical max                   ~71%    ~50%     ~50%     —
```

**S1 is the highest-ROI change**: 1 day of work, 2.6x improvement,
uses existing VLM output field (`target_name_keyword`), no retraining.

---

## 2. What to DO (Priority-Ordered)

### Sprint 1: Evidence (Days 1-5)

#### Day 1: Quick Wins

**T1.1 — Implement target_name_keyword post-filter (S1)**
```
File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → after Cypher returns candidates

Logic:
  if constraints.target_name_keyword and len(candidates) > 3:
      filtered = [c for c in candidates
                  if constraints.target_name_keyword.lower() in c['name'].lower()]
      if filtered:  # graceful: don't filter to empty
          candidates = filtered

Expected: FILLS pool 42→~9, ADJACENT_TO pool 30→~15
```

**T1.2 — Fix continuous_span storey filter (P4.3)**
```
File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → continuous_span branch

Current Cypher:
  WHERE target.is_continuous = true AND target.top_constraint = $top_constraint

Add:
  AND target.storey = $storey   ← base storey filter

Expected: CONTINUOUS pool 110→~46, SSR 47%→~70%
```

**T1.3 — Add material property to Neo4j + wire into P0 Cypher (P5.1-5.2)**
```
File: mscd_demo/src/ifc_engine.py
Location: _create_element_nodes()

P5.1 — Add material property:
  # Extract from psets
  material_value = psets.get("Materials and Finishes", {}).get("Structural Material", "")
  node_props["material"] = material_value

File: mscd_demo/src/v2/retrieval_backend.py
Location: _execute_neo4j() → spatial_triplet branch

P5.2 — Add WHERE clause (graceful — empty string = no filter):
  AND ($object_material = '' OR toLower(ref.material) CONTAINS toLower($object_material))

Expected impact by element type:
  IfcWindow:              0x (all same material — DEAD)
  IfcDoor:              3.8x (10.2% → 38.6% Top-1)
  IfcWallStandardCase:  2.5x (4.2% → 10.5%)
  IfcFurnishingElement: 40x  (0.7% → 26.8%)
```

#### Day 2-3: 3-Way Eval (P10)

**Systems under comparison:**

| System | Extractor | Adapter | Output Schema | Max Priority |
|---|---|---|---|---|
| **Baseline (Gemini)** | `PromptConstraintsExtractor` | None | 7 fields (no spatial_relations) | P1-P8 |
| **LoRA_2** | `LoRAConstraintsExtractor` | `v2_lora_qwen/final/` | 7 fields (no spatial_relations) | P1-P8 |
| **LoRA_3** | `LoRAConstraintsExtractor` | `v3_lora_qwen_20260310_5ep/final/` | 5 fields + spatial_relations | **P0-P8** |

All 3 use the same system prompt (`prompts/constraints_extraction.yaml`).
LoRA_3's advantage: learned `spatial_relations` extraction → fires P0 Cypher.

**Test data choice:**

| Option | Cases | Leakage Risk | Notes |
|---|---|---|---|
| `cases_v3_filtered.jsonl` (v0.4_ap) | 250 | HIGH — overlaps LoRA_2 + LoRA_3 train | Fast, existing format |
| Convert `lora3_test.jsonl` → cases_v3 | 69 | LOW — held-out for LoRA_3 | **Recommended** |
| Cross-IFC (BH + DXA cases) | ~64 | MEDIUM — LoRA_3 saw some | Tests generalization |

**T2.1 — Convert test data**
```
File: NEW mscd_demo/eval/convert_lora3_test.py
Input: data_curation/datasets/synth_v0.5/lora3_test.jsonl (69 held-out cases)
Output: mscd_demo/eval/cases_v3_test.jsonl

Logic: extract from each JSONL record:
  - case_id: generate from index
  - query_text: user message content
  - image_path: from user message image field
  - ground_truth_guid: from assistant message JSON → match to element_index
  - bench fields: ifc_class, storey_name from assistant JSON
```

**T2.2 — Run 3 evaluations** (same Neo4j state, same 69 cases)
```bash
# Baseline (Gemini prompt)
python script/run.py --profile v2_prompt \
  --cases eval/cases_v3_test.jsonl --limit 69

# LoRA_2 (attribute-only adapter)
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v2_lora_qwen/final/ \
  --cases eval/cases_v3_test.jsonl --limit 69

# LoRA_3 (spatial adapter)
python script/run.py --profile v2_lora \
  --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
  --cases eval/cases_v3_test.jsonl --limit 69
```

**T2.3 — Collect metrics per run**
```
For each run, compute:
  - Top-1 Accuracy (exact GUID match)
  - Top-5 Accuracy
  - Mean Reciprocal Rank (MRR)
  - Search Space Reduction (SSR) = 1 - pool_size / total_elements
  - P0 fire rate (% of cases where spatial_triplet was used)
  - Spatial predicate extraction accuracy (vs ground truth)
  - Over-reduction rate (GT dropped from pool)
```

#### Day 4: Generate Plots (P10.5 / P4.5)

**T3.1 — 3-way comparison plots**
```
File: mscd_demo/eval/plot_3way.py (or extend compare_results.py)

Plot 1: Bar chart — Top-1 / Top-5 / MRR across 3 systems
Plot 2: SSR distribution (violin/box) per predicate per system
Plot 3: Pool size waterfall: Total → Storey → P0 → Post-filter
Plot 4: The "money chart" — Entropy Collapse:
        X-axis: pipeline stage (Input → P0 → PostFilter)
        Y-axis: candidate count
        3 lines (Baseline flat, LoRA_2 flat, LoRA_3 drops sharply)
```

#### Day 5: Buffer + Edge Count Pre-check (P4.2)

**T4.1 — Add Neo4j edge pre-check to h2_eval.py**
```
File: mscd_demo/eval/h2_eval.py
Location: top of main()

driver.execute_query("MATCH ()-[r:FILLS]->() RETURN count(r) AS n")
assert n >= 100, f"Graph incomplete: only {n} FILLS edges. Run neo4j_init.sh."
# Same for ADJACENT_TO >= 50
```

### Sprint 2: Demo + Thesis (Days 6-14)

#### Day 6-7: Entropy Collapse Demo Panel

**T5.1 — Add "Entropy Collapse" tab to Streamlit**
```
File: mscd_demo/demo/ui/tab_inference.py (extend existing)

Design:
  Left column:  "Attribute-Only Baseline"
    → Show grid of 46 identical window icons, all gray
    → Caption: "46 candidates — system cannot distinguish"

  Right column: "Neuro-Symbolic (Ours)"
    → Stage 1: 46 icons (storey filter)
    → Stage 2: ~30 icons highlighted (P0 ADJACENT_TO fires, 16 removed)
    → Stage 3: ~9 icons (obj_type post-filter)
    → Final: 1 icon glows green, GUID displayed

  Data source: run a real case through the pipeline,
  log pool size at each stage via RetrievalResult.strategy_actually_used
```

**T5.2 — Post-retrieval result viewer (P7.4)**
```
File: mscd_demo/demo/ui/tab_inference.py

3-column layout after retrieval completes:
  Col 1: Input site photo (as uploaded)
  Col 2: Floorplan patch centered on matched GUID
         → reuse _floorplan_renderer.py render_patch()
         → highlight matched element in red
  Col 3: Neo4j 1-hop subgraph (Graphviz DOT)
         → reuse existing scene graph renderer

All components already exist — just compose into a layout.
```

#### Day 8-14: Thesis Writing

```
Chapter 4 — Method: Neuro-Symbolic Architecture
  4.1 Problem formalization (attribute entropy bottleneck)
  4.2 VLM spatial predicate extraction (LoRA_3 training)
  4.3 Priority cascade query planner
  4.4 Neo4j Cypher execution engine
  4.5 Fallback and graceful degradation

Chapter 5 — Experiments
  5.1 Dataset: synth_v0.5 (1377 train / 69 test)
  5.2 Benchmark: H2 hard-negatives (213 cases)
  5.3 3-way ablation (Baseline / LoRA_2 / LoRA_3)
  5.4 Per-predicate analysis
  5.5 Error analysis: when does P0 fail?

Chapter 6 — Discussion + Future Work
  6.1 Contribution: neuro-symbolic > pure neural for structured retrieval
  6.2 Limitation: FILLS SSR=0% (wall identity problem)
  6.3 Future: IfcRelConnectsPathElements (686 edges, untapped)
  6.4 Future: 2-hop compound queries (simulated 71.5% ceiling)
  6.5 Future: visual grounding with bbox (Qwen2.5-VL native)
  6.6 Future: cross-IFC generalization
```

---

## 3. What to DROP (with reasons)

| Item | Reason to Drop |
|---|---|
| ~~**P5.1-5.2**~~ | ~~REINSTATED~~ — material is dead for windows (1.0x) but alive for doors (3.8x), walls (2.5x), furniture (40x). Keep as quick win (1 day). |
| **P7.2-7.3** (bbox extraction + overlay) | Needs new training data with bbox annotations + new LoRA. Qwen2.5-VL supports native grounding — note as future work (Ch6.5). |
| **P8.1-8.2** (render black-image bug + re-render) | Cosmetic. Existing renders sufficient for thesis figures. |
| **P8.3-8.5** (raise quotas, cross-IFC, compound skeletons) | Nice-to-have. 213 H2 cases + 69 test cases = sufficient evidence. |
| **P9** (4D timestamp) | Low impact, tangential to core question. |
| **P11** (DPO, D1 condition) | Optimization, not contribution. Future work. |
| **new.md SGG schema** | Invalidates all existing data + pipeline. 3-week rewrite for uncertain gain. |
| **new.md dual-track bbox** | Scope explosion. Needs new renderer, new training data, new eval metrics. |
| **new.md LoRA_4** | No time. LoRA_3 (93% acc, 0% FP) is thesis-ready. |
| **P6.1-6.4** (2-hop implementation) | High value but blocked by: need multi-triplet training data → new LoRA. Present as SIMULATED future work with data ceiling proof (71.5%). |

---

## 4. IFC Relationship Taxonomy (Reference for Thesis)

Complete inventory discovered from AdvancedProject.ifc:

```
Relationship                     Count  Used?   Discrimination
──────────────────────────────────────────────────────────────
IfcRelFillsElement                389   ✅ P0   Window/Door→Wall (SSR=0% same-material)
IfcRelConnectsPathElements        686   ✗ NEW   Wall→Wall topology (ATSTART/ATEND/ATPATH)
                                                2-hop enables 71.5% Top-1 (simulated)
IfcRelVoidsElement                427   ✅      Intermediate for FILLS chain
IfcRelContainedInSpatialStructure  10   ✅      Storey assignment
IfcRelAssociatesMaterial         1345   ✗       DEAD for window-hosting walls
IfcRelDefinesByProperties       19877   ✅      Psets (name, type, dimensions)
IfcRelDefinesByType               202   ✗       Type definitions
IfcRelAssignsToGroup              154   ✗       Furniture groups (low value)
IfcRelConnectsPortToElement       139   ✗       MEP ports only
IfcRelAggregates                   17   ✅      Railing decomposition
IfcRelSpaceBoundary                 0   N/A     MISSING in AdvancedProject
                                                (Duplex_A has 264 — room-level filtering)
```

**Key discovery: `IfcRelConnectsPathElements` is the richest untapped signal.**

### 4.1 IfcRelConnectsPathElements Deep Dive

686 wall-to-wall edges with explicit connection semantics:

```
Connection Types (how walls meet):
  ATSTART ← wall's start point connects to another wall
  ATEND   ← wall's end point connects to another wall
  ATPATH  ← wall intersects along its length (T-junction)

Connection type pair distribution:
  ATEND↔ATSTART:   194 edges  (end-to-start chain)
  ATSTART↔ATSTART: 125 edges  (shared start corner)
  ATSTART↔ATEND:    93 edges
  ATPATH↔ATEND:     89 edges  (T-junction)
  ATPATH↔ATSTART:   80 edges  (T-junction)
  ATEND↔ATEND:      67 edges  (shared end corner)
  ATEND↔ATPATH:     26 edges
  ATSTART↔ATPATH:   12 edges
```

Wall-to-wall connectivity graph stats:
```
  389 walls in graph, 681 undirected edges
  Avg degree: 3.5 connections/wall
  Degree distribution:
    degree 1:  19 walls (dead-end)
    degree 2: 140 walls (inline/pass-through)
    degree 3:  90 walls (T-junction)
    degree 4:  62 walls (cross or 2x T)
    degree 5+: 78 walls (complex junctions)
    max: 22 connections (the 85-window exterior wall)
```

### 4.2 Window-Hosting Wall Topology

Only **17 walls** host windows (out of 389 total), but they vary significantly:

```
Wall (windows/floor × floors)     Material Layers            Degree  Storey
────────────────────────────────────────────────────────────────────────────
MockUp Exterior:793582 (17×5=85)  Plaster|Leather,weathered   22     Fl 1-5
MockUp Exterior:793581 (10×5=50)  Plaster|Leather,weathered    9     Fl 1-5
MockUp Exterior:793587 ( 9×5=45)  Plaster|Leather,weathered   10     Fl 1-5
MockUp Exterior:793580 ( 4×5=20)  Plaster|Leather,weathered    4     Fl 1-5
MockUp Exterior:793583 ( 3×5=15)  Plaster|Leather,weathered    3     Fl 1-5
MockUp Exterior:793584 ( 2×5=10)  Plaster|Leather,weathered    —     Fl 1-5
MockUp Exterior:793586 ( 1×5= 5)  Plaster|Leather,weathered    —     Fl 1-5
Ground Floor walls (7 walls, 30)  Brick, Engineering          3-4    Level 1
Roof walls (2 walls, 3)          Render, Beige, Textured       —     Fl 6
```

Key insight: **degree varies 3-22** across the 7 major window-hosting walls.
If the VLM can identify the wall (by position, by number of neighbors, by
corner/T-junction pattern), the pool drops from 46 → 17 max per floor.

### 4.3 2-Hop Spatial Signature Simulation

With a 2-hop query `Window → Wall → ConnectedWalls`, each window gets
a "spatial signature" = (storey, host_wall_guid, neighbor_set):

```
Unique 2-hop signatures: 45 (for 263 windows)
Signature-size distribution:
  pool=1:   8 windows (uniquely identifiable)
  pool=2:   7 signatures (14 windows)
  pool=3:   6 signatures (18 windows)
  pool=4:   7 signatures (28 windows)
  pool=7-9: 7 signatures (59 windows)
  pool=10:  5 signatures (50 windows)
  pool=17:  5 signatures (85 windows)  ← the big exterior wall

  Uniquely identifiable (pool=1):  8/263 (3%)
  Pool ≤ 3:                       40/263 (15%)
```

The bottleneck: the 85-window wall (degree=22) hosts 17 windows/floor —
even with 2-hop, those 17 share the same wall signature.
Adding `obj_type` breaks this further: 17 → avg 1.9 per group.

### 4.4 Feasibility Assessment: What Can the VLM Actually Extract?

| Signal | Pool | Top-1 | VLM Feasibility | Notes |
|---|---|---|---|---|
| Storey only | 46 | 6.8% | ✅ Already done (93% acc) | — |
| + which wall (visual position) | 17 max | 37.5% | ❌ Hard | Wall identity not in VLM schema |
| + window subtype/size | 13 max | 47.2% | ⚠️ Medium | BALANS 15M (1.5m) vs 10M (1.0m) — visually different sizes |
| + wall + obj_type | 6 max | 71.5% | ❌ Hard | Needs wall identification |
| + 2-hop connectivity | 1-3 | ~85% | ❌ Hard | Needs topology reasoning |

**Honest answer**: 14-47% Top-1 is achievable with current + obj_type filter.
71.5% requires wall identification, which is the core **future work** direction.
The gap between 47% and 71% = the value of `IfcRelConnectsPathElements`.

### 4.5 Duplex_A: IfcRelSpaceBoundary (264 instances)

Unlike AdvancedProject (0 space boundaries), Duplex_A has rich room-level data:

```
Element types bounded to spaces:
  IfcWallStandardCase: 90 boundaries
  IfcSlab:             68
  IfcDoor:             28
  IfcCovering:         20
  IfcWindow:           20
  IfcRoof:             16
  IfcWall:              6

18 named spaces: A101, A201, B101, B203, etc.
Each space has 9-19 boundary elements.
```

This means **room-level filtering (space_name) works on Duplex_A** but not AP.
Cross-IFC generalization would demonstrate both code paths.

---

## 5. Expected Results Table (For Thesis)

### 5.1 3-Way Eval (Projected from Data Reality)

| Metric | Baseline (Gemini) | LoRA_2 | LoRA_3 (Ours) |
|---|---|---|---|
| Spatial predicate extracted | 0% | 0% | **93%** |
| P0 fired | 0% | 0% | **~56%** |
| Top-1 Accuracy | ~3-6% | ~5-8% | **~14-20%** |
| Top-5 Accuracy | ~15% | ~20% | **~40-55%** |
| SSR (P0 cases only) | N/A | N/A | **~45%** |
| SSR (overall) | ~60% | ~65% | **~75-80%** |
| Over-reduction (GT lost) | ~20% | ~15% | **<5%** |
| False positive (hallucinated spatial) | N/A | N/A | **0%** |

**Fairness Notes:**
- **System prompt mismatch**: Prompt describes LoRA_2's 7-field schema, but
  LoRA_3 learned to output 5-field + spatial_relations. This is fair — it shows
  LoRA fine-tuning learned capabilities beyond what prompt engineering provides.
- **Optional 4th baseline**: Gemini with LoRA_3-schema prompt (spatial_relations
  in output spec) — isolates fine-tuning benefit from schema benefit.
  Only pursue if time allows after Sprint 1.

### 5.2 Per-Predicate Breakdown (H2 Benchmark)

| Predicate | Cases | Pre-P0 Pool | Post-P0 Pool | Post-Filter Pool | SSR | Top-1 |
|---|---|---|---|---|---|---|
| ADJACENT_TO | 66 | 116.8 | 29.8 | ~15 (S1) | 57%→~87% | ~20% |
| FILLS | 84 | 42.4 | 42.4 | ~9 (S1) | 0%→~79% | ~11% |
| CONTINUOUS | 63 | 227.4 | 110.2 | ~50 (P4.3) | 47%→~78% | ~5% |

### 5.3 Theoretical Ceiling (Simulated, for Future Work section)

| Strategy | Avg Pool | Top-1 | Requires |
|---|---|---|---|
| S0: P0 only (current) | 58.6 | 5.3% | Done |
| S1: + obj_type post-filter | ~25 | ~14% | T1.1 (1 day) |
| S2: + 2-hop ConnectsPath | ~5 | ~30% | Neo4j + LoRA_4 (future) |
| S3: All features (oracle) | 1.9 | 71.5% | Perfect VLM (theoretical) |

---

## 6. Sprint Checklist

```
SPRINT 1: EVIDENCE (Days 1-5)
─────────────────────────────
□ T1.1  target_name_keyword post-filter in retrieval_backend.py     (Day 1)
□ T1.2  Fix continuous_span storey filter                           (Day 1)
□ T1.3  P5.1-5.2: Add material to Neo4j + wire into P0 Cypher      (Day 1)
        (dead for windows, but 3.8x doors, 2.5x walls, 40x furniture)
        See T1.3 implementation details below.
□ T2.1  Convert lora3_test.jsonl → cases_v3 format                 (Day 2)
□ T2.2  Run 3-way eval (Baseline / LoRA_2 / LoRA_3)                (Day 2-3)
□ T2.3  Collect per-run metrics                                     (Day 3)
□ T3.1  Generate 4 thesis plots                                     (Day 4)
□ T4.1  Neo4j edge pre-check in h2_eval.py                         (Day 5)

SPRINT 2: DEMO + THESIS (Days 6-14)
────────────────────────────────────
□ T5.1  Entropy Collapse demo panel                                 (Day 6-7)
□ T5.2  Post-retrieval result viewer (P7.4)                         (Day 7)
□ T6.1  Chapter 4: Method                                           (Day 8-9)
□ T6.2  Chapter 5: Experiments (use real numbers from T2-T3)        (Day 10-11)
□ T6.3  Chapter 6: Discussion + Future Work                         (Day 12-13)
□ T6.4  Revisions and figures                                       (Day 14)
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
   traverses the IFC topology graph)."

Act 3 — THE EVIDENCE (Chapter 5)
  "3-way ablation on 69 held-out cases proves:
   Baseline: 3-6% → LoRA_2: 5-8% → Ours: 14-20% Top-1.
   213-case H2 benchmark: 100% GT-in-pool, 0 fallbacks.
   Zero hallucination rate on spatial predicates."

Act 4 — THE CEILING (Chapter 6)
  "Data simulation shows 71.5% Top-1 is achievable with 2-hop
   topology (IfcRelConnectsPathElements, 686 untapped edges).
   The architecture is ready — only training data limits us."
```

**The story reviewers will remember**: "The system that turned 46 identical
windows into 1 correct match by reasoning about spatial topology."

---

## Appendix A: Compound Spatial Query Design (Future Work Reference)

Preserved from 0307 P6 design for thesis Chapter 6.4:

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
Con: Subject of triplet[1] is ambiguous (window or wall?).

**Option B — Chained triplets (2-hop Cypher)**:
```cypher
MATCH (target:IFCElement)-[:FILLS]->(ref1:IFCElement)-[:ADJACENT_TO]->(ref2:IFCElement)
WHERE target.ifc_type STARTS WITH 'IfcWindow'
  AND ref1.ifc_type STARTS WITH 'IfcWall'
  AND ref1.material CONTAINS 'Concrete'
  AND ref2.ifc_type STARTS WITH 'IfcColumn'
RETURN DISTINCT target
```
Pro: Most precise, single query. Con: Requires schema change + new training data.
**Recommended**: Start with Option A, upgrade to B later.

Prerequisite: multi-triplet training examples (P6.1-6.2) + new LoRA (P6.3).
The `spatial_relations` field is already a list in the schema — architecture is ready.

---

## Appendix B: Key References

```
Adapter checkpoints:
  LoRA_2: models/adapters/v2_lora_qwen/final/
  LoRA_3: models/adapters/v3_lora_qwen_20260310_5ep/final/ (Modal: /mscd-lora/final)

System prompt: prompts/constraints_extraction.yaml
H2 eval results: eval_h2_spatial_triplets/results/h2_p4_4.jsonl
Element index: data_curation/references/element_index.jsonl (1233 elements)
Primary IFC: data_curation/ifc_models/AdvancedProject.ifc (IFC2X3, mm units)
Neo4j: bolt://localhost:7687, pw=password, /tmp/neo4j-community-5.26.0/
```

---

*Last updated: 2026-03-13. Based on verified data from AdvancedProject.ifc,
h2_p4_4.jsonl (213/213 GT-in-pool), and IFC relationship inventory.
Supersedes: 0307_post_mid_plan.md and new.md (both deleted).*
