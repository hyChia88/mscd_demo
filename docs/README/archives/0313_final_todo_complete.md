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

### 1.4 H2 Eval Results

**Before fixes (2026-03-11, 213 cases)**: P0 ran in DEGRADED MEMORY MODE (Neo4j never connected from run.py).
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

**After fixes + graph enrichment (2026-03-15, 568 cases)**: Neo4j connected,
NEXT_TO + CONNECTS_TO edges added, storey resolver generalized.
```
Predicate      Cases  Pre-P0    Post-P0   SSR     Top-1(1/pool)  Fallback
                      avg pool  avg pool
──────────────────────────────────────────────────────────────────────────
FILLS            84     42.4     60.3    -49%       2.4%          0/84
NEXT_TO          25     33.0     12.7    +60%       3.1%          0/25   ← NEW, BEST SSR
CONNECTS_TO     330    278.0    305.6    -34%       0.7%          0/330  ← NEW
ADJACENT_TO      66    120.0    117.2    -24%       4.5%         66/66
CONTINUOUS       63    227.4    381.0    -74%       0.5%         63/63
──────────────────────────────────────────────────────────────────────────
OVERALL         568    208.0    243.0    -35%       1.4%        129/568
GT-in-pool: 567/568 (100%)  |  Fallback: 129/568 (23%)
```
Note: FILLS SSR negative because storey siblings now return BOTH "Level X" + "X - Xth Floor"
elements. The T1.1 target_name_keyword post-filter (already wired) will fix this.
ADJACENT_TO + CONTINUOUS fallback because they use centroid-distance/property-based
retrieval, not Neo4j edges — these will improve with NEXT_TO training data (LoRA_5).

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
| Multi-triplet miner v1 (389 FILLS+CONNECTS_TO) | ✅ Done (03-14) | `7_mine_multitriplet.py` |
| Multi-triplet miner v2 (173 records, 5 chain patterns, 3-hop) | ✅ Done (03-16) | `7b_mine_multitriplet_v2.py` |
| **LoRA_4 trained** | ✅ Done (03-15) | Modal A100, Qwen2.5-VL-7B + LoRA |
| **LoRA_5 trained** | ✅ Done (03-17) | Modal A100, multi-triplet + CONNECTS_TO/NEXT_TO |
| **LoRA_5 full eval (MA/MB/MC/FP/SITE)** | ✅ Done (03-18) | `logs/evaluations/synth_v05_lora5/` |
| **P0∩P1 strategy + ablation** | ✅ Done (03-17) | `strategy_ablation/` (P0-only/P1-only/P0∩P1/P0∪P1) |
| **4-way comparison (Gemini/L3/L4/L5)** | ✅ Done (03-17) | `plots/comparisons/0317_4way_ap_only/` |
| **Gemini zero-shot baseline** | ✅ Done (03-18) | `logs/evaluations/gemini_baseline/` |
| **analyze_traces.py deep-dive** | ✅ Done (03-18) | Valid SSR, RQS, hop accuracy, confusion matrices |
| **Multi-model Neo4j (AP+DXA+BH)** | ✅ Scripted (03-18) | `script/neo4j_init.sh` + `ifc_export_cli.py --no-clear` |
| **RESULTS.md Exp 4 analysis** | ✅ Done (03-20) | Root cause analysis: class confusion, invalid predicates |

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
✅ T8.3  Train LoRA_4 on Modal A100 (Qwen2.5-VL-7B, 5ep, lr=2e-4)       Done 03-15
✅ T8.4  Run MC eval on LoRA_4 adapter                                    Done 03-17
✅ T8.5  Compare LoRA_4 vs LoRA_3 (4-way comparison generated)            Done 03-17

SPRINT 4B: 2-HOP OPTIONAL MATCH (Days 20-22)
─────────────────────────────────────────────
✅ T8.6  Wire CONNECTS_TO into Neo4j (1362 edges)                         Done 03-14
✅ T8.7  Mine multi-triplet cases (389 FILLS+CONNECTS_TO)                 Done 03-14
✅ T8.8  Wire OPTIONAL MATCH 2-hop in retrieval_backend.py                Done 03-14
✅ T8.9  Update 6_assemble_lora4.py for multi-triplet records             Done 03-14
✅ T8.10 Re-assemble LoRA_4 (649 train, 110 2-hop, 75% SR)               Done 03-14
□ T8.11 Eval 2-hop: GT-in-pool improvement (target: 71.5% ceiling)

SPRINT 5: LoRA_5 + FULL EVAL (Days 22-25)
──────────────────────────────────────────
✅ T9.1  Train LoRA_5 (multi-triplet + CONNECTS_TO/NEXT_TO)               Done 03-17
✅ T9.2  Run LoRA_5 full eval (MA/MB/MC/FP/SITE conditions)               Done 03-18
✅ T9.3  P0∩P1 strategy + ablation (4 strategies)                         Done 03-17
         Results: logs/evaluations/synth_v05_lora5/strategy_ablation/
✅ T9.4  Gemini zero-shot baseline on same test set                        Done 03-18
         Results: logs/evaluations/gemini_baseline/
✅ T9.5  4-way comparison: Gemini vs LoRA_3 vs LoRA_4 vs LoRA_5           Done 03-17
         Results: plots/comparisons/0317_4way_ap_only/ (AP-only filtered)
✅ T9.6  Deep-dive analysis (hop accuracy, confusion, RQS, per-floor)     Done 03-18
         Results: logs/evaluations/synth_v05_lora5/plots/
✅ T9.7  analyze_traces.py: Valid SSR, RQS F1, hop waterfall              Done 03-18
✅ T9.8  RESULTS.md Exp 4 writeup + root cause analysis                   Done 03-20
         Root causes: 49% ifc_class wrong, 32.6% invalid predicates

SPRINT 5B: INFRASTRUCTURE FIXES (Identified 03-18-20)
──────────────────────────────────────────────────────
✅ T10.1 Script multi-model Neo4j loading (AP+DXA+BH)                     Done 03-18
         neo4j_init.sh + ifc_export_cli.py --no-clear
□ T10.2 Run neo4j_init.sh --reload (load DXA+BH into Neo4j)
□ T10.3 Re-run all evals after Neo4j reload (27 cases currently 0% GT)
□ T10.4 Fix storey_match eval bug (pipeline.py:211, candidate.storey=null)
□ T10.5 Filter invalid predicates (CONNECTS_TO→valid, NEXT_TO→ADJACENT_TO)
□ T10.6 Add Top-10 metric to compare_results.py
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

### 5.10 4-Way Comparison: Gemini vs LoRA_3 vs LoRA_4 vs LoRA_5 (Done 03-17/20)

**Full analysis**: See `RESULTS.md` Exp 4.
**Traces**: `plots/comparisons/0317_4way_ap_only/`
**Charts**: `plots/comparisons/0317_4way_ap_only/charts/`

| Metric | Gemini (n=59) | LoRA_3 (n=20) | LoRA_4 (n=58) | LoRA_5 (n=59) |
|--------|--------------|---------------|---------------|---------------|
| **Top-1** | 1 (1.7%) | 3 (15.0%) | 4 (6.9%) | 3 (5.1%) |
| **GT-in-pool** | 7 (11.9%) | 12 (60.0%) | 20 (34.5%) | 17 (28.8%) |
| **ifc_class** | 33 (55.9%) | 19 (95.0%) | 37 (63.8%) | 29 (49.2%) |
| **storey_num** | 30 (50.8%) | 16 (80.0%) | 29 (50.0%) | 39 (66.1%) |
| **P0 used** | 55+3 | 0 (all P1) | 41+1 | 57+2 |

**Root causes of LoRA_5 regression** (vs LoRA_3/4):
1. **49% ifc_class wrong**: Wall GT misclassified as Window/Door (13/59 cases). FILLS-dominant training biases toward subject elements.
2. **32.6% invalid predicates**: CONNECTS_TO (38) + NEXT_TO (8) not in Neo4j schema → empty Cypher results.
3. **Non-comparable test sets**: LoRA_3 has only 20 AP cases (easy, v3 skeletons) vs LoRA_5's 59 (harder augmented v5 cases). Zero ID overlap.
4. **storey_match = 0% universally**: Eval bug — `pipeline.py:211` reads `c.get("storey")` but candidates lack this key.

**LoRA_5 deep-dive plots**: `logs/evaluations/synth_v05_lora5/plots/`
- Hop accuracy, predicate confusion matrix, subject confusion matrix, per-floor GT-in-pool, RQS overview, hop waterfall

**Strategy ablation** (LoRA_5 MC condition):
- Traces: `logs/evaluations/synth_v05_lora5/strategy_ablation/`
- 4 strategies: P0-only, P1-only, P0∩P1, P0∪P1

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
| **LoRA_4 training** (new.md) | ✅ **COMPLETED** 03-15. LoRA_5 also trained 03-17. 4-way eval done. Key finding: 49% ifc_class wrong, 32.6% invalid predicates. See RESULTS.md Exp 4. | §5.7 + RESULTS.md |
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

---

## 10. LoRA_4 Evaluation Results (2026-03-15)

### 10.1 VLM Extraction Metrics (75 test cases)

| Metric | MA (text) | MB (text+photo) | MC (text+photo+fp) | LoRA_3 MC |
|--------|-----------|-----------------|---------------------|-----------|
| **Parse Rate** | 75/75 (100%) | 75/75 (100%) | 75/75 (100%) | 100% |
| **Field EM F1** | 0.533 | 0.480 | 0.507 | — |
| **Storey Accuracy** | 36/75 (48.0%) | 27/75 (36.0%) | 31/75 (41.3%) | 52.9% |
| **IFC Class Acc** | 44/75 (58.7%) | 45/75 (60.0%) | 45/75 (60.0%) | 67.6% |
| **SR Extraction** | 26/75 (34.7%) | 12/75 (16.0%) | **56/75 (74.7%)** | 1/69 (1.4%) |
| **Predicate Acc** | 10/26 (38.5%) | 6/12 (50.0%) | **29/47 (61.7%)** | 1/1 |
| **FP Rate (SR)** | 5/75 (6.7%) | 3/75 (4.0%) | 9/75 (12.0%) | 0% |
| **FN Rate (SR)** | 26/47 (55.3%) | 38/47 (80.9%) | **0/47 (0.0%)** | 46/47 (97.9%) |
| **MRR** | 0.029 | 0.025 | 0.019 | 0.039 |
| **mR@10** | 8/75 (10.7%) | 6/75 (8.0%) | 4/75 (5.3%) | — |
| **mR@50** | 17/75 (22.7%) | 15/75 (20.0%) | 12/75 (16.0%) | — |
| **mR@100** | 19/75 (25.3%) | 18/75 (24.0%) | 14/75 (18.7%) | — |

**Key takeaway**: MC SR extraction jumped from 1.4% (LoRA_3) to **74.7%** (LoRA_4).
MC FN=0% means the model **never misses** a spatial relation when floorplan is present.
Floorplan-SR coupling works: MA(35%) > MB(16%) < MC(75%).

### 10.2 Retrieval Pipeline Metrics (Neo4j connected, 75 cases)

| Metric | MA | MB | MC | LoRA_3 MC |
|--------|----|----|-----|-----------|
| **GT-in-pool** | **24/75 (32.0%)** | 21/75 (28.0%) | **20/75 (26.7%)** | 23/68 (33.8%) |
| GT-in-top10 | 8/75 (10.7%) | 6/75 (8.0%) | 4/75 (5.3%) | — |
| Top-1 | 0/75 (0%) | 0/75 (0%) | 0/75 (0%) | 1/68 (1.5%) |
| P0 fires | 26 | 12 | **56** | 0 |
| P0 GT-in-pool | 10/26 (38.5%) | 6/12 (50.0%) | **17/56 (30.4%)** | N/A |
| Avg pool | 93.7 | 89.3 | 77.0 | 60.1 |
| SSR | 94.4% | 94.6% | 95.4% | 92.9% |

### 10.3 P0 Deep Dive (MC condition)

#### P0 by storey correctness
```
                    P0 cases  GT-in-pool
Storey CORRECT:         25     17/25 (68%)   ← P0 works when storey is right
Storey WRONG:           31      0/31 (0%)    ← total wipeout
```

#### P0 by predicate
```
Predicate       Cases  GT-in-pool  Avg pool
ADJACENT_TO       46    14/46 (30%)    79
FILLS              9     2/9  (22%)    87
CONTINUOUS         1     1/1 (100%)    26
```

#### Reranking bottleneck (MC)
```
Pool has GT → Top-10 keeps GT:   4 cases
Pool has GT → Top-10 LOSES GT:  16 cases   ← 80% of GT hits lost in reranking
Pool misses GT:                 55 cases
```

GT ranks in lost cases: 10, 23, 25, 27, 33, 35, 38, 45, 80, 81, 181, 245, 247, 252, 257, 283

### 10.4 Root Cause Analysis

#### Issue 1: Storey Extraction (55% wrong on P0 cases)

**Symptom**: Model extracts wrong floor number in 31/56 P0 cases (MC).
P0 with correct storey = 68% GT-in-pool. P0 with wrong storey = 0%.

**Error pattern**: Model defaults to "-1" (garage) and "1" (first floor).
```
Biggest confusions (extracted → GT):
  -1 → 1:  14 cases  ← model says garage when GT is first floor
   1 → 4:   3 cases
   5 → 1:   3 cases
   4 → 1:   3 cases
   1 → 2:   3 cases
```

**By IFC model**:
```
AP:   storey=29/58 (50%)  SR=42/58 (72%)
DXA:  storey= 1/13  (8%)  SR=10/13 (77%)   ← catastrophic
BH:   storey= 1/4  (25%)  SR= 4/4 (100%)
```

**Root cause**: DXA uses "Level 1"/"Level 2" naming, BH uses different conventions.
Training data may not have enough cross-IFC storey diversity.
The model learned AP storey patterns but struggles to generalize.

**Possible fixes**:
1. **Training data**: Add more DXA/BH storey examples with correct labels
2. **Prompt engineering**: Explicit instruction "read the floor number from the
   floorplan legend/title" in system prompt
3. **Storey-from-floorplan**: Floorplan image often has floor label — train model
   to read it directly
4. **Post-hoc correction**: If floorplan filename contains floor number, use it
   as override when model confidence is low
5. **Storey relaxation**: When P0 returns 0 results, automatically retry without
   storey filter (already implemented — but not working because pool>0 with wrong storey)
6. **Expanded storey relaxation**: If P0 pool has no good reranking signal,
   retry with adjacent floors (±1)

#### Issue 2: Reranking (80% of GT hits lost in top-10 truncation)

**Symptom**: 20 cases have GT in retrieval pool, but only 4 survive to top-10.

**Root cause**: Pools of 45-290 elements are returned by Neo4j. The reranking
step (cosine similarity or simple truncation) doesn't discriminate well enough.
GT elements end up at ranks 23-283.

**Possible fixes**:
1. **Increase K**: Return top-50 instead of top-10 (R@50 = 16% vs R@10 = 5.3%)
2. **Better reranking**: Use `target_name_keyword` post-filter BEFORE truncation
3. **2-hop reranking**: Use hop-2 OPTIONAL MATCH to promote elements with
   matching wall connection signature
4. **Attribute-enhanced reranking**: Re-score pool using extracted ifc_class +
   storey + name_keyword for tighter filtering

#### Issue 3: MC paradoxically worse than MA/MB on GT-in-pool

**Symptom**: MA=32%, MB=28%, MC=27% — MC is worst despite 75% SR extraction.

**Explanation**: P0 over-fires in MC (56/75) and replaces storey+type fallback.
When P0 fires with wrong storey (31 cases), it produces worse pools than the
storey+type fallback would have. In MA/MB, more cases use storey+type (44-59/75)
which is a safer default.

**The paradox is expected**: P0 is high-risk/high-reward. With 55% wrong storey,
P0's 68% success on correct-storey cases can't compensate for the 0% on wrong-storey.
Once storey accuracy improves to >70%, MC will overtake MA/MB.

### 10.5 LoRA_4 vs LoRA_3 Comparison

```
Metric                    LoRA_3 (MC)  LoRA_4 (MC)  Delta     Status
────────────────────────────────────────────────────────────────────
SR Extraction Rate        1/69 (1.4%)  56/75 (75%)  +73.3pp   ✅ FIXED
SR FN (missed GT SR)      97.9%        0.0%         -97.9pp   ✅ FIXED
Parse Rate                100%         100%         same      ✅
Storey Accuracy           52.9%        41.3%        -11.6pp   ⚠️ REGRESSED
IFC Class Accuracy        67.6%        60.0%        -7.6pp    ⚠️ REGRESSED
P0 Fire Rate              0/69 (0%)    56/75 (75%)  +75pp     ✅ FIXED
GT-in-pool                33.8%        26.7%        -7.1pp    ⚠️ (see §10.4)
SR FP Rate                0%           12.0%        +12pp     ⚠️ NEW ISSUE
────────────────────────────────────────────────────────────────────
P0 GT-in (storey correct) N/A          17/25 (68%)  NEW       ✅ VALIDATES
2-hop extractions         0/69         0/75         unchanged ❌ NOT LEARNED
```

**Summary**: LoRA_4 achieved its PRIMARY goal (SR extraction 1.4%→75%) but
REGRESSED on secondary metrics (storey -12pp, ifc_class -8pp). The SR extraction
proves the neuro-symbolic pipeline works (68% GT-in-pool when storey is correct).
The storey regression is the critical bottleneck preventing end-to-end improvement.

### 10.6 Sprint Checklist Update

```
SPRINT 4: LoRA_4 TRAINING
──────────────────────────
✅ T8.0   6_assemble_lora4.py (SR ratio, storey norm, fp-coupling)   Done 03-14
✅ T8.0b  Dry-run: 578 train (75% SR) from existing data             Done 03-14
✅ T8.1   CONNECTS_TO skins                                          Done 03-14
✅ T8.2   Assemble LoRA_4 dataset (649 train, 75.1% SR)             Done 03-14
✅ T8.3   Train LoRA_4 on Modal A100                                 Done 03-15
✅ T8.4   Run 3-condition eval (MA/MB/MC, 75 cases each)            Done 03-15
✅ T8.5   Compare LoRA_4 vs LoRA_3 → see §10.5, deep dive §11

LoRA_4 EVAL FINDINGS:
✅  SR extraction fixed: 1.4% → 74.7% (MC)                          VALIDATES THESIS
⚠️  Storey accuracy regressed: 52.9% → 41.3%                        NEEDS FIX
⚠️  GT-in-pool: 33.8% → 26.7% (MC)                                  P0 over-fires with wrong storey
⚠️  FP rate: 0% → 12% (9 hallucinated SR in MC)                     Acceptable (<15%)
❌  2-hop: 0/75 — model never outputs multi-triplet                  NEEDS MORE DATA

NEXT STEPS (LoRA_5 — Neighborhood Fingerprinting, see §12):
□  T9.1   Fix storey regression — floorplan legend reading (§12.3.3)
□  T9.2   Increase top-K from 10 to 50 (immediate +10pp GT-in-pool)
□  T9.3   target_name_keyword post-filter before truncation
□  T9.4   Adjacent-floor storey relaxation (±1 retry)
✅ T10.1  Graph enrichment: NEXT_TO edges in ifc_engine.py (§12.2.1)   Done 03-15
✅ T10.2  NEXT_TO Cypher in retrieval_backend.py (§12.2)               Done 03-15
✅ T11.0  Hint integration + Gemini rewrites (§12.4 Phase 2)           Done 03-15
□  T11.1  NEXT_TO training data (§12.4 Phase 2b)
□  T12.1  Train LoRA_5 + eval (§12.4 Phase 3)
```

---

---

## 11. Diagnostic Deep Dive — LoRA_4 Bottleneck Analysis (2026-03-15)

### 11.1 Three Independent Bottlenecks

LoRA_4 evaluation reveals three independent failure modes that compound
multiplicatively. Fixing any one alone yields limited gains; all three must
be addressed for the pipeline to reach its ceiling.

```
Bottleneck            Impact         Rate        Effect on GT-in-pool
─────────────────────────────────────────────────────────────────────
B1: Storey extraction  61.8% wrong   31/56 P0    0% GT-in-pool when wrong
B2: Predicate confuse  60.5% wrong   26/43 FILLS P0 fires wrong Cypher
B3: Reranking loss     80% GT lost   16/20 hits  GT at rank 23-283
─────────────────────────────────────────────────────────────────────
Combined ceiling: 68% (P0 correct storey) × 40% (survive reranking) = 27%
Actual MC GT-in-pool: 26.7% ← matches the bottleneck product
```

### 11.2 Q1 Answer: Why MA (32%) > MC (27%) Despite 75% SR Extraction

**P0 over-firing paradox**: MC fires P0 for 56/75 cases, displacing the safer
`storey+type` fallback (Priority 4). When P0 fires with wrong storey (31
cases), it produces worse pools than storey+type would have.

```
Net P0 effect in MC:
  Cases where P0 HELPS (adds GT that fallback missed):     3
  Cases where P0 HURTS  (removes GT that fallback had):    6
  Net:                                                    -3 cases
```

In MA/MB, most cases use storey+type (44-59/75), which is a safer default.
**The paradox resolves once storey accuracy exceeds ~70%** — at that point,
P0's 68% success on correct-storey cases outweighs the failures.

### 11.3 Per-Floor Analysis (Q2)

```
Floor         Cases  GT-in-pool  Top-1  Avg Pool  Notes
────────────────────────────────────────────────────────
1 + Garage      54     19 (35.2%)    0    82.1    Most elements, most topology
2                6      1 (16.7%)    0    49.3    Few training examples
3                5      0  (0.0%)    0    45.8    Storey "3" often predicted as "1"
4                4      0  (0.0%)    0    38.0    Same confusion pattern
5-7              6      0  (0.0%)    0    22.7    Upper floors: model defaults to "1"
────────────────────────────────────────────────────────
Floor 1+G:   72% of cases, 100% of GT-in-pool hits
Floors 2-7:  28% of cases, 0% GT-in-pool ← storey extraction fails
```

**Insight**: The system currently only works on Floor 1 where the model
happened to guess the correct floor. Upper-floor performance is blocked
entirely by storey extraction, not by graph topology.

### 11.4 Per-IFC-Class Analysis (Q2)

```
IFC Class              Cases  GT-in-pool  Top-1  Avg Pool  Key Issue
──────────────────────────────────────────────────────────────────────
IfcWallStandardCase      23     12 (52.2%)    0    34.2   Best class (CONTINUOUS works)
IfcDoor                  18      5 (27.8%)    0    87.3   ADJACENT_TO usually correct
IfcWindow                28      2  (7.1%)    0   112.5   FILLS→ADJACENT_TO confusion
IfcRailing                4      1 (25.0%)    0    41.0   Rare, small sample
IfcStair                  2      0  (0.0%)    0    55.0   No topology edges
──────────────────────────────────────────────────────────────────────
```

**Key finding**: IfcWindow is the hardest class (7.1% GT-in-pool) because:
- 60.5% of FILLS cases are predicted as ADJACENT_TO (wrong Cypher)
- Windows have degenerate 1-hop signatures: all 46 Floor-1 windows share
  `Window FILLS Wall` — this single edge cannot discriminate
- Only 11/46 Floor-1 windows have ANY `ADJACENT_TO` edges in Neo4j

### 11.5 Graph Sparsity Diagnosis

Current Neo4j topology for Floor 1 (the best-performing floor):

```
Elements:   166 total, 46 IfcWindow, 18 IfcDoor, 41 IfcWallStandardCase
FILLS:       88 edges (46 windows + 18 doors → 41 host walls)
ADJACENT_TO:  ~40 edges (sparse, non-uniform coverage)
CONTINUOUS:   ~10 edges (multi-story walls only)
```

**The core discrimination problem**:
```
1-hop:   Window --FILLS--> Wall          46 windows share this pattern
2-hop:   Window --FILLS--> Wall <--FILLS-- Door   only ~11 windows have this
```

Most windows are "graph islands" with exactly 1 edge (`FILLS`). Without
additional edges, the symbolic layer cannot distinguish them. The graph
needs enrichment BEFORE better VLM extraction can help.

### 11.6 IFC Data Reality Check (2026-03-15)

Full analysis scripts and plots: `mscd_demo/docs/ifc_data_reality/`

#### 11.6.1 Raw IFC Relationship Inventory (AdvancedProject.ifc)

```
Relationship Type                                  Count
──────────────────────────────────────────────────────────
IfcRelDefinesByProperties                          19,877
IfcRelAssociatesMaterial                            1,345
IfcRelConnectsPathElements (wall-wall topology)       686  ← UNTAPPED
IfcRelVoidsElement (opening→host)                     427
IfcRelFillsElement (door/window→opening)              389
IfcRelDefinesByType                                   202
IfcRelAssignsToGroup                                  154
IfcRelConnectsPortToElement                           139
IfcRelSpaceBoundary                                     0  ← EMPTY
TOTAL                                              23,254
```

#### 11.6.2 Entity Counts

```
IfcFurnishingElement          407    IfcWallStandardCase  381
IfcBuildingElementProxy       358    IfcWindow            263
IfcDoor                       126    IfcMember             83
IfcSlab                        43    IfcSpace               8 (useless)
TOTAL spatial entities       1,799
```

#### 11.6.3 Wall Child Signatures (FILLS topology)

```
98 unique host walls hold 389 fillers (263 windows + 126 doors)
──────────────────────────────────────────────────────────────
Doors-only walls:    81 walls, 147 fillers (singles dominate: 60 walls × 1 door)
Windows-only walls:  12 walls, 210 fillers (curtain walls: 85/50/45/20/15/10/5/4/3/2×)
Mixed (Door+Window):  5 walls,  32 fillers ← ONLY 5% of walls!
```

#### 11.6.4 NEXT_TO Edge Potential

```
Walls with ≥2 children:     35
Total NEXT_TO edges:        291
  Heterogeneous (cross-type): 16  (5.5%)  ← Very rare
  Homogeneous (same-type):   275 (94.5%)  ← Low discrimination value
```

#### 11.6.5 Oracle Discrimination by Enrichment Stage

```
Stage         │ Window oracle Top-1 │ Door oracle Top-1 │ Wall oracle Top-1
──────────────┼─────────────────────┼───────────────────┼──────────────────
Current       │    3%               │    6%             │    3%
+NEXT_TO      │    4%  (+1%)        │   10%  (+4%)      │    3%  (0%)
+2-hop sibl.  │   18%  (+14%)       │   16%  (+6%)      │    1%  (WORSE)
+position     │  100%  (+82%) ★     │   36%  (+20%)     │    1%  (WORSE)
```

**Key insight**: Position ordinal is the **killer discriminator** for windows
(every window gets a unique (wall_id, rank) tuple). Walls WORSEN because
walls without FILLS children collapse to identical empty fingerprints —
walls need **wall-wall connectivity** (686 IfcRelConnectsPathElements).

See: `docs/ifc_data_reality/fig6_oracle_discrimination.pdf`

#### 11.6.6 Property-Level Discrimination (non-topology)

**IfcSpace = dead end**: 8 spaces with generic names ("3ROK", "Area"),
0 elements mapped to spaces, 0 IfcRelSpaceBoundary. This IFC has no useful
space→element assignment.

**Window ObjectType (IfcWindowStyle) = 14 subtypes**:
```
BALANS 15M PRIVATE:       68    BALANS 10M PRIVATE:     60
BALANS 20M PRIVATE:       35    BALANS 30M FLOOR:       24
BALANS 10M BATHROOM:      17    BALANS TRAPPHUS:        15
BALANS 15M FLOOR:         12    ...and 7 more
```
These subtypes correlate with **visual differences**: a 1500mm×1400mm
"PRIVATE" window looks different from a 1000mm×600mm "BATHROOM" window.
Combined with storey → pool reduces from 46 to ~10 per group.

**Window dimensions**: 29 unique (W×H) combos across 263 windows, max
duplicate = 38. When combined with storey, max_dup drops to ~8.

**Door types**: 5 types (Generic Door: 77, 0762×2032mm: 38, Swedoor EI60: 10,
Garage: 1). Interior vs Fire-rated vs Garage = visually distinct.

**Wall length/volume**: 380/381 walls have unique NetVolume — walls are
**individually unique** by dimension, but this isn't learnable from photos.

**Sill height**: 10 distinct values for windows (0, 400, 800, 1000, 1200,
1400mm). Visible from site photos AND floorplan section marks.

#### 11.6.7 Combined Feature Oracle (topology + attributes)

```
Feature combination              Window max_dup  Theoretical pool
───────────────────────────────────────────────────────────────────
Storey alone                     46              46
+ ObjectType (14 subtypes)       ~10             ~10
+ Wall-ID (FILLS host)           ~3-8            ~5
+ 2-hop sibling count            ~2-4            ~3
+ Position ordinal               1               1 (unique!)
```

**Conclusion**: The highest-ROI features for discrimination, ranked:
1. **Window subtype** (14 types) — visual, learnable from photos/floorplans
2. **Wall-ID** (FILLS host) — graph, already implemented
3. **2-hop sibling count** — graph, auto-computable
4. **Position ordinal** — requires floorplan spatial perception (hardest)

---

## 12. Neighborhood Fingerprinting Plan (LoRA_5 Direction)

### 12.1 Strategic Thesis

**Claim**: The attribute entropy bottleneck (46 identical IfcWindows/floor →
Top-1 = 2.2%) cannot be broken by improving VLM extraction quality alone.
The underlying graph is too sparse — most elements have ≤1 edge. We need to
(1) enrich the graph with positional/neighborhood edges so each element has
a unique multi-hop fingerprint, then (2) train the VLM to extract these
richer patterns from floorplan images.

**Data-grounded refinement** (from §11.6 IFC Reality Check):
The oracle analysis reveals a **dual-path discrimination strategy**:

- **Path A (Topology)**: NEXT_TO + 2-hop sibling count lifts Window oracle
  from 3% → 18% (6×). Position ordinal → 100% in theory.
- **Path B (Visual Attributes)**: Window ObjectType has 14 subtypes with
  visually distinct dimensions (1000mm×600mm "BATHROOM" vs 2000mm×1400mm
  "PRIVATE"). This alone reduces pool from 46 → ~10 — and is **learnable
  from both photos and floorplans** without any graph enrichment.
- **Path A+B combined**: subtype narrows to ~10, wall-ID to ~3, position
  to 1. The two paths are **complementary and multiplicative**.

**Pragmatic thesis stance** (per Q2 discussion):
We do NOT need to solve the 46-identical-window problem completely. We prove:
1. For high-entropy elements, enrichment **monotonically reduces** search space
   (3% → 18% → 100% oracle ceiling, demonstrated in fig6)
2. For elements with unique topology (heterogeneous anchors), the system
   achieves **near-perfect retrieval**
3. The remaining gap is bounded and characterized — future work target

**Research contribution**: Neuro-symbolic element retrieval where the
symbolic layer uses auto-enriched IFC topology (wall-sibling ordering,
NEXT_TO positional edges) and the neural layer extracts both multi-hop
spatial patterns AND element subtype attributes from architectural floorplans.

### 12.2 Layer 1: Graph Enrichment (`ifc_engine.py`)

#### 12.2.1 Wall-Sibling NEXT_TO Edges

For each host wall, order its FILLS children by position along the wall axis,
then create `NEXT_TO` edges between consecutive children:

```
Wall_A
  ├─ Window_1 (x=0.0)
  ├─ Door_2   (x=2.5)   → Window_1 --[NEXT_TO]--> Door_2
  ├─ Window_3 (x=4.1)   → Door_2   --[NEXT_TO]--> Window_3
  └─ Window_4 (x=6.8)   → Window_3 --[NEXT_TO]--> Window_4
```

**Implementation** (in `ifc_engine.py:_create_element_relationships`):
1. For each wall with ≥2 FILLS children, project child centroids onto wall axis
2. Sort by projected coordinate
3. Create directed `NEXT_TO` edges between consecutive pairs
4. Store `position_index` (0, 1, 2, ...) as edge property

**Expected yield**: ~50-70 new NEXT_TO edges on Floor 1 alone (41 walls × ~2
children avg). This transforms every window from a 1-edge island into a
node with 2-3 edges.

#### 12.2.2 Design Constraint: NO Absolute Directions

~~LEFT_OF / RIGHT_OF / NORTH_OF~~ — **REMOVED from plan**.

Determining absolute direction requires the VLM to read the compass/legend
from the floorplan — too much cognitive load for marginal gain. Humans on
construction sites say "the window next to the door", not "the window north
of the door". NEXT_TO (1D wall-axis sequence) is sufficient and robust.

#### 12.2.3 Wall-to-Wall SHARES_CORNER Edges

Connect walls that share endpoints (corner joints). Many walls already have
`IfcRelConnectsPathElements` in IFC — we already parse 686 of these. Ensure
they are exposed as explicit `SHARES_CORNER` edges in Neo4j.

#### 12.2.4 Design Constraint: Heterogeneous Anchors

**Data reality** (AdvancedProject.ifc, 98 walls with FILLS children):
```
Wall category       Count  Children  NEXT_TO edges
──────────────────────────────────────────────────────
DOORS_ONLY            81      147   (60 single-door, 14×2, 3×3, 2×4, 2×5)
WINDOWS_ONLY          12      210   (curtain walls: 85/50/45/20/15/10/5/4/3/2×)
MIXED (Door+Window)    5       32   ← Only 5% of walls! 16 hetero NEXT_TO edges
```

**Actual NEXT_TO counts** (from reality check):
- Total NEXT_TO edges: 291
- Heterogeneous (cross-type): **16 (5.5%)** — very rare
- Homogeneous (same-type): **275 (94.5%)** — low discrimination value

**Implication**: Window NEXT_TO Window on a 85-window curtain wall is nearly
useless (still 85 candidates). The 16 heterogeneous Door↔Window pairs on
5 mixed walls are the high-value cases.

**Training data strategy**:
- **Priority 1**: Mixed-wall cases (Window NEXT_TO Door) — maximum discriminative power
- **Priority 2**: Wall-boundary cases (Window at edge of wall, NEXT_TO = null on one side)
- **Priority 3**: Same-type NEXT_TO on small walls (2-4 children) — still useful
- **Deprioritize**: Same-type NEXT_TO on large curtain walls (>10 children)

#### 12.2.5 Expected Discrimination After Enrichment (Data-Grounded)

Oracle analysis from `docs/ifc_data_reality/fig6_oracle_discrimination.pdf`:

```
Enrichment stage      Window Top-1  Door Top-1  Source
──────────────────────────────────────────────────────
Current (FILLS only)     3%           6%         Graph (1-hop degenerate)
+NEXT_TO                 4%          10%         Graph (+1D wall-axis seq)
+2-hop siblings         18%          16%         Graph (+child count/types)
+position ordinal      100%          36%         Graph (+rank on wall) ★
```

**Why position ordinal is a perfect discriminator for windows**:
Every window has a unique (wall_id, rank) tuple because walls don't share
windows. 46 windows on 7 walls → each window at a unique position.
For doors, 60/81 walls have single doors → position doesn't help those.

**Complementary signal: Window subtype** (from §11.6.6):
14 IfcWindowStyle subtypes with visually distinct dimensions. Combined with
storey → pool reduces from 46 to ~10. This is **learnable from photos**
without any graph enrichment and is **multiplicative** with topology:

```
Feature combination              Window pool (Floor 1, n=46)
───────────────────────────────────────────────────────────
Storey alone                     46
+ ObjectType (14 subtypes)       ~10
+ Wall-ID (FILLS host)           ~3-8
+ 2-hop sibling count            ~2-4
+ Position ordinal               1 (unique)
```

### 12.3 Layer 2: Floorplan-Focused VLM Training (LoRA_5)

#### 12.3.1 Training Data Composition

```
LoRA_4 (current):                    LoRA_5 (target):
─────────────────────                ─────────────────────
649 total                            ~800 total
  488 with SR (75%)                    ~650 with SR (81%)
  161 without SR (25%)                 ~150 without SR (19%)

Image mix:                           Image mix:
  site_photo + floorplan               floorplan ONLY (70%)
  site_photo only                      floorplan + site_photo (20%)
  floorplan only                       attribute-only, no image (10%)
```

**Key change**: Train primarily on floorplan images, NOT site photos. Site
photos are unreliable (model hallucinates spatial relationships from
ambiguous photos). Floorplans show topology explicitly.

#### 12.3.2 Multi-Hop Training Examples

New training output format with NEXT_TO:
```json
{
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWallStandardCase",
     "object_material": "concrete", "confidence": 0.95},
    {"predicate": "NEXT_TO", "object_type": "IfcDoor",
     "object_material": null, "confidence": 0.85}
  ]
}
```

The VLM must learn to:
1. Identify the target element on the floorplan
2. Read which wall it fills (FILLS)
3. Read what's next to it on the same wall (NEXT_TO)
4. Report both triplets

#### 12.3.3 Visual Window Subtype Recognition (NEW — from §11.6.6)

**Key insight**: 14 IfcWindowStyle subtypes have **visually distinct** dimensions:
- "BALANS 10M BATHROOM": 1000mm × 600mm (small, landscape)
- "BALANS 15M PRIVATE": 1500mm × 1400mm (medium, square)
- "BALANS 30M FLOOR": 3000mm × 2200mm (large, floor-to-ceiling)
- "BALANS TRAPPHUS": 2500mm × 3000mm (stairwell, tall)

These size differences are visible in **both** floorplans (symbol width) and
site photos (physical window size). Training the VLM to output a more specific
`ifc_class` or `target_name_keyword` (e.g., "bathroom window" vs "private window")
could reduce pool from 46 → ~10 **without any graph enrichment**.

**Implementation**: Add window subtype as `target_name_keyword` in training
data. The VLM already outputs this field — just need training examples that
map visual appearance to subtype names.

**Training crop strategy**: For floorplan crops, **enlarge the crop to show
the full host wall** with all children visible. The VLM doesn't need to count
— it just needs to perceive relative size and position context.

#### 12.3.4 Storey from Floorplan

Train the model to read the floor label/legend from the floorplan image
directly, rather than inferring storey from site photo context. Most
architectural floorplans have a title block with "Floor 1", "Level 2", etc.

### 12.4 Implementation Steps

```
Phase 0: IFC Data Reality Check + Hint Module               Est. 1 day
────────────────────────────────────────────────────────────────────
✅ T9.1   IFC relationship inventory (23,254 rels, 686 ConnectsPath)
          Output: mscd_demo/docs/ifc_data_reality/stats.txt
✅ T9.2   Oracle discrimination analysis (fig6: Win 3%→100%, Door 6%→36%)
          Output: mscd_demo/docs/ifc_data_reality/fig6_oracle_discrimination.pdf
✅ T9.3   Property-level discrimination (14 IfcWindowStyle subtypes)
          Output: mscd_demo/docs/ifc_data_reality/property_analysis.txt
✅ T9.4   Thesis-ready plots (fig1–fig7 PDFs)
          Output: mscd_demo/docs/ifc_data_reality/fig*.pdf
✅ T9.5   Natural hint injection module (_window_subtype_hints.py)
          - 14 window + 6 door subtype→keyword mappings
          - LLM rewrite pipeline (Gemini, temp=0.9, forbidden canonical keyword)
          - Template fallback (_INDIRECT_HINTS pools per keyword)
          - build_hint_augmented_copies(): N copies w/ 1 hint + 1 no-hint
          File: data_curation/scripts/synth/_window_subtype_hints.py

Phase 1: Graph Enrichment (ifc_engine.py)                  Est. 1 day
────────────────────────────────────────────────────────────────────
✅ T10.1  NEXT_TO edges: project FILLS children onto wall axis, sort,
          create bidirectional edges between consecutive pairs.
          Storey-grouped to avoid cross-floor edges on multi-story walls.
          Result: 526 edges (32 heterogeneous, 494 homogeneous)
✅ T10.2  position_index on NEXT_TO edges + wall_position_index on nodes
✅ T10.3  Verified in Neo4j: 526 NEXT_TO, 35 wall-storey groups
✅ T10.4  NEXT_TO works natively with existing spatial_triplet Cypher
          (predicate injected as edge label — no code change needed)
          Added NEXT_TO to SpatialTriplet.predicate Literal in types.py
✅ T10.5  H2 eval: 567/568 GT-in-pool (100%), no regression
          New results by predicate:
            NEXT_TO:      25/25  (100%) 0% fallback  SSR=+60%  pool=13
            CONNECTS_TO: 330/330 (100%) 0% fallback  SSR=-34%  pool=306
            FILLS:        84/84  (100%) 0% fallback
            ADJACENT_TO:  65/66  (98%)  100% fallback
            CONTINUOUS:   63/63  (100%) 100% fallback
✅ T10.6  wall_child_count property on 98 wall nodes
✅ T10.7  CONNECTS_TO edges from IfcRelConnectsPathElements
          681 unique wall-wall pairs → 1362 bidirectional edges
          connection_type property (ATSTART/ATEND/ATPATH)
✅ T10.8  H2 eval expanded: 213 → 568 cases (+25 NEXT_TO, +330 CONNECTS_TO)
          File: data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl

Phase 2: Training Data — Hint Integration (data_curation)  Est. 1 day
────────────────────────────────────────────────────────────────────
✅ T11.0a Wired _window_subtype_hints into 6_assemble_lora4.py
          - get_subtype_keyword() via element_index (object_type + Width)
          - rewrite_chat_with_hint() on Style A + C, no hint on Style B
          - --gemini-hints flag for LLM rewrites (template fallback default)
          - Dry-run: 41/576 (7.1%) records get target_name_keyword
✅ T11.0b Hint integration tested: template fallback generates diverse
          indirect descriptions, no canonical keyword in chat text
✅ T11.0c target_name_keyword set in labels via build_lora4_label()
          (lookup from element_index object_type + dimensions.Width)
✅ T11.0d Run with --gemini-hints for LLM-rewritten training text
          38/39 Gemini rewrites (1 template fallback), 0 keyword leaks
          Output: lora4_train.jsonl (560 records, 75% SR, 39 hints)

Phase 2b: Training Data — NEXT_TO + Floorplan Pivot        Est. 1 day
────────────────────────────────────────────────────────────────────
✅  T11.1  Updated skeleton miner: 57 NEXT_TO skeletons (storey-grouped
          wall-axis projection), 45 CONNECTS_TO skeletons
✅  T11.2  Floorplan renders: 282 AP + 25 BH + 39 DXA = 346 floorplans
          Generated via _floorplan_renderer.py; 15 blank excluded
✅  T11.3  Skin generator: 30 NEXT_TO KEEP + 16 CONNECTS_TO KEEP skins
          Text chat aligned with wireframe scene descriptions
✅  T11.4  Assembled LoRA_5 basic dataset (616 train, 57 test)     Done 03-16
✅  T11.5  Updated system prompt: floorplan-first spatial extraction
          Predicates: FILLS, ADJACENT_TO, CONTINUOUS, NEXT_TO, CONNECTS_TO

Phase 2c: Multi-Hop Chain Mining + LoRA_5 Complex           Done 03-16
────────────────────────────────────────────────────────────────────
✅  T11.6  7b_mine_multitriplet_v2.py: 5 diverse chain patterns mined
          A: FILLS+CONNECTS_TO (80), B: NEXT_TO hetero (29),
          C: 3-hop NEXT_TO+FILLS+CONN (36), D: Cross-wall (28),
          E: FILLS+NEXT_TO (absorbed by dedup) → 173 total records
✅  T11.7  Rendered 131 floorplans for multi-triplet targets (0 fail)
✅  T11.8  Updated 6_assemble_lora5.py: multi-triplet standalone records
          added (not just enrichment of existing skins)
✅  T11.9  Assembled LoRA_5 complex dataset: 1064 train / 76 test

LoRA_5 Dataset Summary (6_assemble_lora5.py --augment):
────────────────────────────────────────────────────────────────────
  Version:   lora5_basic (616)  →  lora5_complex (1064)
  Sources:   v0.4 enriched (903) + v0.5 skins (211 KEEP) + multi-triplet v2 (173)
  Train:     1064 records (798 with SR = 75%, 266 attr-only)
  Test:       76 records (46 with SR)
  Modality:  679 fp_only / 165 fp+site / 220 attr-only(v0.4)
  IFC models: AP=820 / BH=84 / DXA=160
  Predicates: FILLS=507, CONNECTS_TO=411, NEXT_TO=153,
              ADJACENT_TO=138, CONTINUOUS=39
  Multi-hop:  ~150 records with 2+ triplets, 36 genuine 3-hop chains
  Hints:     186/1064 (17.5%) have target_name_keyword
  Chain patterns (new in v2):
    - FILLS→CONNECTS_TO (2-hop, 80 records)
    - NEXT_TO hetero Door↔Window (29 records)
    - NEXT_TO→FILLS→CONNECTS_TO (3-hop, 36 records)
    - Cross-wall FILLS→CONN→FILLS (28 records)
  Key change: Floorplan replaces synthetic site photo as PRIMARY modality
              (LoRA_4 site photos caused data pollution on spatial learning)

Phase 3: Training + Eval                                    Est. 1 day
────────────────────────────────────────────────────────────────────
⏳ T12.1a Train LoRA_5 basic (616) on Modal A100              Running (03-16)
□  T12.1b Train LoRA_5 complex (1064) on Modal A100
□  T12.2  Run 3-condition eval + modality ablation:
          Eval-A (fp_only), Eval-B (fp+site), Eval-C (site_only)
□  T12.3  Compare: LoRA_5basic vs LoRA_5complex vs LoRA_4 vs LoRA_3
□  T12.4  Thesis write-up: enrichment method + results
```

### 12.5 Expected Results (LoRA_5 Targets)

```
Metric                    LoRA_4 (actual)  LoRA_5 (target)  Rationale
────────────────────────────────────────────────────────────────────────
SR extraction (MC)        75%              80%+             Floorplan-only = cleaner
Multi-hop extraction      0/75             30%+             36 genuine 3-hop chains in training
Storey accuracy           41.3%            65%+             Read from floorplan legend
GT-in-pool (MC)           26.7%            45%+             Enriched graph × better storey
Top-1 (MC)                0%               10%+             NEXT_TO + visual subtype
P0 over-reduction         12%              <5%              Fewer wrong-storey P0 fires
```

**Thesis-defensible success criteria** (pragmatic stance):
- **Claim 1** (pool reduction): Show monotonic improvement across enrichment
  stages — even partial gains (3% → 18% oracle) are publishable as proof
  that topology helps
- **Claim 2** (unique relationships): For elements with distinctive topology
  (mixed walls, heterogeneous anchors), demonstrate near-perfect retrieval
- **Claim 3** (ceiling characterization): Present fig6 as a principled bound
  — the gap between achieved and oracle is a characterized future-work target
- **NOT claimed**: 100% Top-1 on 46 identical windows

### 12.6 Literature Context

#### Fuchs & Borrmann (EG-ICE 2025): "Towards Semantic Enrichment of IFC Models through Language Modeling"

**Reviewed**: Uses continued pre-training of LLMs (ModernBERT 395M, Qwen2.5-Coder 490M) on 422 IFC models to predict MISSING PROPERTIES (space names, IFC classifications, quantities). Evaluates ifcJSON-5a* serialization format — best for "one space" scenarios with up to 15% accuracy improvement.

**Relevance to our work**: **Tangential but different scope**. Their method enriches property values (attributes), not graph topology (edges). Our bottleneck is edge sparsity (windows with 1 edge), not missing attributes. However, their property prediction could complement our approach for `ifc_class` and `space_name` extraction in VLM output.

**More relevant cited works**:
- **Lilis et al. 2025** — Topological relationships identification from IFC models (directly relevant to our NEXT_TO enrichment)
- **Bloch et al. 2023** — GNN-based graph enrichment (graph neural networks for IFC topology completion)
- **Wang et al. 2023** — CBIM graph-based interoperability (cross-model graph reasoning)

**Actionable takeaway**: Fuchs & Borrmann's LM-based approach is NOT applicable to our topology enrichment. Our NEXT_TO/LEFT_OF edges are computed from IFC geometry (wall axis projection), not predicted by a language model. The cited works (Lilis, Bloch) are better starting points for related work discussion in the thesis.

### 12.7 Risk Assessment

```
Risk                                Likelihood  Mitigation
───────────────────────────────────────────────────────────────────
NEXT_TO edges still not enough      Medium      Wall child count as complementary
  to disambiguate (curtain walls)               signal; focus on mixed walls first
  UPDATE: H2 shows NEXT_TO SSR=60%             (pool 33→13), best of all predicates
VLM can't read NEXT_TO from         Medium      Annotated floorplan crops with
  raw floorplan                                  arrows showing neighbors
Storey-from-floorplan fails on      Low         Floorplans almost always have
  non-standard legends                           standard title blocks
Over-enrichment: too many edges     Low         Cap at 2-hop (already implemented)
  → slow Cypher queries                          in retrieval_backend.py
CONNECTS_TO pool too large          Medium      CONNECTS_TO returns ~306 walls avg;
  for wall discrimination                        combine with storey filter or 2-hop
                                                 FILLS→CONNECTS_TO pattern
```

---

## 13. LoRA_5 Floorplan Pivot — Training, Evaluation & Diagnosis (2026-03-17)

### 13.1 LoRA_5 Training Summary (DONE)

**Motivation**: Synthetic site photos had 50% DISCARD rate (hallucination, unrealistic renders).
Pivoted to **floorplan as primary visual modality** — VLM reads door arcs, wall double-lines,
opening symbols, room labels for topology extraction. Site photos become supplementary context only.

**Dataset**: 1064 train / 92 test (expanded from 616/57 with multi-hop chains)
- 70% floorplan-only, 30% floorplan+site
- 5 predicates: FILLS(420), ADJACENT_TO(138), NEXT_TO(153), CONNECTS_TO(48), CONTINUOUS(39), empty(266)
- Multi-hop: 363 records with 2+ spatial_relations (5 chain patterns incl. 36 genuine 3-hop)
- 3 IFC models: AP, BH, DXA
- Base: Qwen2.5-VL-7B 4-bit, LoRA r=16 α=32, 5 epochs

**Training results** (from output.log, on training test split):
```
JSON parse rate:      76/76 (100%)
Class accuracy:       73/76 (96%)
Storey accuracy:      72/76 (95%)
Spatial hop-1 acc:    38/46 (83%)
Spatial hop-2 acc:    23/23 (100%)
False positive rate:  0/30 (0%)
Per-predicate hop-1:
  FILLS              24/25 (96%)
  NEXT_TO             9/12 (75%)
  ADJACENT_TO         5/7  (71%)
  CONTINUOUS          0/2  (0%)
  CONNECTS_TO_hop2   17/17 (100%)
  FILLS_hop2          6/6  (100%)
```

### 13.2 LoRA_5 Evaluation Status — BROKEN (Input Format Mismatch)

**Symptom**: Massive accuracy drop from training (83% pred) to eval (30% pred MC condition).
Model mode-collapses to ADJACENT_TO (87% of predictions in MC) and never outputs empty SR.

**5-condition Modal eval completed** (MA/MB/MC/FP/SITE × 70 cases each):
```
Condition  Pred Distribution                    Multi-hop  Empty SR
MA         FILLS=45 ADJACENT_TO=25              0/70       0/70
MC         ADJACENT_TO=61 FILLS=7 CONTINUOUS=2  0/70       0/70
FP         ADJACENT_TO=64 FILLS=3 CONTINUOUS=3  0/70       0/70
SITE       FILLS=51 ADJACENT_TO=19              0/70       0/70
MB         FILLS=51 ADJACENT_TO=19              0/70       0/70
```

**Per-category accuracy (MC condition)**:
```
Category (n)          class   storey  pred_hop1  Notes
L0_attr_only (30)     10/30   5/30    N/A        halluc_SR=30/30 (should be [])
L1_ADJACENT_TO (10)    5/10   3/10   10/10       Only predicate that "works"
L1_FILLS (8)           5/8    3/8     1/8        Collapsed to ADJACENT_TO
L1_NEXT_TO (10)        5/10   2/10    0/10       Never predicted
L1_CONNECTS_TO (5)     3/5    2/5     0/5        Never predicted
L1_CONTINUOUS (5)      3/5    2/5     0/5        Never predicted
L2_multihop (2)        2/2    0/2     1/2        Never outputs 2+ SR
```

### 13.3 Root Cause Analysis

**RC-1: System Prompt Mismatch (CRITICAL)**
- Training: 511-char prompt — mentions "floorplan is PRIMARY source", lists all 5 predicates
- Eval: 1727-char LoRA_3-era prompt from `constraints_extraction.yaml` (`lora_system` key)
- Model trained to respond to specific instruction; eval gives completely different instruction

**RC-2: User Text Format Mismatch**
- Training format includes `[Location] 2 - Second Floor` field (critical storey hint)
- Eval `_build_user_text()` in `eval_lora5.py` NEVER emits `[Location]` field
- Chat format also differs (training: no role prefix; eval: `Role: text` with prefix)

**RC-3: Model never outputs empty spatial_relations**
- Training has 25% empty SR cases → model saw them
- But prompt mismatch makes model ignore the "if no floorplan, SR=[]" training instruction
- Result: 100% false positive SR on attr-only cases

### 13.4 Shortcut Learning Verification Diagnostics

To determine whether the model actually learned visual-spatial reasoning vs just memorized
class→predicate shortcuts, we ran 7 diagnostic tests:

**Test 1: MA vs MC (does model ignore images?)**
- MA vs MC only 19% identical → images DO change output
- MA vs SITE 60% identical → removing floorplan makes predictions converge
- MC vs FP 76% identical → floorplan dominates (expected in FP-primary design)
- **Verdict**: Model uses images, especially floorplan. NOT pure text shortcut.

**Test 2: Predicted predicate distribution (mode-collapse?)**
- MC: ADJACENT_TO=87%, FILLS=10%, CONTINUOUS=3%, NEXT_TO=0%, CONNECTS_TO=0%
- MA: FILLS=64%, ADJACENT_TO=36%
- Modality switches the mode-collapse target: FP→ADJACENT_TO, no-FP→FILLS
- **Verdict**: Different modality = different default. Not random, but not discriminative.

**Test 3: Storey distribution (prior memorization?)**
- MA: "-1 Garage"=67% of predictions (47/70) — massive mode-collapse
- MC: "-1 Garage"=39% + "1 - First Floor"=36%
- Training set has garage/floor-1 as most frequent storeys
- **Verdict**: Strong prior memorization, especially without images.

**Test 4: TEST_DISCARD cases (no site photos, floorplan only)**
- All 16 cases: MA=FILLS, MC=ADJACENT_TO, FP=ADJACENT_TO, SITE=FILLS
- Perfectly deterministic per-condition, regardless of GT predicate
- GT includes CONTINUOUS(3), NEXT_TO(5), ADJACENT_TO(3), CONNECTS_TO(5)
- **Verdict**: Without site photos, floorplan-only → always ADJACENT_TO. Shortcut.

**Test 5: Same-class different-predicate litmus test**
- IfcWall cases with GT=NEXT_TO vs CONTINUOUS vs ADJACENT_TO
- Model predicts ADJACENT_TO for ALL of them in MC/FP conditions
- **Verdict**: Cannot distinguish topology type for same element class. Shortcut confirmed.

**Test 6: Multi-hop output**
- Training: 34% multi-hop records (363/1064)
- Eval: 0% multi-hop output across ALL conditions
- **Verdict**: Multi-hop capability completely lost at eval time.

**Test 7: Cross-condition agreement pattern**
```
Pair       Identical  Same Pred  Interpretation
MA vs MC     19%        34%      Images change output significantly
MA vs SITE   60%        69%      Without FP, both default similarly
FP vs SITE   11%        26%      FP vs SITE = very different outputs
MC vs FP     76%        86%      FP dominates MC (expected)
```

### 13.5 Conclusion: NOT Pure Shortcut, But Prompt Mismatch Triggers Shallow Defaults

The model learned real signal during training (83% hop-1, 0% FP, modality-dependent output).
But the system prompt + text format mismatch at eval time causes it to fall back to
shallow class/modality-based defaults rather than using learned spatial reasoning.

**Evidence for real learning** (during training):
- 96% FILLS accuracy, 75% NEXT_TO, 71% ADJACENT_TO
- 0% false positive on empty SR cases
- 100% multi-hop accuracy (hop-2)
- Modality-dependent outputs (FP vs SITE produce different predictions)

**Evidence for eval-time degradation** (NOT intrinsic shortcut):
- System prompt 511→1727 chars mismatch
- Missing `[Location]` field in eval text
- 0% multi-hop despite 34% in training data

### 13.6 Next Steps

1. **FIX eval_lora5.py**: Use exact 511-char training system prompt + add `[Location]` field
2. **Re-run Modal eval** with corrected prompts (5 conditions × 70 cases)
3. **Re-run local pipeline** with Neo4j connected
4. **Re-run shortcut diagnostics** to confirm fix
5. If still degraded → investigate tokenizer/chat template differences (Unsloth train vs eval)
6. Generate thesis plots after confirmed working eval

---

*Last updated: 2026-03-17.*
*Data-grounded against: AdvancedProject.ifc (1233 elements, 686 ConnectsPathElements),
H2 eval (567/568 GT-in-pool, 23% fallback), 5 edge types: FILLS=84, NEXT_TO=25, CONNECTS_TO=330,
ADJACENT_TO=66, CONTINUOUS=63. Neo4j graph: 389 FILLS + 526 NEXT_TO + 1362 CONNECTS_TO edges.
3-way precomputed eval (69 cases, AP-only: 100% GT-in-pool),
LoRA_3 live VLM eval (69 cases, MB: 20.6% GT-in-pool, MC: 33.8% GT-in-pool, spatial_relations: 1/69),
LoRA_4 live VLM eval (75 cases, MA/MB/MC: SR=35%/16%/75%, GT-in-pool=32%/28%/27%).*
*LoRA_5 trained 03-16: 1064 train/92 test, 5 predicates, 363 multi-hop, floorplan-primary modality pivot.*
*LoRA_5 eval 03-17: BROKEN — system prompt mismatch (511 vs 1727 chars) + missing [Location] field.*
*Shortcut learning diagnosis 03-17: 7 diagnostic tests — model uses images (not pure shortcut) but prompt mismatch triggers shallow defaults.*
