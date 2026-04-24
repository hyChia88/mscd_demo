# Symbolic Retrieval Backend

Deterministic, template-based retrieval from LoRA_3 JSON → Cypher → candidate pool.
No LLM generation inside the backend; reproducibility by construction.

## Overview

```
LoRA_3 JSON ──► Pydantic (types.py::Constraints)
              │
              ▼
        QueryPlanner.plan()           # constraints_to_query.py
        walks PRIORITY_RULES top↓,
        first rule whose 'requires'
        fields are non-empty fires
              │
              ▼
        List[QueryPlan]  (P0a/P0b/P1/P2/P4/P5/P6/P8, in order)
              │
              ▼
        RetrievalBackend.execute_plan # retrieval_backend.py
          ├── _execute_neo4j  (or _execute_memory, degraded)
          │     ├─ P0 dispatch: multi-anchor vs single-hop
          │     ├─ Relaxation ladder on empty pool:
          │     │    full_fingerprint → no_position → topology_only
          │     │    → no_storey → relaxed (drop weakest SR by conf)
          │     │    → fallback storey+type / type_only
          │     └─ P0 ↔ P1 fusion (default `p0_union_p1`)
          ├── Python post-filter (target_name_keyword, graceful)
          └── Optional CLIP rerank (only real score-based ranking)
              │
              ▼
        final_candidates[:10]  →  EvalTrace.mentioned_guids
```

## Field utilisation

Every field from the LoRA_3 JSON has exactly one role in the backend:

| Field | Role | Mechanism |
|---|---|---|
| `ifc_class` | router + filter | required by P0/P1/P4/P6; `target.ifc_type = $t OR STARTS WITH` |
| `storey_name` | filter | resolved to sibling list, `ANY(s IN $list WHERE CONTAINS s)` |
| `space_name` | router + filter (P1) | `IFCSpace.name CONTAINS` |
| `target_name_keyword` | router (P2) + Python post-filter (P0) | graceful, never filters to empty |
| `spatial_relations[0].predicate` | P0a/P0b router | `CONTINUOUS` → property path; else → edge path |
| `spatial_relations[*].{subject,object}_type` | filter | ifc_type equality / STARTS WITH |
| `spatial_relations[*].object_material` | filter | `ref.material CONTAINS` |
| `spatial_relations[*].object_subtype` | filter | `ref.object_type / ref.name CONTAINS` |
| `spatial_relations[*].direction` | filter | `wall_position_index <` / `>` |
| `spatial_relations[*].distance_mm` | filter (ADJACENT_TO) | edge prop ±200 mm |
| `spatial_relations[*].connection_degree` | filter (CONNECTS_TO) | `COUNT { ... } = $deg` |
| `spatial_relations[*].confidence` | ladder order | drop weakest SR first during relaxation |
| `size_cluster` | filter | `target.size_cluster = $cluster` (equality) |
| `position_context` | filter (if conf ≥ 0.8) | parsed to `position_index`, equality on `wall_position_index` |

**Routing** = which priority rule fires. **Filter** = AND clause in Cypher. **Ladder order** = which filter is dropped first on empty pool.

## Execution details

**P0 multi-anchor vs single-hop dispatch** (`_requires_multi_chain`):
- 2+ triplets, OR any `direction` / `object_subtype`, OR `position_index` present → multi-anchor
- otherwise → single-hop Cypher (simpler, unchanged from Phase-2)

**Multi-anchor AND semantics**:
- `FILLS[0]` promoted to `MATCH` when `NEXT_TO` co-occurs → pins both to the same wall via `nt_r.wall_guid = fi0.guid`
- All `NEXT_TO` edges must share `wall_guid` (same-wall constraint)
- Other predicates stay as `WHERE EXISTS { ... }`

**P0 ∪ P1 fusion** (default `p0_union_p1`):
- P0 Cypher produces topology-matched tier
- P1 = `storey+type` produces safety-net tier
- Final pool = P0 ∪ (P1 − P0). Binary tiering: topology-matched first, storey-mates after.

**Storey sibling resolution**:
- `_resolve_storey_siblings("Floor 1")` → `["level 1", "1 - first floor"]`
- Handles naming heterogeneity across IFC models (AP / BH / DXA) — same `floor_num`, different canonical strings.

## Analysis

### Advantages
1. **Deterministic & reproducible.** Same JSON → same pool. No LLM variance in the symbolic path.
2. **Priority-based coverage.** P0 handles topology-rich queries; P4–P8 keep GT retrievable when fingerprint is weak or missing. Any non-empty `Constraints` object produces at least one executable plan.
3. **Relaxation ladder degrades gracefully.** Empty pool at full fingerprint never crashes — demotes through `no_position → topology_only → no_storey → relaxed → fallback` and records `strategy_actually_used` / `fallback_triggered` for transparency.
4. **Naming-agnostic storey matching.** Sibling list lets "Floor 1", "Level 1", and "1 - First Floor" resolve to the same candidate set across models.
5. **IFC subtype-safe.** `ifc_type = $t OR STARTS WITH $t` handles `IfcWall` ⟂ `IfcWallStandardCase` without requiring the extractor to know subtypes.
6. **P0 ∪ P1 safety net.** When LoRA gets `ifc_class` and `storey_name` right, GT is never lost to over-strict fingerprint — worst case it lands in the P1 tier.
7. **Cheap. No GPU at retrieval time; symbolic layer is a few Cypher round-trips.

### Limitations to improve
1. **No ranking signal.** Every non-(class, storey) field is a binary AND-filter. Candidates passing the same filters are indistinguishable pre-CLIP. `ORDER BY wall_position_index` is graph-state-driven, not evidence-driven.
2. **P0 ∪ P1 is binary tiering.** Topology-matched-first / storey-mates-after — no intra-tier ordering by number of matched fingerprint fields.
3. **Ladder discards signal.** When a filter over-prunes it is **removed**, not demoted to a soft score. A case that narrowly missed `full_fingerprint` loses all partial-match evidence by `relaxed`.
4. **`confidence` fields underused.** Per-triplet `confidence` only orders which SR to drop; never weights evidence combination. Top-level `confidence` is diagnostic-only.
5. **Strict equality on noisy fields.** `size_cluster`, `position_index`, and `direction` use hard equality / inequality. A one-bucket miss from the LoRA classifier fully excludes GT.
6. **Routing on `spatial_relations[0]` only.** P0a vs P0b is decided from the first triplet's predicate. If LoRA emits `[CONTINUOUS, FILLS]` but FILLS is the discriminating one, dispatch is wrong.
7. **P1 rescue has failure modes.** `_get_storey_type_pool` returns `[]` when `storey` or `subject_type` is empty → union degrades to pure P0. Also cannot rescue wrong-`ifc_class` / wrong-`storey` from LoRA.
8. **Post-filter `target_name_keyword` is graceful-to-a-fault.** On empty result it returns the unfiltered pool — GT is not dropped, but also not promoted. The signal is wasted.
9. **CLIP fusion is sequential, not joint.** Symbolic evidence (how many fingerprint fields matched, per-triplet confidence) is not fused into the final ranking — it only decides who is in the pool.
10. **No per-model calibration.** Same thresholds (distance ±200 mm, size ±50 mm, storey sibling resolver) across AP/BH/DXA despite measurable differences. BH needed an elevation fallback inside `ifc_engine.py`; the symbolic layer has no hook.

### Why Oracle is acceptable despite (1)–(10)
When every field is correct, AND-filters collapse the pool to ≤3 and ranking is moot. All ten limitations manifest only under LoRA_3 partial correctness — consistent with the Oracle ≈ 94 % vs LoRA5-r32 ≈ 53 % gap observed in `strategy_ablation_v2`.

## Files

| File | Responsibility |
|---|---|
| `types.py` | Pydantic schema: `Constraints`, `SpatialTriplet`, `QueryPlan`, `RetrievalResult`, `PipelineTrace` |
| `constraints_to_query.py` | `QueryPlanner` — priority rule table, routing, param build |
| `retrieval_backend.py` | `RetrievalBackend` — Neo4j / memory dispatch, multi-anchor, relaxation ladder, P0∪P1 fusion, CLIP |
| `constraints_extractor_lora.py` | LoRA_3 inference → `Constraints` |
| `constraints_extractor_prompt_only.py` | Gemini baseline extractor |
| `pipeline.py` | Glue: condition mask → extractor → planner → backend → `EvalTrace` |
| `condition_mask.py` | Redacts inputs per evaluation condition |
| `metrics.py` | Per-trace metrics (rerank gain etc.) |
| `floorplan_counter.py` | Floorplan-mark counting helper |
