# Insights till Phase 5: (in thesis chap 07)

### 1. Storey and class reach 100% — but via text, not vision
- Chat input almost always states floor and element type explicitly; VLM likely extracts these from text cues, not from floorplan or site photo
- This is acceptable: in real AEC workflows, reporters specify these verbally
- The harder problem is disambiguating same-type elements on the same floor
Thesis focus: spatial disambiguation under geometric repetition, measured end-to-end

### 2. Enriched graph + planner redesign lift the retrieval ceiling
- Oracle experiment: 100% GT-in-Pool, pool reduced by 99%+
- Fingerprint ladder: subtype (L3) is the single largest pool-compression step (45 → 9 median), Spatial topology adds measurable gain over attribute-only filtering (P1-only → full topology: +13pp Top-10)

### 3. Graph-RAG reranking works — but only on coarse pools
- P1-only + rerank: Top-1 0% → 8.3%, MRR tripled
- Full-topology + rerank: Top-1 drops from 6.7% → 1.7%
Key finding: structured extraction and graph-context matching are complementary, not interchangeable

### 4. The pipeline is learnable — continuous improvement is viable
- G0 → G3 → G7 → G8: each generation responds to a diagnosed field-level failure
- Neuro-symbolic split makes errors attributable (extraction vs. planner vs. graph), Worker accept/reject feedback directly produces training signal
- Architecture supports weekly LoRA retraining without pipeline changes

---

# Phase 6 Plan

## Source references (authoritative paths — keep in sync)

### Data
- **IFC models** (`data_curation/ifc_models/`)
  - `AdvancedProject.ifc` — primary (AP, 1233 elements, NO space boundaries) — **Phase 6 focus**
  - `BasicHouse.ifc` — BH (no IfcSpace, no boundaries)
  - `Duplex_A_20110505.ifc` — DXA (18 spaces, 264 boundaries, real room names) — T5 target (deprioritized)
- **Synthetic dataset** (`data_curation/datasets/synth_v0.5_ap/`)
  - `floorplans/` — 200 full floorplan PNGs (one per skeleton) — **T1 counting source (full wall context)**
  - `floorplans_v2/` — 744 wall-centered crops + JSON metadata (`crop_bbox_world`, `target_center_world`) — training-time supervision for OpenCV localization; JSON NOT available at inference (only the patch image is)
  - `skeletons/`, `skins/`, `imgs/`, `renders/`, `train/`, `mappings/`, `staging/`
- **Element index**: `data_curation/references/element_index.jsonl` (1233 elements, centroid)
- **IFC reality audit**: `mscd_demo/docs/ifc_data_reality/`
  - `property_analysis.txt` — real dimension property sets (**T3 k-means input**)
  - `stats.txt` — entity/relationship counts

### Eval cases / experiments
- **Case files** (`mscd_demo/evaluation/cases/`)
  - `cases_v3_test.jsonl` — 69-case test set (wireframe renders)
  - `cases_v3_test_site.jsonl` — site-photo variant
- **Experiments** (`mscd_demo/evaluation/experiments/`)
  - `graph_rag_rerank_ap.py` — current reranker; `_build_prompt()` at lines 515-528 (**T2 target**)
  - `_query_candidate_contexts()` at lines 374-428 (graph fingerprint extraction)
  - `_call_gemini()` at lines 530-560
- **H2 hard-negative eval**: `mscd_demo/eval/h2_eval.py`
- **Trace analysis**: `mscd_demo/eval/analyze_traces.py`
- **Data conversion**: `mscd_demo/eval/convert_lora3_test.py`
- **Plot output dir**: `mscd_demo/docs/plots/phase4_lora6_main/` (**T4 output**)

### Retrieval / constraint backend (`mscd_demo/src/`)
Note: `src/v2/` was renamed to `src/neurosym/` in the Phase 6 cleanup; `metrics_v2.py` → `metrics.py`; `V2Trace` → `PipelineTrace`.

- **Neo4j export**: `src/ifc_engine.py`
  - `_create_element_nodes()` — per-element properties (storey, centroid, ifc_type, `wall_position_index`, `wall_child_total`)
  - `_create_element_relationships()` — FILLS / ADJACENT_TO / CONTINUOUS edges
  - `_resolve_storey_query()` — "Floor 1" → `["level 1", "1 - first floor"]` (1:many)
  - T5 adds `IfcRelSpaceBoundary` parsing here (DXA-only)
- **Schema**: `src/neurosym/types.py`
  - `Constraints` — storey_name, ifc_class, space_name, target_name_keyword, position_context, spatial_relations, target_width_mm, target_height_mm (mm fields → `size_cluster` after T3)
  - `SpatialTriplet` — 8 predicates (FILLS, CONTINUOUS, ADJACENT_TO, NEXT_TO, CONNECTS_TO, ON_TOP_OF, PERPENDICULAR_TO, PARALLEL_TO)
  - `RetrievalResult` — fallback_triggered, strategy_actually_used
  - Legacy fields (`near_keywords`, `relations`, `neighbor_type`) removed; `extra="ignore"` accepts old JSONL
- **Query planner**: `src/neurosym/constraints_to_query.py`
  - Priority table (post-cleanup): 0=spatial_triplet / continuous_span, 1=space+type, 2=name_keyword, 4=storey+type, 5=storey_only, 6=type_only, 8=fallback  (priorities 3 and 7 removed with legacy fields)
  - `_parse_position_context()` at lines 355-367 — regex `"Nth of M openings"` (**T1 consumer**)
  - Dimension routing at lines 347-351
- **Retrieval executor**: `src/neurosym/retrieval_backend.py`
  - `_execute_neo4j()` — spatial_triplet + continuous_span branches
  - `_resolve_storey()` — engine helper wrapper
  - `_post_filter_by_name_keyword()` — Python-side graceful filter
  - Position index WHERE clause at lines 814-818 (exact `wall_position_index` match)
  - Dimension filter at lines 821-830 (±50mm tolerance — **T3 replacement target**)
- **LoRA constraint extractor**: `src/neurosym/constraints_extractor_lora.py` — VLM → Constraints schema; **T1 consumer of OpenCV facts**

### Training (`mscd_demo/training/`)
- `train_lora6.py` — LoRA config at lines 451-462 (`finetune_vision_layers=True`); `ProgressCallback` at lines 478-498 (**T4 extension point**)

### Config
- `mscd_demo/config.yaml` — Neo4j connection (bolt://localhost:7687, pw=password)
- Neo4j Community 5.26.0 at `/tmp/neo4j-community-5.26.0/` (manual start)

### Data-build scripts (`data_curation/scripts/synth/`)
- `1_build_index.py` — IFC → element_index.jsonl (centroid, target_name_keyword)
- `2_hunt_skeletons.py` — skeleton miner (FILLS / ADJACENT_TO / CONTINUOUS)
- `2b_build_h2_hardneg.py` — H2 hard-negative builder

---

## Design principle — tool-augmented VLM

Phase 5 identified the core failure: VLMs are bad at counting and measuring, but good at semantic interpretation. Phase 6 inverts the dependency: **run the deterministic perception tool first, inject its facts into the VLM prompt**, then let VLM focus on semantics.

Pipeline stages:
```
OpenCV counting  →  Neuro VLM  →  Symbolic query  →  Graph-RAG rerank
(deterministic)    (semantic)     (Cypher on        (CoT decompose
                                   structured pool)   on residual)
```

Why OpenCV first:
- VLM stops guessing `position_context` (which it fails on per Phase 5)
- VLM consumes count as a given fact; extraction task simplifies to validation + semantic fields
- Counting becomes a **default feature**, not a tiebreaker — "1 of 1 is information too"
- Graceful degradation: if OpenCV fails, skip the fact and fall through to symbolic + rerank (current Phase 5 behaviour)

Thesis ablation:
`VLM-only` → `VLM + OpenCV counting` → `+ decomposed rerank` → `+ size cluster`

---

## Implementation strategy — ship without retraining

**Ship Phase 6 without retraining.** Pattern A for T1, Pattern B for T3. T2/T5 are already retrain-free. T4 becomes a post-thesis diagnostic for the next training run.

This cuts the critical path from ~2 weeks (with retrain) to ~1 week (pipeline work only). Risk profile drops too — you don't risk regressing the locked Phase 5 numbers with a new training run.

### Retrain-avoidance patterns

**Pattern A (T1) — OpenCV overrides `position_context` post-VLM** (no retrain)
```
VLM → Constraints (with VLM's possibly-wrong position_context)
  ↓
OpenCV runs on same inputs
  ↓
if opencv_confidence > 0.7:
    constraints.position_context = f"{opencv.position} of {opencv.total} openings"
```
- LoRA behaviour unchanged
- Planner gets deterministic position when OpenCV succeeds
- Trade-off: VLM can't cross-validate OpenCV (e.g. "opencv says 3/17 but I see only 12 openings"). If this becomes a visible failure mode, escalate to the retrain path.

**Pattern B (T3) — Bucketize mm → cluster in Python** (no retrain)
```
LoRA emits: target_width_mm=894, target_height_mm=1294   (unchanged schema)
  ↓
Python: cluster_lookup(894, 1294, ifc_class) → "small_window"
  ↓
Symbolic filter: cluster-membership match (replaces ±50mm tolerance)
```
- `Constraints` schema keeps mm fields
- Replaces ±50mm filter at `retrieval_backend.py:821-830`
- Trade-off: model still learns via regression, not classification — weaker gradient signal

### Retrain-only upsides (if you later decide to retrain)

| Upside | Estimated value |
|--------|----------------|
| VLM validates OpenCV facts | Catches OpenCV errors, ~1-2pp Top-1 |
| LoRA learns position as classification | Cleaner gradient than mm regression, ~2-3pp |
| Cleaner schema in thesis narrative | Quality-of-thesis, not quality-of-metric |

Real but marginal. Given thesis timing, skip the retrain; revisit as Phase 6.5 if override-based T1 shows clear failure.

### Retrain-required matrix

| Task | Default plan wanted | Minimum feasible | Retrain needed? |
|------|---------------------|------------------|-----------------|
| T1 OpenCV counting | VLM echoes+validates facts | Pattern A override | ❌ No |
| T2 Decomposed reranker | Gemini prompt engineering | Same | ❌ No |
| T3 Dimension clusters | LoRA emits `size_cluster` | Pattern B bucketize | ❌ No |
| T4 Layer-ratio tracking | Callback during retrain | Dormant until next retrain | — |
| T5 DXA space filtering | LoRA emits `space_name` | Already in schema | ❌ No |

---

## Task list (ordered by pipeline dependency)

### T1 — OpenCV two-image counting — *core Phase 6 feature, do first*
**Effort:** ~3-4 days total, broken into T1.1–T1.4  
**Module:** `mscd_demo/src/neurosym/floorplan_counter.py` (core plumbing already landed)

Target elements are wall-centric openings (doors / windows) — counting is discriminative on every case, not just high-sibling walls.

**Two-image design (solves patch-truncation):**
```
inputs  : floorplan_patch (target-centered, user-provided)
          full_floorplan (storey-level project data)

step 1  : template-match patch inside full floorplan
          → bbox of patch within full; replaces unavailable v2 crop_bbox_world at inference
step 2  : identify target wall
          → wall nearest to patch center; project into full-floorplan coords
step 3  : count on the FULL wall polyline (not cropped)
          → traverse left-to-right, detect door (arc) / window (parallel lines) symbols
          → output (position, total, confidence)
step 4  : cross-check
          → patch-visible openings must be a contiguous subrange of full count
          → confidence drops if mismatch
```

**Symbolic integration (Pattern A — no retrain, already wired):**
- `position_context` → hard filter at `retrieval_backend.py:814-818` (exact `wall_position_index` match) when confidence high
- Medium confidence → soft descriptor passed to reranker
- Low / failed → current fallback behaviour (VLM's own `position_context` or none)

**Current status (2026-04-20):**

| Item | Status |
|---|---|
| Counter plumbing (prompt injection, confidence gating) | ✅ Works |
| Full-storey renderer (`3c_render_full_storeys.py`) | ✅ 3 scoped storeys; Level 1 reuses First Floor's bbox for consistent pixel density |
| Storey calibration sidecar JSON | ✅ Deterministic world→pixel mapping |
| `count_from_full_storey()` F3 entry point | ✅ **2/15 exact-match (13.3%)** on scoped set (frame-mismatch ceiling) |
| `count_from_full_storey_with_wall_bounds()` F4 entry point | ✅ **12/15 exact-match (80.0%)** on scoped set |
| AP canonical validator with `--mode {f3,f4,both}` | ✅ Works end-to-end |
| Wall-endpoint helper (PCA + IFC local-X direction alignment) | ✅ Implemented |
| Counter bug fixes (ordering + filter threshold) | ✅ Applied |
| IFC local-X axis alignment (T1.3 fix) | ✅ Applied — flipped 5 F4 inversions to OK (7→12/15) |
| Annotation visualizer (`annotate_phase6_counter.py`) | ✅ Works — per-case PNG overlays |
| Track B-2 fig03 with OpenCV rows | ⏳ **T1.4 — next step** |

**Scope decision (down from full 33 cases to 15):** attempts to render Floors 2-5 via either (a) `IfcRelContainedInSpatialStructure` union with Level 1 walls or (b) z-overlap scanning failed to produce usable renders. Scoped to the three storeys whose `IfcRelContainedInSpatialStructure` gives complete coverage:

| Storey | Cases | Renders |
|---|---|---|
| 1 - First Floor | 6 | ✅ 736 elements, 64m bbox |
| -1 - Garage | 2 | ✅ 170 elements, 64m bbox |
| Level 1 (holds multi-storey walls) | 7 | ✅ 506 elements, reused First Floor bbox (64m) |
| **Scope total** | **15 / 33** | — |
| Floors 2-5 | 18 | ❌ out-of-scope — renderer under-populates |
| Floor 6 | 0 | — (no canonical wall cases) |

Thesis narrative consequence: validation reports on the **ground-floor subset** (15 cases). Multi-storey-wall renderer coverage becomes Chapter 8 future work.

Current result dor F3/F4:
```
(mscd_demo) root@HYChiaX1:~/cmu/master_thesis# cd /root/cmu/master_thesis
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mscd_demo
python mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py --mode both
[
  {
    "mode": "f3",
    "total_cases": 60,
    "counting_denominator": 15,
    "exact_match": 2,
    "exact_match_rate": 0.13333333333333333,
    "status_breakdown": {
      "n/a_non_counting_predicate": 26,
      "fail_no_calibration": 18,
      "fail_counter_mismatch": 13,
      "skip_no_teacher_slot": 1,
      "ok_exact_match": 2
    },
    "output_path": "mscd_demo/output/lora6_v2_ap_20260331/phase6_ap_canonical_counter_report_f3.jsonl"
  },
  {
    "mode": "f4",
    "total_cases": 60,
    "counting_denominator": 15,
    "exact_match": 7,
    "exact_match_rate": 0.4666666666666667,
    "status_breakdown": {
      "n/a_non_counting_predicate": 26,
      "fail_no_calibration": 18,
      "ok_exact_match": 7,
      "skip_no_teacher_slot": 1,
      "fail_counter_mismatch": 8
    },
    "output_path": "mscd_demo/output/lora6_v2_ap_20260331/phase6_ap_canonical_counter_report_f4.jsonl"
  }
]
```

---

#### T1.1 — Counter diagnostics — ✅ *done (2026-04-20)*

Outcome:
- Pivoted from H2's broken two-image design to **full-storey render + world→pixel calibration**
- New entry point: `FloorplanCounter.count_from_full_storey()` with per-stage debug fields
- New renderer: `data_curation/scripts/synth/3c_render_full_storeys.py` — one PNG + calibration JSON per storey
- New validator: `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py` (G7 teacher labels as GT, no IFC/Neo4j)
- Baseline: **11/33 exact-match (33.3%)** — up from 0/10 on H2
- Failure analysis: [phase6_counter_failure_taxonomy.md](phase6_counter_failure_taxonomy.md)
  - 5 cases: ordering inversion (real bug, 30-min fix)
  - 3 cases: under-count (detection filter too strict, 1h fix)
  - 14 cases: **frame mismatch** (teacher counts per-IFC-wall; OpenCV counts per-visible-line) — not a bug, architectural decision → resolved by T1.5 F4

Blocker uncovered: renderer under-populates Floors 2-5 (multi-storey walls live in IFC "Level 1" container). Addressed in T1.2.5.

#### T1.2 — Counter bug fixes (ordering + filter) — ✅ *done*

- Ordering canonicalisation applied (`wall_origin` = leftmost pixel, tiebreak topmost).
- Opening-detection thresholds loosened (area ≥10, aspect ≥1.3, major ≥8).
- Outcome: F3 exact-match unchanged (2/15) — bugs were subordinate to the frame-mismatch issue; these fixes still make outputs deterministic and catch under-detected openings in F4 mode.

#### T1.2.5 — Renderer scope decision — ✅ *done*

The union-of-contained-elements approach and z-overlap scanning both failed to produce usable Floors 2-5 renders. **Scoped down** to the three storeys whose `IfcRelContainedInSpatialStructure` gives complete coverage: `1 - First Floor`, `-1 - Garage`, `Level 1`. Level 1 reuses First Floor's bbox for consistent pixel density. Floors 2-5 renderer coverage → Chapter 8 future work.

Files:
- `data_curation/scripts/synth/3c_render_full_storeys.py` — renders the 3 scoped storeys + calibration index
- `data_curation/datasets/synth_v0.5_ap/floorplans_full/` — 3 PNGs + 3 JSONs + `calibration.json`

#### T1.3 — Dual-mode validation (F3 + F4) — ✅ *done*

**Counter entry points:**
- `FloorplanCounter.count_from_full_storey()` — F3 soft-signal mode (Hough line)
- `FloorplanCounter.count_from_full_storey_with_wall_bounds()` — F4 oracle mode; clips counting to IFC-supplied wall segment

**Wall-endpoint helper:** `_wall_endpoints_world(host_guid)` in the validator — PCA over 2D vertices of the wall's geometry, with direction aligned to `IfcWall.ObjectPlacement` local-X so endpoint ordering matches teacher's convention. Cached per GUID.

**Validator CLI:** `--mode {f3, f4, both}`
- `f3` → `phase6_ap_canonical_counter_report_f3.jsonl`
- `f4` → `phase6_ap_canonical_counter_report_f4.jsonl`
- `both` (default) → runs both sequentially

**Scope:** 15 cases (6 Floor 1 + 2 Garage + 7 Level 1).

**Final numbers:**
| Mode | Exact-match | Rate | Gate | Notes |
|---|---|---|---|---|
| F3 (soft, no oracle) | 2/15 | 13.3% | below 40% | Frame mismatch ceiling — F3 can't resolve by design (Hough line spans multiple IFC walls) |
| **F4 (oracle wall bounds)** | **12/15** | **80.0%** | **✅ ≥80% target** | Capability ceiling demonstrated |

**F4 remaining failures (3 cases) — dataset ceiling:**
- `AP_SK_158`: teacher 4/10, opencv 5/10 — target pixel on boundary between openings
- `AP_SK_228`: teacher 13/14, opencv 14/14 — same
- `AP_SK_234`: teacher 3/3, opencv 2/3 — same

All three are "target world_xy lands between two adjacent openings" — not algorithmically recoverable without additional target disambiguation (e.g. target bounding-box, not just centroid).

**F3 → F4 gains attributed:**
- +5 from IFC-local-X axis alignment (ordering direction fix) — most important change
- +3 from frame clipping (restricts counting to IFC wall segment)
- +2 from retained F3 wins (SK_096, SK_149)

#### T1.3.5 — Annotation visualizer — ✅ *done*

**Script:** `mscd_demo/evaluation/h2/annotate_phase6_counter.py`  
**CLI:** `--mode {f3, f4, both}` `[--cases SK_xxx,SK_yyy]`  
**Output:** `mscd_demo/output/lora6_v2_ap_20260331/phase6_annotations/{f3,f4}/`

Per-case PNG overlays with:
- Teal line + arrow — detected wall line (arrow at `wall_origin`, shows ordering direction)
- Red circle — opening OpenCV picked as target
- Green circle — teacher's expected target position
- Orange circles (numbered) — other detected openings
- Magenta X — target's world_xy mapped to pixel
- Title bar — case_id, predicate, teacher slot, opencv slot, OK/FAIL

#### T1.5 — Frame-mismatch resolution (F-options) — *architectural decision*

The T1.1 diagnostic uncovered a structural issue separate from the counter's algorithmic bugs: **teacher labels and OpenCV counts live in different frames.**

**Teacher frame** — per-`IfcWall`-GUID, restricted to current storey, ordered along wall's local X-axis. Computed in [6_assemble_lora6.py:310-358](data_curation/scripts/synth/6_assemble_lora6.py#L310-L358) and matches Neo4j's `wall_position_index` / `wall_child_total`.

**OpenCV frame** — per visible Hough line in the rendered floorplan. A single Hough line can span multiple colinear IFC wall elements (adjacent walls sharing an axis across T-junctions or along facades). Also per-storey, but boundary is "visual continuity," not IFC-element identity.

14/33 counter "failures" in T1.1 are not bugs — they are this frame mismatch (e.g. teacher 9/10, OpenCV 19 on what teacher considers one wall but OpenCV sees as two colinear walls). Resolving the mismatch is an architectural choice, not a code fix.

**F1 — IFC injection at inference**
At query time, look up the host wall's GUID → pull its IFC endpoint coords → convert to pixels via calibration → clip OpenCV counting to that pixel range.
- ✅ Frame-aligned, accurate
- ❌ **Circular dependency** — in production we don't know `host_guid` at inference (retrieval is supposed to find it). F1 assumes the answer we're trying to compute.
- **Verdict: not viable for production.**

**F2 — Junction detection in the image**
Detect T-junctions, corners, wall-wall intersections visually (corner detectors, contour analysis). Break Hough lines at junctions so each segment approximates one IFC wall.
- ✅ Vision-only, scales
- ❌ Junction detection has its own false-positive / false-negative rates (door arcs, stair treads, furniture look like junctions). Multi-week tuning project.
- **Verdict: sound direction but out of Phase 6 scope.**

**F3 — Soft-signal mode**
Accept OpenCV's line-count as-is. Don't use it as a hard filter. Pass `(opencv_position, opencv_total)` to the Graph-RAG reranker as context. Reranker sees both OpenCV output and each candidate's Neo4j `wall_position_index` / `wall_child_total`; Gemini reasons over the frame conversion implicitly.
- ✅ Zero extra engineering; preserves vision-only design
- ⚠️ Weakest signal; the reranker must carry the conversion
- **Verdict: current fallback path — keep as default behavior.**

**F4 — Hardcoded wall segment for validation / demo, future work for auto-detect** *(selected)*

At **validation and the thesis demo only**, feed OpenCV the host wall's pixel bounds explicitly. Pull `host_guid` → IFC wall endpoints → pixel bounds via storey calibration. OpenCV counts inside that segment, so its frame matches the teacher frame by construction.

This is **not F1**, because:
- F1 proposes this as the *production inference path* (which is circular)
- F4 is scoped to *validation and demo*, with an explicit thesis disclaimer: "assuming the host wall is known — automatic wall-bound detection from pure vision is future work (F2)."

F4 separates two concerns cleanly:
1. **Can OpenCV count accurately given the right wall?** → measured by F4 validation
2. **Can we identify the right wall from vision alone?** → deferred (F2 future work)

Thesis narrative:
> "The deterministic counter, given correct wall bounds (F4 oracle), achieves ≥80% exact-match on AP canonical — validating the tool-augmented VLM design's core capability. Automatic wall-bound detection from vision (F2) is left as future work."

| Option | Frame-aligned | Vision-only | Ready now | Thesis-ready |
|---|---|---|---|---|
| F1 production | ✅ | ❌ | ❌ circular | ❌ |
| F2 junction | ✅ | ✅ | ❌ multi-week | ❌ |
| F3 soft-signal | ⚠️ reranker converts | ✅ | ✅ | ⚠️ weaker |
| **F4 hardcoded-for-validation** | ✅ | ⚠️ demo-only | ✅ | **✅ cleanest** |

**Phase 6 ships F4 + F3 together:**
- **F4** powers the thesis's core demonstration (validation + demo): measures counter accuracy in isolation, with wall-bound oracle.
- **F3** is the production fallback: when wall bounds unavailable at inference, OpenCV output becomes a soft signal for the reranker.
- **F2** appears in Chapter 8 as future work.

#### F4 execute details — for reference

Counter entry point ([floorplan_counter.py](../../src/neurosym/floorplan_counter.py)):
```python
FloorplanCounter.count_from_full_storey_with_wall_bounds(
    storey_png_path, calibration, target_world_xy, wall_endpoints_world
) -> FloorplanCountResult
```

Counting logic inside the oracle bounds:
```
wall_pixel_line = (world_to_pixel(endpoint_a), world_to_pixel(endpoint_b))
openings = _detect_openings_on_wall(image, wall_pixel_line)
# Clip to IFC wall segment so adjacent colinear walls don't leak
openings = [o for o in openings if -tol <= o['projection'] <= segment_length + tol]
position = nearest_opening_index(openings, target_pixel) + 1
total    = len(openings)
```

Wall-endpoints are resolved by the validator's `_wall_endpoints_world(host_guid)` helper — PCA over the wall's 2D vertices from `ifcopenshell.geom.create_shape`. This robustly handles axis-aligned and angled walls without parsing `IfcExtrudedAreaSolid` internals. Results cached per GUID.

**Thesis positioning** — report both bars:
- **F3** (soft signal, no oracle) → realistic retrieval regime
- **F4** (with wall oracle) → counter's *capability ceiling*

Gap between F3 and F4 directly sizes the future-work opportunity of F2 (junction-detection-based wall segmentation).

#### T1.4 — fig03 Track B-2 refresh (dual-mode bars) — *~0.5 day*

Run `inject_floorplan_counts.py` twice, once per counter mode, to produce two precomputed traces:
- `g8_posctx_dim__ap_eval_opencv_count_f3.jsonl` — soft-signal mode (no oracle)
- `g8_posctx_dim__ap_eval_opencv_count_f4.jsonl` — wall-bound oracle mode

Rescore retrieval against Neo4j (or memory mode with precomputed pool) → two `e2e_phase5_metrics.json` outputs.

Add **two rows** to `MIXED_MODELS` in `evaluation/analysis/create_fair_trackb2_growth_figures.py`:
```python
{
    "key": "g8_opencv_f3",
    "label": "G8 + OpenCV (F3)\n(MM, soft)",
    "display": "G8 + OpenCV F3 (soft signal)",
    "input_regime": "Canonical multimodal + OpenCV F3",
    "source_type": "canonical_json",
    "e2e_json": CANONICAL_ROOT / "metrics" / "g8_posctx_dim_opencv_f3__ap_e2e_phase5_metrics.json",
    "precomputed": CANONICAL_ROOT / "g8_posctx_dim__ap_eval_opencv_count_f3.jsonl",
    "color": "#00695C",
},
{
    "key": "g8_opencv_f4",
    "label": "G8 + OpenCV (F4 oracle)\n(MM, wall oracle)",
    "display": "G8 + OpenCV F4 (wall oracle — capability ceiling)",
    "input_regime": "Canonical multimodal + OpenCV F4 oracle",
    "source_type": "canonical_json",
    "e2e_json": CANONICAL_ROOT / "metrics" / "g8_posctx_dim_opencv_f4__ap_e2e_phase5_metrics.json",
    "precomputed": CANONICAL_ROOT / "g8_posctx_dim__ap_eval_opencv_count_f4.jsonl",
    "color": "#004D40",
},
```

The two rows anchor the thesis narrative:
- **F3 row** = realistic regime (what the system delivers today)
- **F4 row** = capability ceiling (what the counter can do given correct wall identity)
- **Gap between F3 and F4** = size of the future-work opportunity (F2 junction detection)

Regenerate fig03. Expect:
- F3 row ≈ G8 baseline or small lift — OpenCV as soft signal doesn't change hard filtering much
- F4 row > G8 baseline — counter's position match tightens retrieval pool cleanly
- If neither lifts visibly → back to T1.2 (bugs still present)

### T2 — Decomposed Graph-RAG reranker (CoT) — *highest research value*
**Effort:** ~2 days  
**File:** `mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py:515-528` (`_build_prompt`)

With T1 shipping, reranker inputs can include OpenCV counts in descriptors, so CoT questions naturally include positional reasoning when that signal is available.

For implementation, keep the current prompt as baseline and add one new CoT mode:

**Variant — Decomposed CoT questions**
```
Q1: Which candidates share the same wall host?
Q2: Do the opencv counts match the candidates' wall_position_index?
Q3: Among compatible candidates, which matches neighbor type / subtype?
Q4: Rank using answers above.
```

Implementation:
- Modify `_build_prompt()` to support `mode="single_shot"` and `mode="cot"`
- Two-phase Gemini call: answer questions → rank using answers
- Standardize eval on the 60-case AP held-out set (`cases_ap_heldout_e2e.jsonl`; mirrors `lora6_v2_ap_eval_canonical_m_g7.jsonl`)
- Compare MRR / Top-1 against current single-shot prompt

### T3 — Dimension classification from IFC clusters — *medium gain*
**Effort:** ~3-4 days (requires LoRA retraining)

Replace the mm-regression task with cluster classification.

Implementation:
- Mine clusters from `docs/ifc_data_reality/property_analysis.txt` (real IFC dimensions)
- k-means (k=3 or 4) on (width, height) per ifc_class → `dimension_clusters.json`
- Annotate training cases with `size_cluster` labels
- Add `size_cluster` as JSON output field in LoRA prompt schema
- Swap `target_width_mm` / `target_height_mm` in `Constraints` → `size_cluster: Optional[str]`
- Replace ±50mm tolerance filter at `retrieval_backend.py:821-830` with cluster-membership match

### T4 — Update-to-Weight Ratio tracking — *diagnostic only*
**Effort:** ~1 day  
**File:** `training/train_lora6.py:478-498` (extend `ProgressCallback`)

Projection Layer 稳定性至关重要。If this layer changes too much, spatial grounding from pretraining is lost. If it barely changes, fusion isn't adapting.

Implementation:
- Add `LayerUpdateCallback`: at each logging step, `param.grad.norm() / param.data.norm()` per layer group
- Track: `visual.merger.*`, `visual.patch_embed.*`, `lm_head.*`
- Output figure → `mscd_demo/docs/plots/phase4_lora6_main/layer_update_ratio.png`
- Healthy range ~1e-3 to 1e-2. Flag `visual.merger` <1e-4 (not adapting) or >1e-1 (forgetting)
- Remedies if out-of-range: separate LR for `visual.merger` (Learning Rate Discrepancy); or full-FT that layer while LoRA-ing others
- Informs *next* LoRA training run; not a retrieval feature
- Runs alongside T3 LoRA retrain

### T5 — Space filtering (DXA-only) — *hold; do last*
**Effort:** ~1 day once prioritized  
**Scope:** DXA only — deprioritized because Phase 6 focuses on AP and DXA is a separate ablation

Why hold: AP (primary benchmark) has no usable space boundaries. DXA would be a bonus "space-aware IFC" study, not a core Phase 6 result. Run only after T1–T4 land and AP numbers are frozen.

Data audit (reference):
| Model | IfcSpace | RelSpaceBoundary | Wall-related | Names |
|-------|----------|------------------|--------------|-------|
| AP    | 8        | 0                | 0            | gibberish ("3ROK", "Area") |
| BH    | 0        | 0                | 0            | — |
| DXA   | 18       | **264**          | **96**       | **real** ("Foyer", "Living Room", "Bathroom 1", "Hallway") |

Implementation when unblocked:
- Parse `IfcRelSpaceBoundary` in `ifc_engine.py`
- Add `(wall)-[:BORDERS]->(space {name})` edges during Neo4j export
- Add `space_name` filter branch in `retrieval_backend.py`
- Populate existing `space_name` field in Constraints from chat cues
- Re-run DXA-subset eval → measure gain
- Thesis framing: "space-annotated IFC (DXA)" vs "sparse-annotation (AP/BH)" ablation

---

## Deferred / dropped (with rationale)

- ❌ **`wall_facing` (N/S/E/W)** — not all walls fit cardinal directions; diagonal/curved walls break the assumption
- ❌ **`is_external` as filter** — indoor-photo scenarios don't discriminate; only useful as input to wall_facing (dropped)
- ❌ **Geographic prefilter via `crop_bbox_world`** — v2 JSON is training-time reference only; inference reproduces bbox via T1 template matching
- ❌ **Space filtering on AP/BH** — data doesn't support it; no injection because injection ≠ real-world capability
- ❌ **Exact photo↔wall pose estimation** — separate thesis-scale problem (camera pose / visual localization)

---

## Revised pipeline (end-to-end)

```
user inputs:
  chat_history + site_photo + floorplan_patch + full_floorplan (project data)

╔═══════════════════════════════════════════════════════════════════╗
║ Stage 1 — OpenCV (T1)                                              ║
║   patch + full_floorplan → template match → target wall → count    ║
║   output: {position, total, confidence}                            ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ injected as structured prompt context
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 2 — Neuro VLM                                                ║
║   chat + photos + opencv_facts → Constraints                       ║
║   extracts: storey, ifc_class, space_name, material, size_cluster  ║
║   echoes  : position_context (validated against opencv)            ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ Constraints
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 3 — Symbolic query                                           ║
║   Priority-0 spatial_triplet with position_context hard filter     ║
║     + storey + ifc_class + (T3 size_cluster) + (T5 space_name)     ║
║   output: pool of ~10-30 candidates                                ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ candidates + graph fingerprints
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 4 — Decomposed Graph-RAG rerank (T2)                         ║
║   CoT questions over residual pool → Top-1                         ║
╚═══════════════════════════════════════════════════════════════════╝
```

Graceful degradation: every stage has a fallback when the upstream signal is absent or low-confidence.

---

## Execution order

| # | Task | Status |
|---|------|--------|
| 1a | T1.1 Counter diagnostics | ✅ done |
| 1b | T1.2 Counter bug fixes (ordering + filter) | ✅ done — outputs deterministic; F3 rate unchanged (frame mismatch dominates) |
| 1c | T1.2.5 Renderer scope decision (Floor 1 + Garage + Level 1, 15 cases) | ✅ done — Floors 2-5 deferred to Ch. 8 future work |
| 1d | T1.3 Dual-mode validation (F3 + F4) | ✅ done — F3 13.3%, **F4 80.0%** on 15-case scoped set |
| 1e | T1.3.5 Annotation visualizer | ✅ done — per-case PNG overlays |
| 1f | T1.4 fig03 Track B-2 refresh (F3 + F4 bars) | ⏳ **Next** — inject counter into G8 precomputed, rescore, add 2 bars |
| 1g | T1.5 Frame-mismatch architectural decision | ✅ baked in — F4+F3 ship; F2 junction detection = Ch. 8 future work |
| 2 | T2 Decomposed reranker (CoT) | ⏳ After T1.4 |
| 3 | T3 Dimension clusters (Pattern-B bucketize) | ⏳ After T2 |
| 4 | T4 Layer-ratio tracking (diagnostic) | ⏳ Dormant until next retrain |
| 5 | T5 DXA space filtering | ⏳ Hold until AP numbers frozen |

**Milestone 1 (in progress):** T1.3 F4 run → T1.4 fig03 dual-bar refresh. First measurable Phase 6 numbers.

**Milestone 2:** T2 decomposed reranker + T3 dimension clusters. Full Phase 6 feature set active.

**Milestone 3 (optional):** T5 DXA bonus study if time allows.

---

## Reference: current reranker prompt (to be extended in T2)
```python
def _build_prompt(case: dict, descriptions: Sequence[str], letters: Sequence[str]) -> str:
    query_text = case.get("query_text") or _flatten_chat(case.get("inputs") or {})
    example = " ".join(letters[: min(len(letters), 8)])
    return (
        "Match the construction-site evidence to the best BIM candidate.\n\n"
        "Use the site photo, floorplan patch, and query text. "
        "Prefer candidates whose type, storey, host wall, slot position, and left/right neighbors best match the evidence.\n\n"
        f"Query:\n{query_text}\n\n"
        "Candidates:\n"
        + "\n".join(descriptions)
        + "\n\nReturn only the ranked candidate letters from best to worst, separated by spaces.\n"
        f"Example: {example}\n"
        "Do not return JSON. Do not explain."
    )
```
