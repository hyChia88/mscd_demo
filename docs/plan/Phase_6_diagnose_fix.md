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
  - `AdvancedProject.ifc` — primary (AP, 1233 elements) — **Phase 6 focus**
  - `BasicHouse.ifc` — BH
  - `Duplex_A_20110505.ifc` — DXA (18 spaces, 264 boundaries, real room names) — T5 target
- **Synthetic dataset** (`data_curation/datasets/synth_v0.5_ap/`)
  - `floorplans/` — 200 full floorplan PNGs (one per skeleton)
  - `floorplans_full/` — 3 scoped full-storey renders + calibration (G9 input)
  - `skeletons/skeletons.jsonl` — target_guid, host_guid, patch_center_xyz per case
- **IFC reality audit**: `mscd_demo/docs/ifc_data_reality/property_analysis.txt`
- **Dimension anchors**: `mscd_demo/prompts/dimension_anchors.json` (13 windows, 4 doors) — raw unique (w, h) counts
- **Size-cluster taxonomy**: `mscd_demo/prompts/size_cluster_taxonomy.json` — semantic labels (`window_M_1480x1380`, `door_S_760x2030`, …)

### Eval cases / experiments
- **Case files** (`mscd_demo/evaluation/cases/`)
  - `cases_ap_heldout_e2e.jsonl` — 60-case AP held-out
- **G9 training labels** (`data_curation/datasets/synth_v0.5_ap/train/`)
  - `lora6_v2_ap_train_canonical_m_g7.jsonl` — source for G9 train split (238 cases, 157 with dims)
  - `lora6_v2_ap_eval_canonical_m_g7.jsonl` — source for G9 eval split (60 cases, 38 with dims)
- **Validator**: `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py` (`--mode {f3,f4,both}`)
- **Reranker**: `mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py` (`--prompt-mode {single_shot,cot}`)
- **Plot output dir**: `mscd_demo/docs/plots/phase4_lora6_main/`

### Retrieval / constraint backend (`mscd_demo/src/`)
Note: `src/v2/` was renamed to `src/neurosym/`; `metrics_v2.py` → `metrics.py`; `V2Trace` → `PipelineTrace`.

- **Neo4j export**: `src/ifc_engine.py` — `_create_element_nodes()`, `_create_element_relationships()`, `_resolve_storey_query()`. G9 adds `size_cluster` property per IfcWindow/IfcDoor.
- **Schema**: `src/neurosym/types.py` — `Constraints`, `SpatialTriplet`, `RetrievalResult`. G9 adds `size_cluster: Optional[str]`; `target_width_mm/height_mm` retained as deprecated.
- **Query planner**: `src/neurosym/constraints_to_query.py` — priority table: 0=spatial_triplet/continuous_span, 1=space+type, 2=name_keyword, 4=storey+type, 5=storey_only, 6=type_only, 8=fallback. G9 replaces ±50mm dimension routing with `size_cluster` equality.
- **Retrieval executor**: `src/neurosym/retrieval_backend.py` — dimension filter at lines 821-830 replaced by `target.size_cluster = $target_size_cluster`.
- **LoRA extractor**: `src/neurosym/constraints_extractor_lora.py` — VLM → Constraints; consumes `[OpenCV Counting]` prompt block (T1 hybrid).
- **Floorplan counter**: `src/neurosym/floorplan_counter.py` — F3 (`count_from_full_storey`) + F4 (`count_from_full_storey_with_wall_bounds`).

### Training (`mscd_demo/training/`)
- `train_lora6.py` — LoRA config; G9 adapter trained via Modal.

### Config
- `mscd_demo/config.yaml` — Neo4j at `bolt://localhost:7687` pw=password
- `mscd_demo/prompts/graphrag_rerank.yaml` — static reranker prompt text
- Neo4j Community 5.26.0 at `/tmp/neo4j-community-5.26.0/` (manual start)

### Data-build scripts (`data_curation/scripts/synth/`)
- `1_build_index.py`, `2_hunt_skeletons.py`, `3c_render_full_storeys.py`
- `7_build_dimension_anchors.py` → writes `mscd_demo/prompts/dimension_anchors.json`
- `8_build_size_cluster_taxonomy.py` *(G9.1)* → writes `mscd_demo/prompts/size_cluster_taxonomy.json`
- `9_assemble_lora9.py` *(G9.2, new)* → writes G9 train/eval JSONL

---

## Design principle — tool-augmented VLM

Phase 5 identified the core failure: VLMs are bad at counting and measuring, but good at semantic interpretation. Phase 6 inverts the dependency: **run the deterministic perception tool first, inject its facts into the VLM prompt**, then let VLM focus on semantics.

Pipeline stages:
```
OpenCV counting  →  Neuro VLM (G9)  →  Symbolic query  →  Graph-RAG rerank
(deterministic)    (semantic +         (Cypher over       (single-shot)
                    cluster clf)        structured pool)
```

Ablation path: `G8 baseline` → `+ OpenCV F4 override` → `+ Graph-RAG rerank` → `G9 (retrain with OpenCV facts + cluster classification)`.

---

## G9 — OpenCV counter + size-cluster classification (T1 + T3 combined)

G9 is a retrain that bakes Phase 6's two real modeling changes into one adapter:
- **Input change (T1 hybrid):** `[OpenCV Counting]` block injected into the user prompt during training. F4 oracle values used as training labels (wall bounds known); F3 runs at inference with the F4-override path when confidence is high.
- **Output change (T3):** `target_width_mm` / `target_height_mm` deprecated; LoRA emits `size_cluster: str` (classification). Retrieval filter becomes equality on `target.size_cluster`.

### Scope decisions
- **mm fields kept as deprecated** in `Constraints` — no tokens emitted, no training-capacity cost; remains importable from legacy traces (`extra="ignore"`).
- **Hybrid OpenCV supervision** — training gets clean F4 labels; inference uses F3 by default with the `merge_position_context()` override when OpenCV confidence ≥ 0.8.
- **Semantic cluster labels** — e.g. `window_M_1480x1380`, `door_S_760x2030`. Readable in thesis tables; loaded from `size_cluster_taxonomy.json`.

### Execution steps

| # | Task | Effort | Output |
|---|------|--------|--------|
| G9.1 | Build `size_cluster_taxonomy.json` from existing `dimension_anchors.json` — assign semantic labels + order by frequency | 30 min | `mscd_demo/src/neurosym/size_cluster_taxonomy.json` |
| G9.2 | `6_assemble_lora9.py` — training-set assembly: `[OpenCV Counting]` in user prompt (F4 oracle), `size_cluster` in assistant output | 2-3h | script |
| G9.3 | Regenerate train + eval JSONL | 30 min | `lora6_v2_ap_{train,eval}_canonical_m_g9.jsonl` |
| G9.4 | Extend `ifc_engine.py` — Neo4j stores `size_cluster` per IfcWindow/IfcDoor via taxonomy lookup | 30 min | Neo4j reimport |
| G9.5 | Update `retrieval_backend.py` — replace ±50mm filter with `target.size_cluster = $target_size_cluster` | 20 min | backend patch |
| G9.6 | Update `types.py` — add `size_cluster: Optional[str]`; keep mm fields, mark deprecated | 10 min | schema patch |
| G9.7 | **Modal training** — copy G8 config, train G9 adapter on new data; **T4 layer-ratio tracking runs here** (extends `ProgressCallback`, logs `update/weight` per layer group, emits `layer_update_ratio_g9.png`) | 4-6h GPU | `/mscd-lora-v6-g9/best` + T4 diagnostic plot |
| G9.8 | **Modal eval** — produce `g9__ap_eval.jsonl` | 1h GPU | precomputed |
| G9.9 | Run retrieval + score vs G8 baseline | 30 min | metrics JSON |
| G9.10 | Add G9 row to fig03 | 15 min | fig03 refresh |

**Total:** ~8-10h engineering + 5-7h GPU.

### Expected result (hypothesis)

| System | Top-1 | MRR | Notes |
|---|---|---|---|
| G8 baseline | 6.7% | 0.1104 | LoRA6 G8, no OpenCV, mm regression (0% emitted) |
| G8 + OpenCV F4 override | 6.7% | 0.1126 | T1.4 — override only affects 16/60, no Top-1 lift |
| G8 + OpenCV F4 + single-shot rerank | **11.7%** | **0.1612** | T2 shipped |
| G9 (OpenCV F4 supervision + cluster clf) | *target >G8+rerank* | *target >0.17* | retrain brings structured fields into VLM supervision |

Retrain payoff hypothesized on two fronts: (a) dimension field accuracy jumps from ~10% mm-regression to ~70% cluster-classification; (b) LoRA learns to consume OpenCV facts rather than guess position.

### T1.x and T3.x artifacts already landed (reused by G9)
- `src/neurosym/floorplan_counter.py` — F3 + F4 entry points, ordering canonicalized, PCA + IFC local-X alignment. Counter validated at **F3 13.3% / F4 80.0%** on 15-case scoped set (Floor 1 + Garage + Level 1).
- `data_curation/scripts/synth/3c_render_full_storeys.py` — 3 scoped full-storey PNGs + calibration index.
- `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py` — `--mode {f3,f4,both}` validator with IFC wall-endpoint helper.
- `mscd_demo/evaluation/h2/annotate_phase6_counter.py` — per-case PNG overlays for thesis figures.
- `mscd_demo/evaluation/analysis/inject_floorplan_counts.py` — `--mode {f3,f4}` injector for post-hoc overrides on non-G9 traces.
- `data_curation/scripts/synth/7_build_dimension_anchors.py` — anchors mined from IFC (13 windows, 4 doors) → `mscd_demo/prompts/dimension_anchors.json`
- `data_curation/scripts/synth/8_build_size_cluster_taxonomy.py` — semantic labels → `mscd_demo/prompts/size_cluster_taxonomy.json`

### Frame-mismatch decision (retained as future-work signpost)

OpenCV's visible-line count can span multiple colinear IFC walls. Four options considered; Phase 6 ships **F4 + F3**:

| Option | Frame-aligned | Vision-only | Verdict |
|---|---|---|---|
| F1 — IFC injection at inference | ✅ | ❌ | Circular dependency in production; not viable |
| F2 — Junction detection | ✅ | ✅ | Multi-week; **Chapter 8 future work** |
| F3 — Soft signal to reranker | ⚠️ | ✅ | Default inference path (13.3% exact match counter-level) |
| **F4 — Oracle wall bounds** | ✅ | demo/training only | **Capability ceiling (80.0%); used for G9 supervision** |

Gap between F3 and F4 sizes the F2 future-work opportunity.

### Scope limitations (honest disclosure)

- **Counter scope:** 15 / 33 canonical wall-counting cases (Floor 1 + Garage + Level 1). Floors 2-5 out of scope due to IFC `IfcRelContainedInSpatialStructure` under-populating those storeys' element lists — deferred to Chapter 8.
- **3 F4 failures** are target-pixel-boundary ambiguity (target world_xy lands between two adjacent openings) — dataset ceiling.

---

## T2 — Graph-RAG reranker prompt modes — ✅ *done (2026-04-21)*

Symbolic backend only — no retrain; runs on any precomputed trace (G8+OpenCV F4 today, G9 when ready).

**Files:** `mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py`, `mscd_demo/prompts/graphrag_rerank.yaml`

Modes:
- `--prompt-mode single_shot` (default) — baseline rerank, one Gemini call
- `--prompt-mode cot` — ablation, two-call decomposed reasoning

**Result on frozen G8 + OpenCV F4 trace:**

| System | Top-10 | Top-1 | MRR@10 |
|---|---:|---:|---:|
| G8 + OpenCV F4 | 30.0% | 6.7% | 0.1126 |
| + single-shot rerank | 30.0% | **11.7%** | **0.1612** |
| + CoT rerank | 30.0% | 8.3% | 0.1289 |
| P1-only control | 20.0% | 0.0% | 0.0321 |
| P1-only + single-shot rerank | 20.0% | 6.7% | 0.0970 |
| P1-only + CoT rerank | 20.0% | 3.3% | 0.0661 |

**Verdict:** ship `single_shot`; keep `cot` as ablation. CoT positive but weaker + costlier.

**Artifacts:**
- `mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/phase6_f4_{single_shot,cot,t2_comparison}/`
- `mscd_demo/docs/plots/phase4_lora6_main/fig11_phase6_f4_graph_rag_rerank_{comparison.png,summary.md,significant_cases.md}`

Prompt assembly:
- Static text → `mscd_demo/prompts/graphrag_rerank.yaml` (`system_instruction`, `single_shot_user`, `cot_reasoning_user`, `cot_rank_user`)
- Dynamic Python-assembled fields → `{evidence_block}` (OpenCV position facts), `{query_text}`, `{candidate_block}`, `{example}`, `{cot_reasoning}`

T2 plugs into whatever precomputed trace is supplied via `--trace-jsonl` — will be rerun on G9 output when available.

---

## T4 — Update-to-Weight Ratio tracking — *diagnostic, runs alongside G9 training*

**File:** `training/train_lora6.py:478-498` (extend `ProgressCallback`)

Projection Layer 稳定性至关重要。If this layer changes too much, spatial grounding from pretraining is lost. If it barely changes, fusion isn't adapting.

- Add `LayerUpdateCallback`: `param.grad.norm() / param.data.norm()` per layer group each logging step
- Track: `visual.merger.*`, `visual.patch_embed.*`, `lm_head.*`
- Healthy range: ~1e-3 to 1e-2; flag `visual.merger` <1e-4 or >1e-1
- Figure → `docs/plots/phase4_lora6_main/layer_update_ratio_g9.png`
- Runs during G9.7 Modal training; no standalone run

---

## T5 — DXA space filtering — *held until AP/G9 numbers frozen*

**Scope:** DXA only (AP has 0 usable `IfcRelSpaceBoundary`; no injection).

Data audit:
| Model | IfcSpace | RelSpaceBoundary | Wall-related | Names |
|-------|----------|------------------|--------------|-------|
| AP    | 8        | 0                | 0            | gibberish |
| BH    | 0        | 0                | 0            | — |
| DXA   | 18       | **264**          | **96**       | **real** |

Implementation when unblocked:
- Parse `IfcRelSpaceBoundary` in `ifc_engine.py`
- Add `(wall)-[:BORDERS]->(space {name})` edges during Neo4j export
- Add `space_name` filter branch in `retrieval_backend.py`
- Re-run DXA-subset eval → thesis framing "space-annotated IFC (DXA) vs sparse (AP/BH)"

---

## Deferred / dropped

- ❌ `wall_facing` (N/S/E/W) — diagonal/curved walls break the assumption
- ❌ `is_external` as filter — indoor-photo scenarios don't discriminate
- ❌ Geographic prefilter via v2 `crop_bbox_world` — not available at inference
- ❌ Space filtering on AP/BH — data doesn't support it
- ❌ Photo↔wall pose estimation — separate thesis-scale problem
- ❌ Retrain-avoidance Pattern A override in production (kept as T1.4 post-hoc tool only) — superseded by G9 proper retrain

---

## Revised pipeline (end-to-end)

```
user inputs:
  chat_history + site_photo + floorplan_patch + full_floorplan

╔═══════════════════════════════════════════════════════════════════╗
║ Stage 1 — OpenCV (F3 default, F4 in demo/validation)              ║
║   full_floorplan + skeleton/chat → wall detection + counting      ║
║   output: {position, total, confidence}                            ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ injected as [OpenCV Counting] block
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 2 — Neuro VLM (G9)                                           ║
║   chat + photos + opencv_facts → Constraints                       ║
║   emits: storey, ifc_class, space_name, size_cluster,              ║
║          position_context (validated against opencv), triplets     ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ Constraints
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 3 — Symbolic query                                           ║
║   P0 spatial_triplet + size_cluster equality + wall_position_index ║
║   → pool of ~10-30 candidates                                      ║
╚═══════════════════════════════════════════════════════════════════╝
                              ↓ candidates
╔═══════════════════════════════════════════════════════════════════╗
║ Stage 4 — Graph-RAG rerank (single-shot)                           ║
║   candidate fingerprints + opencv evidence → Top-1                  ║
╚═══════════════════════════════════════════════════════════════════╝
```

Graceful degradation: every stage falls back when upstream signal is absent or low-confidence.

---

## Execution order

| # | Task | Status |
|---|------|--------|
| 1 | **G9 — OpenCV counter + size-cluster classification (T1+T3 combined)** | ⏳ next |
| 2 | T2 Graph-RAG reranker (symbolic backend) | ✅ done — reruns on G9 trace when available |
| 3 | T4 Layer-ratio tracking | ⏳ piggy-backs on G9 training |
| 4 | T5 DXA space filtering | ⏳ hold until G9 numbers frozen |

**Milestone:** G9.1-G9.6 CPU work → G9.7-G9.8 Modal training+eval → G9.9 retrieval → G9.10 fig03 + T2 rerun on G9 trace → Phase 6 complete.

---

## G9 actual result (2026-04-22) — regression, root-caused

### Track A (per-field extraction accuracy vs GT, n=60)

| Field | G8 | G9 | Notes |
|---|---|---|---|
| `storey_name` | 60/60 | 60/60 | text cue, stable |
| `ifc_class` | 60/60 | 60/60 | text cue, stable |
| `target_name_keyword` | 9/31 | 11/31 | small G9 gain |
| `position_context` (emitted) | 0/59 | 57/59 | G9 massive emission gain |
| `position_context` (exact) | 0/59 | 13/59 | still noisy; soft signal only |
| `spatial_relations` (set) | 55/60 | 53/60 | marginal regression |
| **`size_cluster`** | n/a | **12/38 = 31.6%** | new field, 26/38 wrong |

### Track B-2 (end-to-end retrieval, scored via `score_opencv_rescored_traces.py`)

| System | GT-in-top10 | Top-1 | MRR |
|---|---:|---:|---:|
| g7_position_context | 30.0% (18/60) | 6.7% | 0.102 |
| g8_posctx_dim | 30.0% (18/60) | 6.7% | 0.110 |
| **g9_opencv_cluster** | **25.0% (15/60)** | 6.7% | 0.106 |

Note: `gt_in_pool=100%` quoted for G7/G8 in prior tables came from the older full-pool scorer (~118-elem final pool). `score_opencv_rescored_traces.py` measures top-10 membership — apples-to-apples, G8 was already 30%, not 100%.

### Root cause — precision-recall imbalance on a destructive filter

Accounting across the 60 cases under the G9 `target.size_cluster = $val` equality filter:

| Case group | Count | Effect vs G8 |
|---|---:|---|
| G9 doesn't emit `size_cluster` | 22 | neutral (no filter fires, same as G8) |
| G9 emits *correct* cluster | 12 | gain (pool narrows, GT kept) |
| G9 emits *wrong* cluster | 26 | **loss (GT hard-excluded from pool)** |

Net: +12 gains vs −26 losses = **−14 cases** → top-10 30% → 20–25%.

The trap: G8's "~10% mm-regression" was actually ~0% *emission rate*, so the G8 dimension filter almost never fired. "32% > 10%" compares classifier accuracies, not retrieval contributions. On a hard-equality filter, anything below ~70% precision is net-negative — one wrong prediction eliminates GT; one correct prediction only narrows the pool. G9 sits well below that threshold.

Typical G9 miss pattern (same size class, adjacent bucket):
- `door_M_920x2130` ↔ `door_M_760x2030`
- `window_M_2000x1400` ↔ `window_M_1500x1400`
- `window_S_1000x1400` ↔ `window_S_1000x600`

→ visually similar floorplan footprints; VLM has no explicit pixel→mm scale signal to disambiguate.

### Pipeline bugs found and fixed during diagnosis

| Loc | Bug | Status |
|---|---|---|
| `training/eval.py:649` | `constraints` dict dropped `size_cluster` at persist time → 0/60 in output JSONL despite adapter emitting it | ✅ patched 2026-04-22 |
| `script/run.py:392` | `--precomputed` loader didn't forward `size_cluster` into `Constraints` → retrieval never applied the filter in first G9 run | ✅ patched 2026-04-22 |
| `score_opencv_rescored_traces.py` vs legacy scorer | `gt_in_pool` measured different pools (top-10 vs full-118) → misread as "100% → 20%" | documented; Track B-2 now uses top-10 scorer consistently |

---

## Phase 6.1 — Minimal pipeline-proof plan (AP only)

### Design principles (generalised from G9 + keyword post-mortem)

Two losses share one root cause: **noisy text/classifier outputs put behind hard equality filters in retrieval erase the signal entirely**. G9 `size_cluster` (32% accuracy → −14 cases) and `target_name_keyword` (0/31 GT-keyword↔GT-IFC-name overlap, see "6.1.0 audit" below) both exhibited it.

1. **Filter ↔ rerank decoupling.** A field's tolerance for noise is opposite in the two stages. Hard-equality retrieval needs ≥70% precision or it's net-negative; the Gemini reranker is *designed* to consume noisy free text. Same field, two roles, two routes — never the same code path.
2. **Mixture of specialists.** VLM can't measure mm from thumbnails; retraining the VLM on the same input hits the same ceiling. Fix by **routing each signal to the tool that's cheapest for it**, then fusing through the existing post-hoc override hook.
3. **Soft signals before harder schemas.** Before adding a new structured field (classifier, taxonomy, retrain), ask whether the signal can survive as rerank-only context. If yes, ship it as soft evidence and only escalate to a structured field if a measured rerank ablation shows it's emission-limited rather than schema-limited.

### Design decision: mixture of specialists

VLM can't measure mm from thumbnails; retraining the VLM on the same input hits the same ceiling. Fix by **routing each signal to the tool that's cheapest for it**, then fusing through the existing post-hoc override hook.

```
┌────────────────────────────────────────────────────┐
│  VLM (G9 LoRA, unchanged)                          │
│    → storey, class, space, keyword, relations,     │
│      position_context                              │
└────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────┐
│  OpenCV counter (existing)                         │
│    → target localization + ordinal position        │
└────────────────────────────────────────────────────┘
┌────────────────────────────────────────────────────┐
│  NEW: ResNet-18 classifier on 192×192 crop         │
│    → size_cluster + confidence                     │
└────────────────────────────────────────────────────┘
              ↓
   inject_size_cluster.py (post-hoc JSONL rewrite)
              ↓
   Local retrieval + graph-RAG rerank
```

Scope: **AP only** — 454 labeled W/D crops is enough to prove the pipeline. BH/DXA extension is post-proof.

### Why this shape

- VLM stays frozen — no regression risk on storey/class/relations (already 100% / near-100%)
- Cluster classification becomes recognition at consistent scale, not measurement under missing scale — the exact task CNNs are good at
- Plugs into the *existing* post-hoc override hook (same pattern as `inject_floorplan_counts.py`); zero pipeline surgery
- Every component is independently swappable/ablatable for the thesis

### 6.1.0 audit — `target_name_keyword` is filter-stage-dead (2026-04-28)

Investigation conducted before scoping 6.1.1 surfaced a parallel failure to G9, sharing the same root cause:

| Measurement (60 AP held-out cases) | Result |
|---|---|
| GT-label CONTAINS GT-IFC-name (oracle keyword) | **0 / 31** |
| LoRA G7 emits keyword exactly matching GT label | 12 / 28 (43%) |
| LoRA G8 emits keyword exactly matching GT label | 9 / 31 (29%) |
| LoRA G7/G8 keyword CONTAINS the GT-element's IFC name | **0 / 28**, **0 / 31** |
| Gemini-baseline keyword vocabulary | `{"window","wall","door","wall junction"}` — redundant with `ifc_class` |

Vocabularies don't intersect: GT labels are human descriptors (`"floor-to-ceiling window"`, `"bathroom window"`); IFC names are Revit family/type strings (`"BALANS Fixed Single Window:BALANS 30M FLOOR (SH = 0)"`). [`_post_filter_by_name_keyword`](mscd_demo/src/neurosym/retrieval_backend.py#L942) cannot fire correctly even at oracle prediction. Self-diagnosed in [README.md:98](mscd_demo/src/neurosym/README.md#L98) ("graceful-to-a-fault — signal is wasted").

**Decision: filter/rerank decoupling.** Apply principle #1.
- **Retrieval side:** drop the post-filter and the priority-2 `name_keyword` strategy. Rename field to `target_description` to encode rerank-only intent.
- **Reranker side:** surface the descriptor in `_structured_evidence`. Gemini already sees IFC vocab on the candidate side via `_name_hint`; adding the human descriptor on the evidence side lets it bridge `"floor-to-ceiling window"` ↔ `"BALANS 30M FLOOR (SH = 0)"` — the one task LLMs are designed for.
- **Not pursued:** keyword classifier (Option C in audit). Information overlaps existing fields (`space_name` covers bedroom/bathroom; `size_cluster` covers floor-to-ceiling; `fire_rating` pset covers fire-rated). Adding a duplicative classifier risks repeating G9's regression on a field whose signal is already structurally encoded.

### Tasks

Targets revised 2026-04-28: the classifier now predicts **`size_band` (6-way)**, not the full 15-way `size_cluster`. See "Band-collapse decision" below.

| # | Task | File(s) | Effort | Status |
|---|---|---|---:|---|
| 6.1.0 | **Filter/rerank decoupling for keyword.** Drop `_post_filter_by_name_keyword` + priority-2 `name_keyword` strategy. Surface descriptor in `_structured_evidence`. | `src/neurosym/retrieval_backend.py`, `src/neurosym/types.py`, `src/neurosym/constraints_to_query.py`, `evaluation/experiments/graph_rag_rerank_ap.py` | 1–2 h | ✅ done |
| 6.1.1 | **Mode-controlled `size_cluster`.** Three modes (off/soft/hard) via `retrieval.size_cluster_mode` in `config.yaml`. Default `soft` (ORDER BY match). Switch flips to `hard` once classifier hits ≥70% precision. | `src/neurosym/retrieval_backend.py`, `src/pipeline_base.py`, `config.yaml` | 2–3 h | ✅ done |
| 6.1.2 | **Build crop dataset.** 192×192 patches centred on each in-scope W/D's centroid (mm→m unit fix). Manifest carries both `size_cluster` (15-way) and `size_band` (6-way). Stratified split is by `size_band`. **Result: 195 crops; train=155 / val=17 / test=23.** | [10_build_cluster_crops.py](data_curation/scripts/synth/10_build_cluster_crops.py) | 3–4 h | ✅ done |
| 6.1.3 | **Train ResNet-18 on `size_band`.** ImageNet-pretrained, 6-way head. Augs: rot90/180/270, ±8px translate, brightness/contrast jitter. Weighted CE handles class imbalance. Best-checkpoint by `min(val_loss)` (later ties win). **Result: test acc 82.6%, beats G9 (apples-to-apples on bands) 55.3% by +27pp.** | [train_cluster_classifier.py](mscd_demo/training/train_cluster_classifier.py); checkpoint at `mscd_demo/models/cluster_classifier_ap/best.pt` | 1–2 h GPU | ✅ done (2026-04-29) |
| 6.1.4 | **Inference wrapper.** `SizeBandClassifier` engine in `src/neurosym/`. `.predict(storey, world_xy_mm) → BandPrediction(band, confidence, logits)`. Caches PNG per storey for fast repeated cropping. | [src/neurosym/cluster_classifier.py](mscd_demo/src/neurosym/cluster_classifier.py) | 2 h | ✅ done |
| 6.1.5 | **Post-hoc inject script.** Reads G9 JSONL; oracle-centroid lookup → ResNet → confidence-gated injection (default `--min-confidence 0.6`). Writes `size_band`, `size_band_confidence`, `size_band_source` into `constraints`. | [evaluation/analysis/inject_size_band.py](mscd_demo/evaluation/analysis/inject_size_band.py) | 1–2 h | ✅ done |
| 6.1.6 | **Schema + retrieval mode for `size_band`.** `Constraints.size_band/_confidence/_source` added. Planner emits `target_size_band` param. Retrieval Cypher uses `STARTS WITH '{band}_'` (no Neo4j migration; reuses existing `size_cluster` column). Config flag `retrieval.size_band_mode` (default `hard` since ResNet 83% precision). **Bonus fix:** `_get_storey_type_pool` now also applies the band filter so the P1 fallback doesn't swamp the band-filtered P0 in p0_union_p1 mode. | [types.py](mscd_demo/src/neurosym/types.py), [constraints_to_query.py](mscd_demo/src/neurosym/constraints_to_query.py), [retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py), [config.yaml](mscd_demo/config.yaml), [pipeline_base.py](mscd_demo/src/pipeline_base.py), [script/run.py](mscd_demo/script/run.py) | 30 min | ✅ done |
| 6.1.7 | **Run retrieval + rerank + score.** See "Phase 6.1.7 results (2026-04-29)" below. | existing | 1 h | 🟡 partial — multiple ablations run, fusion variant pending rerank |
| 6.1.8 | **Thesis artifacts.** Update fig03 with G9 + ResNet rows; 1-page Phase 6.1 narrative. | `docs/plots/phase4_lora6_main/` | 2 h | ⏳ |

**Total:** ~1.5–2 days engineering, no Modal GPU on the critical path (ResNet-18 trains on CPU or a single T4 in minutes).

### Pass conditions

| Gate | Metric | Must hit |
|---|---|---|
| 6.1.0 alone | Track B-2 Top-1 with rerank | ≥ 11.7% (no regression vs current G8+F4+rerank); inspect rerank delta on the 31 keyword-bearing cases for descriptor signal |
| 6.1.1 alone | Track B-2 Top-10 | ≥ 30% — recovers to G8 baseline |
| 6.1.1 + 6.1.5 + 6.1.6 | Cluster accuracy on held-out W/D test split | ≥ 80% |
| 6.1.1 + 6.1.5 + 6.1.6 | Track B-2 Top-10 | **> 30%** (beats G8+rerank) |
| 6.1.1 + 6.1.5 + 6.1.6 + graph-RAG rerank | Track B-2 Top-1 | **≥ 12%** (beats G8+rerank's 11.7%) |

**The pipeline is "proven" when Top-1 > 11.7% with rerank on top.** That's the minimum bar; anything higher is upside.

### 6.1.0 + 6.1.1 result and empirical case for ResNet (2026-04-28)

Both quick-wins shipped and were measured on the AP held-out (n=60). They are **not** sufficient on their own; the data forces the ResNet path.

#### Implementation summary

| Change | File | Behaviour |
|---|---|---|
| 6.1.0 keyword filter killed | [retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py), [constraints_to_query.py](mscd_demo/src/neurosym/constraints_to_query.py) | Post-filter + priority-2 strategy removed. Field is rerank-only descriptor. |
| 6.1.0/6.1.1 evidence surfaced | [graph_rag_rerank_ap.py](mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py), [graphrag_rerank.yaml](mscd_demo/prompts/graphrag_rerank.yaml) | `target_description`, `size_cluster`, `extracted_space` rendered conditionally; bridging instruction added when fields populated. Each candidate also surfaces its own `size_cluster` in the description. |
| 6.1.1 size_cluster filter mode-controlled | [retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py), [pipeline_base.py](mscd_demo/src/pipeline_base.py), [config.yaml](mscd_demo/config.yaml) | Single config flag `retrieval.size_cluster_mode: off\|soft\|hard`. Default `soft`. Soft mode applies `ORDER BY CASE WHEN size_cluster = $val THEN 0 ELSE 1 END, pos`. Switch flips to `hard` once a higher-precision classifier replaces the field. |
| Mode rename | producer + 4 downstream consumers | `g7_pipeline` → `full_topology` (model-agnostic). Loaders auto-normalise legacy summary JSONs. |
| `--label` CLI | [graph_rag_rerank_ap.py](mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py) | Plot/summary headings parameterised; default `G7` for back-compat. Post-hoc relabeller at [relabel_rerank_plot.py](mscd_demo/evaluation/analysis/relabel_rerank_plot.py). |
| Cypher syntax bugfix | [retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py) | Soft-mode `ORDER BY` referenced bare `target.size_cluster` under `RETURN DISTINCT`, which Cypher rejects. 6 IfcDoor cases were silently dropped from the v1 trace (wrong-size-cluster cases that exercised the new ORDER BY). Fixed by aliasing `target.size_cluster AS size_cluster` in RETURN. |

#### Final v2 results (n=60 confirmed, no silent drops)

| System | Top-10 | Top-1 | MRR@10 | Δ vs G8+rerank baseline |
|---|---|---|---|---|
| G9 retrieval (size_cluster_mode = `soft`) | 21.7% | 6.7% | 0.0929 | −8.3pp Top-10 |
| G9 retrieval + rerank@30 (descriptor + size_cluster surfaced) | 23.3% | 5.0% | 0.0880 | −6.7pp Top-1 |
| **G8 + F4 + rerank@10 (reference baseline)** | **30.0%** | **11.7%** | **0.1612** | — |
| G9 retrieval (size_cluster_mode = `hard`, prior G9 baseline) | 25.0% | 6.7% | 0.1057 | −5pp Top-10 |
| G9 retrieval (size_cluster_mode = `off`, all signal removed) | 20.0% | 1.7% | 0.0659 | −10pp Top-10 |

#### What this proves

1. **Soft mode is the correct intermediate.** Off-mode (Top-10 20%) is worst; hard-mode (25%) excludes GT for 26 cases; soft-mode (21.7%) compromises in the middle but doesn't break recall — GT-in-pool stays at 100%.
2. **The rerank cannot rescue what retrieval cannot rank.** GT-rank median in soft-mode pool is ~30; even at top_k=30 the rerank's Top-1 went *down* (6.7% → 5.0%). Adding noisy candidates dilutes Gemini's attention; the descriptor surfacing was neutral.
3. **G9 VLM precision is the ceiling, not the data.** Per-cluster recall analysis ([investigation 2026-04-28](mscd_demo/output/lora6_v2_ap_20260331/g9_opencv_cluster__ap_eval.jsonl)) shows `window_M_1500x1400` at 11.1% recall *with 22 training examples* — the most-trained cluster fails. 62% of wrong predictions are size-class swaps (S↔M↔L↔XL), the discriminator the VLM cannot see at floorplan-thumbnail resolution.

#### Empirical case for ResNet (Phase 6.1.2 onwards)

| Path | Expected Top-1 ceiling | Wall-clock | Risk |
|---|---|---|---|
| Soft mode + bigger top_k | ≤ 7% | done | none — measured |
| Retrain G10 with collapsed taxonomy | ≤ 12% (50-60% cluster accuracy) | 4–6h | VLM still has no scale signal — same root failure |
| **ResNet-18 on consistent-scale crops** | **15–20% (target ≥ 80% cluster accuracy)** | **1.5-2 days** | known design risks (height-axis ambiguity, calibration drift) — both pre-flagged |

The mode-switch infrastructure is already in place. When 6.1.5's `inject_size_cluster.py` produces a precomputed JSONL with classifier-overridden values, flip `size_cluster_mode: "hard"` in config — no code change required.

### Band-collapse decision (2026-04-28)

The taxonomy was originally 15 fine-grained clusters (e.g. `window_M_1500x1400`, `window_M_2000x1400`). G9 emit accuracy on this taxonomy was 31.6%; per-cluster recall analysis on AP held-out shows the most-trained cluster (`window_M_1500x1400`, n=22 train) at **11.1% recall**. This isn't a data-quantity problem — it's a perception-modality problem: the VLM cannot distinguish 1500×1400 from 2000×1400 at floorplan-thumbnail resolution.

**The Phase 6.1 classifier targets `size_band` (6-way) instead of `size_cluster` (15-way):**

| Band | Constituent clusters | Total / Train / Val / Test |
|---|---|---|
| `door_M` | 760×2030, 920×2130 | 113 / 90 / 11 / 12 |
| `door_L` | 2010×2400 | 9 / 7 / 0 / 2 |
| `window_S` | 1000×600, 1000×1400, 1600×600 | 20 / 16 / 2 / 2 |
| `window_M` | 1500×1400, 2000×1400, 1500×2200, 2500×1200 | 30 / 24 / 3 / 3 |
| `window_L` | 2000×2200, 2500×2200 | 8 / 6 / 0 / 2 |
| `window_XL` | 3000×2200, 2500×3000 | 15 / 12 / 1 / 2 |

**Mitigation B chosen:** keep 6-way + weighted CE (inverse-frequency) rather than collapse `window_L` into `window_XL` or fold long-tail into `other`. Two minority classes (door_L=9, window_L=8) get all-train val splits; the test set still has 2 samples each for confusion-matrix evaluation. Class imbalance is handled by `nn.CrossEntropyLoss(weight=...)` at training time. The thesis story is clean: "the classifier learns the granularity the visual modality can support, no finer."

The full 15-way `size_cluster` field is **retained on disk and on Neo4j** for ablations and as a higher-precision input pathway for future work (e.g. when an OCR-based dimension extractor lands).

**Retrieval consumes bands via prefix matching** (`target.size_cluster STARTS WITH 'window_M_'`) — no Neo4j migration; the existing per-element `size_cluster` property covers it.

### Known scope limits (honest disclosure for thesis)

- **AP-only.** BH and DXA extend the classifier trivially (same crop pipeline, larger labeled set) but are deferred until AP proves the architecture.

- **Oracle world_xy at inference (mirrors F4 pattern).** The ResNet eats 192×192 crops centred on the target element's pixel coordinates. At eval time we read the GT element's centroid from `element_index.jsonl` (mm→m unit conversion) and crop there. This is the same scope concession F4 makes ("oracle wall bounds — demo/training only — capability ceiling"). In real production, the centroid would come from one of:
  1. **OpenCV F4's pixel-detected opening centre** — already pipelined; reverse-mapping to world coordinates is the integration we'd add next.
  2. **VLM emitting a pixel position** — G9 already emits `position_context` (e.g. "3rd of 17 openings"); coupling that to a coarse pixel estimate is plausible but unvalidated.

  Both are out-of-scope for proving Q1 (does ResNet beat VLM on dimensions?) and Q2 (does it bridge the retrieval gap?). The thesis will frame this section as: *"the classifier proves the perception ceiling for this modality; integrating the centroid-extraction step is a separate engineering problem."*

- **In-scope storeys only.** 3 of 6 AP storeys have full-floorplan renders + calibration (Garage, First Floor, Level 1). 195/389 W/D elements survive the storey filter. Floors 2–5 are the same skip we already have in F3/F4 counter scope (IFC `IfcRelContainedInSpatialStructure` under-populates those storeys). Extending the renderer is straightforward but deferred.

- **Height-axis disambiguation, partially mitigated.** Floorplans are top-down — the classifier cannot directly see element height (1400mm vs 2200mm window of equal width are distinguishable by area in IFC but not visually from above). Within bands this matters less; across bands it matters more. Confusion matrix from 6.1.3 will tell us which band-pairs (likely `window_M`↔`window_L`) need attention.

- **No augmentation of training data with synthetic crops.** Train n=155 is small. We rely on ImageNet pretrained weights for visual features and use weighted CE for class imbalance. If 6.1.3's confusion matrix shows minority classes (door_L=7 train, window_L=6 train) collapsing into majority classes, the move would be either (a) augment those bands with controlled crops, or (b) further-collapse to 4-way (`{door_M, door_L} → door`, `{window_S, M} → window_small`, `{L, XL} → window_large`). Let the data decide.

### What we're explicitly NOT doing

- ❌ No VLM retrain. G9 stays frozen.
- ❌ No prompt-block injection. Post-hoc JSONL rewrite is equivalent for retrieval, with no GPU cost.
- ❌ No OpenCV dimension measurement (dropped in favor of the classifier, which handles both width and learned pattern without calibration assumptions).
- ❌ No BH/DXA until AP proves out.
- ❌ No multi-axis crop fusion. If the height axis needs a dedicated model, that's Chapter 8.
- ❌ No keyword/description classifier. Field becomes rerank-only soft text (6.1.0); structured pivot is deferred and only revisited if a rerank ablation shows emission-limited rather than schema-limited signal.
- ❌ No 15-way classifier. The 31.6% G9 ceiling and per-cluster recall analysis (window_M_1500x1400 = 11.1% with 22 train samples) make 15-way classification a capacity problem, not a data problem. The 6-way `size_band` matches perception capability to retrieval granularity.
- ❌ No production world_xy extraction. Held-out eval uses oracle GT centroid (mirrors F4's "oracle wall bounds" pattern). Real-inference coupling to OpenCV F4's detected opening pixel is future work.

### Deliverables

1. `cluster_classifier_ap.pt` — trained weights + inference wrapper
2. `inject_size_cluster.py` — post-hoc override script
3. Confusion matrix + per-class metrics table (Track A)
4. fig03 updated with G9+classifier row (Track B-2)
5. Schema patch: `target_name_keyword` → `target_description`; rerank evidence-block update; ablation table comparing rerank Top-1 with vs without descriptor on the 31 keyword-bearing cases
6. 1-page Phase 6.1 narrative for thesis chapter 7 — frame as **two parallel applications of filter↔rerank decoupling** (size_cluster + descriptor)

**Milestone:** Phase 6.1 closes when 6.1.6 passes its gates and fig03 / thesis prose reflect the pipeline-proof result. 6.1.0 is the prerequisite quick-win that ships with no ML dependency.

---

## Phase 6.1.3-6.1.7 progress log (2026-04-29)

### 6.1.3 — ResNet training (✅ done)

**Pre-fix run** (best-by-`max(val_acc)` strict `>` tie-break): test 73.9%. Checkpoint locked in at epoch 3 (under-converged) due to `va_acc=0.882` tie with epoch 28.

**Post-fix run** (best-by-`min(val_loss)`, `<=` tie-break): test **82.6%**. Confusion matrix from [test_metrics.json](mscd_demo/models/cluster_classifier_ap/test_metrics.json):

```
              door_L  door_M  window_S  window_M  window_L  window_XL
door_L          1       0        0         1         0          0     1/2
door_M          0      12        0         0         0          0     12/12 ✅
window_S        0       0        2         0         0          0     2/2 ✅
window_M        0       0        1         2         0          0     2/3
window_L        0       0        0         1         0          1     0/2 ⚠
window_XL       0       0        0         0         0          2     2/2 ✅
```

`window_L` is the perception ceiling (n=2 test, both confused with neighbouring bands; conf 0.38, 0.54). Confidence-gating at ≥0.6 routes those cases to the rerank-only soft path automatically — they're not poisoned into the hard filter.

**Apples-to-apples G9→band score** (computed by [score_g9_size_band.py](mscd_demo/evaluation/analysis/score_g9_size_band.py)): 21/38 = 55.3%. **ResNet beats G9 by +27.3pp on bands** (Q1 firmly answered).

### 6.1.7 — retrieval + rerank ablation results

All runs n=60, p0_union_p1, top_k=10 except where noted.

| # | Pipeline | Top-10 retr | Top-1 retr | Top-1 +rerank | MRR rerank | Notes |
|---|---|---|---|---|---|---|
| A | G8 (no F4, no inject) | 30.0% | 6.7% | — | — | Phase 5 baseline |
| B | **G8 + F4 + rerank** | 30.0% | 6.7% | **11.7%** | **0.1612** | **Phase 6 production baseline** |
| C | G9 + (size_cluster_mode=`hard`) | 25.0% | 6.7% | — | — | Original G9 — −14 cases vs G8 hard-filter regression |
| D | G9 + (size_cluster_mode=`soft`) + rerank@30 | 23.3% | 5.0% | 5.0% | 0.0880 | Soft-mode test (Phase 6.1.1) |
| E | G9 + ResNet `size_band` (hard) + rerank | 26.7% | 6.7% | 6.7% | 0.1184 | First end-to-end ResNet integration |
| F | G9 + ResNet `size_band` + F4 + rerank | 26.7% | 6.7% | **3.3%** | 0.0920 | **Conflict regression** — F4×ResNet interfere |
| G | G9 + ResNet `size_band` + F4 + **fusion-rerank** | 26.7% | 6.7% | *pending* | *pending* | Fix 1+2 (2026-04-29) — confidence-aware + per-candidate fusion |

**Diagnosis of E→F regression (the F4×ResNet conflict):** size_band hard filter narrows the candidate pool to the GT band, but the resulting pool spans multiple walls. F4's `position_context` ("3rd of 17 openings on the same wall") is wall-scoped — it applies only to the GT element's wall. After band narrowing, candidates are split across walls and F4's slot pointer matches several wall-specific positions ambiguously. Gemini cannot tell which "3rd of 17" is the right wall.

**The pipeline issue (P1 swamping):** v1 of E showed pool size unchanged (~76) because `_get_storey_type_pool` was returning the unfiltered storey+type pool, which the union appended to the band-filtered P0. Fixed in v2 — pool size for size_band cases dropped to median 33; Top-10 climbed 21.7% → 26.7%, three new cases entered Top-10. Diagnosis + patch documented in [retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py) `_get_storey_type_pool`.

### Fix 1 + Fix 2 (2026-04-29, pending rerank rerun)

Confidence-and-fusion-aware reranking, two layered changes:

| Layer | What | File |
|---|---|---|
| **Fix 1** — confidence in evidence | `_fmt_evidence()` helper surfaces `(conf=…, src=…)` next to `size_band` and `position_context` in the rerank prompt's `evidence_block`. Gemini's prior is "trust higher conf more"; we measured this previously failing because conf was hidden. | [graph_rag_rerank_ap.py](mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py) `_structured_evidence` |
| **Fix 2** — per-candidate fusion score | `_candidate_match_signals()` computes deterministic match flags + a confidence-weighted fusion score per candidate (band_match × ResNet_conf, slot_match × F4_conf, weighted average). Top-K candidates are now sorted by fusion DESC before letter assignment, so Gemini sees a pre-ranked list. Match flags + fusion score appear in candidate descriptions. | [graph_rag_rerank_ap.py](mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py) `_candidate_match_signals`, `_format_candidate_description`, `_run_mode` |
| Prompt nudge | Tells Gemini that `(conf=, src=)` and `fusion=` are deterministic pre-scores; lean on the ordering when fusion margins are wide, fall back to image evidence when tight. | [graphrag_rerank.yaml](mscd_demo/prompts/graphrag_rerank.yaml) `single_shot_user` |

This is the response to the diagnosis "F4 and ResNet conflict because Gemini can't fuse them" — instead of asking Gemini to fuse, we fuse in Python (deterministic, debuggable) and surface the fused result. Gemini's role narrows to "verify fused top with image evidence".

### Latest file paths (single source of truth, 2026-04-29)

**Code (live-pipeline):**
- [src/neurosym/cluster_classifier.py](mscd_demo/src/neurosym/cluster_classifier.py) — `SizeBandClassifier` engine
- [src/neurosym/types.py](mscd_demo/src/neurosym/types.py) — `Constraints.size_band/_confidence/_source`
- [src/neurosym/constraints_to_query.py](mscd_demo/src/neurosym/constraints_to_query.py) — `target_size_band` planner param
- [src/neurosym/retrieval_backend.py](mscd_demo/src/neurosym/retrieval_backend.py) — `size_band_mode` constructor; STARTS WITH match in spatial_triplet single-hop + multi-anchor + `_get_storey_type_pool`
- [src/pipeline_base.py](mscd_demo/src/pipeline_base.py) — wires `size_band_mode` from config
- [script/run.py](mscd_demo/script/run.py) — `--precomputed` loader forwards `size_band` fields into `Constraints`
- [config.yaml](mscd_demo/config.yaml) — `retrieval.size_cluster_mode: soft` + `retrieval.size_band_mode: hard`

**Code (eval harnesses):**
- [evaluation/analysis/inject_size_band.py](mscd_demo/evaluation/analysis/inject_size_band.py) — oracle-centroid → ResNet → JSONL rewrite
- [evaluation/analysis/inject_floorplan_counts.py](mscd_demo/evaluation/analysis/inject_floorplan_counts.py) — F3/F4 OpenCV count override (existed)
- [evaluation/analysis/score_g9_size_band.py](mscd_demo/evaluation/analysis/score_g9_size_band.py) — G9 cluster→band collapse + accuracy
- [evaluation/analysis/relabel_rerank_plot.py](mscd_demo/evaluation/analysis/relabel_rerank_plot.py) — post-hoc plot/markdown relabel for saved rerank summaries
- [evaluation/experiments/graph_rag_rerank_ap.py](mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py) — reranker (now with Fix 1+2 fusion)
- [training/train_cluster_classifier.py](mscd_demo/training/train_cluster_classifier.py) — ResNet-18 trainer

**Data + dataset:**
- [data_curation/scripts/synth/10_build_cluster_crops.py](data_curation/scripts/synth/10_build_cluster_crops.py) — crop manifest builder (mm→m unit fix; band stratified split)
- `data_curation/datasets/synth_v0.5_ap/cluster_crops_ap/manifest.jsonl` — 195 crops train=155 / val=17 / test=23
- `data_curation/datasets/synth_v0.5_ap/cluster_crops_ap/manifest_summary.json` — per-band split breakdown

**Model artefacts:**
- `mscd_demo/models/cluster_classifier_ap/best.pt` — ResNet-18 weights
- `mscd_demo/models/cluster_classifier_ap/test_metrics.json` — confusion matrix + per-class P/R/F1
- `mscd_demo/models/cluster_classifier_ap/history.json` — per-epoch train/val curves

**Eval traces (precomputed JSONLs, in publication order):**
- `mscd_demo/output/lora6_v2_ap_20260331/g9_opencv_cluster__ap_eval.jsonl` — G9 raw emit
- `mscd_demo/output/lora6_v2_ap_20260331/g9_resnet_band__ap_eval.jsonl` — G9 + ResNet inject
- `mscd_demo/output/lora6_v2_ap_20260331/g9_resnet_band_f4__ap_eval.jsonl` — G9 + ResNet + F4 inject

**Retrieval traces:**
- `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase6_1_g9_resnet_band/` — v1 (broken P1 fallback, n=54)
- `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase6_1_g9_resnet_band_v2/` — **v2 with P1 fix (n=60, Top-10 26.7%)**
- `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase6_1_g9_resnet_f4/` — G9+ResNet+F4 (Top-10 26.7%)

**Rerank artefacts:**
- `…/graph_rag_rerank/phase6_f4_single_shot/` — G8+F4+rerank baseline (Top-1 11.7%)
- `…/graph_rag_rerank/phase6_1_g9_soft_top30_v2/` — G9 soft + descriptor (Top-1 5.0%)
- `…/graph_rag_rerank/phase6_1_g9_resnet_band_v2/` — G9+ResNet (Top-1 6.7%)
- `…/graph_rag_rerank/phase6_1_g9_resnet_f4/` — G9+ResNet+F4, no fusion (Top-1 3.3%, conflict)
- `…/graph_rag_rerank/phase6_1_g9_resnet_f4_fused/` — **pending: Fix 1+2 fusion rerun**

### Open question

Does fusion-aware reranking (Fix 1+2) clear the 11.7% baseline? Possible outcomes:

| Top-1 | Read | Action |
|---|---|---|
| ≥ 12% | Fusion solved the F4×ResNet conflict — Phase 6.1 ships. | 6.1.8 thesis artifacts |
| 9-11% | Major recovery from 3.3% but short of baseline. | One micro-fix per the per-case rank delta vs run F |
| 6-8% | Confidence/fusion didn't lift. F4×ResNet structural conflict needs architectural change (e.g. wall-scoped band filter). | Document as negative result; ship G8+F4 as final. |
| < 5% | Bug — fusion sort flipping correct cases. | Inspect per-case rank changes vs run E. |

### Phase 6.1 framing for thesis (if fusion rerun lands in 9-11% band)

Two scientifically valuable findings, both ship-worthy as a chapter:

1. **Q1 (positive):** ResNet-18 on consistent-scale 192×192 crops achieves 82.6% on `size_band` vs 55.3% for the VLM (apples-to-apples). Validates the perception-specialist architecture for spatial-dimension classification.

2. **Q2 (negative-but-informative):** Stacking band-classifier output on top of OpenCV F4 position-context evidence in a graph-RAG rerank does **not** improve end-to-end Top-1. The two narrowing axes (size, position) interfere because F4 is wall-scoped while band filtering scrambles wall coherence. The rerank stage cannot fuse signals it can't structurally compose. Mitigations explored: confidence annotations (Fix 1), Python-side deterministic fusion (Fix 2). Their effectiveness is the remaining open measurement.

This is a clean Phase 6.1 chapter. Q1 succeeds and is publishable; Q2's negative result identifies a *structural* limitation in single-shot rerank, motivating Chapter 8 future work on a wall-scoped band filter or junction-aware retrieval.
