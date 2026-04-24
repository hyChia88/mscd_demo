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

### Tasks

| # | Task | File(s) | Effort |
|---|---|---|---:|
| 6.1.1 | **Soft rerank.** Drop hard `size_cluster` WHERE clause. Add cluster-match bonus in graph-RAG rerank scoring. Unblocks G9 regression regardless of the classifier. | `src/neurosym/retrieval_backend.py`, graph-RAG rerank module | 2–3 h |
| 6.1.2 | **Build crop dataset.** For each IfcWindow / IfcDoor in AP with a known `size_cluster`, crop 192×192 patch centered on element (fixed px/mm from calibration). Stratified 80/10/10 split. Save `{crop_path, label}` manifest. | new `data_curation/scripts/synth/10_build_cluster_crops.py` | 3–4 h |
| 6.1.3 | **Train ResNet-18 classifier.** PyTorch, small aug (rot90/180/270, ±8px position jitter, brightness jitter). Weighted CE for rare clusters. Collapse clusters with <5 instances into `other`. Report per-class precision/recall. | new `training/train_cluster_classifier.py` | 4–6 h CPU/GPU (trains in minutes) |
| 6.1.4 | **Inference wrapper.** At runtime: given target `(guid, world_xy)` → crop from full-storey floorplan → classifier → `(size_cluster, confidence)`. Shared `crop_target_patch()` with training. | `src/neurosym/cluster_classifier.py` (new) | 2 h |
| 6.1.5 | **Post-hoc override script.** Read G9 precomputed JSONL; for each in-scope W/D case, replace model's `size_cluster` with classifier's output (carry confidence). Mirror of existing `inject_floorplan_counts.py`. | new `evaluation/analysis/inject_size_cluster.py` | 2 h |
| 6.1.6 | **Run retrieval + rerank + score.** Reuse existing `script/run.py` + `graph_rag_rerank_ap.py` + `score_opencv_rescored_traces.py`. Produce Track A (cluster accuracy) + Track B-2 (Top-k, MRR). | existing | 1 h |
| 6.1.7 | **Thesis artifacts.** Update fig03 with the G9+classifier row; draft 1-page Phase 6.1 narrative: regression → root cause → mixture-of-specialists fix → result. | `docs/plots/phase4_lora6_main/` | 2 h |

**Total:** ~1.5–2 days engineering, no Modal GPU on the critical path (ResNet-18 trains on CPU or a single T4 in minutes).

### Pass conditions

| Gate | Metric | Must hit |
|---|---|---|
| 6.1.1 alone | Track B-2 Top-10 | ≥ 30% — recovers to G8 baseline |
| 6.1.1 + 6.1.5 + 6.1.6 | Cluster accuracy on held-out W/D test split | ≥ 80% |
| 6.1.1 + 6.1.5 + 6.1.6 | Track B-2 Top-10 | **> 30%** (beats G8+rerank) |
| 6.1.1 + 6.1.5 + 6.1.6 + graph-RAG rerank | Track B-2 Top-1 | **≥ 12%** (beats G8+rerank's 11.7%) |

**The pipeline is "proven" when Top-1 > 11.7% with rerank on top.** That's the minimum bar; anything higher is upside.

### Known scope limits (honest disclosure for thesis)

- **AP-only.** BH and DXA extend the classifier trivially (same crop pipeline, larger labeled set) but are deferred until AP proves the architecture.
- **Height-axis disambiguation.** Floorplans are top-down → classifier can't see height. Two mitigations, chosen during 6.1.3 based on confusion matrix:
  - If confusion is concentrated on height-only cluster pairs (e.g. `window_S_1000x1400` vs `window_S_1000x600`): collapse to width-major taxonomy for the classifier. Retrieval tolerates this because the rerank step picks among height variants using the site photo.
  - If confusion is mild: keep the full 15-way taxonomy.
- **F3/F4 scope (floors 2–5).** Doesn't apply to the classifier — it only needs the full-storey floorplan and the target's world_xy, both of which are available for all storeys. This is a side benefit over Step 2 (OpenCV measurement).

### What we're explicitly NOT doing

- ❌ No VLM retrain. G9 stays frozen.
- ❌ No prompt-block injection. Post-hoc JSONL rewrite is equivalent for retrieval, with no GPU cost.
- ❌ No OpenCV dimension measurement (dropped in favor of the classifier, which handles both width and learned pattern without calibration assumptions).
- ❌ No BH/DXA until AP proves out.
- ❌ No multi-axis crop fusion. If the height axis needs a dedicated model, that's Chapter 8.

### Deliverables

1. `cluster_classifier_ap.pt` — trained weights + inference wrapper
2. `inject_size_cluster.py` — post-hoc override script
3. Confusion matrix + per-class metrics table (Track A)
4. fig03 updated with G9+classifier row (Track B-2)
5. 1-page Phase 6.1 narrative for thesis chapter 7

**Milestone:** Phase 6.1 closes when 6.1.6 passes its gates and fig03 / thesis prose reflect the pipeline-proof result.
