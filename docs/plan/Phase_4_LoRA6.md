# Phase 4 — LoRA6: Region-Grounded Training Signal

> Consolidated plan — updated 2026-04-02
> Supersedes: `Research_Pivot_Summary.md`, `LoRAV-v2 Data mining IFC_Region-Grounded_Retrieval.md`, `LoRA6-v2 Implementation.md`, `LoRA6-v2 Evaluation Plan — Dual-Track Reuse of the Old Framework.md`

---

## 4.1 Motivation & Pivot

### 4.1.1 The Core Contradiction

Four diagnostics revealed that the system's bottleneck is training signal quality, not model capacity or planner design.

| Diagnostic | Finding | Implication |
|---|---|---|
| Q1: Positional fingerprint uniqueness | 389/389 FILLS elements = **100% unique** in IFC graph | Graph has all discriminative signal needed |
| Q2: Training data output distribution | 1,064 samples → only **37 unique outputs**, top-10 templates cover 78% | Model learned templates because labels offered nothing else |
| Q3: Type-group Oracle GT-in-Pool | Type-group alone: 60–68%. Needs correct storey to reach ~83% | Storey accuracy is the real gate |
| Q4: RTV zero-shot verification | Chat+desc: 0/10. Chat+floorplan+desc: 2/10 | Zero-shot RTV not viable as primary strategy |

```
IFC Graph:    389 elements → 389 unique positional fingerprints (100%)
                                    ↓ current data pipeline compresses to
Training Data: 1,064 samples → 37 unique spatial_relations outputs (3.5%)
                                    ↓ model faithfully learns what it's given
LoRA5 Output:  116 test cases → 14 unique patterns (template collapse)
```

The model isn't failing to learn — it's faithfully learning a collapsed supervision signal. Gemini zero-shot produces 61 unique patterns without any training, confirming VLM architecture is capable. LoRA5 fine-tuning actually reduced output diversity.

### 4.1.2 Strategy Selection

| Strategy | Verdict | Rationale |
|---|---|---|
| **Training Signal Fix (Plan A)** | **CHOSEN** | Fixes root cause (Q2), uses proven signal (Q1), no unvalidated deps |
| RTV Verification (Plan B) | Demoted to future work | Q4 shows 0–2/10 success; needs working G2L first |
| Type-Group Relaxation | Retained as supplement | Q3 ceiling ~68% without storey fix |

### 4.1.3 Architecture: Region-Grounded Supervision

The key insight: `wall_region_index.jsonl` (IFC enrichment sidecar) serves triple duty — rendering, training labels, and future G2L candidate descriptions.

```
                    wall_region_index.jsonl
                    (IFC enrichment sidecar)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         [Rendering]   [Training]    [Future: G2L]
         Region-aware  Position-aware  Candidate
         visual crops  GT labels       descriptions
```

Design principles:
- IFC is enriched by mined topological relations and region-grounded visual anchors
- Retrieval target remains the IFC entity; regions are for perception grounding and supervision
- Region layer is a sidecar index, NOT Neo4j first-class nodes — no graph migration risk
- Dual-layer GT: `retrieval_gt = owner_guid` (IFC element), `perception_gt = region_id` (local patch)

---

## 4.2 Data Mining & Assembly

### 4.2.1 Region Enrichment (AP, complete)

`wall_region_enrichment.py` → `wall_region_index_ap_20260331_c.jsonl`: **2,277 regions**

| Region Type | Count | Source Relation | Anchors |
|---|---|---|---|
| `opening_perimeter_patch` | 389 | FILLS | door/window → host wall |
| `wall_junction_patch` | 1,362 | CONNECTS_TO | wall → connected wall |
| `between_openings_patch` | 526 | NEXT_TO | filler → neighbor filler |
| `slab_crossing_patch` | 0 | CONTINUOUS | **not yet implemented** |

Each region carries: `region_id`, `region_kind`, `owner_guid`, `host_guid`, `anchor_guid`, `storey_band`, `patch_center_xyz`, `locatability_score`, `source_relation`.

### 4.2.2 Skeleton Mining & Judging (AP canonical, frozen)

312 skeletons mined → 297 KEEP after LLM judge + manual review.

| Predicate | Skeletons | KEEP | DISCARD |
|---|---|---|---|
| CONNECTS_TO | 73 | 69 | 4 |
| FILLS | 75 | 69 | 6 |
| NEXT_TO | 102 | 97 | 5 |
| ADJACENT_TO | 62 | 62 | 0 |
| **CONTINUOUS** | **0** | **0** | **—** |
| **Total** | **312** | **297** | **15** |

Skeleton dual-layer GT: 250/312 carry `region_kind` + `anchor_guid` from wall_region_index.

### 4.2.3 Three Canonical Relation Families

LoRA6-v2 is built around three primary relation families:
- **FILLS**: door/window fills host wall (opening_perimeter_patch)
- **CONNECTS_TO**: wall-to-wall junction (wall_junction_patch)
- **NEXT_TO**: consecutive openings on same wall (between_openings_patch)

ADJACENT_TO is included (62 KEEP) but treated as supplementary — cross-type proximity, not wall-grounded topology.

### 4.2.4 Assembly Pipeline

`6_assemble_lora6.py` consumes skins, site images, and floorplans to produce AP-only train/eval JSONLs.

**Data splits (frozen):**

| Set | Records | Notes |
|---|---|---|
| Train canonical | 237 | M-scale, site+floorplan+chat |
| Eval canonical | 60 | M-scale, site+floorplan+chat, held-out |
| Train augmented | 753 | Text + modality + scale augmentation |
| **Total train** | **990** | 237 canonical + 753 augmented |

Split rules: grouped by `skeleton_id`, seed 42, 80/20, stratified by predicate.

**Augmentation policy (train-only):**
- Text: T1 (all), T2 (40%), T3 (20% of Tier 2/3 only)
- Modality: canonical `site+FP+chat` (all), `FP+chat` (25%), `site+chat` (20%)
- Scale: M canonical, selective S/L for train only

**Input schema**: site image + floorplan(M) + chat. No global render, no wireframe.

**Output schema (LoRA6 label):**
```json
{
  "storey_name": "2",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWall", "object_material": "Plaster", "confidence": 1.0},
    {"predicate": "NEXT_TO", "object_type": "IfcWindow", "object_material": null, "confidence": 1.0},
    {"predicate": "NEXT_TO", "object_type": "IfcWindow", "object_material": null, "confidence": 1.0}
  ]
}
```

Valid predicates: `FILLS`, `ADJACENT_TO`, `CONTINUOUS`, `NEXT_TO`, `CONNECTS_TO`

### 4.2.5 BH/DXA Status: Abandoned

BH (25 KEEP) and DXA (39 KEEP) completed LLM judging but were never assembled into training data. Both abandoned (2026-04-02) due to persistent data quality issues. All mining focused exclusively on AP.

### 4.2.6 Known Data Gaps

| Gap | Severity | Status |
|---|---|---|
| **CONTINUOUS = 0 training cases** | Critical | Next mining target (see §4.5) |
| **Template collapse**: only 19 unique output templates | Critical | `between_openings_patch` position data exists but not in labels |
| Storey coverage skewed to Level 1 | Moderate | Addressable via targeted mining |
| IfcDoor under-covered for FILLS/NEXT_TO | Moderate | Low priority |

---

## 4.3 Implementation & Training

### 4.3.1 Planner Architecture

The query planner operates in two priority tiers:
- **P0 (spatial)**: uses extracted spatial_relations for graph-based retrieval
- **P1 (attribute)**: storey + ifc_class fallback

Strategy: `p0_union_p1` (P0 pool ∪ P1 pool) is the mainline — matches P0 ranking quality while preserving 100% GT-in-Pool safety.

**Phase 3 planner upgrade (2026-04-01):**
- `_execute_multi_anchor`: AND-intersection Cypher for any combination of FILLS + NEXT_TO + ADJACENT_TO + CONNECTS_TO
- Same-wall constraint: all NEXT_TO edges share `wall_guid`, all neighbors distinct
- Material filter on FILLS, ADJACENT_TO, CONNECTS_TO
- `ORDER BY target.wall_position_index` for ranking within pool
- `_relax_multi_anchor`: drops lowest-confidence SR one at a time; falls back to single-hop

### 4.3.2 Experiment Matrix (Core Sweep + Late Retrains)

The core G0-G3 sweep uses Qwen2.5-VL base model, 3 epochs, batch_size=2, grad_accum=4. The late-round G4/G6 retrains were exported separately and are summarized from the recovered adapter metadata plus realized evaluation results.

| Group | Train Data | LR | LoRA r/α | Key Variation |
|---|---|---|---|---|
| G0 canonical | 237 canonical | 1e-4 | r=16, α=32 | Baseline (no aug) |
| G1 fullaug | 990 augmented | 1e-4 | r=16, α=32 | Full augmentation |
| G2 fullaug-lowlr | 990 augmented | 5e-5 | r=16, α=32 | Lower learning rate |
| G3 fullaug-r32 | 990 augmented | 1e-4 | r=32, α=64 | Larger LoRA capacity |

**Late-round Phase 4 retrains (evaluated with the same Track A / B-1 / B-2 protocol):**

| Group | Exported adapter | LoRA r/α | Recovered signal | Eval role |
|---|---|---|---|---|
| G4 ultimate | `best` (`checkpoint-200` selected as best) | r=32, α=64 | Late-round retrain after first evaluation round | Current Track A winner |
| G6 baseline | `checkpoint-20` | r=32, α=64 | Deliberately weak / underfit control; trainer state shows best checkpoint at step 20 | Failure-reference baseline |

Adapters stored at `mscd_demo/models/lora6_v2_ap_20260331/<group>/{best,checkpoint-*}`

### 4.3.3 Inference Prompt Alignment

System prompt (shared by training and inference):
> "You are a construction site assistant that extracts IFC search constraints from multimodal evidence. Use the floorplan and site photo to reason about storey, element type, and spatial relations. Output valid JSON only."

Gemini zero-shot uses the same schema with an explicit `SCHEMA_HINT` block listing valid predicates and field types. A critical bug was found and fixed (2026-04-01): the original SCHEMA_HINT had stale LoRA2 fields, causing Gemini to output invalid predicates (`BESIDE`, `IS_NEXT_TO`).

---

## 4.4 Evaluation & Results

### 4.4.1 Evaluation Framework

Command:
```
Track A:
modal run mscd_demo/training/eval.py --adapter-dir /mscd-lora-v6-g7-position-context/best --cases /data/ap_eval.jsonl

modal volume get mscd-checkpoints /mscd-lora/eval_constraints_mscd-lora-v6-g7-position-context__best.jsonl mscd_demo/output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl

Track B-2:
MPLCONFIGDIR=/tmp/matplotlib-mscd python mscd_demo/script/run.py --profile v2_lora --cases mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl --precomputed mscd_demo/output/lora6_v2_ap_20260331/g7_position_context__ap_eval.jsonl --output_dir mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase5_g7/g7_position_context --config mscd_demo/config.yaml --profiles mscd_demo/profiles.yaml --p0-strategy p0_union_p1
```

Three evaluation tracks, coordinated:

| Track | Purpose | Benchmark | n |
|---|---|---|---|
| **A** | Intermediate extraction quality | AP held-out eval | 60 |
| **B-1** | External generalization (end-to-end) | Unified benchmark | 116 |
| **B-2** | Strict AP downstream (end-to-end) | AP held-out e2e | 60 |

### 4.4.2 Track A — Intermediate Extraction Quality (AP held-out, n=60)

| Model | Parse | Class | Storey | Hop-1 | Hop-2 | Pred P | Pred R | Dir | NEXT_TO | ADJ |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 canonical | 100% | 100% | 100% | 83.3% | 93.9% | 94.2% | 84.5% | 60.7% | 55.0% | 91.7% |
| G1 fullaug | 100% | 100% | 100% | 78.3% | 93.9% | 89.0% | 90.5% | 76.8% | 40.0% | 91.7% |
| G2 fullaug-lowlr | 100% | 100% | 100% | 85.0% | 93.9% | 92.0% | 89.7% | 73.2% | 60.0% | 91.7% |
| G3 fullaug-r32 | 100% | 100% | 100% | 80.0% | 93.9% | 89.1% | 91.4% | 78.6% | 45.0% | 91.7% |
| **G4 ultimate** | **100%** | **100%** | **100%** | **86.7%** | **93.9%** | **91.3%** | **81.0%** | 57.1% | **65.0%** | 91.7% |
| G6 baseline | 96.7% | 96.7% | 78.3% | 23.3% | 6.1% | 51.1% | 39.7% | 17.9% | 30.0% | 50.0% |
| Gemini AP (pre-fix) | 100% | 63.3% | 1.7% | 1.7% | 0.0% | 3.4% | 0.9% | 0.0% | 0.0% | 8.3% |
| Gemini AP v2 (schema-fixed) | 100% | 63.3% | 0.0% | 30.0% | 0.0% | 74.2% | 39.7% | 0.0% | 0.0% | 33.3% |

**Current winner: G4 Ultimate.** It reaches the best Hop-1 accuracy at 86.7% and the best NEXT_TO Hop-1 at 65.0%, which is the primary ranking rule for Track A.

**How to read the late-round retrains:**
- G4 improved over the original G0-G3 sweep on Hop-1 (+1.7pp over G2) but traded away predicate recall and direction accuracy; it is the strongest extractor, not the most balanced one.
- G3 still has the highest predicate recall (91.4%) and direction accuracy (78.6%), making it the strongest pre-retrain reference for relation completeness.
- G6 behaves as an underfit control: parse drops to 96.7%, storey collapses to 78.3%, and Hop-1 to 23.3%.
- Gemini AP v2 fixes the schema problem and recovers Hop-1 to 30.0%, but still fails storey grounding and directional reasoning.

**Ablation takeaways from the original G0-G3 sweep:**
- G0 vs G1 (augmentation): G1 gains predicate recall (+6.0pp) and direction accuracy (+16.1pp), but loses Hop-1 (−5.0pp).
- G1 vs G2 (learning rate): lower LR recovers Hop-1 (+6.7pp) and precision (+3.0pp) while maintaining high recall.
- G1 vs G3 (capacity): r=32 improves predicate recall (+0.9pp) and direction (+1.8pp), but does not beat G2 on Hop-1.

### 4.4.3 Track B-1 — Unified End-to-End (n=116)

| Model | GT-in-Pool | Top-10 | Top-1 | MRR | Avg Pool | Med Pool | Reduction | Storey Acc | IFC Acc | SR Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| Gemini Unified | 30.2% | 9.5% | 2.6% | 0.0415 | 208.0 | 48.0 | 65.0% | 73.3 | 57.8 | 15.5 |
| G2 fullaug-lowlr | 36.2% | 6.0% | 0.9% | 0.0206 | 105.6 | 100.0 | 71.6% | 0.0% | 26.7 | 52.6 |
| G4 ultimate | 31.9% | 4.3% | 0.9% | 0.0175 | 100.3 | 100.0 | 70.6% | 0.0% | 12.9 | 13.8 |
| G6 baseline | 44.8% | 5.2% | 1.7% | 0.0261 | 192.0 | 100.0 | 64.5% | 11.2 | 6.9 | 16.4 |

Unified remains a secondary benchmark because of known label/storey mismatches, but the external-generalization pattern is still informative:
- **Gemini Unified** ranks best when it succeeds (Top-10 9.5%, Top-1 2.6%, MRR 0.0415), but has lower GT-in-Pool.
- **G6** unexpectedly achieves the highest GT-in-Pool (44.8%), but with a very large average pool (192.0) and weaker ranking quality than Gemini.
- **G2** remains the cleanest LoRA trade-off on this benchmark by combining the best LoRA Top-10 (6.0%) with a much smaller pool than G6, while **G4** does not transfer well despite winning Track A.

### 4.4.4 Track B-2 — AP Held-Out End-to-End (n=60, p0_union_p1)

| Model | GT-pool | Top-10 | Top-1 | MRR | Avg Pool | Med Pool | Reduction | Storey Acc | IFC Acc | SR Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| G0 canonical | 100.0% | 25.0% | 1.7% | 0.0503 | 118.3 | 76.0 | 92.9% | 100.0 | 100.0 | 100.0 |
| G1 fullaug | 100.0% | 23.3% | 3.3% | 0.0645 | 118.3 | 76.0 | 92.9% | 100.0 | 100.0 | 100.0 |
| G2 fullaug-lowlr | 100.0% | 20.0% | 1.7% | 0.0524 | 118.3 | 76.0 | 92.9% | 100.0 | 100.0 | 100.0 |
| **G3 fullaug-r32** | **100.0%** | **26.7%** | 1.7% | **0.0641** | **117.4** | 76.0 | **93.0%** | 100.0 | 100.0 | 100.0 |
| G4 ultimate | 100.0% | 23.3% | 0.0% | 0.0324 | 118.3 | 76.0 | 92.9% | 100.0 | 100.0 | 100.0 |
| G6 baseline | 81.7% | 23.3% | 1.7% | 0.0515 | 138.4 | 76.0 | 91.7% | 80.0 | 96.7 | 96.7 |
| Gemini AP | 91.7% | 21.7% | 0.0% | 0.0482 | 108.9 | 76.0 | 93.5% | 95.0 | 100.0 | 68.3 |

**Key observations:**
- **G3** remains the strict AP downstream winner: best Top-10 (26.7%) and best MRR (0.0641) under the deployed planner.
- **G1** retains the best Top-1 (3.3%), even though its Top-10 is below G3.
- **G4** keeps perfect GT-in-Pool safety, but ranking collapses: Top-1 drops to 0.0% and MRR to 0.0324 despite winning Track A.
- **G6** recovers a reasonable Top-10 (23.3%) but loses the P1 safety net entirely on 11 cases, dropping GT-in-Pool to 81.7%.
- **Gemini AP** remains below the best LoRA models on strict downstream retrieval and still loses GT in pool on 8.3% of cases.

### 4.4.5 Oracle Suite — AP Held-Out

**Phase 2A — current-system strategy search:**

| Strategy | Top-10 | Top-1 | MRR | Avg Pool | Med Pool | Reduction |
|---|---|---|---|---|---|---|
| p0_only | 30.0% | 1.7% | 0.0710 | 65.5 | 46.0 | 96.1% |
| p1_only | 16.7% | 0.0% | 0.0392 | 107.2 | 76.0 | 93.6% |
| p0_intersect_p1 | 30.0% | 1.7% | 0.0710 | 65.5 | 46.0 | 96.1% |
| **p0_union_p1** | **30.0%** | **1.7%** | **0.0710** | 115.5 | 76.0 | 93.1% |

`p0_union_p1` remains the deployed mainline because it matches the best ranking metrics while keeping the safest recall behavior in the corresponding trace analysis.

**Phase 2B — topology-faithful oracle (P1-only upper bound vs full-topology union):**

| Oracle setting | Top-10 | Top-1 | MRR | Avg Pool | Med Pool | Reduction |
|---|---|---|---|---|---|---|
| p1_only_upper_bound | 16.7% | 0.0% | 0.0392 | 107.2 | 76.0 | 93.6% |
| **full_topology_union** | **30.0%** | **1.7%** | **0.0710** | 115.5 | 76.0 | 93.1% |

**Benefit of spatial relations (P1-only -> full-topology):**

| Metric | P1-only | Full-topology | Delta |
|---|---|---|---|
| Top-10 | 16.7% | 30.0% | **+13.3pp** |
| Top-1 | 0.0% | 1.7% | +1.7pp |
| MRR | 0.0392 | 0.0710 | +0.0318 |

**By topology universe (P1-only -> full-topology Top-10):**

| Universe | Cases | P1-only | Full-topology | Delta |
|---|---|---|---|---|
| U1 Wall-Connectivity | 14 | 14.3% | 35.7% | +21.4pp |
| U2 Adjacency-Singleton | 12 | 25.0% | 16.7% | -8.3pp |
| U3 Opening-Paired | 10 | 20.0% | 60.0% | **+40.0pp** |
| U4 Symmetric-Triad | 21 | 14.3% | 19.0% | +4.7pp |
| U5 Mixed-Triad | 2 | 0.0% | 50.0% | **+50.0pp** |
| U6 Rare/Edge | 1 | 0.0% | 0.0% | 0.0pp |

Spatial relations help most on **U3 Opening-Paired** and **U5 Mixed-Triad**, exactly the multi-anchor FILLS+NEXT_TO families that motivated the planner work.

### 4.4.6 Oracle-vs-Model Gap Analysis

| Reference | Top-10 | Top-1 | MRR | Med Pool | Reduction |
|---|---|---|---|---|---|
| p1_only_upper_bound | 16.7% | 0.0% | 0.0392 | 76.0 | 93.6% |
| **full_topology_union** | **30.0%** | **1.7%** | **0.0710** | 76.0 | 93.1% |
| **G3 fullaug-r32** | **26.7%** | **1.7%** | **0.0641** | 76.0 | 93.0% |
| G4 ultimate | 23.3% | 0.0% | 0.0324 | 76.0 | 92.9% |
| Gemini AP | 21.7% | 0.0% | 0.0482 | 76.0 | 93.5% |

The current oracle-vs-model picture is tighter and more precise than the earlier Phase 3 ceiling estimate:
- Full-topology oracle improves over P1-only by **+13.3pp Top-10** and **+0.0318 MRR**.
- **G3** closes most of that gap: 26.7% vs 30.0% Top-10, and 0.0641 vs 0.0710 MRR.
- **G4** proves the opposite trade-off: best intermediate extractor, but noticeably weaker realized ranking.
- The remaining model gap is now concentrated in topology execution on U3/U5 and in model-side extraction consistency, not in basic planner availability.

### 4.4.7 Planner Optimization Experiments

| Experiment | Quantitative result | Status |
|---|---|---|
| Current-system strategy search | `p0_union_p1` matches the best Top-10 / MRR (30.0 / 0.0710) while remaining the safest deployable strategy | **Adopted mainline** |
| Topology-faithful oracle | Full-topology beats P1-only by +13.3pp Top-10 and +0.0318 MRR | **Confirmed** |
| Late-round retrains | G4 improves Track A to 86.7% Hop-1, but G3 still wins strict downstream at 26.7% Top-10 | **Completed** |

The planner story is now stable: current query logic is good enough to expose topology value, and the remaining gains depend more on realized extraction quality than on further symbolic rewrites.

### 4.4.8 Model Selection

| Thesis claim | Recommended model | Rationale |
|---|---|---|
| Neural extraction quality | **G4** | Current Track A winner: Hop-1 86.7%, NEXT_TO Hop-1 65.0% |
| Strict AP downstream | **G3** | Best Top-10 (26.7%) and best MRR (0.0641) on Track B-2 |
| Primary carrier | G4, with G3 noted | Use G4 for extraction claims; explicitly note G3 as the best realized downstream model |

### 4.4.9 AP Held-Out Topology Structure (n=60)

The benchmark is a **flat multi-anchor** benchmark, not a deep multi-hop benchmark.

| Family | Signature | Count |
|---|---|---|
| singleton:CONNECTS_TO | CONNECTS_TO×1 | 14 |
| singleton:ADJACENT_TO | ADJACENT_TO×1 | 12 |
| paired:FILLS+NEXT_TO | FILLS+NEXT_TO×1 | 10 |
| triad:FILLS+NEXT_TO+NEXT_TO | FILLS+NEXT_TO×2 | 21 |
| triad mixed-anchor | FILLS+NEXT_TO×2 (diff types) | 2 |
| Other | — | 1 |

The dominant challenge is the 21 FILLS+NEXT_TO×2 triads: 46 identical windows per floor requiring multi-anchor AND to discriminate.

### 4.4.10 Summary Table — All Results

| Metric | LoRA5 baseline | G4 (Track A winner) | G3 (downstream best) | Full-topology oracle | Gemini AP v2 |
|---|---|---|---|---|---|
| Hop-1 (Track A) | — | **86.7%** | 80.0% | — | 30.0% |
| Predicate Recall (Track A) | — | 81.0% | **91.4%** | — | 39.7% |
| Direction Acc (Track A) | — | 57.1% | **78.6%** | — | 0.0% |
| GT-in-Pool (B-2) | 53.4%* | **100.0%** | **100.0%** | — | 91.7% |
| Top-10 (B-2) | — | 23.3% | **26.7%** | **30.0%** | 21.7% |
| Top-1 (B-2) | — | 0.0% | **1.7%** | **1.7%** | 0.0% |
| MRR (B-2) | — | 0.0324 | **0.0641** | **0.0710** | 0.0482 |

*LoRA5 baseline is from the prior unified benchmark and is not directly comparable to AP-only Track B-2; it is retained only as a historical anchor for GT-in-Pool.

### 4.4.11 Final Thesis Figure Package

The final plot package for Chapter 4 is stored under:
- `mscd_demo/docs/plots/phase4_lora6_main/`
- `mscd_demo/docs/plots/phase4_lora6_appendix/`

Caption text files are stored alongside the figures:
- `mscd_demo/docs/plots/phase4_lora6_main/phase4_main_captions.txt`
- `mscd_demo/docs/plots/phase4_lora6_appendix/phase4_appendix_captions.txt`

**Recommended main-text figures (primary narrative):**

| Figure | File | Role in thesis |
|---|---|---|
| Fig. 1 | `fig01_topology_overview.png` | Establishes AP held-out as a flat multi-anchor topology benchmark rather than a deep-hop benchmark |
| Fig. 2v2 | `fig02_v2_extraction_vs_downstream_tradeoff.png` | Shows the model-selection tension between Track A extraction quality and Track B-2 downstream ranking |
| Fig. 4v2 | `fig04_v2_oracle_dashboard.png` | Consolidates oracle strategy search, P1-only vs full-topology gain, and topology-family benefit |
| Fig. 6 | `fig06_oracle_vs_model_gap.png` | Quantifies the remaining gap between the oracle planner ceiling and realized model performance |

**Optional fifth main-text figure:**

| Figure | File | When to use |
|---|---|---|
| Fig. 7 | `fig07_oracle_progression_waterfall.png` | Use when a progression-style narrative is preferred to explain `P1 -> topology -> oracle -> best model` |

**Recommended appendix-only figures:**

| Figure | File | Role |
|---|---|---|
| Fig. 2 | `fig02_trackA_intermediate_comparison.png` | Full Track A matrix for model-by-metric inspection |
| Fig. 3 | `fig03_trackB2_strict_downstream.png` | Full Track B-2 grouped comparison |
| Fig. 4 | `fig04_oracle_strategy_sweep.png` | Standalone oracle strategy sweep |
| Fig. 5 | `fig05_p1_vs_full_topology_benefit.png` | Direct P1-only vs full-topology comparison |
| Fig. A1 | `figA1_topology_slice_benefit_by_universe.png` | Universe-wise topology benefit |
| Fig. A2 | `figA2_topology_slice_benefit_by_multiplicity.png` | Multiplicity-wise topology benefit |
| Fig. A3 | `figA3_trackB1_external_generalization.png` | Secondary unified benchmark comparison |
| Fig. A4 | `figA4_phase4_new_models_zoomin.png` | Zoom-in for late-round retrains G3/G4/G6 |

**Current interpretation of the figure package:**
- `G4` is the strongest intermediate extractor on Track A.
- `G3` remains the strongest strict AP downstream retriever on Track B-2.
- `p0_union_p1` is the safest executable query strategy under current planner logic.
- The largest topology-driven gains come from `U3` (opening-paired) and `U5` (mixed triad), which justifies the topology-aware planner focus.

---

## 4.5 Status & Next Steps

### Current State (2026-04-03)

```
Region enrichment MVP        ✅ DONE  — 2,277 regions (AP)
Skeleton mining + judging    ✅ DONE  — 297 KEEP (AP canonical, frozen)
Assembly pipeline            ✅ DONE  — 237 train + 753 aug + 60 eval
G0–G6 training/eval          ✅ DONE  — G4=Track A winner, G3=downstream best, G6=underfit control
Evaluation (Tracks A/B-1/B-2)✅ DONE  — All tracks complete, late retrains integrated
Planner optimization         ✅ DONE  — p0_union_p1 fixed as mainline; topology value quantified
Template collapse fix         ❌ BLOCKED — position context not flowing into labels
CONTINUOUS mining             ❌ NOT STARTED — 0 training cases, biggest gap
```

### Post-Mining Target

| Predicate | Current | After CONTINUOUS |
|---|---|---|
| CONNECTS_TO | 69 | 69 |
| FILLS | 69 | 69 |
| NEXT_TO | 97 | 97 |
| ADJACENT_TO | 62 | 62 |
| **CONTINUOUS (NOT IMPLEMENTED YET)** | **0** | **~25–30** |

### Defense Narrative

> LoRA5's template collapse is not a model capacity issue — the IFC graph contains 100% unique positional fingerprints, but the training pipeline compressed this to 37 output templates. The model faithfully learned a collapsed signal. By rebuilding supervision with region-grounded labels mined from the enriched graph, we established that a topology-faithful oracle reaches **30.0% Top-10 and 0.0710 MRR** under the deployed planner, while the best realized model (G3) reaches **26.7% Top-10 and 0.0641 MRR**. The remaining gap is now small but still extraction-limited, not planner-limited — validating a deeper principle: **the symbolic layer should not only serve runtime retrieval, but also guide the training signal for the neural layer.**

---

## Appendix: Artifact Locations

### Data
- Wall region index: `data_curation/references/wall_region_index_ap_20260331_c.jsonl`
- Skeletons: `data_curation/datasets/synth_v0.5_ap/skeletons/skeletons.jsonl`
- Skins (judged): `data_curation/datasets/synth_v0.5_ap/skins/skins_integrated.jsonl`
- Train canonical: `data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_train_canonical_m.jsonl` (237)
- Train augmented: `data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_train_aug.jsonl` (753)
- Eval canonical: `data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_eval_canonical_m.jsonl` (60)

### Models
- Adapters: `mscd_demo/models/lora6_v2_ap_20260331/{g0_canonical,g1_fullaug,g2_fullaug_lowlr,g3_fullaug_r32,g4_ultimate}/best`
- Late baseline checkpoint: `mscd_demo/models/lora6_v2_ap_20260331/g6_baseline/checkpoint-20`

### Evaluation Outputs
- Metrics: `mscd_demo/output/lora6_v2_ap_20260331/metrics/`
- Oracle: `mscd_demo/output/lora6_v2_ap_20260331/oracle_ap_heldout/`
- Phase 3 traces: `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase3_fixed/`
- Oracle Phase 3: `mscd_demo/output/lora6_v2_ap_20260331/oracle_phase3_fixed/`
- Topology analysis: `mscd_demo/output/lora6_v2_ap_20260331/topology_analysis/`
- Main plot package: `mscd_demo/docs/plots/phase4_lora6_main/`
- Appendix plot package: `mscd_demo/docs/plots/phase4_lora6_appendix/`
- Main captions: `mscd_demo/docs/plots/phase4_lora6_main/phase4_main_captions.txt`
- Appendix captions: `mscd_demo/docs/plots/phase4_lora6_appendix/phase4_appendix_captions.txt`

### Evaluation Scripts
- Track A scorer: `mscd_demo/evaluation/analysis/score_ap_track.py`
- Track B-1 scorer: `mscd_demo/evaluation/analysis/score_unified_track.py`
- Gemini AP runner: `mscd_demo/evaluation/inference/eval_gemini_ap.py`
- Gemini unified runner: `mscd_demo/evaluation/inference/eval_gemini_unified.py`
- Oracle AP: `mscd_demo/evaluation/oracle_ap_heldout.py`
- Topology analysis: `mscd_demo/evaluation/analysis/analyze_ap_heldout_topology.py`
- AP e2e case builder: `mscd_demo/evaluation/build_ap_heldout_e2e_cases.py`
