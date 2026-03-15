# MSCD Thesis Evaluation Protocol (Engineering-Grounded Revision)

> Revised 2026-03-15 after LoRA_4 training complete.
> Aligned to actual codebase, schemas, eval infrastructure, and data.
> Original idealized plan by Gemini; this revision by engineering review.

---

## Notation & Definitions

| Symbol | Meaning |
|--------|---------|
| GT-in-pool | Ground-truth element appears in candidate pool returned by retrieval |
| SSR | Search Space Reduction = (pre_pool - post_pool) / pre_pool |
| Top-1 | Correct element is rank-1 after reranking |
| P0 | Priority-0 query (spatial_triplet / continuous_span via Neo4j) |
| Over-reduction | P0 fires but GT is NOT in the returned pool (false negative) |

---

## Group 2: 4-Tier System Ablation

**Goal**: Prove — with controlled experiments on the SAME test set — that
attribute-only retrieval hits a mathematical ceiling, and that the neuro-symbolic
2-hop architecture is the only way to break through.

### Tier 1 (Red): Zero-shot VLM Constraint Extraction

* **What it is**: Run the same eval pipeline (constraint extraction -> Neo4j
  retrieval) but using the **base Qwen2.5-VL-7B-Instruct without any LoRA
  adapter**. The model sees the same inputs (site photo + floorplan + chat)
  but has never been finetuned on AEC constraint extraction.
* **Execution**:
  ```bash
  # Use eval_lora4.py but skip adapter loading (base model only)
  modal run training/eval_lora4.py \
    --adapter-dir NONE \
    --cases /data/v05_test.jsonl \
    --condition-override MC
  ```
  Implementation: add a `NONE` sentinel in `eval_lora4.py` that skips
  `PeftModel.from_pretrained()` and runs base model inference directly.
* **Actual output schema**: Same prompt, but base model will likely produce
  malformed JSON or hallucinated field values since it was never trained on
  our `{storey_name, ifc_class, spatial_relations: [...]}` schema.
* **Expected result**: Parse rate < 30%. Of parsed outputs, `spatial_relations`
  will be empty or nonsensical. Pipeline falls back to Priority 4-8 (storey+type),
  yielding GT-in-pool ~ 50% but avg pool ~ 130 (no better than random within pool).
* **Script**: `training/eval_lora4.py` (needs `NONE` adapter sentinel — ~10 lines)
* **Effort**: 30 min
* **Status**: NOT YET BUILT

### Tier 2 (Orange): Vector RAG Baseline (Embedding Similarity)

* **What it is**: Standard dense retrieval. Embed all 1233 IFC elements from
  `element_index.jsonl` as text chunks, embed the user query, retrieve Top-K
  by cosine similarity. No graph, no topology.
* **Execution**:
  ```bash
  # 1. Build FAISS index from element_index.jsonl
  python eval/baselines/build_vector_index.py \
    --elements ../data_curation/references/element_index.jsonl \
    --output eval/baselines/faiss_index/

  # 2. Run retrieval on test cases
  python eval/baselines/vector_rag_eval.py \
    --index eval/baselines/faiss_index/ \
    --cases eval/cases_v4_test.jsonl \
    --output logs/evaluations/vector_rag_results.jsonl
  ```
* **Element text template** (what gets embedded):
  ```
  {ifc_class} on {storey_name}. Material: {material}. Name: {name}.
  Dimensions: width={width}, height={height}.
  ```
* **Embedding model**: `text-embedding-3-small` (OpenAI) or `Alibaba-NLP/gte-Qwen2-1.5B-instruct` (local, free).
* **Expected result**: "Semantic collapse" — 46 identical IfcWindows on the same
  floor produce nearly identical embeddings. Cosine similarity cannot distinguish
  them. Top-1 ~ 2-5%, GT-in-pool(Top-10) ~ 20-30%.
* **Why include**: Demonstrates that dense retrieval fundamentally cannot solve
  the attribute entropy problem. The embeddings CANNOT encode topology because
  the text chunks contain no relational information.
* **Script**: `eval/baselines/build_vector_index.py` + `eval/baselines/vector_rag_eval.py`
* **Effort**: 2-3 hours
* **Status**: PROBABLY DO (not yet built, but adds strong thesis argument)

### Tier 3 (Yellow): LoRA_3 — Finetuned but Attribute-Dominated

* **What it is**: Our previous LoRA_3 model. It HAS `spatial_relations` in its
  schema, but in practice only extracted SR for **1 out of 69 test cases (1.4%)**
  because 56% of training data had empty SR -> model learned "default to []".
  Effectively attribute-only in deployment.
* **Execution**: Already have results from LoRA_3 eval (03-14).
  ```
  # Existing precomputed results:
  logs/evaluations/synth_v04/eval_constraints_final_MB.jsonl
  logs/evaluations/synth_v04/eval_constraints_final_MC.jsonl
  eval/results/lora3_MB/
  eval/results/lora3_MC/
  ```
* **Actual output schema** (LoRA_3):
  ```json
  {
    "storey_name": "1 - First Floor",
    "ifc_class": "IfcWindow",
    "space_name": null,
    "target_name_keyword": null,
    "spatial_relations": []
  }
  ```
  Note: `spatial_relations: []` in 68/69 cases. Model CAN output SR but almost
  never does because training distribution was 56% empty.
* **Results from 3-way precomputed eval (AP-only, n=50)**:
  ```
  GT-in-pool: 30/49 (61%)    P0 fires: 0/49 (0%)
  Avg pool:   77.4            SSR: 93.8%
  ```
  Drops from storey+type fallback when ifc_class is wrong. P0 never fires because
  SR is always empty.
* **Script**: Results already exist. Re-run with `eval.sh` if needed.
* **Effort**: 0 (done)
* **Status**: DONE

### Tier 4 (Green): LoRA_4 — Neuro-Symbolic 2-Hop (Proposed System)

* **What it is**: The full proposed system. LoRA_4 trained on 75% SR ratio data
  with 4 predicates (FILLS, ADJACENT_TO, CONTINUOUS, CONNECTS_TO), 94 multi-triplet
  2-hop records, modality dropout, and 30% metadata masking.
* **Execution**:
  ```bash
  cd mscd_demo
  ./training/eval_lora4.sh --step modal   # Modal GPU extraction (MA/MB/MC)
  ./training/eval_lora4.sh --step local   # Local Neo4j retrieval + scoring
  ```
* **Actual output schema** (LoRA_4):
  ```json
  {
    "storey_name": "1",
    "ifc_class": "IfcWindow",
    "space_name": null,
    "target_name_keyword": null,
    "spatial_relations": [
      {"predicate": "FILLS", "object_type": "IfcWallStandardCase",
       "object_material": "Concrete", "confidence": 0.8},
      {"predicate": "CONNECTS_TO", "object_type": "IfcWallStandardCase",
       "object_material": null, "confidence": 0.8}
    ]
  }
  ```
  Key differences from Gemini's idealized SGG schema:
  - No `node_id`, no `role`, no `bbox` fields (not trained on these)
  - `spatial_relations` is a flat list of `{predicate, object_type, object_material, confidence}`
  - Storey normalized to floor number only ("1" not "1 - First Floor")
  - `object_type` is the IFC class of the anchor, not a specific entity reference
* **Cypher compilation** (what the Python agent builds from the JSON above):
  ```cypher
  -- 2-hop OPTIONAL MATCH (hop-1 hard filter, hop-2 soft rerank)
  MATCH (target:Element)-[:FILLS]->(ref:Element)
  WHERE target.ifc_type STARTS WITH 'IfcWindow'
    AND ref.ifc_type STARTS WITH 'IfcWallStandardCase'
    AND target.storey =~ '(?i).*1.*'
  OPTIONAL MATCH (ref)-[:CONNECTS_TO]->(ref2:Element)
  WHERE ref2.ifc_type STARTS WITH 'IfcWallStandardCase'
  RETURN DISTINCT target, (ref2 IS NOT NULL) AS has_hop2
  ORDER BY has_hop2 DESC
  ```
* **Expected result** (target metrics from LoRA_4 plan):
  ```
  SR extraction (MC): >40/75 (53%+)      (was 1/69 in LoRA_3)
  P0 fire rate (MC):  >50%               (was 0% in LoRA_3)
  GT-in-pool (MC):    >55%               (was 61% attribute-only)
  Over-reduction:     <5%                 (OPTIONAL MATCH safety net)
  SSR:                >93%               (maintain LoRA_3 level)
  ```
* **Script**: `training/eval_lora4.py` + `training/eval_lora4.sh`
* **Effort**: Run time only (~45 min Modal + 10 min local)
* **Status**: READY TO RUN

---

## Group 3: Defense Narrative — 6 Experiments

### Experiment 1: Attribute Entropy Quantification ("The Problem")

* **Goal**: Quantify the homogeneity crisis in industrialized construction IFC data.
* **Execution**: Cypher queries on the loaded Neo4j graph (AdvancedProject.ifc,
  1233 elements). Count elements per (storey, ifc_type) bucket. Calculate
  attribute-only Top-1 probability = 1/bucket_size.
  ```cypher
  MATCH (e:Element)
  WITH e.storey AS storey, e.ifc_type AS ifc_type, count(e) AS n
  WHERE n > 1
  RETURN storey, ifc_type, n, round(1.0/n * 100, 1) AS top1_pct
  ORDER BY n DESC
  ```
* **Data source**: Neo4j graph + `element_index.jsonl` (1233 elements).
  Also: H2 eval `pre_pool_size` field already contains per-case attribute pool sizes.
* **Expected output**: Table + bar chart showing:
  - 46 identical IfcWindows on storey "Level 1" (Top-1 = 2.2%)
  - 34 identical IfcDoors on storey "1 - First Floor" (Top-1 = 2.9%)
  - Weighted average attribute-only Top-1 across all H2 cases: ~3%
* **Thesis figure**: "Attribute Entropy by Element Type and Storey" bar chart.
* **Script**: `eval/experiments/exp1_attribute_entropy.py` (Cypher + matplotlib)
* **Effort**: 30 min
* **Status**: NOT YET BUILT (but H2 eval already has partial data)

### Experiment 2: Oracle Upper Bound ("The Ceiling")

* **Goal**: Establish the theoretical maximum achievable by the retrieval layer
  when given perfect constraints. Answers: "How good CAN this system be?"
* **Execution**: Use ground-truth constraints from skeleton metadata (not VLM
  extraction). Feed perfect `{ifc_class, storey_name, spatial_relations}` into
  the retrieval pipeline.
  ```bash
  python script/run.py --profile v2_lora \
    --cases eval/cases_v4_test.jsonl \
    --precomputed eval/precomputed_oracle.jsonl
  ```
* **Existing results** (AP-only, n=50, from 03-14):
  ```
  GT-in-pool:    49/49 (100%)    P0 fires: 48/48 (100%)
  Over-reduction: 0%             SSR: 94.7%
  Avg pool:      66.0
  ```
* **Also**: H2 hard-negative eval (83 oracle cases): 83/83 GT-in-pool (100%).
* **Thesis figure**: Oracle line on all comparison charts (dashed horizontal).
* **Script**: `eval/build_precomputed.py` (oracle) + `eval_h2_spatial_triplets/h2_eval.py`
* **Effort**: 0 (done)
* **Status**: DONE

### Experiment 3: Uniqueness by Hop — the L-Curve ("Why 2-Hop?")

* **Goal**: Answer the committee question: "Why exactly 2 hops? Why not 1 or 3?"
  Show diminishing returns via L-shaped candidate pool curve.
* **Execution**: For each H2 FILLS case (28 cases where hop chain is
  Window -[FILLS]-> Wall -[CONNECTS_TO]-> Wall2), run 4 progressive queries:
  ```
  0-hop: MATCH (t:Element) WHERE t.ifc_type = $type AND t.storey =~ $storey
         -> returns N candidates (attribute pool)

  1-hop: MATCH (t)-[:FILLS]->(ref) WHERE ref.ifc_type = $ref_type
         -> returns M candidates (topology-filtered)

  2-hop: ... OPTIONAL MATCH (ref)-[:CONNECTS_TO]->(ref2)
         RETURN DISTINCT t ORDER BY has_hop2 DESC
         -> returns K candidates with GT promoted to top

  3-hop: ... OPTIONAL MATCH (ref2)-[:CONNECTS_TO]->(ref3)
         -> returns K' candidates (expect K' ~ K, diminishing returns)
  ```
* **Expected output**: L-shaped curve showing:
  ```
  Hop 0: avg 43 candidates (attribute pool)
  Hop 1: avg  4 candidates (FILLS narrows to host wall's elements)
  Hop 2: avg  1-2 candidates (CONNECTS_TO disambiguates wall identity)
  Hop 3: avg  1-2 candidates (no further gain, but higher hallucination risk)
  ```
* **Thesis figure**: "Candidate Pool Size vs Hop Depth" line chart with
  L-shaped curve. X-axis: hop depth (0,1,2,3). Y-axis: mean pool size.
* **Script**: `eval/experiments/exp3_hop_uniqueness.py` (extend h2_eval.py logic)
* **Effort**: 1 hour
* **Status**: NOT YET BUILT

### Experiment 4: Modality Ablation MA/MB/MC ("The Floorplan Matters")

* **Goal**: Prove that floorplan input is critical for spatial relation extraction.
  Without floorplan (MA/MB), the VLM cannot extract topology from occluded site
  photos alone.
* **Execution**:
  ```bash
  # Already wired into eval_lora4.sh:
  ./training/eval_lora4.sh --step modal   # Runs MA, MB, MC on Modal GPU
  ./training/eval_lora4.sh --step local   # Local retrieval for each condition
  ```
  Conditions (from `profiles.yaml`):
  - **MA**: Text-only (chat + 4D metadata, no images at all)
  - **MB**: Text + site photo (no floorplan)
  - **MC**: Text + site photo + floorplan (full system)
* **Metrics to compare across conditions**:
  - SR extraction rate (% of cases where `spatial_relations != []`)
  - P0 fire rate (% of cases where Priority-0 Cypher executes)
  - GT-in-pool rate
  - Over-reduction rate
* **Expected result**:
  ```
  Condition  SR Rate   P0 Fires  GT-in-pool
  MA         ~5%       ~0%       ~45%  (storey+type fallback)
  MB         ~20%      ~15%      ~50%  (some SR from site photo, many wrong)
  MC         ~60%+     ~50%+     ~65%+ (floorplan enables reliable SR)
  ```
* **Thesis figure**: Grouped bar chart, 3 clusters (MA/MB/MC), 4 bars each
  (SR rate, P0 fires, GT-in-pool, SSR).
* **Script**: `training/eval_lora4.sh` (modal + local steps) + `script/compare_results.py`
* **Effort**: Run time only (~45 min Modal + 10 min local + 5 min plots)
* **Status**: READY TO RUN

### Experiment 5: Fallback Stress Test ("The Safety Net")

* **Goal**: Prove that OPTIONAL MATCH provides industrial robustness. When the
  VLM hallucinates hop-2, the system degrades gracefully (returns larger pool
  with GT still present) instead of catastrophically (empty result, GT lost).
* **Execution**: Take oracle constraints from H2 FILLS cases. For each case:
  1. **Correct run**: Original 2-hop constraints -> measure pool & GT-in-pool
  2. **Corrupted run**: Flip hop-2 `object_type` to a wrong class (e.g.,
     `IfcWallStandardCase` -> `IfcSlab`), simulating VLM hallucination
  3. **Hard MATCH run**: Same corrupted constraints but replace OPTIONAL MATCH
     with hard MATCH -> expect empty result (GT lost)
  ```python
  # Pseudocode for each H2 FILLS case:
  correct_pool = run_2hop_optional(correct_constraints)   # GT in pool
  corrupt_pool = run_2hop_optional(corrupted_constraints)  # GT still in pool (hop-1 saves it)
  hard_pool    = run_2hop_hard_match(corrupted_constraints) # GT LOST (0 results)
  ```
* **Expected result**:
  ```
  Condition         GT-in-pool  Avg Pool  Over-reduction
  Correct (oracle)  100%        4.2       0%
  Corrupted+OPTIONAL 100%      12.8       0%  (graceful degradation)
  Corrupted+HARD      0%        0.0     100%  (catastrophic failure)
  ```
* **Thesis figure**: 3-column comparison table + bar chart showing GT-in-pool
  rate under correct/corrupted conditions.
* **Script**: `eval/experiments/exp5_fallback_stress.py`
* **Effort**: 1 hour
* **Status**: BUILT

### Experiment 6: Full System Comparison ("The Final Scoreboard")

* **Goal**: The thesis climax. 4-way comparison on the same test set showing
  the progression from baseline to proposed system.
* **Execution**: Aggregate results from Tiers 1-4 on the 75 v0.5 topology
  test cases (or the 83 H2 hard-negative cases for a tougher benchmark).
* **Systems compared**:
  | System | Adapter | SR Extraction | Retrieval |
  |--------|---------|---------------|-----------|
  | Zero-shot VLM | None (base Qwen2.5-VL) | Garbage | Storey+type fallback |
  | Vector RAG | N/A | N/A | FAISS cosine Top-K |
  | LoRA_3 (attr-only) | v3_lora_qwen | 1.4% (1/69) | Storey+type fallback |
  | LoRA_4 (proposed) | v4_lora_qwen | 53%+ target | 2-hop OPTIONAL MATCH |
  | Oracle (ceiling) | N/A (GT constraints) | 100% | 2-hop OPTIONAL MATCH |

* **Expected result**:
  ```
  System           GT-in-pool  Avg Pool  Top-1 Est  SSR
  Zero-shot VLM    ~40%        ~130      ~1%        ~0%
  Vector RAG       ~25%        Top-10    ~3%        N/A
  LoRA_3           ~61%         77       ~5%        93.8%
  LoRA_4           ~70%+        ~40      ~20%+      95%+
  Oracle           100%         66       ~50%+      94.7%
  ```
* **Thesis figure**: Multi-metric grouped bar chart (5 systems x 4 metrics).
  The visual story: LoRA_4 bar approaches Oracle bar, all baselines far below.
* **Script**: `script/compare_results.py --thesis` (auto-discovers result dirs)
* **Effort**: Aggregation only (all individual runs done above)
* **Status**: READY once Tiers 1-4 are run

---

## Implementation Status & Priority

| # | Experiment | Script | Status | Effort | Priority |
|---|-----------|--------|--------|--------|----------|
| - | **LoRA_4 quick smoke test** | `eval_lora4.sh --step quick` | READY | 5 min | **P0 (do first)** |
| 4 | Modality ablation (MA/MB/MC) | `eval_lora4.sh --step modal+local` | READY | 45 min | **P0** |
| 2 | Oracle upper bound | `precomputed_oracle.jsonl` + `h2_eval.py` | DONE | 0 | **P0** |
| 6 | Full 5-way comparison | `compare_results.py --thesis` | READY (needs runs) | 5 min | **P0** |
| 1 | Attribute entropy | `exp1_attribute_entropy.py` | BUILT | run only | **P1** |
| 3 | Hop uniqueness L-curve | `exp3_hop_uniqueness.py` | BUILT | run only | **P1** |
| 5 | Fallback stress test | `exp5_fallback_stress.py` | BUILT | run only | **P1** |
| T1 | Zero-shot VLM baseline | `eval_lora4.py --adapter-dir NONE` | BUILT | run only | **P1** |
| T2 | Vector RAG baseline | `build_vector_index.py` + `vector_rag_eval.py` | BUILT | run only | **P2 (probably do)** |

### Execution Order

All commands run from `mscd_demo/` project root.

```bash
cd mscd_demo

# ═══════════════════════════════════════════════════════════════════════
# Phase 0 — Smoke test (5 min)
# Verifies LoRA_4 adapter loads on Modal and extracts SR on 3 cases.
# Do this FIRST to catch adapter issues before spending GPU time.
# ═══════════════════════════════════════════════════════════════════════

./training/eval_lora4.sh --step quick

# ═══════════════════════════════════════════════════════════════════════
# Phase 1 — Core LoRA_4 evaluation (1 hr)
# Modal GPU extraction (MA/MB/MC) -> local Neo4j retrieval -> comparison
# ═══════════════════════════════════════════════════════════════════════

# Step 1: Extract constraints on Modal A100 (3 runs: MA, MB, MC)
# Output: logs/evaluations/synth_v05_lora4/eval_constraints_final_{MA,MB,MC}.jsonl
./training/eval_lora4.sh --step modal

# Step 2: Feed precomputed constraints through Neo4j retrieval + scoring
# Requires Neo4j running: /tmp/neo4j-community-5.26.0/bin/neo4j start
# Output: logs/evaluations/synth_v05_lora4/traces_*_v2_lora_{MA,MB,MC}.jsonl
./training/eval_lora4.sh --step local

# Step 3: H2 hard-negative eval (83 oracle topology cases)
# Tests retrieval layer with perfect constraints (independent of VLM)
# Output: logs/evaluations/synth_v05_lora4/h2_results_*.jsonl + plot
./training/eval_lora4.sh --step h2

# Step 4: Generate LoRA_4 vs LoRA_3 comparison charts
# Output: logs/comparisons/lora4_vs_lora3/*.png
./training/eval_lora4.sh --step compare

# ═══════════════════════════════════════════════════════════════════════
# Phase 2 — Thesis experiments (requires Neo4j running)
# These run locally, no Modal GPU needed.
# ═══════════════════════════════════════════════════════════════════════

# Exp 1: Attribute entropy quantification
# Counts duplicate (storey, type) buckets. Shows 46 identical IfcWindows.
conda run -n mscd_demo python eval/experiments/exp1_attribute_entropy.py \
    --plot docs/plots/exp1_entropy.png \
    --h2 ../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl

# Exp 3: Hop uniqueness L-curve ("Why 2-hop?")
# Runs 0/1/2/3-hop progressive Cypher on 28 FILLS cases.
# Shows L-shaped pool size: 43 -> 4 -> 1-2 -> 1-2
conda run -n mscd_demo python eval/experiments/exp3_hop_uniqueness.py \
    --plot docs/plots/exp3_lcurve.png

# Exp 5: Fallback stress test (OPTIONAL MATCH robustness)
# Injects wrong object_type into hop-2, compares OPTIONAL vs HARD MATCH.
conda run -n mscd_demo python eval/experiments/exp5_fallback_stress.py \
    --plot docs/plots/exp5_fallback.png

# ═══════════════════════════════════════════════════════════════════════
# Phase 3 — Baselines for 5-way system comparison
# ═══════════════════════════════════════════════════════════════════════

# --- Tier 1: Zero-shot VLM baseline (Modal GPU, no LoRA adapter) ---
# Runs base Qwen2.5-VL-7B without finetuning. Expects garbage JSON output.
# --adapter-dir NONE triggers zero-shot mode (skips PeftModel loading).
modal run training/eval_lora4.py \
    --adapter-dir NONE \
    --cases /data/v05_test.jsonl \
    --condition-override MC

# Download zero-shot results
modal volume get mscd-checkpoints \
    /mscd-lora-v4/eval_constraints_zeroshot_MC.jsonl \
    logs/evaluations/synth_v05_lora4/

# Run local pipeline on zero-shot constraints
conda run -n mscd_demo python -u script/run.py \
    --profile v2_lora \
    --cases eval/cases_v4_test.jsonl \
    --precomputed logs/evaluations/synth_v05_lora4/eval_constraints_zeroshot_MC.jsonl \
    --output_dir logs/evaluations/synth_v05_lora4 \
    --condition-override MC

# --- Tier 2: Vector RAG baseline (local, no GPU needed) ---
# Step 1: Build FAISS index from element_index.jsonl
# Uses local sentence-transformers (free). For OpenAI: --model openai
pip install sentence-transformers faiss-cpu  # one-time install

conda run -n mscd_demo python eval/baselines/build_vector_index.py \
    --elements ../data_curation/references/element_index.jsonl \
    --output eval/baselines/faiss_index/ \
    --model local

# Step 2: Run cosine similarity retrieval on test cases
conda run -n mscd_demo python eval/baselines/vector_rag_eval.py \
    --index eval/baselines/faiss_index/ \
    --cases eval/cases_v4_test.jsonl \
    --top-k 10 \
    --output logs/evaluations/vector_rag_results.jsonl \
    --plot docs/plots/vector_rag_baseline.png

# Optional: also run on H2 hard-negative cases for tougher benchmark
conda run -n mscd_demo python eval/baselines/vector_rag_eval.py \
    --index eval/baselines/faiss_index/ \
    --h2 ../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl \
    --top-k 10 \
    --output logs/evaluations/vector_rag_h2_results.jsonl

# ═══════════════════════════════════════════════════════════════════════
# Phase 4 — Final thesis comparison (5-system scoreboard)
# Aggregates all results into one multi-metric chart.
# ═══════════════════════════════════════════════════════════════════════

conda run -n mscd_demo python script/compare_results.py --thesis

# ═══════════════════════════════════════════════════════════════════════
# Optional: Analyze individual trace files in detail
# ═══════════════════════════════════════════════════════════════════════

# Per-strategy/per-predicate breakdown of a single trace
conda run -n mscd_demo python eval/analyze_traces.py \
    logs/evaluations/synth_v05_lora4/traces_*_v2_lora_MC.jsonl

# View latest traces
ls -lt logs/evaluations/synth_v05_lora4/traces_*.jsonl | head -10

# View generated plots
ls docs/plots/exp*.png docs/plots/vector_rag*.png
```

---

## Key Schema Correction (vs. Gemini's Original Plan)

The original plan proposed an SGG-style schema with `entities[]`, `node_id`,
`role`, `onsite_image_bbox`, and `floorplan_bbox`. **This is NOT what LoRA_4
was trained on.** Our actual schema is constraint-based, not scene-graph-based:

```
Gemini's idealized schema (NOT IMPLEMENTED):
{
  "entities": [
    {"node_id": "E1", "ifc_class": "IfcWindow", "role": "target",
     "onsite_image_bbox": [210,340,450,600], "floorplan_bbox": [50,100,80,120]},
    {"node_id": "E2", "ifc_class": "IfcWall", "role": "anchor_1"}
  ],
  "spatial_triplets": [
    {"subject_id": "E1", "predicate": "FILLS", "object_id": "E2"}
  ]
}

Our actual LoRA_4 schema (WHAT THE MODEL OUTPUTS):
{
  "storey_name": "1",
  "ifc_class": "IfcWindow",
  "space_name": null,
  "target_name_keyword": null,
  "spatial_relations": [
    {"predicate": "FILLS", "object_type": "IfcWallStandardCase",
     "object_material": "Concrete", "confidence": 0.8},
    {"predicate": "CONNECTS_TO", "object_type": "IfcWallStandardCase",
     "object_material": null, "confidence": 0.8}
  ]
}
```

**Why the difference matters for the thesis**: Our schema is "constraint-based
extraction" (extract type + predicate + anchor type), not "scene graph generation"
(extract entities with bounding boxes + relations). The thesis should describe it
accurately as constraint extraction that gets compiled into graph queries, NOT as
visual grounding with bounding boxes. The Cypher compiler in `constraints_to_query.py`
bridges the gap between extracted constraints and graph traversal.

---

## File References

### Evaluation Infrastructure

| File | Role |
|------|------|
| `training/eval_lora4.py` | Modal GPU inference (max_new_tokens=512, SR/hop-2 logging, `--adapter-dir NONE` for zero-shot) |
| `training/eval_lora4.sh` | Orchestrator (steps: quick/modal/local/h2/compare/full) |
| `training/eval.py` | LoRA_3 eval (max_new_tokens=256, adapter=/mscd-lora/final) |
| `eval_h2_spatial_triplets/h2_eval.py` | H2 hard-negative eval (83 oracle cases) |
| `eval/analyze_traces.py` | Per-strategy/per-predicate breakdown of trace files |
| `script/run.py` | Local eval runner (--precomputed, --condition-override) |
| `script/compare_results.py` | N-way comparison charts (--thesis mode) |

### Experiment Scripts

| File | Experiment | What it measures |
|------|-----------|-----------------|
| `eval/experiments/exp1_attribute_entropy.py` | Exp 1 | Cypher counts duplicate (storey, type) buckets, outputs entropy bar chart |
| `eval/experiments/exp3_hop_uniqueness.py` | Exp 3 | 0/1/2/3-hop progressive Cypher on 28 FILLS cases, outputs L-curve |
| `eval/experiments/exp5_fallback_stress.py` | Exp 5 | Error injection + OPTIONAL vs HARD MATCH comparison |

### Baseline Scripts

| File | Tier | What it does |
|------|------|-------------|
| `eval/baselines/build_vector_index.py` | Tier 2 | Embeds 1233 elements into FAISS (OpenAI or local sentence-transformers) |
| `eval/baselines/vector_rag_eval.py` | Tier 2 | Cosine similarity Top-K retrieval, GT-in-pool and Top-1 measurement |

### Data Files

| File | Contents |
|------|---------|
| `eval/precomputed_oracle.jsonl` | Oracle constraints (ground-truth, 100% accurate) |
| `eval/cases_v4_test.jsonl` | 75 v0.5 topology test cases (auto-converted from lora4_test.jsonl) |
| `../data_curation/datasets/synth_v0.5/eval/h2_hard_negatives.jsonl` | 83 H2 hard-negative cases |
| `../data_curation/references/element_index.jsonl` | 1233 IFC elements (for Vector RAG embedding) |
| `models/adapters/v4_lora_qwen_20260315_013030/final/` | Downloaded LoRA_4 adapter |

### Output Directories

| Directory | Contents |
|-----------|---------|
| `logs/evaluations/synth_v05_lora4/` | LoRA_4 precomputed constraints + trace files |
| `logs/comparisons/lora4_vs_lora3/` | Comparison charts (LoRA_4 vs LoRA_3) |
| `eval/baselines/faiss_index/` | FAISS index + metadata for Vector RAG |
| `docs/plots/` | Thesis figures (exp1, exp3, exp5, vector_rag) |
