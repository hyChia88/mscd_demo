# Evaluation Results

---

## Exp 1: V1 Agent Pipeline on synth_v0.2

**Date:** 2026-02-07 | **Dataset:** synth_v0.2 (43 cases) | **LLM:** Gemini 2.5 Flash

| Mode | Top-1 | Top-3 | F1 | GUID Matches |
|------|-------|-------|-----|--------------|
| memory | **0.326** | 0.372 | **0.347** | 16 |
| neo4j | 0.302 | 0.326 | 0.314 | 14 |
| memory+clip | 0.256 | 0.326 | 0.287 | 14 |
| neo4j+clip | 0.256 | 0.279 | 0.267 | 12 |

Memory mode is best. Adding Neo4j/CLIP hurts — Neo4j fallback adds noise, CLIP reranking finds the right type but wrong instance.
However I suppose sample too small to judge.

---

## Exp 2: V2 Prompt Condition Ablations on synth_v0.2

**Date:** 2026-02-07 | **Dataset:** synth_v0.2 (43 cases) | **Profile:** `v2_prompt`

| Condition | Cases | Top-1 | Top-K | SSR | Field F1 |
|-----------|-------|-------|-------|-----|----------|
| **A1** (clear + 4D) | 6 | **0.500** | **0.500** | 0.984 | **0.722** |
| A2 (blurred + 4D) | 6 | 0.167 | 0.167 | 0.867 | 0.500 |
| A3 (blurred + 4D enhanced) | 2 | 0.000 | 0.000 | 0.970 | 0.333 |
| B1 (blurred + img) | 7 | 0.000 | 0.143 | 0.964 | 0.286 |
| B2 (blurred + img + CLIP) | 5 | 0.000 | 0.000 | 0.947 | 0.133 |
| B3 (clear + img) | 3 | 0.000 | 0.000 | 0.922 | 0.389 |
| C1 (clear + floorplan) | 7 | 0.143 | **0.286** | 0.974 | 0.476 |
| C2 (blurred + img + floorplan) | 5 | 0.000 | 0.000 | 0.940 | N/A |
| C3 (clear + all) | 2 | 0.000 | 0.000 | 0.810 | 0.667 |

### Key Findings

1. **Clear text + 4D is the strongest signal** — A1 = 50% Top-1, beating best V1 (32.6%).
2. **Chat blurring degrades extraction** — A1→A2: Top-1 drops 0.500→0.167, F1 drops 0.722→0.500.
3. **Images alone don't help V2** — All B conditions = 0% Top-1 (prompt extractor is text-only).
4. **Pool=100 strongly correlates with miss** — fallback strategy is too broad.
5. **Parse rate is 100%** but field accuracy varies — bottleneck is extraction quality, not parsing.
6. **V2 A1 > V1 memory** (+17.4pp Top-1), but V2 with degraded inputs underperforms V1.

---

## Exp 3: V2 Prompt Baseline on synth_v0.3

**Date:** 2026-02-14 | **Dataset:** synth_v0.3 (84 cases) | **Profile:** `v2_prompt`
**Traces:** `logs/evaluations/traces_20260214_210555_v2_prompt.jsonl`

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 0.0357 (3/84) |
| Top-K Accuracy | 0.0595 (5/84) |
| Search Space Reduction | 0.9319 |
| Field EM F1 | 0.2135 |
| Parse Rate | 1.0000 |

### Success/Failure Analysis

**Top-1 hits: 3/84 (3.6%)** — all Top-1 hits are also Top-K.

| Factor | Success pattern | Insight |
|--------|----------------|---------|
| IFC Class | Wall(5), Door(1), Railing(1), Slab(1) | Walls dominate when chat says "cracks", "surface", "partition wall" |
| Pool size | Mean 17.9 (vs 65 overall) | Fewer candidates = easier to rank #1 |
| Chat keywords | "fire doors", "handrail", "hairline cracks" | Explicit element keywords make extraction trivial |
| Storey | Level 1(3), Garage(3) | Common storey names the LLM recognizes |

**What fails (76+ cases):** Vague/deictic text ("Right here.", "Check this."), high candidate density (k=181), uncommon storey names.

### Takeaway for LoRA

The prompt-only extractor only works when chat literally names the element type. LoRA should teach the model to infer `ifc_class` from visual cues (cracked slab, damaged window) instead of text keywords — exactly what Style B (vague/deictic) augmentation trains for.

---

## Limitations

1. **Small per-condition samples in v0.2** — A3/C3 have only 2 cases each.
2. **No vision model in V2 prompt** — B-group conditions are effectively text-only.
3. **Neo4j was not running** for V1 neo4j modes (fell back to memory).
4. **v0.3 Top-1 drop** (3.6% vs v0.2's 11.6% overall) is expected — v0.3 cases are intentionally harder with vague/deictic text.

4. The current A/B/C grouping assigns different cases to each modality condition rather than testing the same case under multiple evidence configurations. As a result, observed accuracy differences between modality groups may be confounded by case-level difficulty variation (e.g., candidate density k). A controlled modality-ablation study — where the same skeleton is tested with text-only, image+text, and full multimodal inputs — is needed to isolate the true effect of visual evidence.
---

## Appendix: Run Commands

```bash
# V1 on synth_v0.2
./run_mcp.sh --all -d synth

# V2 condition ablations on synth_v0.3
CASES=../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
for cond in A1 A2 A3 B1 B2 B3 C1 C2 C3; do
  python script/run.py --profile v2_prompt --cases $CASES --condition $cond
done

# V2 prompt baseline (all conditions)
python script/run.py --profile v2_prompt --cases $CASES
```
---
## Final Evaluation Framework

### Overview

The evaluation is organised into two complementary experiment groups addressing
the two research questions (RQs). Both groups share the same IFC model
(AdvancedProject.ifc, 1 233 elements) and Neo4j graph state (edges loaded via
`neo4j_init.sh`).

---

### Experiment Group 1 — Agentic vs. Structured Pipeline (RQ1)

**Research Question:** Does a structured neuro-symbolic pipeline outperform a
free-form LLM agent in element retrieval accuracy and operational efficiency?

#### Systems Under Test

| System | Architecture | LLM | Retrieval Backend |
|---|---|---|---|
| **V1 Agent** | Free-form tool-calling agent (ReAct) | Gemini 2.5 Flash | Memory / Neo4j |
| **V2 Structured** | Constrained extraction → priority-cascade query planner | Gemini 2.5 Flash (prompt) | Neo4j |

#### Independent Variables

- Pipeline architecture: V1 (agent) vs V2 (structured)
- Retrieval backend: Memory vs Neo4j
- CLIP reranking: enabled / disabled

#### Metrics

| Metric | Definition | What It Proves |
|---|---|---|
| **Top-1 Accuracy** | `GT_GUID == candidates[0].guid` | End-to-end retrieval precision |
| **Top-K Accuracy** (K=3) | `GT_GUID ∈ candidates[:K]` | Retrieval recall within short list |
| **SSR** | `1 − (final_pool / initial_pool)` | Search space pruning efficiency |
| **Over-Reduction Rate** | Fraction of cases where GT pruned from pool | Safety of pruning |
| **Parse Rate** | Fraction of valid JSON constraint outputs | Pipeline reliability |
| **Avg Latency (ms)** | End-to-end wall time per case | Operational efficiency |
| **Avg Tool Calls** | Mean tool invocations per case (V1 only) | Agent verbosity / cost |
| **Escalation Rate** | Fraction requiring human fallback | Autonomy |

#### Test Data

- `synth_v0.4_ap` cases (250, condition-stratified: A1–C3)
- Modality conditions: A (text+4D), B (text+image), C (text+floorplan)
- Difficulty tiers: X1 (clear text), X2 (blurred/vague), X3 (degraded+enhanced)

#### Profiles

```
v1_baseline       — V1 agent, memory retrieval, no CLIP
v1_full           — V1 agent, Neo4j, CLIP reranking
v2_prompt         — V2 structured, Gemini prompt extraction, Neo4j
v2_prompt_clip    — V2 structured, Gemini prompt, Neo4j + CLIP
```

#### Expected Outputs

- Overall accuracy table (Top-1, Top-K, SSR, latency) per profile
- Condition-wise breakdown (A/B/C × 1/2/3) heatmap
- Density-vs-accuracy scatter (candidate pool size effect)
- Modality gain analysis (paired ablation: MA vs MB vs MC)

---

### Experiment Group 2 — Neuro-Symbolic 3-Way Comparison (RQ1 + RQ2)

**Research Question:** Does LoRA fine-tuning with structured spatial triplet
supervision enable VLMs to extract topological constraints that break the
attribute entropy bottleneck, and does deterministic graph traversal eliminate
retrieval hallucination?

#### Systems Under Test

| System | Extractor | Spatial Capability | Max Priority | Adapter |
|---|---|---|---|---|
| **Baseline (Gemini)** | `PromptConstraintsExtractor` | None — prompt-only, no spatial_relations | P1–P8 | None |
| **LoRA_2** | `LoRAConstraintsExtractor` | Attribute-only — 7-field schema, `spatial_relations=[]` | P1–P8 | `v2_lora_qwen/final/` |
| **LoRA_3** | `LoRAConstraintsExtractor` | **Spatial triplets** — 5-field + `spatial_relations` | **P0–P8** | `v3_lora_qwen_20260310_5ep/final/` |

All three systems use the same system prompt, Neo4j graph state, and test cases.
The only variable is the constraints extractor (and its learned capabilities).

#### Independent Variables

- Extractor: Gemini prompt / LoRA_2 / LoRA_3
- Confidence threshold: 0.3 / 0.5 / 0.7 / 0.9 (LoRA_3 only — controls P0 activation)
- Modality condition: MA (text+4D+image+floorplan)

#### Metrics — Retrieval Performance

| Metric | Definition | What It Proves |
|---|---|---|
| **Top-1 Accuracy** | Exact GUID match on rank-1 candidate | Overall system effectiveness |
| **Top-K Accuracy** (K=3,5) | GT in top-K candidates | Retrieval depth |
| **SSR** | Search space reduction ratio | Query planner efficiency |
| **Over-Reduction Rate** | GT pruned from final pool | Safety of spatial filtering |
| **GT-in-Pool Rate** | GT present in candidate pool (before ranking) | Symbolic layer completeness |

#### Metrics — VLM Extraction Quality

| Metric | Definition | What It Proves |
|---|---|---|
| **Constraints Parse Rate** | Valid JSON output / total cases | Extraction reliability |
| **Field EM F1** | Per-field exact match (storey, ifc_class, space_name) | Attribute extraction accuracy |
| **Spatial Predicate Accuracy** | Correct predicate / total topology cases | Triplet extraction precision |
| **Spatial False Positive Rate** | Spurious `spatial_relations` on attribute-only cases / total attr cases | Anti-hallucination effectiveness |
| **mR@100 (per-predicate)** | Mean recall per predicate type at K=100 | Whether VLM learned topology vs language frequency |

#### Metrics — Query Planner Behaviour

| Metric | Definition | What It Proves |
|---|---|---|
| **P0 Activation Rate** | Fraction of cases where Priority-0 spatial Cypher fires | Confidence gate effectiveness |
| **Query Strategy Distribution** | Histogram of priority rules used (P0–P8) | Pipeline routing behaviour |
| **Fallback Rate** | Fraction where P0 fails → degrades to P1–P8 | Symbolic layer robustness |
| **Avg Pool Size (per predicate)** | Mean candidates returned by FILLS / ADJACENT_TO / CONTINUOUS | Discriminative power of each predicate |

#### Test Data

| Dataset | Cases | Leakage | Purpose |
|---|---|---|---|
| **Primary**: `lora3_test.jsonl` converted to eval format | 69 | Low (held-out) | Fair 3-way comparison |
| **Secondary**: `h2_hard_negatives.jsonl` | 213 | Topology-only | Stress test on attribute-identical elements |
| **Supplementary**: Cross-IFC (BH + DXA) | ~64 | Medium | Generalisation test |

#### Profiles

```
v2_prompt         — Baseline Gemini, prompt extraction, Neo4j
v2_lora (LoRA_2)  — LoRA_2 adapter, attribute-only output, Neo4j
v2_lora (LoRA_3)  — LoRA_3 adapter, spatial triplet output, Neo4j
```

#### Confidence Threshold Sweep (LoRA_3 only)

Sweep `CONFIDENCE_THRESHOLD ∈ {0.3, 0.5, 0.7, 0.9}` to characterise the
precision-recall trade-off of the quality gate:

```
Low threshold  (0.3): high P0 activation, higher risk of wrong Cypher
High threshold (0.9): low P0 activation, more P1–P8 fallbacks
```

Report: Top-1 vs threshold curve, P0 activation rate vs threshold.

#### Expected Outputs

- 3-way accuracy table (Top-1, Top-K, SSR, parse rate) per system
- Per-predicate pool size comparison (FILLS, ADJACENT_TO, CONTINUOUS)
- Query strategy distribution bar chart (P0–P8 per system)
- Confidence threshold sweep curve (LoRA_3)
- mR@100 per predicate (proves visual topology learning, not frequency bias)
- H2 stress test results: GT-in-pool rate per predicate

---

### Experiment Group 3 — Ablation Studies

#### 3a. Modality Ablation (controlled)

Hold the same 50 cases constant, vary input modalities:

| Condition | Text | 4D Metadata | Site Photo | Floorplan |
|---|---|---|---|---|
| MA  | ✓ | ✓ | ✓ | ✓ |
| MB  | ✓ | ✓ | ✓ | ✗ |
| MC  | ✓ | ✓ | ✗ | ✗ |
| MA− | ✓ | ✗ | ✓ | ✓ |

**Metric**: Paired ΔTop-1 between conditions on same cases.
**Proves**: Which modality contributes most to spatial triplet extraction.

#### 3b. Component Ablation

| Variant | Spatial Triplets | Neo4j | CLIP Rerank |
|---|---|---|---|
| Full (LoRA_3) | ✓ | ✓ | ✗ |
| − Spatial | ✗ (force `spatial_relations=[]`) | ✓ | ✗ |
| − Neo4j | ✓ | ✗ (memory fallback) | ✗ |
| + CLIP | ✓ | ✓ | ✓ |

**Proves**: Marginal contribution of each component.

---

### Execution Plan

```
Step 1: Ensure Neo4j edges loaded (P4.1 ✅)
Step 2: Convert lora3_test.jsonl → cases_v3 eval format (P10.1)
Step 3: Run 3 systems × same cases × same Neo4j (P10.2–P10.4)
Step 4: Generate comparison plots (P10.5)
Step 5: Run H2 stress test on LoRA_3 (P4.4)
Step 6: Confidence threshold sweep (LoRA_3 on H2)
Step 7: Modality ablation (MA/MB/MC/MA−) on LoRA_3
Step 8: Compile thesis figures + tables
```

### Run Commands

```bash
# Held-out test set (69 cases, converted from lora3_test.jsonl)
CASES=../data_curation/datasets/synth_v0.5/eval/cases_v3_test.jsonl

# Group 2: 3-way comparison
python script/run.py --profile v2_prompt --cases $CASES --condition_override MA
python script/run.py --profile v2_lora --cases $CASES --condition_override MA \
  --adapter_path models/adapters/v2_lora_qwen/final/
python script/run.py --profile v2_lora --cases $CASES --condition_override MA \
  --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/

# Comparison plots
python script/compare_results.py \
  --traces <gemini>.jsonl  --label "Baseline (Gemini)" \
  --traces <lora2>.jsonl   --label "LoRA_2" \
  --traces <lora3>.jsonl   --label "LoRA_3" \
  --plots --output plots/comparisons/3way

# H2 stress test
python eval/h2_eval.py --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/

# Confidence threshold sweep
for t in 0.3 0.5 0.7 0.9; do
  python eval/h2_eval.py --confidence_threshold $t \
    --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
    --output logs/evaluations/h2_t${t}.jsonl
done
```

---

## Exp 4: 4-Way Comparison — Gemini vs LoRA_3 vs LoRA_4 vs LoRA_5 (AP-only)

**Date:** 2026-03-17/18 | **Test set:** `eval/cases_v3_test.jsonl` (69 cases, AP-only subset)

**Trace files:**
| Label | Trace File | Cases |
|-------|-----------|-------|
| Gemini | [`traces_gemini_MC_ap_only.jsonl`](plots/comparisons/0317_4way_ap_only/traces_gemini_MC_ap_only.jsonl) | 59 |
| LoRA_3 | [`traces_lora3_MC_ap_only.jsonl`](plots/comparisons/0317_4way_ap_only/traces_lora3_MC_ap_only.jsonl) | 20 |
| LoRA_4 | [`traces_lora4_MC_ap_only.jsonl`](plots/comparisons/0317_4way_ap_only/traces_lora4_MC_ap_only.jsonl) | 58 |
| LoRA_5 | [`traces_lora5_MC_ap_only.jsonl`](plots/comparisons/0317_4way_ap_only/traces_lora5_MC_ap_only.jsonl) | 59 |

**Charts (AP-only filtered):** [`plots/comparisons/0317_4way_ap_only/charts/`](plots/comparisons/0317_4way_ap_only/charts/)
| Chart | File | What It Shows |
|-------|------|---------------|
| Overall Metrics | [`1_overall_metrics.png`](plots/comparisons/0317_4way_ap_only/charts/1_overall_metrics.png) | Top-1, Name Match, Storey Match, Valid SSR, Over-Reduction |
| Condition Comparison | [`2_condition_comparison.png`](plots/comparisons/0317_4way_ap_only/charts/2_condition_comparison.png) | Per-condition (MA/MB/MC) accuracy bars |
| Search Space Reduction | [`3_search_space_reduction.png`](plots/comparisons/0317_4way_ap_only/charts/3_search_space_reduction.png) | Pool size box plots (valid vs over-reduced) |
| Efficiency | [`4_efficiency_comparison.png`](plots/comparisons/0317_4way_ap_only/charts/4_efficiency_comparison.png) | Latency and cost per model |
| Accuracy Heatmap | [`5_accuracy_heatmap.png`](plots/comparisons/0317_4way_ap_only/charts/5_accuracy_heatmap.png) | Model × condition accuracy grid |
| Accuracy Heatmap (detail) | [`6_accuracy_heatmap_details.png`](plots/comparisons/0317_4way_ap_only/charts/6_accuracy_heatmap_details.png) | Detailed per-case breakdown |
| Full Condition Heatmap | [`6b_full_condition_heatmap.png`](plots/comparisons/0317_4way_ap_only/charts/6b_full_condition_heatmap.png) | All conditions expanded |
| Modality Gain | [`7_modality_gain.png`](plots/comparisons/0317_4way_ap_only/charts/7_modality_gain.png) | MA vs MB vs MC paired delta |
| Difficulty Degradation | [`8_difficulty_degradation.png`](plots/comparisons/0317_4way_ap_only/charts/8_difficulty_degradation.png) | Accuracy by difficulty tier |
| Query Plan Distribution | [`13_query_plan_distribution.png`](plots/comparisons/0317_4way_ap_only/charts/13_query_plan_distribution.png) | P0–P8 strategy usage per model |

**Charts (all cases, unfiltered):** [`plots/comparisons/0317_4way_comparison/`](plots/comparisons/0317_4way_comparison/)

**LoRA_5 deep-dive plots:** [`logs/evaluations/synth_v05_lora5/plots/`](logs/evaluations/synth_v05_lora5/plots/)
| Chart | File | What It Shows |
|-------|------|---------------|
| Per-Floor GT-in-Pool (MC) | [`per_floor_gt_in_pool_MC.png`](logs/evaluations/synth_v05_lora5/plots/per_floor_gt_in_pool_MC.png) | GT-in-pool rate per storey |
| Per-Floor Retrieval (MC) | [`per_floor_retrieval_MC.png`](logs/evaluations/synth_v05_lora5/plots/per_floor_retrieval_MC.png) | Pool size per storey |
| Multi-Condition Floor | [`per_floor_multi_condition.png`](logs/evaluations/synth_v05_lora5/plots/per_floor_multi_condition.png) | MA/MB/MC per-floor comparison |
| RQS Overview | [`rqs_overview.png`](logs/evaluations/synth_v05_lora5/plots/rqs_overview.png) | GT-recall × Valid SSR (F1-style) |
| Hop Accuracy (MC) | [`hop_accuracy_MC.png`](logs/evaluations/synth_v05_lora5/plots/hop_accuracy_MC.png) | Per-hop field accuracy (subject/predicate/object) |
| Predicate Confusion (MC) | [`predicate_confusion_MC.png`](logs/evaluations/synth_v05_lora5/plots/predicate_confusion_MC.png) | Predicted vs GT predicate heatmap |
| Subject Confusion (MC) | [`subject_confusion_MC.png`](logs/evaluations/synth_v05_lora5/plots/subject_confusion_MC.png) | Predicted vs GT subject type heatmap |
| Hop Waterfall (MC) | [`hop_waterfall_MC.png`](logs/evaluations/synth_v05_lora5/plots/hop_waterfall_MC.png) | Per-case hop-by-hop correctness |

**Strategy ablation traces:** [`logs/evaluations/synth_v05_lora5/strategy_ablation/`](logs/evaluations/synth_v05_lora5/strategy_ablation/)
| Strategy | Trace File |
|----------|-----------|
| P0-only | [`traces_..._MC_p0_only.jsonl`](logs/evaluations/synth_v05_lora5/strategy_ablation/traces_20260317_222542_v2_lora_MC_p0_only.jsonl) |
| P1-only | [`traces_..._MC_p1_only.jsonl`](logs/evaluations/synth_v05_lora5/strategy_ablation/traces_20260317_222632_v2_lora_MC_p1_only.jsonl) |
| P0∩P1 | [`traces_..._MC_p0_intersect_p1.jsonl`](logs/evaluations/synth_v05_lora5/strategy_ablation/traces_20260317_222717_v2_lora_MC_p0_intersect_p1.jsonl) |
| P0∪P1 | [`traces_..._MC_p0_union_p1.jsonl`](logs/evaluations/synth_v05_lora5/strategy_ablation/traces_20260317_222758_v2_lora_MC_p0_union_p1.jsonl) |

**LoRA_5 evaluation logs:** [`logs/evaluations/synth_v05_lora5/`](logs/evaluations/synth_v05_lora5/)
| File | Description |
|------|-------------|
| [`eval_lora5_20260318_010906.log`](logs/evaluations/synth_v05_lora5/eval_lora5_20260318_010906.log) | Latest full eval run log |
| [`lora5_metrics_latest.csv`](logs/evaluations/synth_v05_lora5/lora5_metrics_latest.csv) | Summary metrics CSV |
| [`eval_constraints_final_MC.jsonl`](logs/evaluations/synth_v05_lora5/eval_constraints_final_MC.jsonl) | Precomputed constraints (MC condition) |

### 4.1 Overall Metrics

| Metric | Gemini (n=59) | LoRA_3 (n=20) | LoRA_4 (n=58) | LoRA_5 (n=59) |
|--------|--------------|---------------|---------------|---------------|
| **Top-1** | 1 (1.7%) | 3 (15.0%) | 4 (6.9%) | 3 (5.1%) |
| **GT-in-pool** | 7 (11.9%) | 12 (60.0%) | 20 (34.5%) | 17 (28.8%) |
| **name_match** | 26 (44.1%) | 17 (85.0%) | 37 (63.8%) | 31 (52.5%) |
| **storey_match** | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| **ifc_class correct** | 33 (55.9%) | 19 (95.0%) | 37 (63.8%) | 29 (49.2%) |
| **storey_num correct** | 30 (50.8%) | 16 (80.0%) | 29 (50.0%) | 39 (66.1%) |
| **SR extracted** | 58/59 | 0/20 | 42/58 | 59/59 |
| **P0 strategy used** | 55+3 | 0 (all storey+type) | 41+1 | 57+2 |

### 4.2 Why LoRA_3 Appears Better (15.0% vs 5.1%)

**Root cause: Non-comparable test sets + different retrieval strategies.**

1. **Different test sets**: LoRA_3 runs on only 20 AP cases (v3 skeletons), while LoRA_5 runs on 59 AP cases (v5 skeletons including harder augmented cases). There is **zero overlap** in scenario IDs — the LoRA_3 IDs follow `SYNTH_V3_XXX_AP_SK_XXX` while LoRA_5 uses `_augB`/`_augC`/`V05_AP_SK_*` variants.

2. **LoRA_3 has no spatial_relations**: It uses the simpler `storey+type` strategy (Priority 1) for all 20 cases. With 95% ifc_class accuracy and 80% storey accuracy, the simple `WHERE storey = X AND type = Y` Cypher works reliably.

3. **LoRA_5 uses P0 spatial_triplet for 57/59 cases**: This is more powerful in theory but introduces 3 additional failure modes:
   - **Invalid predicates**: LoRA_5 generates `CONNECTS_TO` (38 occurrences) and `NEXT_TO` (8 occurrences) which are NOT in the Neo4j schema (`FILLS`, `CONTINUOUS`, `ADJACENT_TO` only). These produce empty Cypher results.
   - **Subject/object type errors**: With only 49.2% ifc_class accuracy, the Cypher `WHERE node.ifc_type = $wrong_type` filters out the GT element.
   - **P0∩P1 intersection**: When P0 returns wrong elements and P1 returns correct elements, the intersection can be empty or miss GT.

4. **LoRA_3's 3 Top-1 wins** are structurally easy: 2 are singletons (pool=1, only 1 slab on that floor) and 1 is a small pool with lucky ranking.

**Conclusion**: LoRA_3's higher Top-1% is an artifact of a smaller, easier test set and a simpler retrieval strategy, not better extraction quality. A fair comparison would require running all models on the same test set with the same retrieval strategy.

### 4.3 Why storey_match = 0% Universally

**Root cause: Eval pipeline bug — candidate storey field is always null.**

The storey_match metric checks whether the GT storey string appears among the top-10 candidates' storey values (see `src/common/evaluation.py:259-261`). However, the pipeline constructs candidates at `src/v2/pipeline.py:211` using `c.get("storey")` — but the retrieval result dicts from `retrieval_backend.execute_plan()` do not include a `"storey"` key. They include `ref_storey` (the reference element's storey in spatial queries) but not the target element's own storey. This means `candidate.storey = null` for every candidate, causing storey_match to always be False.

**Actual storey prediction accuracy** (comparing floor numbers from constraints vs GT):

| Model | Storey Correct | Rate | Common Errors |
|-------|---------------|------|---------------|
| Gemini | 30/59 | 50.8% | Predicts "Floor 5"/"Floor 6" for lower floors |
| LoRA_3 | 16/20 | 80.0% | 3× predicts "Garage" for non-garage elements |
| LoRA_4 | 29/58 | 50.0% | 7× predicts "-1" for Level 1 elements |
| LoRA_5 | 39/59 | 66.1% | 3× predicts "1" for Garage elements, 2× "5" for Floor 2 |

LoRA_5 actually has the best absolute storey count (39 correct) but the wrong storey still cascades into P0 Cypher failures because `_resolve_storey()` converts the number to a canonical name used in the WHERE clause.

### 4.4 LoRA_5 Failure Taxonomy (59 AP-only cases)

| Category | Count | Description |
|----------|-------|-------------|
| **A: Top-1 success** | 3 (5.1%) | GT is rank-1 candidate |
| **B: GT in pool, not Top-1** | 14 (23.7%) | Retrieval works, reranking absent |
| **C: ifc_class wrong** | 23 (39.0%) | Cypher filters by wrong element type |
| **D: Storey wrong** | 13 (22.0%) | Floor number wrong → wrong storey candidates |
| **E: Other (large pool, no discrimination)** | 6 (10.2%) | Correct predicate but pool too large for ranking |

**Predicate distribution by hop position (LoRA_5, 59 AP-only cases)**:

All 5 predicates (FILLS, ADJACENT_TO, CONTINUOUS, CONNECTS_TO, NEXT_TO) are valid Neo4j edge types, loaded in Sprint 4B from `IfcRelConnectsPathElements` (1362 CONNECTS_TO edges) and filler ordering (~200 NEXT_TO edges).

**Hop-1** (determines Cypher edge traversal):

| Predicate | Count | Subject→Object | Semantically correct? |
|-----------|-------|----------------|----------------------|
| FILLS | 31 | Window→Wall (27), Door→Wall (4) | Yes — matches Neo4j FILLS edges |
| ADJACENT_TO | 16 | Wall→ (9), Door→ (4), Window→ (2), Stair→ (1) | Mostly yes |
| NEXT_TO | 8 | Door→Door/Window (6), Window→Window (2) | Yes — consecutive fillers |
| CONNECTS_TO | 2 | Wall→Wall (2) | Yes — wall path connections |
| CONTINUOUS | 2 | Wall→ (2) | Yes — multi-storey span |

**Hop-2** (soft re-rank via OPTIONAL MATCH, does not filter):

| Predicate | Count | Semantics |
|-----------|-------|-----------|
| CONNECTS_TO | 36 | 2-hop: e.g., Window→FILLS→Wall→CONNECTS_TO→Wall2 |
| FILLS | 5 | Reverse hop |

> **Key finding**: Hop-1 predicates are largely correct — FILLS with Window/Door subjects dominates (31/59). The 36 hop-2 CONNECTS_TO form valid 2-hop chains where the CONNECTS_TO hop itself is Wall→Wall (correct Neo4j semantics). **The dominant failure mode is ifc_class confusion (49.2% wrong), not predicate selection.** Fixing type extraction would have the largest single impact.

**ifc_class confusion matrix (LoRA_5, 59 cases)**:

| GT → Pred | IfcWindow | IfcDoor | IfcWallStdCase | IfcStair | Other |
|-----------|-----------|---------|----------------|----------|-------|
| IfcWindow | **correct** | 2 | 3 | — | — |
| IfcDoor | 2 | **correct** | 7 | 1 | — |
| IfcWallStdCase | 6 | 7 | **correct** | — | — |
| IfcSlab | 1 | — | — | — | — |
| IfcRailing | — | 1 | — | — | — |

Walls are the biggest victim — 13/59 cases where a Wall GT is misclassified as Window or Door. This is because LoRA_5 was trained with FILLS-dominant data (windows/doors fill walls), so it biases toward predicting the subject of a FILLS relation rather than the GT target.

### 4.5 Key Insights

1. **LoRA_5's P0 spatial strategy is more ambitious but less reliable than LoRA_3's simpler P1 storey+type**. The spatial triplet approach has higher theoretical ceiling (can disambiguate within type+storey groups) but currently suffers from 51% wrong ifc_class, making the simpler approach win on this test set.

2. **The critical bottleneck is ifc_class accuracy, not storey or predicate accuracy**. LoRA_5's storey accuracy (66.1%) is reasonable, and hop-1 predicates are largely correct (FILLS=31, ADJACENT_TO=16, with correct subject types). But ifc_class accuracy (49.2%) means half the Cypher queries filter by the wrong element type — guaranteed GT miss. Wall↔Door confusion accounts for 13/59 cases.

3. **Hop-1 predicates are mostly correct; hop-2 CONNECTS_TO is valid 2-hop**. All 5 predicates (FILLS, ADJACENT_TO, CONTINUOUS, CONNECTS_TO, NEXT_TO) are valid Neo4j edge types (loaded from `IfcRelConnectsPathElements` + filler ordering). The 36 hop-2 CONNECTS_TO instances form correct 2-hop chains (`Window→FILLS→Wall→CONNECTS_TO→Wall2`). The model has learned the predicate vocabulary — the primary failure mode is ifc_class confusion, not predicate selection.

4. **storey_match=0% is an eval harness bug, not a model bug**. Candidate dicts from `retrieval_backend` lack the `"storey"` key. Fix: populate `storey` from `engine.get_element_storey(guid)` in the candidate construction step, or use `ref_storey` as a proxy.

5. **LoRA_3 vs LoRA_5 is not a fair comparison** due to different test sets (20 vs 59 cases, zero ID overlap) and different retrieval strategies (storey+type vs spatial_triplet). Future work should run all models on the same test set with the same strategy for a controlled comparison.

### 4.6 Actionable Next Steps

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| **P0** | Fix ifc_class extraction: add more Wall/Door/Window disambiguation examples in training data (currently 13/59 Wall GTs misclassified as Door/Window). ifc_class is the dominant failure mode at 49.2% error rate | Largest single improvement — fixes 39% of failures |
| **P1** | Fix storey field in candidate dict (`pipeline.py:211`) | Unblocks storey_match metric |
| **P2** | Add ifc_class confusion-aware fallback: if P0 returns 0 candidates, retry with broader type match | Recovers some of the 39% class-wrong cases |
| **P3** | Leverage 2-hop chains for reranking: LoRA_5 already generates valid hop-2 CONNECTS_TO (36 cases). Verify that OPTIONAL MATCH reranking actually changes candidate order in practice | Potentially improves Top-1 for 14 GT-in-pool cases |
| **P4** | Run LoRA_3 adapter on the same v5 test set (59 cases) for fair comparison | Establishes true baseline |