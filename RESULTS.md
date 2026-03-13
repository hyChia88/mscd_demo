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
  --plots --output logs/comparisons/3way

# H2 stress test
python eval/h2_eval.py --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/

# Confidence threshold sweep
for t in 0.3 0.5 0.7 0.9; do
  python eval/h2_eval.py --confidence_threshold $t \
    --adapter_path models/adapters/v3_lora_qwen_20260310_5ep/final/ \
    --output logs/evaluations/h2_t${t}.jsonl
done
```