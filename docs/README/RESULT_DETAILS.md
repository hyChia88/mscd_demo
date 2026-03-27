# Result Details — Neuro-Symbolic IFC Element Retrieval

> **Companion to:** [RESULT_OVERVIEW.md](RESULT_OVERVIEW.md)
> **Last updated:** 2026-03-26
> **IFC Models:** AdvancedProject (AP, 1,257 elements, 7 storeys), BasicHouse (BH, 97 elements, 2 storeys), Duplex_A (DXA, 181 elements, 3 storeys)
> **Graph State:** Neo4j Community 5.26.0 — all 3 models loaded

---

## Table of Contents
1. [Metric Definitions](#1-metric-definitions)
2. [Experiment Group 1 — V1 Agent vs V2 Structured Pipeline](#2-experiment-group-1)
3. [Experiment Group 2 — Early System Comparison (AP-only, n=69)](#3-experiment-group-2)
4. [Experiment Group 3 — LoRA5 Deep-Dive & Ablation (n=70)](#4-experiment-group-3)
5. [Experiment Group 4 — 4-Way Model Comparison (AP-only, n=59)](#5-experiment-group-4)
6. [Experiment Group 5 — Unified Cross-Model Eval (n=116)](#6-experiment-group-5)
7. [Cross-Version Analysis: LoRA2 to LoRA5](#7-cross-version-analysis)
8. [Oracle Upper-Bound Analysis](#8-oracle-upper-bound)
9. [Shortcut Learning Analysis](#9-shortcut-learning)
10. [Multi-Hop Analysis](#10-multi-hop-analysis)
11. [Input Pattern & User Guide Analysis](#11-input-pattern-analysis)
12. [Threats to Validity](#12-threats-to-validity)
13. [Data Artefacts Index](#13-data-artefacts-index)

---

## 1  Metric Definitions

### Tier 1 — Primary Metrics

| Metric | Formula | What It Answers |
|--------|---------|-----------------|
| **Top-1 Accuracy** | `GT_GUID == candidates[0].guid` | Can the system find the exact element? |
| **MRR@10** | Mean reciprocal rank within top-10 | How well does ranking place GT near top? |
| **RWR** (Recall-Weighted Reduction) | `mean(1[GT in pool] * (1 - pool/N))` | Joint pool quality — penalises over-reduction |
| **GT-in-Pool** | `count(GT in pool) / total_cases` | How safe is the symbolic filtering? |
| **Avg Pool Size** | Mean pool size when GT is in pool | Discriminative power of queries |

> **Why RWR?** SSR can be high even when GT is pruned. RWR assigns 0 credit when GT is lost, and `1 - pool/N` when GT is kept. Equivalent to `GT-in-Pool * Valid-SSR`.

### Tier 2 — Diagnostic Metrics

| Metric | What It Answers |
|--------|-----------------|
| **storey_acc** | Is the VLM predicting the correct floor? |
| **ifc_class_acc** | Is the VLM predicting the correct element type? |
| **predicate_acc** | Is the spatial predicate correct? |
| **P0 Activation Rate** | How often does spatial Cypher fire? |

### Dropped Metrics

| Dropped | Reason |
|---------|--------|
| Parse Rate | 100% everywhere — not informative |
| Field EM F1 | Masks per-field variation |
| SSR (unconditional) | Misleading when GT is lost; use RWR |
| Over-Reduction Rate | = 1 - GT-in-Pool, redundant |
| Top-K@3/5 | MRR@10 is more informative |

---

## 2  Experiment Group 1 — V1 Agent vs V2 Structured Pipeline {#2-experiment-group-1}

**RQ:** Does a structured pipeline outperform a free-form LLM agent?
**Dataset:** synth_v0.2 (43 cases) | **LLM:** Gemini 2.5 Flash

| System | Top-1 | MRR@10 | GT-in-Pool | GUID Matches |
|--------|-------|--------|------------|--------------|
| V1 Agent (memory) | **32.6%** | 0.347 | — | 16/43 |
| V1 Agent (neo4j) | 30.2% | 0.314 | — | 14/43 |
| V2 Structured (A1: clear+4D) | **50.0%** | — | — | 3/6 |
| V2 Structured (all conditions) | 11.6% | — | — | 5/43 |

### Per-Condition Breakdown (V2)

| Condition | n | Top-1 | Finding |
|-----------|---|-------|---------|
| A1 (clear text + 4D) | 6 | 50.0% | Text metadata is strongest signal |
| A2 (blurred + 4D) | 6 | 16.7% | Chat blurring halves accuracy |
| C1 (clear + floorplan) | 7 | 14.3% | Floorplan alone has limited value |
| B1-B3 (image-based) | 15 | 0% | V2 prompt extractor is text-only |

**Takeaway:** V2 Structured (A1) achieves +17.4pp Top-1 over V1 Agent when input quality is high. But V2 degrades sharply with vague inputs — motivating LoRA fine-tuning.

> **Limitation:** Small per-condition samples (n=2-7). This is a directional pilot.

---

## 3  Experiment Group 2 — Early System Comparison (AP-only, n=69) {#3-experiment-group-2}

### Systems Under Test

| System | Extractor | Spatial | Adapter |
|--------|-----------|---------|---------|
| Baseline (skeleton attrs) | Skeleton-derived storey+type | None (P4 only) | — |
| LoRA-label (train) | LoRA2 adapter | Attribute-only (P1-P8) | v2_lora_qwen |
| Oracle (GT spatial) | Skeleton attrs + GT triplets | Full (P0) | — |
| LoRA3 (site MC) | LoRA3 adapter | Spatial (P0-P8) | v3_lora_qwen |

### Results

| System | Top-1 | MRR@10 | RWR | GT-in-Pool | Avg Pool |
|--------|-------|--------|-----|------------|----------|
| Baseline | 4.3% | 0.098 | 0.774 | 84.1% | 102 |
| LoRA-label | 0.0% | 0.035 | 0.513 | 54.5% | 75 |
| Oracle | 3.4% | 0.100 | 0.872 | **91.5%** | 59 |
| LoRA3 site MC | 1.5% | 0.039 | 0.314 | 33.8% | 89 |

> **Note:** Baseline/Oracle use skeleton-derived labels (reference element attributes), not GT target attributes. In 9/69 cases, skeleton ifc_class differs from GT target's. The 84.1% is a skeleton-attribute upper bound.

### Search Space Reduction

All systems achieve 91-95% SSR, reducing 1,233 elements to 59-102 candidates. The key is **whether GT survives** — captured by RWR, not SSR.

### P0 Fire Rate vs GT-in-Pool

| System | P0 Fire Rate | GT-in-Pool |
|--------|-------------|------------|
| Oracle | 96.6% | 91.5% |
| LoRA-label | 52.7% | 54.5% |
| Baseline | 0% | 84.1% |
| LoRA3 MC | 0% | 33.8% |

The Oracle's narrow gap (96.6% fire, 91.5% GT) confirms Cypher queries are precise. The gap vs LoRA3 is entirely extraction errors.

---

## 4  Experiment Group 3 — LoRA5 Deep-Dive & Ablation (n=70) {#4-experiment-group-3}

**Dataset:** synth_v0.5 (70 cases, augmented) | **Adapter:** LoRA5

### 4.1  Modality Ablation

| Condition | Description | GT-in-Pool | Top-1 | RWR |
|-----------|-------------|------------|-------|-----|
| MB (text+image) | Site photo + text | 23 (32.9%) | 1 (1.4%) | 0.314 |
| SITE (photo only) | Site photo, no text | 23 (32.9%) | 1 (1.4%) | 0.314 |
| MA (all modalities) | Text + 4D + image + FP | 20 (28.6%) | 2 (2.9%) | 0.273 |
| MC (text+floorplan) | Floorplan + text | 17 (24.3%) | 1 (1.4%) | 0.233 |
| FP (floorplan only) | Floorplan, no text | 16 (22.9%) | 1 (1.4%) | 0.219 |

- Site photos outperform floorplans for GT-in-Pool: **+8.6pp**
- Adding floorplan to site photo (MA vs MB) does NOT improve GT-in-Pool
- The model cannot yet fuse cross-modal spatial cues

### 4.2  Per-Field Extraction Accuracy (MC, n=70)

| Field | Accuracy | Notes |
|-------|----------|-------|
| storey_num | 55.7% | Reasonable |
| ifc_class | 47.1% | Primary bottleneck; Wall-Door confusion |
| predicate | 40.0% (16/40) | FILLS=80%, ADJACENT_TO=60% |
| object_type | 62.5% | Better than subject type |
| keyword | 100% | When GT has keyword, extraction is reliable |

### 4.3  Predicate Confusion Matrix (MC)

| GT / Predicted | ADJACENT_TO | FILLS | CONTINUOUS | NEXT_TO | CONNECTS_TO |
|----------------|-------------|-------|------------|---------|-------------|
| ADJACENT_TO | **6** | 2 | 1 | 1 | — |
| FILLS | 1 | **8** | — | 1 | — |
| CONTINUOUS | 1 | 2 | — | 1 | 1 |
| CONNECTS_TO | 4 | — | — | 1 | — |
| NEXT_TO | 2 | 7 | — | 1 | — |

**FILLS is learnable** (8/10 = 80%). ADJACENT_TO is moderate (6/10 = 60%). CONNECTS_TO and NEXT_TO have wrong subject types (e.g., IfcWindow CONNECTS_TO IfcWall when CONNECTS_TO only exists between walls).

### 4.4  Per-Hop Field Accuracy (MC)

| Hop | Subject | Predicate | Object |
|-----|---------|-----------|--------|
| Hop 1 (n=70) | 79% | 23% | 36% |
| Hop 2 (n=39) | 28% | 5% | 5% |
| Hop 3 (n=5) | 0% | 0% | 0% |

Multi-hop extraction is not viable. Predicate extraction is the weakest link.

### 4.5  Failure Taxonomy (59 AP-only cases)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| A: Top-1 success | 3 | 5.1% | GT is rank-1 |
| B: GT in pool, not Top-1 | 14 | 23.7% | Retrieval works, reranking needed |
| C: ifc_class wrong | 23 | 39.0% | Cypher filters by wrong type |
| D: Storey wrong | 13 | 22.0% | Wrong floor -> wrong candidates |
| E: Large pool | 6 | 10.2% | Correct predicate but pool too big |

### 4.6  Hop-1 Predicate Distribution (59 cases)

| Predicate | Count | Subject->Object | In Neo4j? |
|-----------|-------|----------------|-----------|
| FILLS | 31 | Window->Wall (27), Door->Wall (4) | Yes |
| ADJACENT_TO | 16 | Wall (9), Door (4), Window (2) | Yes |
| NEXT_TO | 8 | Door->Door/Window (6) | Yes |
| CONNECTS_TO | 2 | Wall->Wall (2) | Yes |
| CONTINUOUS | 2 | Wall (2) | Yes |

Hop-1 predicates are largely correct. The primary failure is ifc_class confusion (49.2% wrong), not predicate selection.

---

## 5  Experiment Group 4 — 4-Way Model Comparison (AP-only, n=59) {#5-experiment-group-4}

| Metric | Gemini (n=59) | LoRA3 (n=20) | LoRA4 (n=58) | LoRA5 (n=59) |
|--------|--------------|---------------|---------------|---------------|
| Top-1 | 1 (1.7%) | 3 (15.0%) | 4 (6.9%) | 3 (5.1%) |
| GT-in-Pool | 7 (11.9%) | 12 (60.0%) | 20 (34.5%) | 17 (28.8%) |
| ifc_class correct | 33 (55.9%) | 19 (95.0%) | 37 (63.8%) | 29 (49.2%) |
| storey_num correct | 30 (50.8%) | 16 (80.0%) | 29 (50.0%) | 39 (66.1%) |
| SR extracted | 58/59 | 0/20 | 42/58 | 59/59 |
| P0 used | 55 | 0 | 41 | 57 |

> **Caveat:** LoRA3 runs on only 20 cases (easier skeletons) with no spatial extraction (uses P1 storey+type). Its higher Top-1 reflects a smaller, easier test set — not superior extraction.

### Why LoRA3 Appears Better (15.0% vs 5.1%)

1. **Different test sets:** LoRA3 = 20 easy cases, LoRA5 = 59 harder cases (zero ID overlap)
2. **LoRA3 uses simpler P1 strategy:** With 95% ifc_class + 80% storey, simple `WHERE storey=X AND type=Y` works reliably
3. **LoRA5 uses ambitious P0 spatial:** More powerful but introduces 3 failure modes (invalid predicates, type errors, intersection pruning)
4. **LoRA3's 3 wins are structurally easy:** 2 are singletons (pool=1)

### ifc_class Confusion Matrix (LoRA5, 59 cases)

| GT / Predicted | IfcWindow | IfcDoor | IfcWallStdCase | IfcStair |
|----------------|-----------|---------|----------------|----------|
| IfcWindow | correct | 2 | 3 | — |
| IfcDoor | 2 | correct | 7 | 1 |
| IfcWallStdCase | 6 | 7 | correct | — |
| IfcSlab | 1 | — | — | — |

Walls are the biggest victim — 13/59 Wall GTs misclassified as Window or Door (FILLS-dominant training bias).

---

## 6  Experiment Group 5 — Unified Cross-Model Eval (n=116) {#6-experiment-group-5}

### 6.1  Systems Under Test

| System | Conditions | Spatial | Notes |
|--------|------------|---------|-------|
| Gemini | FP, MC | SR ~93% | Prompt-based, no fine-tuning |
| LoRA2 | FP, MC | SR 0% | Attribute-only |
| LoRA5-r16 | FP, MC | SR 100% | Standard rank |
| LoRA5-r32 | FP, MC | SR 100% | Higher rank |

### 6.2  Primary Results (FP, p0 union p1)

| System | GT-in-Pool | Top-10 | Top-1 | Avg Pool |
|--------|-----------|--------|-------|----------|
| LoRA5-r32 | **53.4%** (62/116) | 20.7% | 4.3% | 73 |
| LoRA5-r16 | 52.6% (61/116) | 24.1% | 4.3% | 71 |
| Gemini | 50.9% (59/116) | **25.9%** | 4.3% | 81 |
| LoRA2 | 36.2% (42/116) | 21.6% | 2.6% | 80 |

### 6.3  Per-IFC-Model Breakdown (FP, p0 union p1)

| System | Total (116) | AP (76) | BH (23) | DXA (17) |
|--------|------------|---------|---------|----------|
| LoRA5-r32 | 53.4% | 48.7% | **78.3%** | 41.2% |
| LoRA5-r16 | 52.6% | 47.4% | **78.3%** | 41.2% |
| Gemini | 50.9% | 43.4% | **78.3%** | **47.1%** |
| LoRA2 | 36.2% | 32.9% | 56.5% | 23.5% |

- BH jumped from ~22% to 78.3% after elevation fallback fix
- Gemini leads on DXA (47%) — generalises better to unseen models
- LoRA5-r32 leads on AP (+5pp over Gemini)

### 6.4  Per-Field Extraction Accuracy

| System | storey_acc | ifc_class_acc | SR_rate |
|--------|-----------|--------------|---------|
| gemini_FP | 68.1% | 62.1% | 93.1% |
| lora2_FP | 67.2% | 62.9% | 0.0% |
| lora5r16_FP | **81.9%** | 62.9% | 100.0% |
| lora5r32_FP | **81.9%** | **63.8%** | 100.0% |

LoRA5 achieves highest storey accuracy (81.9%) — simplified numeric format ("1" vs "1 - First Floor") is easier to learn. All models have similar ifc_class accuracy (62-65%).

### 6.5  Strategy Ablation

#### LoRA5-r32 (FP, n=116)

| Strategy | GT-in-Pool | Top-1 | Avg Pool |
|----------|-----------|-------|----------|
| p0_only | 25.9% | 3.4% | 40 |
| p1_only | **42.2%** | 2.6% | 68 |
| p0 intersect p1 | 25.9% | 3.4% | 39 |
| **p0 union p1** | **42.2%** | **3.4%** | 70 |

> Note: These sub-tables are from strategy_ablation_v2 (pre BH-fix). Relative ordering unchanged; absolute numbers ~11pp lower than primary results (v3 post-fix).

#### Cross-Model Set Analysis

| Model | Both P0&P1 find GT | Only P0 | Only P1 | Neither |
|-------|-------------------|---------|---------|---------|
| LoRA5-r32 | 30 | **0** | 19 | 67 |
| LoRA5-r16 | 37 | 0 | 11 | 68 |
| Gemini | 38 | 2 | 2 | 74 |
| LoRA2 | 33 | 0 | 0 | 83 |

**Key insight:** For LoRA5, P0 is a strict subset of P1 — spatial never uniquely recovers GT. P0's value is pool compression (1.8x), not GT discovery. Gemini is the only model where P0 uniquely recovers GT (2 cases).

#### Recommended Strategies

| Model | Strategy | GT-in-Pool | Top-1 | Rationale |
|-------|---------|-----------|-------|-----------|
| LoRA5-r32 | p0 union p1 | **53.4%** | 20.7% / 4.3% | Preserves P1 pool; P0 adds compression |
| LoRA5-r16 | p0 union p1 | 52.6% | 24.1% / 4.3% | Best Top-10 among LoRA models |
| Gemini | p0 union p1 | 50.9% | **25.9%** / 4.3% | Best Top-10 overall; unique P0 recovery |
| LoRA2 | any | 36.2% | 21.6% / 2.6% | Strategy-invariant (0% SR) |

---

## 7  Cross-Version Analysis: LoRA2 to LoRA5 {#7-cross-version-analysis}

### Training Configuration

| | LoRA2 | LoRA3 | LoRA4 | LoRA5 |
|---|---|---|---|---|
| Training samples | 933 | 1,377 | 553 | 616 |
| Epochs | 3 | 3 | 5 | 5 |
| Predicates | 0 | 3 (F/A/C) | 4 (+CONNECTS_TO) | 5 (+NEXT_TO) |
| SR ratio | 0% | 44% | 75% | ~75% |
| IFC models | 3 | 3 | ~1 (AP) | 3 |
| LoRA rank | 16 | 16 | 16 | 16 |

### Why LoRA5 Underperforms LoRA2 on Top-1

#### Cause 1: Multi-Task Capacity Conflict
LoRA2 learns 1 task (attributes). LoRA5 learns 3 tasks (attributes + predicates + object types) through the same r=16 adapter. Spatial supervision competes with attribute extraction:
- LoRA2: ifc_class >60%
- LoRA5: ifc_class = 49.2%

#### Cause 2: SR Ratio Too Aggressive (75%)
| Version | SR Ratio | SR Output Rate | Result |
|---------|----------|---------------|--------|
| LoRA3 | 44% | 0% (never outputs SR) | Too conservative |
| LoRA5 | 75% | 100% (always outputs SR) | Too aggressive — 30 false positives |

#### Cause 3: Spatial Label Quality
- FILLS: from IfcRelFillsElement — reliable
- ADJACENT_TO: centroid distance < 1500mm — noisy (arbitrary threshold)
- CONNECTS_TO: reliable but Wall-Wall only
- NEXT_TO: few samples, underspecified semantics

#### Cause 4: Predicate Vocabulary Imbalance
| Predicate | Approx Training Samples | Sufficient? |
|-----------|------------------------|-------------|
| ADJACENT_TO | ~182 | OK |
| FILLS | ~147 | OK |
| CONNECTS_TO | ~124 | Borderline |
| CONTINUOUS | ~56 | Not enough |
| NEXT_TO | <50 | Not enough |

### LoRA2 Modality Ablation (synth_v04, n=50)

| Condition | Top-1 | Top-K | SSR |
|-----------|-------|-------|-----|
| MC (text + FP) | **12.0%** | **20.0%** | 84.4% |
| MC- (FP, no 4D) | 10.0% | 18.0% | 84.6% |
| MA (all) | 8.0% | 16.0% | 74.1% |
| MB (text + site) | 6.0% | 14.0% | 78.5% |

> **Confound:** BH has only 53 elements — storey+type filtering is trivially effective. On AP-only, LoRA2 Top-1 is ~5%, comparable to LoRA5.

---

## 8  Oracle Upper-Bound Analysis {#8-oracle-upper-bound}

**Method:** Look up GT element in Neo4j, read its actual attributes and edges, run Cypher with those ground-truth properties. Guarantees GT is always returned.

**Scope:** n=100 cases (AP=70, BH=20, DXA=10)

| Stage | n | Avg Pool | Reduction | GT-in-Pool |
|-------|---|---------|-----------|------------|
| Full elements | 100 | 917 | — | 100% |
| P1 (storey+type) | 100 | 47 | -95% | 98% |
| 1-hop spatial | 78 | 39 | -16% | 100% |
| 2-hop spatial | 71 | 33 | -16% | 100% |
| 2-hop + material | 69 | 31 | -6% | 100% |

### Per-Predicate Discrimination Power

| Best 1-Hop Edge | n | P1 Pool | 1-Hop Pool | Reduction |
|-----------------|---|---------|------------|-----------|
| ADJACENT_TO | 25 | 77 | 32 | **-58%** |
| FILLS | 6 | 36 | 21 | **-41%** |
| CONTINUOUS | 7 | 37 | 32 | -12% |
| NEXT_TO | 32 | 39 | 36 | -9% |
| CONNECTS_TO | 8 | 93 | 92 | ~0% |

**Value hierarchy:** ADJACENT_TO > FILLS > CONTINUOUS > NEXT_TO >> CONNECTS_TO

Training data should prioritize heterogeneous cross-type predicates (ADJACENT_TO, FILLS).

---

## 9  Shortcut Learning Analysis {#9-shortcut-learning}

> **Plot:** [E1_shortcut_learning_evidence.png](../../evaluation/experiment_plots/E1_shortcut_learning_evidence.png)

### The "Image as Trigger" Effect

| Comparison | storey | ifc_class | SR predicate | Meaning |
|-----------|--------|-----------|-------------|---------|
| MA to FP (text-only vs floorplan) | 100% same | 51% same | 20% same | Image *presence* changes output |
| FP to MC (floorplan vs multi-crop) | 100% same | 94% same | 81% same | Image *content* does not matter |

When any image is provided, LoRA5 switches to its dominant template:
```
FILLS -> IfcWallStandardCase(Plaster) + CONNECTS_TO -> IfcWallStandardCase(Leather, weathered)
```
This appears in **48 of 50** multi-hop extractions.

### Template Diversity

| Metric | LoRA5-r32 | LoRA5-r16 | LoRA2 | Gemini |
|--------|-----------|-----------|-------|--------|
| Unique SR patterns (of 116) | 14 | 14 | 1 (empty) | **61** |
| Shannon entropy (% of max) | 44% | 48% | N/A | **76%** |
| FP to MC SR identity | 81% | 72% | 100% | **23%** |

LoRA5 collapses to ~14 templates. Gemini produces 61 distinct patterns and only 23% survive unchanged when image changes — evidence of partial visual grounding.

### 5-Point Evidence Summary

| # | Test | Result | Meaning |
|---|------|--------|---------|
| 1 | FP to MC SR identity | 81% (LoRA5) vs 23% (Gemini) | Image content has ~0 effect on LoRA5 |
| 2 | MA to FP SR identity | 20% | Image presence = mode switch |
| 3 | Template diversity | 14 vs 61 patterns | LoRA5 collapsed to training marginals |
| 4 | Dominant template | 48/50 multi-hop identical | Direct copy from majority training pattern |
| 5 | Cross-IFC-model invariance | Same templates for AP/BH/DXA | SR independent of building geometry |

### Diagnosis
LoRA5 was fine-tuned on `skins_multitriplet.jsonl` (389 records) where Pattern A (FILLS + CONNECTS_TO) was the majority. The model over-fit to the training data marginal distribution rather than learning to condition on image content. Classic **shortcut learning** (Geirhos et al., 2020).

### Gemini: Partial Grounding
Gemini shows genuine input sensitivity (61 patterns, 23% identity). But:
- Hallucinates SR on **89% of attribute-only cases** (76 cases with no GT SR)
- Predicate accuracy on 40 spatial cases: only **30%**

Gemini reads the image but lacks domain precision for correct IFC graph predicates.

---

## 10  Multi-Hop Analysis {#10-multi-hop-analysis}

> **Plot:** [E2_multihop_analysis.png](../../evaluation/experiment_plots/E2_multihop_analysis.png)

### Can Multi-Hop Be Correctly Identified?

**No.** The eval set has **0 multi-hop GT cases** — all 40 spatial cases have exactly 1 SR.

- Hop-1 predicate accuracy: LoRA5-r32 = 47.5%, LoRA5-r16 = 32.5%, Gemini = 30.0%
- All models hallucinate multi-hop heavily:
  - LoRA5-r32: 50/116 (43%) extracted as multi-hop, **48 hallucinated**
  - Gemini: 56/116 (48%) extracted as multi-hop, **all 56 hallucinated**

### Does Multi-Hop Help Retrieval?

| Model | Single-hop GIP | Multi-hop GIP | Delta |
|-------|---------------|--------------|-------|
| LoRA5-r32 | 59.1% | 46.0% | **-13.1pp** |
| LoRA5-r16 | 47.0% | 60.0% | +13.0pp |
| Gemini | 44.2% | 58.9% | +14.7pp |

For LoRA5-r32, multi-hop **hurts**. The hallucinated hop-2 `CONNECTS_TO->Wall` matches 100% of pool candidates (every wall connects to another wall), providing zero discriminative power.

### Hallucination on Attribute-Only Cases (n=76)

- LoRA5: **100%** hallucinate at least one SR
- Gemini: **89%** hallucinate SR
- LoRA2: **0%** (useful negative control)

### Architectural Soundness

The 2-hop Cypher uses `OPTIONAL MATCH` — hop-2 never reduces the pool, only reorders. Hallucinated hop-2 is at worst neutral. The bottleneck is extraction quality, not architecture.

---

## 11  Input Pattern Analysis {#11-input-pattern-analysis}

> **Plot:** [E3_input_analysis_user_guide.png](../../evaluation/experiment_plots/E3_input_analysis_user_guide.png)

### Text Feature Prevalence

| Feature | Cases | Prevalence |
|---------|-------|-----------|
| Element type mention | 41/116 | 35% |
| Floor/storey mention | 29/116 | 21% |
| Spatial keywords | 3/116 | 3% |
| Material keywords | 6/116 | 5% |
| Empty chat text | 40/116 | 34% |

### Type-Mention Lift: +23pp

| Condition | LoRA5-r32 GIP | Gemini GIP |
|-----------|--------------|------------|
| Type mentioned (n=41) | **68.3%** | **53.7%** |
| Type not mentioned (n=75) | 45.3% | 34.7% |
| Lift | **+23.0pp** | **+19.0pp** |

Lifts ifc_class accuracy from ~71% to 85%, cascading through Cypher to improve pool formation.

### Floor Mention: Confounded

| Condition | LoRA5-r32 GIP | Gemini GIP |
|-----------|--------------|------------|
| Floor mentioned (n=29) | 41.4% | 34.5% |
| Floor not mentioned (n=87) | 57.5% | 55.2% |

Floor mention correlates with *lower* GIP. Likely confound: cases with explicit floor mention are harder (multi-storey disambiguation). System already infers storey from task metadata.

### task_status Coverage

Only 25% (29/116) have meaningful task_status. 75% are "N/A".

### GT Spatial Relation Diversity

Only 10 unique GT patterns (9 meaningful + EMPTY). 76/116 cases have no spatial relation at all.

---

## 12  Threats to Validity {#12-threats-to-validity}

### Internal Validity

1. **Cross-version confound:** LoRA2 tested on 50 cases (3 models), LoRA5 on 70 AP-only cases. BH has only 53 elements making Top-1 artificially high. AP-only LoRA2 Top-1 is ~5%, comparable to LoRA5.

2. **Baseline label is misleading:** "Baseline (GT labels)" uses skeleton-derived constraints, not GT target attributes. 9/69 cases have ifc_class mismatch.

3. **Class mismatch by design:** In 23/70 LoRA5 cases, label ifc_class differs from GT target's. Storey+type-only retrieval has theoretical ceiling of ~67% GT-in-Pool. Spatial relations are structurally necessary for the remaining 33%.

4. **Storey as hidden failure (ADDRESSED):** P0 intersect P1 amplified storey errors. Fixed by switching default to p0 union p1 (+19.8pp recovery).

### External Validity

5. **Test set bias:** 97% Tier-3 (hard) cases. Results = stress-test, not production accuracy.
6. **IFC type coverage:** Only 6 element types (no IfcColumn, IfcBeam, IfcCurtainWall, IfcStair).
7. **Storey concentration:** 51% on Level 1, 16% Garage. Upper floors under-tested.
8. **Synthetic data:** All cases from skeleton mining + LLM augmentation, not real queries.
9. **No embedding baseline:** No direct comparison with vector-DB/dense retrieval.

### Positive Indicators

10. **Non-trivial successes:** All 17 GT-in-pool cases have pool 15-141 (no singletons).
11. **Systematic failures:** Failures cluster around identifiable causes, supporting interpretability claim.
12. **No type blackout:** All IFC types have both successes and failures.

---

## 13  Data Artefacts Index {#13-data-artefacts-index}

### Evaluation Cases

| File | Cases | Description |
|------|-------|-------------|
| `evaluation/cases/cases_v5_test.jsonl` | 70 | LoRA5 test set |
| `evaluation/cases/cases_unified_test.jsonl` | 116 | Unified test set (AP+BH+DXA) |
| `evaluation/cases/precomputed/precomputed_baseline.jsonl` | 69 | GT-label baseline |

### Key Trace Files

| Experiment | Location |
|------------|----------|
| Group 2 traces | `evaluation/results/` |
| Group 3 (LoRA5 ablation) | `output/synth_v05_lora5/` |
| Group 4 (4-way) | `plots/comparisons/0317_4way_ap_only/` |
| Group 5 (unified, 8 runs) | `output/unified/traces/` |
| Group 5 (strategy ablation, 16 runs) | `output/unified/strategy_ablation_v2/` |

### Constraint Files

| File | Description |
|------|-------------|
| `logs/evaluation_output/unified/eval_constraints_lora5r32_{FP,MC}.jsonl` | LoRA5-r32 constraints |
| `logs/evaluation_output/unified/eval_constraints_gemini_{FP,MC}.jsonl` | Gemini constraints |
| `logs/evaluation_output/synth_v05_lora5/eval_constraints_final_MA.jsonl` | LoRA5 text-only (70 cases) |

### Plot Directories

| Directory | Contents |
|-----------|----------|
| `evaluation/experiment_plots/` | E1-E3: Shortcut learning, multi-hop, input analysis |
| `evaluation/plots/` | T1-T5: Thesis-ready system comparison figures |
| `output/synth_v05_lora5/plots/` | LoRA5 deep-dive (confusion matrices, waterfall) |
| `output/unified/plots/` | U1-U10: Unified eval plots |
| `plots/comparisons/0317_4way_ap_only/charts/` | 4-way comparison charts |

### Research Questions Mapping

| RQ | Answer | Key Evidence |
|----|--------|-------------|
| **RQ1:** Can multimodal info assist spatial localisation? | Supported with caveats | Modality crossover (S4.1), Oracle ceiling (S8), LoRA2 attribute improvement |
| **RQ2:** Can schema alignment produce hallucination-resistant output? | Supported — two-layer model | 100% parse rate, symbolic guardrail catches invalid triplets, typed error attribution |
