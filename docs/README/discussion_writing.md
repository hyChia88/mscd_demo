# Thesis Discussion: Deep-Dive Analysis & Insights

> Compiled 2026-03-26. Draws on experiments E1–E3 from
> `evaluation/analysis/experiment_plots.py`.
> All numbers reference the **unified eval set** (n = 116, 3 IFC models,
> p0∪p1 strategy, FP condition unless noted).

---

## 1  Why Gemini Outperforms LoRA₅ Despite Lower Storey Accuracy

### 1.1  The Paradox

| Diagnostic Metric | LoRA₅-r32 | Gemini |
|-------------------|-----------|--------|
| Storey accuracy   | 76.7%     | 66.4%  |
| ifc_class accuracy| 75.9%     | 75.0%  |
| SR extraction rate| 100%      | 93%    |
| **GT-in-Pool**    | **53.4%** | **51.7%** |
| **Top-10 / MRR@10** | lower  | **higher** |

LoRA₅ leads on every *diagnostic* field accuracy yet trails on ranking
metrics. The intuitive explanation — "better storey → better retrieval" — is
wrong.

### 1.2  Root Cause: The Bottleneck Hierarchy

Through per-case flip analysis (n = 116 paired traces), the actual bottleneck
ordering is:

> **ifc_class accuracy > SR quality > storey accuracy**

Evidence:
- **6 of 11 Gemini-only GT-in-Pool wins** trace to LoRA₅ predicting the
  wrong `ifc_class`. When the Cypher node-type filter selects the wrong
  element type, no amount of correct storey or spatial information can recover
  the GT element.
- Storey errors are **absorbed by the p0∪p1 union strategy**: P1
  (`storey + type`) covers cases where P0 spatial fails, and vice versa.
  Among 12 cases where LoRA₅ has the correct storey but Gemini does not,
  GT-in-Pool is 7/12 for LoRA₅ and 6/12 for Gemini — only a 1-case
  difference.
- Gemini's **contextual spatial relations** (diverse ADJACENT_TO
  predictions) provide a soft reranking signal that LoRA₅'s memorised
  templates cannot. This explains the Top-10 / MRR advantage.

### 1.3  Implication

Storey accuracy is necessary but not the primary lever for improving
end-to-end retrieval. Optimisation effort should prioritise:

1. **ifc_class extraction** — the hard gate on Cypher filtering.
2. **Spatial relation diversity** — genuine input-grounded SR helps ranking.
3. **Storey** — already above 75% for the best model; diminishing returns.

---

## 2  Shortcut Learning in LoRA₅ Spatial Extraction

> **Plot: E1 — Shortcut Learning Evidence**
> `plots/experiments/E1_shortcut_learning_evidence.png`

![E1 Shortcut Learning Evidence](../../plots/experiments/E1_shortcut_learning_evidence.png)

### 2.1  The "Image as Trigger" Effect

We compare constraint outputs under three input modalities for LoRA₅-r32:

| Comparison | storey | ifc_class | SR predicate | Interpretation |
|-----------|--------|-----------|-------------|----------------|
| **MA ↔ FP** (text-only vs floorplan) | 100% same | 51% same | 20% same | Image *presence* changes output |
| **FP ↔ MC** (floorplan vs multi-crop) | 100% same | 94% same | 81% same | Image *content* does not matter |

When no image is provided (MA), LoRA₅ defaults to `ADJACENT_TO` as the
dominant predicate (68% of cases). When *any* image is provided — whether a
floor plan or a multi-crop site photo — the model switches to its dominant
training template:

```
FILLS → IfcWallStandardCase(Plaster) + CONNECTS_TO → IfcWallStandardCase(Leather, weathered)
```

This template appears in **48 of 50** multi-hop extractions. The model
learned: *"image present → emit the most frequent training pattern"*, not
*"read the image → extract what is visible"*.

### 2.2  Template Diversity as a Memorisation Fingerprint

| Metric | LoRA₅-r32 | LoRA₅-r16 | LoRA₂ | Gemini |
|--------|-----------|-----------|-------|--------|
| Unique SR patterns (of 116) | 14 | 14 | 1 (empty) | **61** |
| Shannon entropy (% of max) | 44% | 48% | N/A | **76%** |
| FP ↔ MC SR identity | 81% | 72% | 100% | **23%** |

LoRA₅ collapses to ~14 templates regardless of input. Gemini produces 61
distinct patterns and only 23% survive unchanged when the image modality
changes — evidence of partial but real visual grounding.

### 2.3  Diagnosis

LoRA₅ was fine-tuned on `skins_multitriplet.jsonl` (389 records), where
Pattern A (`FILLS + CONNECTS_TO`) constituted the majority. The model
over-fit to the training data marginal distribution rather than learning to
condition on image content. This is a classic instance of **shortcut
learning** (Geirhos et al., 2020): the model exploits spurious correlations
(presence/absence of image tokens) rather than the intended causal pathway
(spatial layout → spatial relation).

### 2.4  Gemini: Partial Grounding, Still Over-Hallucinates

Gemini shows genuine input sensitivity (61 unique patterns, 23% identity
rate), meaning it does extract *something* from images. However:
- It hallucinates SR on **89% of attribute-only cases** (76 cases where the
  GT has no spatial relation at all).
- Predicate accuracy on the 40 spatial cases is only **30%**.

Gemini reads the image but lacks the domain precision to map visual
adjacency to the correct IFC graph predicate.

---

## 3  Multi-Hop Spatial Relations: Extraction & Impact

> **Plot: E2 — Multi-Hop Analysis**
> `plots/experiments/E2_multihop_analysis.png`

![E2 Multi-Hop Analysis](../../plots/experiments/E2_multihop_analysis.png)

### 3.1  Can Multi-Hop Be Correctly Identified?

**No — and the eval set cannot fully measure it.**

- The unified eval set contains **0 multi-hop GT cases**: all 40 spatial
  cases have exactly 1 spatial relation. There is no ground truth to evaluate
  multi-hop identification against.
- Hop-1 predicate accuracy is mediocre: LoRA₅-r32 = **47.5%**, LoRA₅-r16 =
  32.5%, Gemini = 30.0%.
- All models hallucinate multi-hop heavily:
  - LoRA₅-r32: 50/116 (43%) extracted as multi-hop, of which **48 are
    hallucinated** (GT has ≤1 SR)
  - Gemini: 56/116 (48%) extracted as multi-hop, **all 56 hallucinated**

### 3.2  Does Multi-Hop Help Retrieval?

Mixed, and likely confounded:

| Model | Single-hop GIP | Multi-hop GIP | Δ |
|-------|---------------|--------------|---|
| LoRA₅-r32 | 59.1% | 46.0% | **−13.1pp** |
| LoRA₅-r16 | 47.0% | 60.0% | +13.0pp |
| Gemini | 44.2% | 58.9% | +14.7pp |

For LoRA₅-r32, multi-hop **hurts** retrieval. The hallucinated
`CONNECTS_TO → IfcWallStandardCase` hop-2 produces a useless reranking
signal: 100% of pool candidates match `has_hop2 = True` (every wall connects
to another wall), so the `ORDER BY has_hop2 DESC` clause provides zero
discriminative power.

### 3.3  Hallucination on Attribute-Only Cases

On the 76 cases where the GT has *no* spatial relation:
- LoRA₅: **100%** hallucinate at least one SR (36–39% hallucinate multi-hop)
- Gemini: **89%** hallucinate SR (49% multi-hop)
- LoRA₂: **0%** (never extracts SR — acts as a useful negative control)

### 3.4  Architectural Soundness

The 2-hop Cypher design is architecturally correct: the `OPTIONAL MATCH`
clause for hop-2 **never reduces the candidate pool** relative to single-hop.
It only reorders candidates, with `has_hop2 = True` ranked first. This means
hallucinated hop-2 is at worst neutral (when `has_hop2` is non-discriminative)
and at best beneficial (when the second hop genuinely narrows ranking).

The bottleneck is extraction quality, not architecture.

### 3.5  Recommendation

1. Multi-hop extraction is not reliable enough to deploy as a feature today.
2. Priority: improve hop-1 predicate accuracy (currently 30–48%) before
   investing in multi-hop.
3. The eval set needs multi-hop GT labels (from `skins_multitriplet.jsonl`
   skeletons) to properly measure this capability.
4. Consider a confidence-gated approach: only execute hop-2 Cypher when
   `confidence ≥ 0.8` and the predicate is in a validated set.

---

## 4  FP vs MC: What Visual Inputs Drive Accuracy

### 4.1  MC Consistently Improves GT-in-Pool

| Model | FP GIP | MC GIP | Δ |
|-------|--------|--------|---|
| LoRA₅-r32 | 53.4% | 56.0% | +2.6pp |
| LoRA₅-r16 | 50.9% | 53.4% | +2.5pp |
| LoRA₂     | 28.4% | 37.1% | **+8.7pp** |
| Gemini    | 51.7% | 54.3% | +2.6pp |

LoRA₂ benefits most because its baseline is weakest and MC images provide
the clearest IFC class signal.

### 4.2  71% of MC Gains Come from ifc_class Correction

Among cases that flip from miss → hit when switching FP → MC:
- **71%** are driven by ifc_class correction (model identifies the correct
  element type from the site photo)
- **16%** are driven by storey correction
- **13%** are driven by both class and SR changes

This confirms the bottleneck hierarchy: **ifc_class is the gating factor**.
Site photos help because element appearance (window frame, door handle, wall
surface) provides strong class signal that floor plans cannot.

### 4.3  LoRA₅ Is Input-Insensitive

LoRA₅ changes only 3 cases between FP and MC (FP ↔ MC identity = 75%).
Its memorised templates overpower any visual signal from the actual images.
Gemini changes 27 cases (identity = 15.5%), confirming it genuinely processes
image content.

---

## 5  Input Pattern Analysis: What Users Should Provide

> **Plot: E3 — Input Pattern Analysis & User Guidance**
> `plots/experiments/E3_input_analysis_user_guide.png`

![E3 Input Analysis](../../plots/experiments/E3_input_analysis_user_guide.png)

### 5.1  Text Feature Prevalence in the Eval Set

| Text Feature | Cases with feature | Prevalence |
|-------------|-------------------|-----------|
| Element type in chat text ("window", "door", "wall") | 41/116 | 35% |
| Floor/storey in chat text or task metadata | 29/116 | 21% |
| Spatial keywords ("next to", "near") | 3/116 | 3% |
| Material keywords ("brick", "plaster") | 6/116 | 5% |

Most conversations are sparse — 40/116 (34%) have *empty* chat text (only
task metadata and images). This is realistic for construction site
communication but means the system must extract most information from images
and metadata.

### 5.2  The Type-Mention Lift: +23pp GT-in-Pool

The single most impactful user action is **mentioning the element type** in
the conversation text:

| Condition | LoRA₅-r32 GIP | Gemini GIP |
|-----------|--------------|------------|
| Type mentioned (n = 41) | **68.3%** | **53.7%** |
| Type not mentioned (n = 75) | 45.3% | 34.7% |
| **Lift** | **+23.0pp** | **+19.0pp** |

This lifts ifc_class extraction accuracy from ~71% → 85% for LoRA₅, which
cascades through the Cypher filter to improve pool formation.

### 5.3  Floor Mention: Modest Effect

| Condition | LoRA₅-r32 GIP | Gemini GIP |
|-----------|--------------|------------|
| Floor mentioned (n = 29) | 41.4% | 34.5% |
| Floor not mentioned (n = 87) | 57.5% | 55.2% |

Counter-intuitively, floor mention *correlates with lower GIP*. This is
likely a confound: cases with explicit floor mentions tend to be harder cases
(multi-storey buildings where storey disambiguation is critical). The system
already infers storey from `task_status` metadata ("TASK_831: Floor 2
Finishing Works") with high accuracy, so explicit floor mention in chat
provides limited additional signal.

### 5.4  User Input Priority Guide

Based on the quantitative analysis:

| Priority | Input | Source | Impact | Recommendation |
|----------|-------|--------|--------|----------------|
| ★★★ | **Element type** | User text | +23pp GIP | Always mention: "the window", "this wall" |
| ★★☆ | **Floor / storey** | Text + task metadata | +5pp storey accuracy | Mention if not in metadata: "on Floor 3" |
| ★★☆ | **Multiple photos** | MC images | +3–9pp GIP | Provide when available; helps class identification |
| ★☆☆ | **Spatial context** | VLM (unreliable) | ±0pp (noisy) | System can auto-extract but current accuracy is low |
| ☆☆☆ | **Material** | VLM / text | Rare signal | Only mention for disambiguation ("the *brick* wall") |

### 5.5  Design Recommendation: Guided Input Prompts

Based on these findings, the conversational interface should:

1. **Prompt for element type** when not detected in the initial message:
   *"What type of element are you asking about? (window, door, wall, ...)"*
2. **Auto-extract floor from metadata** (`task_status`, `project_phase`)
   rather than relying on user mention.
3. **Accept but not require spatial descriptions** — the system architecture
   supports spatial queries, but current VLM extraction is too unreliable to
   make this a hard dependency.
4. **Request site photos when available** — even low-quality site photos
   improve class identification vs floor plan alone.

---

## 6  Shortcut Learning Verification: Evidence Summary

The shortcut learning hypothesis for LoRA₅ is supported by five converging
lines of evidence:

| # | Test | Result | Implication |
|---|------|--------|-------------|
| 1 | FP ↔ MC SR identity | 81% (LoRA₅) vs 23% (Gemini) | Image content has ~0 effect on LoRA₅ SR output |
| 2 | MA ↔ FP SR identity | 20% (image *presence* changes output) | Model uses image as mode switch, not information source |
| 3 | Template diversity | 14 patterns (LoRA₅) vs 61 (Gemini) | LoRA₅ collapsed to training distribution marginals |
| 4 | Dominant template | 48/50 multi-hop = `FILLS→Wall + CONNECTS_TO→Wall` | Direct copy from majority training pattern |
| 5 | Cross-IFC-model invariance | Same templates for AP/BH/DXA | SR is independent of building geometry |

### 6.1  Potential Further Verification (Future Work)

| Method | Effort | What It Proves |
|--------|--------|----------------|
| **Blank image test** | Medium (GPU) | Zero visual grounding if output = FP output with blank image |
| **Cross-swap test** | Medium (GPU) | Text-only shortcut if Case A text + Case B image → same output |
| **Attention heatmap** (GradCAM) | High | Where model "looks" — uniform = no grounding |
| **Gemini marking test** | Low (API) | Whether Gemini can localise elements in floorplan |

---

## 7  Consolidated Bottleneck Model

Synthesising all findings into a single causal model:

```
User Input  ──→  VLM Extraction  ──→  Query Planning  ──→  Graph Retrieval  ──→  Ranking
    │                  │                     │                    │                  │
    │           ┌──────┴──────┐              │              ┌─────┴─────┐            │
    │           │             │              │              │           │            │
    │      ifc_class     spatial_rel         │         P0 (spatial)  P1 (attr)       │
    │       ★★★            ★☆☆            │           ★★☆          ★★☆         │
    │   (71% of flips)  (47% pred acc)       │                                       │
    │                                        │                                       │
    ▼                                        ▼                                       ▼
 Element type    ──────────────────→  Correct node-type  ────────────────────→  GT in pool
 in text (+23pp)                      in Cypher WHERE                           (53% → 68%)
```

**The single highest-ROI intervention is improving ifc_class extraction** —
whether through better training data, user-explicit type mentions, or
ensemble strategies. Spatial relations remain architecturally valuable but
are blocked by VLM extraction quality, not by system design.
