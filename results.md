# Results — Neuro-Symbolic IFC Element Retrieval

> Consolidated from `RESULT_OVERVIEW.md`, `RESULT_DETAILS.md`, and `RESULTS_FINAL.md`.
> Two experiment generations are reported side-by-side:
> - **Track U** — Unified cross-model evaluation (n=116, AP+BH+DXA), LoRA2/LoRA5/Gemini, last updated 2026-03-26.
> - **Track G** — AP held-out deep dive (n=60), LoRA6 family G0–G9 + Gemini. G0–G8 last updated 2026-04-08; G9 size-band + rerank extensions added 2026-04-29.
>
> Track G is the newer evidence and supersedes Track U on AP. Track U remains the canonical cross-IFC-model story (BH + DXA generalisation).
>
> **IFC Models:** AdvancedProject (AP, 1,257 elements, 7 storeys), BasicHouse (BH, 97 elements, 2 storeys), Duplex_A (DXA, 181 elements, 3 storeys).
> **Graph:** Neo4j Community 5.26.0. Track G uses the enriched (phase-5) graph (Fix-3 `width_mm`/`height_mm` on Window/Door nodes, Fix-4 `ADJACENT_TO` edges with `distance_mm`).

---

## Table of Contents

- [0. Executive Summary](#0-executive-summary)
- [1. Metric Definitions](#1-metric-definitions)
- **Track U — Unified Cross-Model Eval (n=116)**
  - [U1. Headline Results](#u1-headline-results)
  - [U2. Per-IFC-Model Breakdown](#u2-per-ifc-model-breakdown)
  - [U3. Per-Field Extraction Accuracy](#u3-per-field-extraction-accuracy)
  - [U4. Strategy Ablation (p0 vs p1 vs union)](#u4-strategy-ablation)
  - [U5. Earlier Experiment Groups 1–4](#u5-earlier-experiment-groups)
  - [U6. Cross-Version Analysis LoRA2→LoRA5](#u6-cross-version-analysis)
- **Track G — AP Held-Out Deep Dive (n=60, LoRA6 G-series)**
  - [G1. Track A — Extraction Quality](#g1-track-a-extraction-quality)
  - [G2. Track B2 — Downstream Retrieval](#g2-track-b2-downstream-retrieval)
  - [G3. Graph-RAG Reranker](#g3-graph-rag-reranker)
  - [G4. Oracle Waterfall (theoretical ceiling)](#g4-oracle-waterfall)
  - [G5. Gap Analysis: Live System vs Oracle](#g5-gap-analysis)
  - [G6. G9 — Extended Architecture (final iteration)](#g6-g9-extended-architecture)
- **Diagnostics & Cross-cutting**
  - [D1. Shortcut-Learning Analysis](#d1-shortcut-learning)
  - [D2. Multi-Hop Analysis](#d2-multi-hop)
  - [D3. Input Pattern & User-Guide Analysis](#d3-input-patterns)
  - [D4. User Input Priority Guide](#d4-user-input-priority)
  - [D5. Bottleneck Hierarchy](#d5-bottleneck-hierarchy)
  - [D6. Two-Layer Hallucination Resistance](#d6-hallucination)
- [9. Threats to Validity](#9-threats-to-validity)
- [10. File / Trace Provenance & Artefact Index](#10-provenance)
- [11. Research Questions Mapping](#11-rq-mapping)

---

## 0  Executive Summary

**Track U winner (cross-model, n=116, FP, p0 ∪ p1):**

| System | GT-in-Pool | Top-10 | Top-1 | Avg Pool |
|--------|-----------|--------|-------|----------|
| **LoRA5-r32** | **53.4%** (62/116) | 20.7% | 4.3% | 73 |
| LoRA5-r16 | 52.6% (61/116) | 24.1% | 4.3% | 71 |
| Gemini | 50.9% (59/116) | **25.9%** | 4.3% | 81 |
| LoRA2 | 36.2% (42/116) | 21.6% | 2.6% | 80 |

Oracle upper bound on the unified set: 100% GT-in-Pool, pool compressed from 917 → 31 elements (96.6% reduction).

**Track G results (AP held-out, n=60, `p0_union_p1`):**

| System | GT-Pool | Top-1 | Top-10 | MRR@10 | Med Pool | Configuration |
|--------|--------:|------:|-------:|-------:|---------:|---------------|
| **G8 PosCtx+Dim** (best) | **100%** | **6.7%** | **30.0%** | **0.1104** | 76 | enriched graph + dim labels |
| G7 PosCtx | 100% | 6.7% | 26.7% | 0.1015 | 76 | enriched graph + position_context |
| G1 FullAug | 100% | 3.3% | 23.3% | 0.0645 | 76 | phase-3 graph, text augmentation |
| G6 Baseline (no topology) | 81.7% | 1.7% | 23.3% | 0.0515 | 76 | phase-3 graph, attribute-only |
| Gemini v2 | 91.7% | 1.7% | 18.3% | 0.0557 | 76 | zero-shot, v2_lora profile |
| **G9 (final extended)** | 98.3% | 3.3% | 26.7% | 0.0920 | 76 | enriched + ResNet size-band + Graph-RAG rerank |

**G9 is the most-engineered iteration we ran** — G8 PosCtx+Dim baseline plus a ResNet size-band classifier (trained on OpenCV-supervised floorplan element counts, freeze-4 backbone) plus a Graph-RAG reranker pass on the top-10 shortlist. **It does not surpass G8.** All four metrics regress: the ResNet hard filter loses 1 GT case (100% → 98.3% GT-in-pool), and the Graph-RAG reranker halves Top-1 (6.7% → 3.3%) and shaves Top-10 (30.0% → 26.7%) and MRR (0.1104 → 0.0920). Useful negative result confirming the size-cluster + rerank direction is not the unlock — see [§G3](#g3-graph-rag-reranker) and [§G6](#g6-g9-extended-architecture).

Oracle ceiling (perfect constraints, same Cypher): median pool 9 at L3, Top-1 ≈ 35%, Top-10 ≈ 67%.

**Five headline findings:**

1. **The symbolic layer is sound — the bottleneck is VLM extraction.** Oracle = 100% GT-in-pool. The 53.4% (LoRA5) / 100% (G8 on AP) gap to the live model is entirely wrong predicates / wrong types / wrong storeys.
2. **`ifc_class` is the critical bottleneck**, not storey. 71% of miss→hit flips on the unified set come from fixing `ifc_class`. All models cluster at 62–65% accuracy → dataset-level ceiling.
3. **Spatial adds pool compression, not GT discovery (for LoRA5).** P0 is a strict subset of P1 for GT recovery; P0's value is the ~1.8× pool size reduction that helps downstream ranking.
4. **Shortcut learning in LoRA5 spatial extraction.** Only 14 unique SR patterns across 116 cases (vs Gemini's 61). 48/50 multi-hop extractions are the same template `FILLS→Wall + CONNECTS_TO→Wall`.
5. **Type mention is the biggest user-side lever:** +23pp GT-in-pool when the user says "the window" / "this wall". Asking for the element type is the highest-ROI UX change.

**Track G additional finding:** G8 (PosCtx + dimension labels) on the enriched (phase-5) graph is the strongest LoRA so far — but the live system still leaves a 76→9 pool gap to oracle. Closing it requires the model to emit `object_subtype`, `distance_mm`, `connection_degree`, and `wall_position_index` as **structured fields** rather than embedded in free text. All four are already wired in the retrieval Cypher.

---

## 1  Metric Definitions

### Tier 1 — Primary

| Metric | Formula | What It Answers |
|--------|---------|-----------------|
| **Top-1 Accuracy** | `GT_GUID == candidates[0].guid` | Can the system find the exact element? |
| **MRR@10** | Mean reciprocal rank within top-10 | How well does ranking place GT near top? |
| **RWR** (Recall-Weighted Reduction) | `mean(1[GT in pool] * (1 - pool/N))` | Joint pool quality — penalises over-reduction |
| **GT-in-Pool** | `count(GT in pool) / total_cases` | How safe is the symbolic filtering? |
| **Avg / Med Pool Size** | Mean / median pool size when GT is in pool | Discriminative power of queries |
| **SSR%** | `1 - pool / N_total` (1257 for AP, etc.) | Average search-space reduction vs L0 |

> **Why RWR?** SSR can be high even when GT is pruned. RWR assigns 0 credit when GT is lost, and `1 - pool/N` when GT is kept. Equivalent to `GT-in-Pool × Valid-SSR`.

### Tier 2 — Diagnostic

| Metric | What It Answers |
|--------|-----------------|
| **Parse%** | JSON parse success rate (model output valid JSON) |
| **Class%** | `ifc_class` matches ground truth |
| **Storey%** | `storey_name` matches ground truth (normalised) |
| **Hop1% / Hop2%** | `spatial_relations[i]` (predicate + object_type + direction) exact match. "Hop" = SR slot index, not graph traversal |
| **PredPrec / PredRec** | Predicate-set precision / recall across all SRs |
| **P0 Activation Rate** | How often spatial Cypher fires |

### Dropped Metrics

| Dropped | Reason |
|---------|--------|
| Parse Rate | 100% across all trained models — not informative |
| Field EM F1 | Masks per-field variation |
| Unconditional SSR | Misleading when GT is lost; use RWR |
| Over-Reduction Rate | = 1 − GT-in-Pool, redundant |
| Top-K@3 / @5 | MRR@10 is more informative |

---

# Track U — Unified Cross-Model Eval

Test set: `cases_unified_test.jsonl` (n=116, AP=70, BH=23, DXA=23). Plots: `evaluation/experiment_plots/E1–E3`, `evaluation/plots/T1–T5`, `output/unified/plots/U1–U10`.

## U1  Headline Results <a id="u1-headline-results"></a>

**Best results (FP condition, p0 union p1):**

| System | GT-in-Pool | Top-10 | Top-1 | Avg Pool |
|--------|-----------|--------|-------|----------|
| **LoRA5-r32** | **53.4%** (62/116) | 20.7% | 4.3% | 73 |
| LoRA5-r16 | 52.6% (61/116) | 24.1% | 4.3% | 71 |
| Gemini | 50.9% (59/116) | **25.9%** | 4.3% | 81 |
| LoRA2 | 36.2% (42/116) | 21.6% | 2.6% | 80 |

## U2  Per-IFC-Model Breakdown <a id="u2-per-ifc-model-breakdown"></a>

**(FP, p0 ∪ p1)**

| System | Total (116) | AP (76) | BH (23) | DXA (17) |
|--------|------------|---------|---------|----------|
| LoRA5-r32 | 53.4% | 48.7% | **78.3%** | 41.2% |
| LoRA5-r16 | 52.6% | 47.4% | **78.3%** | 41.2% |
| Gemini | 50.9% | 43.4% | **78.3%** | **47.1%** |
| LoRA2 | 36.2% | 32.9% | 56.5% | 23.5% |

- BH jumped from ~22% to 78.3% after the elevation-fallback fix in `ifc_engine.py`.
- Gemini leads on DXA (47%) — generalises better to unseen models.
- LoRA5-r32 leads on AP (+5pp over Gemini).

## U3  Per-Field Extraction Accuracy <a id="u3-per-field-extraction-accuracy"></a>

| System | storey_acc | ifc_class_acc | SR_rate |
|--------|-----------|--------------|---------|
| gemini_FP | 68.1% | 62.1% | 93.1% |
| lora2_FP | 67.2% | 62.9% | 0.0% |
| lora5r16_FP | **81.9%** | 62.9% | 100.0% |
| lora5r32_FP | **81.9%** | **63.8%** | 100.0% |

LoRA5 achieves the highest storey accuracy (81.9%) — the simplified numeric format ("1" vs "1 – First Floor") is easier to learn. All models cluster around 62–65% `ifc_class` accuracy, suggesting a dataset-level ceiling.

## U4  Strategy Ablation <a id="u4-strategy-ablation"></a>

### LoRA5-r32 (FP, n=116)

| Strategy | GT-in-Pool | Top-1 | Avg Pool |
|----------|-----------|-------|----------|
| p0_only | 25.9% | 3.4% | 40 |
| p1_only | **42.2%** | 2.6% | 68 |
| p0 ∩ p1 | 25.9% | 3.4% | 39 |
| **p0 ∪ p1** | **42.2%** | **3.4%** | 70 |

> Sub-table is from `strategy_ablation_v2` (pre-BH-fix). Relative ordering unchanged; absolute numbers ~11pp lower than the post-fix primary table.

### Cross-Model set analysis

| Model | Both P0&P1 find GT | Only P0 | Only P1 | Neither |
|-------|-------------------|---------|---------|---------|
| LoRA5-r32 | 30 | **0** | 19 | 67 |
| LoRA5-r16 | 37 | 0 | 11 | 68 |
| Gemini | 38 | 2 | 2 | 74 |
| LoRA2 | 33 | 0 | 0 | 83 |

**Key insight:** For LoRA5, P0 is a strict subset of P1 — spatial never uniquely recovers GT. P0's value is pool compression (~1.8×), not GT discovery. Gemini is the only model where P0 uniquely recovers GT (2 cases).

### Recommended Strategies

| Model | Strategy | GT-in-Pool | Top-1 | Rationale |
|-------|---------|-----------|-------|-----------|
| LoRA5-r32 | p0 ∪ p1 | **53.4%** | 4.3% | Preserves P1 pool; P0 adds compression |
| LoRA5-r16 | p0 ∪ p1 | 52.6% | 4.3% | Best Top-10 among LoRA models |
| Gemini | p0 ∪ p1 | 50.9% | **4.3%** | Best Top-10 overall; unique P0 recovery |
| LoRA2 | any | 36.2% | 2.6% | Strategy-invariant (0% SR) |

## U5  Earlier Experiment Groups 1–4 <a id="u5-earlier-experiment-groups"></a>

### Group 1 — V1 Agent vs V2 Structured Pipeline (synth_v0.2, n=43, Gemini 2.5 Flash)

| System | Top-1 | MRR@10 | GUID Matches |
|--------|-------|--------|--------------|
| V1 Agent (memory) | **32.6%** | 0.347 | 16/43 |
| V1 Agent (neo4j) | 30.2% | 0.314 | 14/43 |
| V2 Structured (A1: clear+4D, n=6) | **50.0%** | — | 3/6 |
| V2 Structured (all conditions) | 11.6% | — | 5/43 |

Per-condition (V2): A1 50%, A2 16.7% (chat blurring halves accuracy), C1 14.3%, B1–B3 0% (V2 prompt extractor was text-only).

> Small per-condition samples (n=2–7). This is a directional pilot. V2 (A1) achieves +17.4pp Top-1 over V1 when input quality is high but degrades sharply with vague inputs — motivating LoRA fine-tuning.

### Group 2 — Early System Comparison (AP-only, n=69)

| System | Top-1 | MRR@10 | RWR | GT-in-Pool | Avg Pool |
|--------|-------|--------|-----|------------|----------|
| Baseline (skeleton attrs, P4) | 4.3% | 0.098 | 0.774 | 84.1% | 102 |
| LoRA-label (LoRA2, P1–P8) | 0.0% | 0.035 | 0.513 | 54.5% | 75 |
| Oracle (GT triplets, P0) | 3.4% | 0.100 | 0.872 | **91.5%** | 59 |
| LoRA3 site MC | 1.5% | 0.039 | 0.314 | 33.8% | 89 |

P0 fire rate vs GT-in-pool: Oracle 96.6% / 91.5%; LoRA-label 52.7% / 54.5%; Baseline 0% / 84.1%; LoRA3 MC 0% / 33.8%. The Oracle's narrow gap confirms Cypher queries are precise — the gap vs LoRA3 is entirely extraction errors.

> Baseline/Oracle use skeleton-derived labels (reference element attributes), not GT target attributes. In 9/69 cases, skeleton `ifc_class` differs from GT target's. The 84.1% is a skeleton-attribute upper bound.

### Group 3 — LoRA5 Deep-Dive & Ablation (synth_v0.5, n=70)

**Modality ablation:**

| Condition | GT-in-Pool | Top-1 | RWR |
|-----------|------------|-------|-----|
| MB (text + site photo) | 32.9% | 1.4% | 0.314 |
| SITE (photo only) | 32.9% | 1.4% | 0.314 |
| MA (all modalities) | 28.6% | 2.9% | 0.273 |
| MC (text + floorplan) | 24.3% | 1.4% | 0.233 |
| FP (floorplan only) | 22.9% | 1.4% | 0.219 |

Site photos beat floorplans for GT-in-pool by +8.6pp. Adding floorplan to site photo (MA vs MB) does NOT improve GT-in-pool — the model can't yet fuse cross-modal cues.

**Per-field accuracy (MC, n=70):**

| Field | Accuracy | Notes |
|-------|----------|-------|
| storey_num | 55.7% | Reasonable |
| ifc_class | 47.1% | Primary bottleneck; Wall-Door confusion |
| predicate | 40.0% (16/40) | FILLS=80%, ADJACENT_TO=60% |
| object_type | 62.5% | Better than subject type |
| keyword | 100% | When GT has keyword, extraction is reliable |

**Predicate confusion (MC):** FILLS is learnable (8/10 = 80%). ADJACENT_TO is moderate (6/10 = 60%). CONNECTS_TO and NEXT_TO often have wrong subject types (e.g., IfcWindow CONNECTS_TO IfcWall when CONNECTS_TO only exists between walls).

**Per-hop accuracy:** Hop 1 — subject 79%, predicate 23%, object 36%. Hop 2 (n=39) — 28% / 5% / 5%. Hop 3 (n=5) — all 0%. **Multi-hop extraction is not viable.** Predicate extraction is the weakest link.

**Failure taxonomy (59 AP-only cases):** A Top-1 success 5.1%, B GT in pool but not Top-1 23.7%, C wrong ifc_class 39.0%, D wrong storey 22.0%, E large pool 10.2%.

### Group 4 — 4-Way Model Comparison (AP-only, n=59)

| Metric | Gemini (59) | LoRA3 (20) | LoRA4 (58) | LoRA5 (59) |
|--------|------------|-----------:|-----------:|-----------:|
| Top-1 | 1.7% | 15.0% | 6.9% | 5.1% |
| GT-in-Pool | 11.9% | 60.0% | 34.5% | 28.8% |
| ifc_class correct | 55.9% | 95.0% | 63.8% | 49.2% |
| storey_num correct | 50.8% | 80.0% | 50.0% | 66.1% |
| SR extracted | 58/59 | 0/20 | 42/58 | 59/59 |
| P0 used | 55 | 0 | 41 | 57 |

> **Why LoRA3 appears better (15.0% vs 5.1%):** different test sets (LoRA3 = 20 easy cases, LoRA5 = 59 harder cases, zero ID overlap); LoRA3 uses simpler P1; LoRA3's 3 wins are structurally easy (2 singletons, pool=1).

**ifc_class confusion (LoRA5, 59 cases):** Walls are the biggest victim — 13/59 Wall GTs misclassified as Window or Door (FILLS-dominant training bias).

## U6  Cross-Version Analysis LoRA2 → LoRA5 <a id="u6-cross-version-analysis"></a>

| | LoRA2 | LoRA3 | LoRA4 | LoRA5 |
|---|------:|------:|------:|------:|
| Training samples | 933 | 1,377 | 553 | 616 |
| Epochs | 3 | 3 | 5 | 5 |
| Predicates | 0 | 3 (F/A/C) | 4 (+CONNECTS_TO) | 5 (+NEXT_TO) |
| SR ratio | 0% | 44% | 75% | ~75% |
| IFC models | 3 | 3 | ~1 (AP) | 3 |
| LoRA rank | 16 | 16 | 16 | 16 |

### Why LoRA5 underperforms LoRA2 on Top-1

1. **Multi-task capacity conflict.** LoRA2 learns 1 task (attributes), LoRA5 learns 3 (attributes + predicates + object types) through the same r=16 adapter. Spatial supervision competes with attribute extraction → `ifc_class` drops from >60% → 49.2%.
2. **SR ratio too aggressive (75%).** LoRA3 (44%) outputs SR 0% of the time (too conservative); LoRA5 (75%) outputs SR 100% of the time (too aggressive — 30 false positives).
3. **Spatial label quality.** FILLS is reliable. ADJACENT_TO is noisy (centroid distance < 1500mm is an arbitrary threshold). CONNECTS_TO is Wall-Wall only. NEXT_TO has few samples and underspecified semantics.
4. **Predicate vocabulary imbalance.** ADJACENT_TO ~182 OK, FILLS ~147 OK, CONNECTS_TO ~124 borderline, CONTINUOUS ~56 not enough, NEXT_TO <50 not enough.

### LoRA2 modality ablation (synth_v0.4, n=50)

| Condition | Top-1 | Top-K | SSR |
|-----------|-------|-------|-----|
| MC (text + FP) | **12.0%** | **20.0%** | 84.4% |
| MC- (FP, no 4D) | 10.0% | 18.0% | 84.6% |
| MA (all) | 8.0% | 16.0% | 74.1% |
| MB (text + site) | 6.0% | 14.0% | 78.5% |

> Confound: BH has only 53 elements — storey+type filtering is trivially effective there. On AP-only, LoRA2 Top-1 is ~5%, comparable to LoRA5.

---

# Track G — AP Held-Out Deep Dive (LoRA6 G-series, n=60)

Test set: `cases_ap_heldout_e2e.jsonl` — 60 cases, AP model, all Tier-3 (topology-annotated). All `p0_union_p1`. Gemini v2 baseline: `gemini_ap_v2__ap_eval.jsonl` run with `v2_lora` profile (fair comparison to G-series).

## G1  Track A — Extraction Quality <a id="g1-track-a-extraction-quality"></a>

| Model | Parse% | Class% | Storey% | Hop1% | Hop2% | PredPrec | PredRec | Notes |
|-------|-------:|-------:|--------:|------:|------:|---------:|--------:|-------|
| G0 Canonical | 100% | 100% | 100% | 83.3% | 93.9% | 0.94 | 0.84 | Canonical training, r=16 |
| G1 FullAug | 100% | 100% | 100% | 78.3% | 93.9% | 0.89 | 0.91 | +text augmentation |
| G2 FullAug LowLR | 100% | 100% | 100% | 85.0% | 93.9% | 0.92 | 0.90 | Lower LR |
| G3 FullAug r32 | 100% | 100% | 100% | 80.0% | 93.9% | 0.89 | 0.91 | r=32 rank |
| G4 Ultimate | 100% | 100% | 100% | 86.7% | 93.9% | 0.91 | 0.81 | All augmentations |
| G6 Baseline | 97% | 97% | 78% | 23.3% | 6.1% | 0.51 | 0.40 | No topology training |
| G7 PosCtx | 100% | 100% | 100% | 78.3% | 87.9% | 0.92 | 0.93 | + position_context labels |
| **G8 PosCtx+Dim** | **100%** | **100%** | **100%** | **81.7%** | **93.9%** | **0.92** | **0.93** | + dimension labels (Fix-3) |
| Gemini v2 | 100% | 63% | 0% | 30.0% | 0.0% | 0.74 | 0.40 | Zero-shot, no storey norm |

**Observations:**
- All trained models parse and classify perfectly; Gemini fails storey normalisation (outputs "First Floor" not canonical "1 – First Floor") and `ifc_class` (63%).
- G8 improves Hop2 from 87.9% → 93.9% over G7: dimension labels help correctly identify the second SR anchor.
- G6 (no topology training) collapses to near-zero SR extraction — confirms topology labels are essential.

## G2  Track B2 — Downstream Retrieval <a id="g2-track-b2-downstream-retrieval"></a>

> G1–G6: phase3_fixed traces (enriched graph has no effect — these models don't emit the new fields).
> G7/G8/Gemini: phase5 traces on enriched graph, `v2_lora` profile.

| Model | GT-Pool% | Top-1% | Top-10% | MRR@10 | Med Pool | SSR% | Graph |
|-------|--------:|-------:|--------:|-------:|---------:|-----:|-------|
| G0 Canonical | 100% | 1.7% | 25.0% | 0.0503 | 76 | 92.9% | phase3 |
| G1 FullAug | 100% | 3.3% | 23.3% | 0.0645 | 76 | 92.9% | phase3 |
| G2 FullAug LowLR | 100% | 1.7% | 20.0% | 0.0524 | 76 | 92.9% | phase3 |
| G3 FullAug r32 | 100% | 1.7% | 26.7% | 0.0641 | 76 | 93.0% | phase3 |
| G4 Ultimate | 100% | 0.0% | 23.3% | 0.0324 | 76 | 92.9% | phase3 |
| G6 Baseline | 81.7% | 1.7% | 23.3% | 0.0515 | 76 | 91.7% | phase3 |
| G7 PosCtx | 100% | 6.7% | 26.7% | 0.1015 | 76 | 93.0% | **phase5** |
| **G8 PosCtx+Dim** | **100%** | **6.7%** | **30.0%** | **0.1104** | 76 | 92.9% | **phase5** |
| Gemini v2 | 91.7% | 1.7% | 18.3% | 0.0557 | 76 | 93.4% | **phase5** |
| G9 OpenCV cluster (phase5) | 100% | 6.7% | 25.0% | 0.1057 | 76 | 92.9% | phase5 |
| G9 ResNet band v2 | 98.3% | 6.7% | 26.7% | 0.1020 | — | 93.5% | phase6.1 |
| G9 ResNet+F4 (no rerank) | 98.3% | 6.7% | 26.7% | 0.1041 | — | 93.5% | phase6.1 |
| **G9 ResNet+F4 + Graph-RAG rerank** (final) | **98.3%** | **3.3%** | **26.7%** | **0.0920** | — | 93.5% | phase6.1 |
| G9 ResNet+F4 fused + Graph-RAG | 98.3% | 8.3% | 26.7% | 0.1244 | — | 93.5% | phase6.1 |
| G9 soft size top-30 | 100% | 6.7% | 21.7% | 0.0929 | — | 92.9% | phase6.1 |
| G9 no hard filter | 100% | 1.7% | 20.0% | 0.0659 | — | 92.3% | phase6.1 |

**Observations:**
- G8 achieves the best Top-10 (30.0%) and MRR (0.1104) — enriched graph + dimension labels give +3.3pp Top-10 and +0.009 MRR over G7.
- G7/G8 Top-1 (6.7%) is identical — the G8 improvement is in ranking within the top-10, not in hitting rank-1 more often.
- Median pool = 76 across all models: `p0_union_p1` always falls back to P1 (46 elements) ∪ P0; the union inflates the pool.
- Gemini 91.7% GT-in-pool (not the previously reported 56.7%) — the earlier 56.7% was a **profile bug** (`v2_prompt` fired P2 `name_keyword` on generic words like "window").
- G6 81.7% GT-in-pool: no SR extraction → P0 fires `spatial_triplet` with empty relations → storey+type fallback sometimes misses on storey mismatch.

## G3  Graph-RAG Reranker (phase5, top-10 shortlist) <a id="g3-graph-rag-reranker"></a>

| System | Top-10% | Top-1% | MRR@10 | Δ Top-1 vs base |
|--------|--------:|-------:|-------:|---------------:|
| G7 full-topology | 26.7% | 6.7% | 0.104 | baseline |
| G7 + Graph-RAG | 26.7% | 1.7% | 0.073 | **−5.0pp** |
| G8 full-topology | 30.0% | 6.7% | 0.113 | baseline |
| G8 + Graph-RAG | 30.0% | 1.7% | 0.082 | **−5.0pp** |
| P1-only (coarse) | 20.0% | 0.0% | 0.032 | baseline |
| P1 + Graph-RAG | 20.0% | **8.3%** | **0.101** | **+8.3pp** |

**Per-family (G8 + Graph-RAG):**

| Family | n | Base Top-1 | Reranked Top-1 | Verdict |
|--------|--:|----------:|---------------:|---------|
| singleton:ADJACENT_TO | 12 | 16.7% | 0.0% | **Degraded** |
| triad:FILLS+NEXT_TO+NEXT_TO | 21 | 9.5% | 4.8% | Degraded |
| paired:FILLS+NEXT_TO | 10 | 0.0% | 0.0% | No change |
| singleton:CONNECTS_TO | 14 | 0.0% | 0.0% | No change |

**Observations:**
- Graph-RAG **hurts** full-topology pipelines (−5pp Top-1): the topology-filtered pool already encodes the spatial signal; Gemini re-scoring on graph context is redundant and adds noise.
- Graph-RAG **helps** P1-only coarse pools (+8.3pp Top-1): when no topology filtering has occurred, Gemini's graph context adds the first discriminating signal.
- ADJACENT_TO cases are worst-hit: millimetre distance reasoning from text descriptions is unreliable.
- The G9 ResNet+F4 + Graph-RAG configuration repeats this pattern on a more aggressive setup (size-band hard filter + rerank), with the same direction: −3.4pp Top-1 vs the G9-without-rerank baseline. See [§G6](#g6-g9-extended-architecture).
- **Recommendation:** Apply Graph-RAG only on P1-only / coarse pools, or as a final step after pool ≤ 10 from L3 fingerprint. Do not stack it on top of topology-filtered + size-band-filtered pools.

## G4  Oracle Waterfall — Theoretical Ceiling <a id="g4-oracle-waterfall"></a>

> Perfect constraint extraction, same Cypher as live system. Method: look up GT element in Neo4j, read its actual attributes and edges, run Cypher with those ground-truth properties.

### Track G version (n=60, AP held-out)

| Layer | What it adds | Median Pool | n | Origin |
|-------|-------------|------------:|--:|--------|
| L0 | no filter | 1257 | 60 | — |
| L1 | storey + ifc_type (IFC attrs) | 46 | 60 | IFC attribute |
| L2 | topology type-only (pred + obj_type) | 45 | 60 | topology |
| **L3** | **+ fingerprint** (subtype / material / distance) | **9** | **60** | **topology** |
| L4 | + exact position slot | 1 | 35 | topology |
| L5 | + dimensions ±50mm | 12 | 38 | enriched graph |
| L6 | multi-anchor AND (2+ SRs) | 45 | 33 | topology |
| L7 | p0 ∪ p1 (live default) | 46 | 60 | live system |

**Per-predicate at L2 and L3:**

| Predicate | Origin | L2 median | L3 median | Reduction | n |
|-----------|--------|----------:|----------:|----------:|--:|
| FILLS | IFC-native | 46 | 46 | 0% | 38 |
| CONNECTS_TO | IFC-native | 108 | 58 | 46% | 22 |
| NEXT_TO | Author-added | 45 | 9 | 80% | 35 |
| ADJACENT_TO | Author-added | 40 | 8 | 80% | 30 |

**Observations:**
- **L2→L3 is the dominant signal** (45→9, 80% reduction): fingerprint details collapse the pool; this is the layer the live model fails to leverage.
- **FILLS gives zero L2→L3 reduction** (46→46): all windows on a floor fill walls; predicate type alone is degenerate without position slot.
- **Author-added edges** (NEXT_TO, ADJACENT_TO) outperform IFC-native at L3: centroid distance and wall projection are highly discriminating.
- **L4 position slot = pool 1** for 35 cases: structured `wall_position_index` would nearly solve retrieval for those cases.
- **L7 = L1** (p0 ∪ p1 = 46): the union always equals P1 because P0 is a subset of P1 when storey/type are correct.

### Track U version (n=100, AP=70 / BH=20 / DXA=10)

| Stage | n | Avg Pool | Reduction | GT-in-Pool |
|-------|--:|---------:|----------:|------------|
| Full elements | 100 | 917 | — | 100% |
| P1 (storey+type) | 100 | 47 | −95% | 98% |
| 1-hop spatial | 78 | 39 | −16% | 100% |
| 2-hop spatial | 71 | 33 | −16% | 100% |
| 2-hop + material | 69 | 31 | −6% | 100% |

**Per-predicate discrimination (Track U):**

| Best 1-Hop Edge | n | P1 Pool | 1-Hop Pool | Reduction |
|-----------------|--:|--------:|-----------:|----------:|
| ADJACENT_TO | 25 | 77 | 32 | **−58%** |
| FILLS | 6 | 36 | 21 | **−41%** |
| CONTINUOUS | 7 | 37 | 32 | −12% |
| NEXT_TO | 32 | 39 | 36 | −9% |
| CONNECTS_TO | 8 | 93 | 92 | ~0% |

Value hierarchy: **ADJACENT_TO > FILLS > CONTINUOUS > NEXT_TO ≫ CONNECTS_TO**. Training data should prioritise heterogeneous cross-type predicates.

## G5  Gap Analysis: Live System vs Oracle <a id="g5-gap-analysis"></a>

| Metric | Live G8 | Oracle L3 | Gap |
|--------|--------:|----------:|----:|
| Median pool | 76 | 9 | **−67 elements** |
| Top-10% | 30.0% | ~67%* | ~−37pp |
| Top-1% | 6.7% | ~35%* | ~−28pp |

*Oracle L3 rank estimates based on pool size.

**Root causes of the gap, priority-ordered:**

1. **L3 fingerprint not reached** — the model emits `object_subtype` inside free-text `position_context`, not as the structured `spatial_relations[].object_subtype` field. Fix-1 wired the Cypher filter but G8 doesn't populate the field.
2. **`distance_mm` dormant** — Fix-4 added the ADJACENT_TO edge property and Cypher filter, but G8 doesn't emit `distance_mm` as a structured float.
3. **`connection_degree` dormant** — Fix-2 same situation.
4. **`wall_position_index` not parsed** — G8 improved position_context text accuracy (+6pp Hop2) but the planner doesn't extract a structured integer. L4 (pool=1) is unreachable.
5. **p0 ∪ p1 inflates the pool** — even when P0 returns a precise 9-element set, P1 appends 37 more elements. L7 = L1 = 46.

## G6  G9 — Extended Architecture (final iteration) <a id="g6-g9-extended-architecture"></a>

G9 is the final, most-engineered iteration of the LoRA6 line. It takes the G8 PosCtx+Dim model as the constraint extractor and bolts two additional stages onto the retrieval pipeline:

```
G9 = G8 PosCtx+Dim  +  ResNet size-band classifier  +  Graph-RAG reranker
     (constraint     (per-candidate size prior on    (Gemini reranks the
      extractor)      W/D classes, OpenCV-supervised   top-10 shortlist
                      training labels, freeze-4)       on graph context)
```

The ResNet size-band classifier operates **only on candidates classified as IfcWindow / IfcDoor** and only when the LoRA extractor's `ifc_class` confidence ≥ 0.6 and the predicted storey matches. The inject report (`g9_resnet_band__inject_report.json`) shows that of 60 cases: 17 received injection, 22 skipped (non-W/D), 18 skipped (off-storey), 3 skipped (low confidence). The Graph-RAG reranker then re-orders the top-10 shortlist using Gemini over candidate-level graph context (host wall, NEXT_TO neighbours, position slot).

### G9 vs G8 baseline — headline comparison

| | G8 PosCtx+Dim | **G9 final** (ResNet+F4 + Graph-RAG) | Δ |
|---|---:|---:|---:|
| GT-in-Pool | 100% | 98.3% | −1.7pp |
| Top-1 | 6.7% | 3.3% | **−3.4pp** |
| Top-10 | 30.0% | 26.7% | −3.3pp |
| MRR@10 | 0.1104 | 0.0920 | −0.0184 |
| Avg SSR | 92.89% | 93.49% | +0.60pp |
| Median pool | 76 | 76 | 0 |
| Latency | 88ms | ~400ms (incl. Gemini call) | +4.5× |

**G9 does not beat G8 on any retrieval metric.** It modestly reduces the average pool size (higher SSR), but pays for it by dropping one GT case and degrading Top-1 / Top-10 / MRR.

### Component-level attribution

The G9 variant grid (see §G2 expanded table) lets us isolate which component does what:

| Component added on G8 | GT-Pool | Top-1 | Top-10 | MRR | Conclusion |
|---|---:|---:|---:|---:|---|
| Nothing (G8 baseline) | 100% | 6.7% | 30.0% | 0.1104 | reference |
| OpenCV count snap (f3 / f4) | 100% | 6.7% | 30.0% | 0.1104 | neutral |
| ResNet size-band (hard filter) | 98.3% | 6.7% | 26.7% | 0.1041 | −1 GT, −3.3pp Top-10 |
| ResNet size-band, no hard cutoff | 100% | 1.7% | 20.0% | 0.0659 | catastrophic when soft |
| Soft size-band + top-30 widen | 100% | 6.7% | 21.7% | 0.0929 | −8.3pp Top-10 |
| ResNet + **Graph-RAG rerank** | 98.3% | **3.3%** | 26.7% | 0.0920 | rerank halves Top-1 |
| ResNet + Graph-RAG **fused** | 98.3% | **8.3%** | 26.7% | 0.1244 | only positive G9 row |

**Findings:**
1. **OpenCV element-count snap is neutral.** It changes nothing because the LoRA extractor's `target_width_mm` / `target_height_mm` already covers the same signal at the constraint layer.
2. **ResNet size-band hard filter over-prunes.** It buys ~0.6pp SSR but at the cost of 1 GT case — net negative.
3. **Soft size-band is worse than hard.** Without a hard cutoff, the size prior diffuses into low-rank positions and pushes GT down the list (Top-1 collapse).
4. **Graph-RAG reranking on a topology-filtered pool hurts (consistent with §G3).** Adding it on top of G9's already-filtered pool degrades Top-1 by 3.4pp.
5. **The single positive G9 cell is the "fused" rerank** (ResNet score + Gemini score blended rather than replaced) — that one gets Top-1 8.3% and MRR 0.1244, narrowly beating G8 on MRR (+0.014) while losing 1 GT case. It's a borderline result on n=60, not a robust win.

### What this rules out

- Adding a visual size-band classifier downstream of the LoRA extractor does not help on AP held-out. The size signal that matters is already captured by `target_width_mm` / `target_height_mm` at the constraint stage.
- Graph-RAG reranking on a topology-filtered shortlist is unhelpful, repeating the §G3 finding on G7/G8.
- The unlock direction proposed in §G5 (emit `object_subtype`, `distance_mm`, `connection_degree`, `wall_position_index` as structured fields) **remains open** — G9 explored a different axis.

---

# Diagnostics & Cross-cutting

## D1  Shortcut-Learning Analysis <a id="d1-shortcut-learning"></a>

> Plot: [`evaluation/experiment_plots/E1_shortcut_learning_evidence.png`](evaluation/experiment_plots/E1_shortcut_learning_evidence.png)

### "Image as trigger" effect

| Comparison | storey | ifc_class | SR predicate | Meaning |
|-----------|--------|-----------|-------------|---------|
| MA → FP (text-only vs floorplan) | 100% same | 51% same | 20% same | Image *presence* changes output |
| FP → MC (floorplan vs multi-crop) | 100% same | 94% same | 81% same | Image *content* does not matter |

When any image is provided, LoRA5 switches to its dominant template:

```
FILLS → IfcWallStandardCase(Plaster) + CONNECTS_TO → IfcWallStandardCase(Leather, weathered)
```

This appears in **48 of 50** multi-hop extractions.

### Template diversity

| Metric | LoRA5-r32 | LoRA5-r16 | LoRA2 | Gemini |
|--------|----------:|----------:|------:|-------:|
| Unique SR patterns (of 116) | 14 | 14 | 1 (empty) | **61** |
| Shannon entropy (% of max) | 44% | 48% | N/A | **76%** |
| FP → MC SR identity | 81% | 72% | 100% | **23%** |

LoRA5 collapses to ~14 templates. Gemini produces 61 distinct patterns and only 23% survive unchanged when image changes — evidence of partial visual grounding.

### 5-point evidence summary

| # | Test | Result | Meaning |
|---|------|--------|---------|
| 1 | FP→MC SR identity | 81% (LoRA5) vs 23% (Gemini) | Image content has ~0 effect on LoRA5 |
| 2 | MA→FP SR identity | 20% | Image presence = mode switch |
| 3 | Template diversity | 14 vs 61 patterns | LoRA5 collapsed to training marginals |
| 4 | Dominant template | 48/50 multi-hop identical | Direct copy from majority training pattern |
| 5 | Cross-IFC-model invariance | Same templates for AP/BH/DXA | SR independent of building geometry |

### Diagnosis

LoRA5 was fine-tuned on `skins_multitriplet.jsonl` (389 records) where Pattern A (FILLS + CONNECTS_TO) was the majority. The model over-fit to the training-data marginal distribution rather than learning to condition on image content. Classic **shortcut learning** (Geirhos et al., 2020).

### Gemini: partial grounding

Gemini shows genuine input sensitivity (61 patterns, 23% identity). But:
- Hallucinates SR on **89% of attribute-only cases** (76 cases with no GT SR).
- Predicate accuracy on 40 spatial cases: only **30%**.

Gemini reads the image but lacks domain precision for correct IFC graph predicates.

## D2  Multi-Hop Analysis <a id="d2-multi-hop"></a>

> Plot: [`evaluation/experiment_plots/E2_multihop_analysis.png`](evaluation/experiment_plots/E2_multihop_analysis.png)

### Can multi-hop be correctly identified? — No.

The eval set has **0 multi-hop GT cases** — all 40 spatial cases have exactly 1 SR.

- Hop-1 predicate accuracy: LoRA5-r32 47.5%, LoRA5-r16 32.5%, Gemini 30.0%.
- All models hallucinate multi-hop heavily:
  - LoRA5-r32: 50/116 (43%) extracted as multi-hop, **48 hallucinated**.
  - Gemini: 56/116 (48%) extracted as multi-hop, **all 56 hallucinated**.

### Does multi-hop help retrieval?

| Model | Single-hop GIP | Multi-hop GIP | Δ |
|-------|---------------:|--------------:|--:|
| LoRA5-r32 | 59.1% | 46.0% | **−13.1pp** |
| LoRA5-r16 | 47.0% | 60.0% | +13.0pp |
| Gemini | 44.2% | 58.9% | +14.7pp |

For LoRA5-r32 multi-hop **hurts** — the hallucinated hop-2 `CONNECTS_TO → Wall` matches 100% of pool candidates (every wall connects to another wall) and provides zero discriminative power.

### Hallucination on attribute-only cases (n=76)

- LoRA5: **100%** hallucinate at least one SR
- Gemini: **89%** hallucinate SR
- LoRA2: **0%** (useful negative control)

### Architectural soundness

The 2-hop Cypher uses `OPTIONAL MATCH` — hop-2 never reduces the pool, only reorders. Hallucinated hop-2 is at worst neutral. The bottleneck is extraction quality, not architecture.

## D3  Input Pattern & User-Guide Analysis <a id="d3-input-patterns"></a>

> Plot: [`evaluation/experiment_plots/E3_input_analysis_user_guide.png`](evaluation/experiment_plots/E3_input_analysis_user_guide.png)

### Text feature prevalence

| Feature | Cases | Prevalence |
|---------|------:|-----------:|
| Element type mention | 41/116 | 35% |
| Floor / storey mention | 29/116 | 21% |
| Spatial keywords | 3/116 | 3% |
| Material keywords | 6/116 | 5% |
| Empty chat text | 40/116 | 34% |

### Type-mention lift: +23pp

| Condition | LoRA5-r32 GIP | Gemini GIP |
|-----------|--------------:|-----------:|
| Type mentioned (n=41) | **68.3%** | **53.7%** |
| Type not mentioned (n=75) | 45.3% | 34.7% |
| **Lift** | **+23.0pp** | **+19.0pp** |

Type mention lifts `ifc_class` accuracy from ~71% to 85%, cascading through Cypher to improve pool formation.

### Floor mention: confounded

| Condition | LoRA5-r32 GIP | Gemini GIP |
|-----------|--------------:|-----------:|
| Floor mentioned (n=29) | 41.4% | 34.5% |
| Floor not mentioned (n=87) | 57.5% | 55.2% |

Floor mention correlates with *lower* GIP. Likely confound: cases with explicit floor mention are harder (multi-storey disambiguation). System already infers storey from task metadata.

### task_status coverage

Only 25% (29/116) have meaningful `task_status`. 75% are "N/A".

### GT spatial-relation diversity

Only 10 unique GT patterns (9 meaningful + EMPTY). 76/116 cases have no spatial relation at all.

## D4  User Input Priority Guide <a id="d4-user-input-priority"></a>

| Priority | What to provide | Impact | How |
|----------|----------------|-------:|-----|
| Highest | **Element type** | +23pp GIP | Say "the window", "this wall" |
| Medium | **Floor / storey** | +5pp storey acc | Say "on Floor 3" if not in metadata |
| Medium | **Multiple photos** | +3–9pp GIP | Helps class identification |
| Low | **Spatial context** | ±0pp (noisy) | Current VLM accuracy too low |
| Lowest | **Material** | Rare signal | Only for disambiguation |

**Design recommendation:** prompt users for element type when not detected. Auto-extract floor from task metadata. Accept but don't require spatial descriptions.

## D5  Bottleneck Hierarchy <a id="d5-bottleneck-hierarchy"></a>

```
ifc_class accuracy  >>>  SR quality  >  storey accuracy
   (the hard gate)      (pool compress)    (absorbed by p0 ∪ p1)
```

Wrong element type = guaranteed miss. Wrong storey is absorbed by the union strategy. Wrong spatial relation = wasted pool compression but GT survives via P1.

### Gemini vs LoRA5 paradox

LoRA5 leads on every diagnostic metric (storey 77% vs 66%, ifc_class 76% vs 75%, SR rate 100% vs 93%) but Gemini achieves higher Top-10 / MRR. Why? Gemini's **diverse spatial relations** (61 unique patterns) provide real reranking signal. LoRA5's memorised templates (14 patterns) provide zero discriminative power.

## D6  Two-Layer Hallucination Resistance <a id="d6-hallucination"></a>

```
Layer 1 — Schema:   100% valid JSON output                (SOLVED)
Layer 2 — Symbolic: Invalid triplets → empty Cypher → fallback  (DETECTABLE)
Gap:                Valid-but-wrong triplets → wrong pool (SILENT failure)
```

### Training pipeline is sound

- LoRA2 proves fine-tuning works (+8.4pp Top-1 over Gemini prompt baseline).
- LoRA5 proves spatial output is learnable (0% → 100% SR extraction rate).
- Degradation explained by capacity conflict, SR ratio too aggressive (75%), label noise.

---

## 9  Threats to Validity <a id="9-threats-to-validity"></a>

### Internal validity

1. **Cross-version confound.** LoRA2 tested on 50 cases (3 models), LoRA5 on 70 AP-only cases. BH has only 53 elements, making Top-1 artificially high. AP-only LoRA2 Top-1 is ~5%, comparable to LoRA5.
2. **Baseline label is misleading.** "Baseline (GT labels)" uses skeleton-derived constraints, not GT target attributes. 9/69 cases have `ifc_class` mismatch.
3. **Class mismatch by design.** In 23/70 LoRA5 cases, label `ifc_class` differs from GT target's. Storey+type-only retrieval has a theoretical ceiling of ~67% GT-in-pool. Spatial relations are structurally necessary for the remaining 33%.
4. **Storey as hidden failure (addressed).** P0 ∩ P1 amplified storey errors. Fixed by switching the default to p0 ∪ p1 (+19.8pp recovery).

### External validity

5. **Test set bias.** 97% Tier-3 (hard) cases. Results are stress-test performance, not production accuracy.
6. **IFC type coverage.** Only 6 element types (no IfcColumn, IfcBeam, IfcCurtainWall, IfcStair).
7. **Storey concentration.** 51% on Level 1, 16% Garage. Upper floors under-tested.
8. **Synthetic data.** All cases from skeleton mining + LLM augmentation, not real queries.
9. **No embedding baseline.** No direct comparison with vector-DB / dense retrieval.

### Positive indicators

10. **Non-trivial successes.** All GT-in-pool cases on the LoRA5 ablation set have pool 15–141 (no singletons).
11. **Systematic failures.** Failures cluster around identifiable causes, supporting the interpretability claim.
12. **No type blackout.** All IFC types have both successes and failures.

---

## 10  File / Trace Provenance & Artefact Index <a id="10-provenance"></a>

### Track G (AP held-out) provenance

| Model | Precomputed | Traces | Graph |
|-------|------------|--------|-------|
| G0–G6 | `g*__ap_eval.jsonl` | `ap_e2e_phase3_fixed/g*/` | phase3 |
| G7 | `g7_position_context__ap_eval.jsonl` | `ap_e2e_phase5_g8/g7_position_context/` | **phase5** |
| G8 | `g8_posctx_dim__ap_eval.jsonl` | `ap_e2e_phase5_g8/g8_posctx_dim/` | **phase5** |
| Gemini v2 | `gemini_ap_v2__ap_eval.jsonl` | `ap_e2e_phase5_g8/gemini_ap_v2/` | **phase5** |
| G9 OpenCV cluster | `g9_opencv_cluster__ap_eval.jsonl` | `ap_e2e_phase5_g9/` | phase5 |
| G9 ResNet band | `g9_resnet_band__ap_eval.jsonl` + `g9_resnet_band__inject_report.json` | `ap_e2e_phase6_1_g9_resnet_band(_v2)/` | phase6.1 |
| G9 ResNet+F4 | `g9_resnet_band_f4__ap_eval.jsonl` | `ap_e2e_phase6_1_g9_resnet_f4/` | phase6.1 |
| G9 ResNet+F4 + Graph-RAG (final) | — | `graph_rag_rerank/phase6_1_g9_resnet_f4/` | phase6.1 |
| G9 ResNet+F4 fused | — | `graph_rag_rerank/phase6_1_g9_resnet_f4_fused/` | phase6.1 |
| G9 soft size top-30 | — | `ap_e2e_phase6_1_g9_soft_size(_v2)/`, `graph_rag_rerank/phase6_1_g9_soft_top30(_v2)/` | phase6.1 |
| G9 no hard filter | — | `ap_e2e_phase6_1_g9_no_hard_filter/` | phase6.1 |

> **Deprecated / legacy** (do not use):
> - `ap_e2e_phase3_fixed/{g7_position_context, g8_posctx_dim, gemini_ap*}/` → moved to `ap_e2e_phase3_fixed/legacy/`
> - `ap_e2e_phase5_g8/legacy_wrong_profile_gemini_ap_v2prompt/` — Gemini run with `v2_prompt` profile (causes 56.7% GT-in-pool due to name_keyword trap).
> - `graph_rag_rerank/20260405_*/` → moved to `graph_rag_rerank/legacy/` (used phase3 G7 traces).

### Evaluation cases

| File | Cases | Description |
|------|------:|-------------|
| `evaluation/cases/cases_v5_test.jsonl` | 70 | LoRA5 test set |
| `evaluation/cases/cases_unified_test.jsonl` | 116 | Unified test set (AP+BH+DXA) |
| `evaluation/cases/cases_ap_heldout_e2e.jsonl` | 60 | AP held-out for G-series |
| `evaluation/cases/precomputed/precomputed_baseline.jsonl` | 69 | GT-label baseline |

### Key trace files

| Experiment | Location |
|------------|----------|
| Group 2 traces | `evaluation/results/` |
| Group 3 (LoRA5 ablation) | `output/synth_v05_lora5/` |
| Group 4 (4-way) | `plots/comparisons/0317_4way_ap_only/` |
| Group 5 (unified, 8 runs) | `output/unified/traces/` |
| Group 5 (strategy ablation, 16 runs) | `output/unified/strategy_ablation_v2/` |
| G-series (AP held-out) | `output/ap_e2e_phase5_g8/` |

### Constraint files

| File | Description |
|------|-------------|
| `logs/evaluation_output/unified/eval_constraints_lora5r32_{FP,MC}.jsonl` | LoRA5-r32 constraints |
| `logs/evaluation_output/unified/eval_constraints_gemini_{FP,MC}.jsonl` | Gemini constraints |
| `logs/evaluation_output/synth_v05_lora5/eval_constraints_final_MA.jsonl` | LoRA5 text-only (70 cases) |

### Plot directories

| Directory | Contents |
|-----------|----------|
| `evaluation/experiment_plots/` | E1–E3: shortcut learning, multi-hop, input analysis |
| `evaluation/plots/` | T1–T5: thesis-ready system comparison figures |
| `output/synth_v05_lora5/plots/` | LoRA5 deep-dive (confusion matrices, waterfall) |
| `output/unified/plots/` | U1–U10: unified eval plots |
| `plots/comparisons/0317_4way_ap_only/charts/` | 4-way comparison charts |
| `docs/plots/phase4_lora6_main/` and `..._appendix/` | Phase-4 / G-series plots |

### G-series figure inventory

| Figure | Description |
|--------|-------------|
| `fig02_v2_extraction_vs_downstream_tradeoff` | Track A extraction vs Track B2 downstream scatter |
| `fig03_trackB2_strict_downstream` | G0–G8 + Gemini Top-10 / Top-1 / MRR bar chart |
| `fig04_v2_oracle_dashboard` | Oracle waterfall L0–L7 |
| `fig05_p1_vs_full_topology_benefit` | P1-only vs full topology pool comparison |
| `fig06_oracle_vs_model_gap` | Gap between oracle L3 and live models |
| `fig11_graph_rag_rerank_comparison` | G7 / G8 / P1 + Graph-RAG comparison |
| `figA6_graph_rag_rerank_comparison` | Appendix version |

---

## 11  Research Questions Mapping <a id="11-rq-mapping"></a>

| RQ | Answer | Key Evidence |
|----|--------|--------------|
| **RQ1:** Can multimodal info assist spatial localisation? | Supported with caveats | Modality crossover (Group 3.1), Oracle ceiling (G4), LoRA2 attribute improvement |
| **RQ2:** Can schema alignment produce hallucination-resistant output? | Supported — two-layer model | 100% parse rate, symbolic guardrail catches invalid triplets, typed error attribution (D6) |

---

## 12  Conclusions

1. **G8 PosCtx+Dim is the best LoRA model on AP held-out** (Top-10 = 30.0%, MRR = 0.110, 100% GT-in-pool) — dimension labels and the enriched graph improve downstream retrieval over G7 (+3.3pp Top-10).
2. **LoRA5-r32 is the best LoRA model on the unified cross-IFC set** (GT-in-pool 53.4%, Top-10 20.7%); Gemini leads Top-10 (25.9%) thanks to diverse SR patterns.
3. **G9 — the final extended architecture (G8 + ResNet size-band + Graph-RAG rerank) — does not beat G8.** Top-1 drops 6.7%→3.3%, Top-10 30.0%→26.7%, MRR 0.1104→0.0920, GT-in-pool 100%→98.3%. Useful negative result: visual size-cluster classification and downstream reranking are not the unlock direction on this benchmark. See [§G6](#g6-g9-extended-architecture).
4. **The bottleneck is structured output, not graph or wiring.** Oracle L3 shows pool = 9 is achievable, but reaching it requires `object_subtype`, `distance_mm`, `connection_degree`, and `wall_position_index` as structured JSON primitives. All four are wired in retrieval but dormant in G8 / G9 (embedded in free text or substituted by a downstream classifier).
5. **Graph-RAG is only beneficial on coarse (P1-only) pools** — applying it to topology-filtered pools degrades Top-1 by 5pp, and stacking it on top of size-band-filtered pools (G9) degrades it further.
6. **Gemini is a weak baseline for this task** — zero-shot Gemini cannot normalise storey names or learn the schema's intent for `target_name_keyword`; trained LoRA models outperform it on all downstream metrics by large margins.
7. **Open path (post-G9):** training a model to output `object_subtype`, `distance_mm`, `connection_degree`, and `wall_position_index` as structured fields — combined with the already-wired Cypher — would close the 76→9 pool gap and unlock L3 / L4 retrieval. G9 demonstrated this is **not** substitutable by a downstream visual classifier.

### Next steps (high ROI first)

| Action | Expected impact |
|--------|----------------|
| **UI prompt for element type** | +23pp GIP (zero cost) |
| **Structured fields** (`object_subtype`, `distance_mm`, `connection_degree`, `wall_position_index`) — next-gen LoRA | Unlock oracle L3 / L4 — close 76→9 pool gap. G9 confirmed a downstream visual classifier is *not* a substitute. |
| **Add rare-type training data** (IfcSlab, IfcStair) | ~+4pp GIP |
| **Fix shortcut learning** (more negative examples, balanced SR ratio ~50%) | Unlock real spatial signal |
| **Attribute-matching reranker** | Recover GT from pool (75–82% of GT-in-pool cases lost at ranking) |
| Higher LoRA rank (r=64) or staged training | Recover ifc_class accuracy while keeping spatial |
| Confidence gating on SR | Only execute hop-2 when confidence ≥ 0.8 |
| Text-based spatial extraction | Parse "next to", "near the" from user text |

### FP-only strategy is viable

FP→MC delta is only +2.6pp. Floorplan-only is deployable with the improvement roadmap:
1. Type prompt (+23pp) → ~68% GIP
2. Rare-type data → ~72%
3. Fix shortcut learning → ~75%+
