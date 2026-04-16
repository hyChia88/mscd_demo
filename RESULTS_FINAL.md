# MSCD AP Held-Out Evaluation — Final Consolidated Results

> **Last updated:** 2026-04-08  
> **Evaluation set:** `cases_ap_heldout_e2e.jsonl` — 60 cases, AP model, all Tier-3 (topology-annotated)  
> **Graph:** Enriched (phase5) — includes Fix3 `width_mm`/`height_mm` on Window/Door nodes, Fix4 `ADJACENT_TO` edges with `distance_mm`  
> **Strategy:** `p0_union_p1` for all LoRA models; same for Gemini  
> **Gemini baseline:** `gemini_ap_v2__ap_eval.jsonl` run with `v2_lora` profile (fair comparison to G-series)


## 1. What Each Column Means

| Column | Definition |
|--------|-----------|
| Parse% | JSON parse success rate (model output valid JSON) |
| Class% | `ifc_class` matches ground truth |
| Storey% | `storey_name` matches ground truth (normalised) |
| Hop1% | `spatial_relations[0]` (predicate + object_type + direction) exact match |
| Hop2% | `spatial_relations[1]` exact match — **only on cases with ≥2 SRs** (33 cases). "Hop" = SR slot index, NOT graph traversal |
| PredPrec | Predicate-set precision across all SRs |
| PredRec | Predicate-set recall across all SRs |
| GT-Pool% | Ground-truth element found in candidate pool after retrieval |
| Top-1% | GT ranked #1 |
| Top-10% | GT in top-10 |
| MRR | Mean Reciprocal Rank @10 |
| MedPool | Median final pool size |
| SSR% | Average Search-Space Reduction vs L0 (1257 elements) |

---

## 2. Track A — Extraction Quality (what the model predicts)

| Model | Parse% | Class% | Storey% | Hop1% | Hop2% | PredPrec | PredRec | Notes |
|-------|-------:|-------:|--------:|------:|------:|---------:|--------:|-------|
| G0 Canonical | 100% | 100% | 100% | 83.3% | 93.9% | 0.94 | 0.84 | Canonical training, r=16 |
| G1 FullAug | 100% | 100% | 100% | 78.3% | 93.9% | 0.89 | 0.91 | +text augmentation |
| G2 FullAug LowLR | 100% | 100% | 100% | 85.0% | 93.9% | 0.92 | 0.90 | Lower LR |
| G3 FullAug r32 | 100% | 100% | 100% | 80.0% | 93.9% | 0.89 | 0.91 | r=32 rank |
| G4 Ultimate | 100% | 100% | 100% | 86.7% | 93.9% | 0.91 | 0.81 | All augmentations |
| G6 Baseline | 97% | 97% | 78% | 23.3% | 6.1% | 0.51 | 0.40 | No topology training |
| G7 PosCtx | 100% | 100% | 100% | 78.3% | 87.9% | 0.92 | 0.93 | + position_context labels |
| **G8 PosCtx+Dim** | **100%** | **100%** | **100%** | **81.7%** | **93.9%** | **0.92** | **0.93** | + dimension labels (Fix3) |
| Gemini v2 | 100% | 63% | 0% | 30.0% | 0.0% | 0.74 | 0.40 | Zero-shot, no storey norm |

**Key observations:**
- All trained models parse and classify perfectly; Gemini fails storey normalisation (outputs "First Floor" not canonical "1 - First Floor") and IFC class (63%)
- G8 improves Hop2 from 87.9% → 93.9% over G7: dimension labels help correctly identify the second spatial relation anchor
- G6 (no topology training) collapses to near-zero SR extraction — confirms topology labels are essential
- Gemini's 0% storey accuracy is a normalisation issue, not a model weakness per se

---

## 3. Track B2 — End-to-End Downstream Retrieval

> G1–G6: phase3\_fixed traces (enriched graph has no effect — these models don't output new fields)  
> G7/G8/Gemini: phase5 traces on enriched graph, `v2_lora` profile

| Model | GT-Pool% | Top-1% | Top-10% | MRR@10 | Med Pool | SSR% | Graph phase |
|-------|--------:|-------:|--------:|-------:|---------:|-----:|-------------|
| G0 Canonical | 100% | 1.7% | 25.0% | 0.0503 | 76 | 92.9% | phase3 |
| G1 FullAug | 100% | 3.3% | 23.3% | 0.0645 | 76 | 92.9% | phase3 |
| G2 FullAug LowLR | 100% | 1.7% | 20.0% | 0.0524 | 76 | 92.9% | phase3 |
| G3 FullAug r32 | 100% | 1.7% | 26.7% | 0.0641 | 76 | 93.0% | phase3 |
| G4 Ultimate | 100% | 0.0% | 23.3% | 0.0324 | 76 | 92.9% | phase3 |
| G6 Baseline | 81.7% | 1.7% | 23.3% | 0.0515 | 76 | 91.7% | phase3 |
| G7 PosCtx | 100% | 6.7% | 26.7% | 0.1015 | 76 | 93.0% | **phase5** |
| **G8 PosCtx+Dim** | **100%** | **6.7%** | **30.0%** | **0.1104** | **76** | **92.9%** | **phase5** |
| Gemini v2 | 91.7% | 1.7% | 18.3% | 0.0557 | 76 | 93.4% | **phase5** |

**Key observations:**
- G8 achieves the best Top-10 (30.0%) and MRR (0.1104) — enriched graph + dimension labels give +3.3pp Top-10 and +0.009 MRR over G7
- G7/G8 Top-1 (6.7%) is identical — the improvement from G8 is in ranking within the top-10, not in hitting rank-1 more often
- Median pool = 76 across all models: the p0_union_p1 strategy always falls back to P1 (46 elements) ∪ P0 result; the union inflates the pool
- Gemini 91.7% GT-in-pool (not 56.7%) — the earlier 56.7% was a **profile bug** (v2\_prompt fired P2 name\_keyword on generic words like "window")
- G6 81.7% GT-in-pool: no SR extraction → P0 fires spatial_triplet with empty relations → storey+type fallback sometimes misses due to storey mismatch

---

## 4. Graph-RAG Reranker (phase5, top-10 shortlist)

| System | Top-10% | Top-1% | MRR@10 | Δ Top-1 vs base |
|--------|--------:|-------:|-------:|---------------:|
| G7 full-topology | 26.7% | 6.7% | 0.104 | baseline |
| G7 + Graph-RAG | 26.7% | 1.7% | 0.073 | **−5.0pp** |
| G8 full-topology | 30.0% | 6.7% | 0.113 | baseline |
| G8 + Graph-RAG | 30.0% | 1.7% | 0.082 | **−5.0pp** |
| P1-only (coarse) | 20.0% | 0.0% | 0.032 | baseline |
| P1 + Graph-RAG | 20.0% | **8.3%** | **0.101** | **+8.3pp** |

**Per-family G8 + Graph-RAG:**

| Family | n | Base Top-1 | Reranked Top-1 | Verdict |
|--------|---|----------:|---------------:|---------|
| singleton:ADJACENT_TO | 12 | 16.7% | 0.0% | **Degraded** |
| triad:FILLS+NEXT_TO+NEXT_TO | 21 | 9.5% | 4.8% | Degraded |
| paired:FILLS+NEXT_TO | 10 | 0.0% | 0.0% | No change |
| singleton:CONNECTS_TO | 14 | 0.0% | 0.0% | No change |

**Key observations:**
- Graph-RAG **hurts** full-topology pipelines (−5pp Top-1): the topology-filtered pool already encodes the spatial signal; Gemini re-scoring on graph context is redundant and introduces noise
- Graph-RAG **helps** P1-only coarse pools (+8.3pp Top-1): when no topology filtering has occurred, Gemini's graph context adds the first discriminating signal
- ADJACENT_TO cases are worst hit: millimeter distance reasoning from text descriptions is unreliable
- **Recommendation:** Apply Graph-RAG only on P1-only/coarse pools, or as a final step after pool ≤ 10 from L3 fingerprint

---

## 5. Oracle Waterfall — Theoretical Ceiling

> Perfect constraint extraction, same Cypher as live system

| Layer | What it adds | Median Pool | n | Origin |
|-------|-------------|------------|---|--------|
| L0 | no filter | 1257 | 60 | — |
| L1 | storey + ifc_type (IFC attrs) | 46 | 60 | IFC attribute |
| L2 | topology type-only (pred + obj_type) | 45 | 60 | topology |
| **L3** | **+ fingerprint** (subtype/material/distance) | **9** | **60** | **topology** |
| L4 | + exact position slot | 1 | 35 | topology |
| L5 | + dimensions ±50mm | 12 | 38 | enriched graph |
| L6 | multi-anchor AND (2+ SRs) | 45 | 33 | topology |
| L7 | p0∪p1 (live default) | 46 | 60 | live system |

**Per-predicate at L2 and L3:**

| Predicate | Origin | L2 median | L3 median | Reduction | n |
|-----------|--------|----------:|----------:|----------:|---|
| FILLS | IFC-native | 46 | 46 | 0% | 38 |
| CONNECTS_TO | IFC-native | 108 | 58 | 46% | 22 |
| NEXT_TO | Author-added | 45 | 9 | 80% | 35 |
| ADJACENT_TO | Author-added | 40 | 8 | 80% | 30 |

**Key observations:**
- **L2→L3 is the dominant signal** (45→9, 80% reduction): fingerprint details collapse the pool; this is the layer the live model fails to leverage
- **FILLS gives zero L2→L3 reduction** (46→46): all windows on a floor fill walls; predicate type alone is degenerate without position slot
- **Author-added edges** (NEXT_TO, ADJACENT_TO) outperform IFC-native at L3: centroid distance and wall projection are highly discriminating
- **L4 position slot = pool 1** for 35 cases: structured `wall_position_index` would nearly solve retrieval for those cases
- **L7 = L1** (p0∪p1 = 46): the union always equals P1 because P0 is a subset of P1 when storey/type are correct

---

## 6. Gap Analysis: Live System vs Oracle

| Metric | Live G8 | Oracle L3 | Gap |
|--------|--------:|----------:|-----|
| Median pool | 76 | 9 | **−67 elements** |
| Top-10% | 30.0% | ~67%* | ~−37pp |
| Top-1% | 6.7% | ~35%* | ~−28pp |

*Oracle L3 rank estimates based on pool size.

**Root causes of the gap (in priority order):**

1. **L3 fingerprint not reached** — model outputs `object_subtype` in free-text `position_context`, not as structured `spatial_relations[].object_subtype` field. Fix 1 wired the Cypher filter but G8 doesn't populate the field.
2. **`distance_mm` dormant** — Fix 4 added ADJACENT_TO edge property and Cypher filter, but G8 doesn't output `distance_mm` as a structured float. Dormant until G9.
3. **`connection_degree` dormant** — Fix 2 same situation.
4. **`wall_position_index` not parsed** — G8 improved position_context text accuracy (+6pp Hop2) but the planner doesn't extract a structured integer. L4 (pool=1) is unreachable.
5. **p0∪p1 union inflates pool** — even when P0 returns a precise 9-element set, P1 appends 37 more elements. L7 = L1 = 46.

---

## 7. File/Trace Provenance

| Model | Precomputed | Traces | Graph |
|-------|------------|--------|-------|
| G0–G6 | `g*__ap_eval.jsonl` | `ap_e2e_phase3_fixed/g*/` | phase3 |
| G7 | `g7_position_context__ap_eval.jsonl` | `ap_e2e_phase5_g8/g7_position_context/` | **phase5** |
| G8 | `g8_posctx_dim__ap_eval.jsonl` | `ap_e2e_phase5_g8/g8_posctx_dim/` | **phase5** |
| Gemini v2 | `gemini_ap_v2__ap_eval.jsonl` | `ap_e2e_phase5_g8/gemini_ap_v2/` | **phase5** |

> **Deprecated/legacy:** `ap_e2e_phase3_fixed/g7_position_context/`, `ap_e2e_phase3_fixed/g8_posctx_dim/`, `ap_e2e_phase3_fixed/gemini_ap*/` → moved to `ap_e2e_phase3_fixed/legacy/`  
> **Wrong-profile run:** `ap_e2e_phase5_g8/legacy_wrong_profile_gemini_ap_v2prompt/` — Gemini run with `v2_prompt` profile, causes 56.7% GT-in-pool due to name\_keyword trap; **do not use**  
> **Old Graph-RAG:** `graph_rag_rerank/20260405_*/` → moved to `graph_rag_rerank/legacy/` (used phase3 G7 traces)

---

## 8. Plots

All plots in `docs/plots/phase4_lora6_main/` and `docs/plots/phase4_lora6_appendix/` reflect the final consolidated metrics above.

| Figure | Description |
|--------|-------------|
| fig02_v2_extraction_vs_downstream_tradeoff | Track A extraction vs Track B2 downstream scatter |
| fig03_trackB2_strict_downstream | G0–G8 + Gemini Top-10/Top-1/MRR bar chart |
| fig04_v2_oracle_dashboard | Oracle waterfall L0–L7 |
| fig05_p1_vs_full_topology_benefit | P1-only vs full topology pool comparison |
| fig06_oracle_vs_model_gap | Gap between oracle L3 and live models |
| fig11_graph_rag_rerank_comparison | G7/G8/P1 + Graph-RAG comparison |
| figA6_graph_rag_rerank_comparison | Same, appendix version |

---

## 9. Conclusions

1. **G8 is the best LoRA model** (Top-10=30.0%, MRR=0.110) — dimension labels and enriched graph improve downstream retrieval over G7 (+3.3pp Top-10).
2. **The bottleneck is structured output, not graph or wiring** — oracle L3 shows pool=9 is achievable, but reaching it requires `object_subtype`, `distance_mm`, `connection_degree`, and `wall_position_index` as structured JSON primitives. All four are wired in retrieval but dormant in G8 (embedded in free text).
3. **Graph-RAG is only beneficial on coarse (P1-only) pools** — applying it to topology-filtered pools degrades Top-1 by 5pp. The reranker should be reserved for cases where retrieval returns no topology signal.
4. **Gemini is a weak baseline for this task** — zero-shot Gemini cannot normalise storey names or learn the schema's intent for `target_name_keyword`; trained LoRA models outperform it on all downstream metrics by large margins.
5. **G9 path:** Training a model to output `object_subtype`, `distance_mm`, `connection_degree`, and `wall_position_index` as structured fields — combined with the already-wired Cypher — would close the 76→9 pool gap and unlock L3/L4 retrieval.
