# Phase 5 Final

## Scope

- Phase 5 focus:
  - finalize LoRA6 AP held-out post-hoc analysis
  - upgrade planner to consume richer fingerprint
  - train and evaluate `G7_position_context`
- Primary benchmark:
  - `AP held-out`
  - bug-fixed oracle line only

---

## Thesis-Ready Results

### 1. Oracle Ceiling and Information Loss

| Item | Result |
| --- | --- |
| Oracle phase3 fixed | `Top-10 40.0%`, `Top-1 5.0%`, `MRR@10 0.1279` |
| `never_unique_even_at_L4` | `33/60` |
| `unique_at_L3` | `9/60` |
| `unique_at_L4_only` | `18/60` |
| `Top-10=YES, Top-1=NO` subset | `21` |
| exact-slot applicable | `16/21` |
| exact-slot -> Top-1 | `15/16` |
| `query_not_using_available_info` | `24` |
| `true_graph_ambiguity` | `30` |
| `ground_truth_not_collected` | `3` |

Key interpretation:

- `40.0% / 5.0%` is the current planner ceiling, not the full graph-theoretical ceiling.
- About half the benchmark is structurally ambiguous even under full fingerprint.
- The most actionable gap is the `24` cases where richer graph information exists but the current planner does not use it.
- `position_context` is the highest-value enrichment target; `direction + object_subtype` is second.

### 2. Model Diagnostics: Template Collapse vs Shortcut-Like Behavior

#### Collapse diversity

| Source | Pred Only | Pred+Obj | Pred+Obj+Dir | SR Full | Label Full |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT | 5 | 13 | 16 | 35 | 45 |
| G3 | 4 | 10 | 12 | 30 | 41 |
| G4 | 5 | 10 | 12 | 21 | 38 |
| G7 | 6 | 13 | 15 | 31 | 42 |

#### Matched-case identity when GT differs

| Model | Pairs | Pred+Obj Same | Pred+Obj+Dir Same | SR Full Same | Label Full Same |
| --- | ---: | ---: | ---: | ---: | ---: |
| G3 | 37 | 37.8% | 37.8% | 8.1% | 0.0% |
| G4 | 37 | 51.3% | 51.3% | 16.2% | 5.4% |
| G7 | 37 | 32.4% | 32.4% | 10.8% | 5.4% |

Key interpretation:

- LoRA6 is no longer in extreme collapse mode, but residual template collapse remains.
- `G4` is more compressed than `G3` on discriminative output space.
- `G7` no longer supports the earlier collapse-to-attributes claim; after fixing the formal eval path, its diversity is close to `G3` and clearly above `G4` on `SR Full`.
- Current evidence supports:
  - `residual template collapse`: yes
  - `shortcut-like / low sensitivity behavior`: yes
  - `pure shortcut learning` as a strong causal claim: not yet proven

### 3. G3 vs G4 Tension

| System | Hop-1 | Pred R | Dir | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle phase3 fixed | - | - | - | 40.0 | 5.0 | 0.1279 |
| G3 | 80.0 | 91.4 | 78.6 | 26.7 | 1.7 | 0.0641 |
| G4 | 86.7 | 81.0 | 57.1 | 23.3 | 0.0 | 0.0324 |
| G7 | 78.3 | 93.1 | 82.1 | 23.3 | 3.3 | 0.0681 |

Key interpretation:

- `G4` is the better intermediate extractor.
- `G7` is the current Track A winner under the formal scorer.
- `G3` is still the best realized downstream retriever if `Top-10` is the main metric.
- `G7` is now the best AP held-out downstream system on `Top-1` and `MRR@10`.
- This supports the thesis claim that extraction accuracy alone does not guarantee downstream retrieval quality.

### 4. G7 Pre-Train Audit

Assembly command already rerun:

```bash
python data_curation/scripts/synth/6_assemble_lora6.py --g7-profile --enable-scale-aug
```

#### Current G7 dataset state

| Split | n | position_context_cov | direction_cov | object_subtype_cov |
| --- | ---: | ---: | ---: | ---: |
| train canonical | 237 | 0.5316 | 1.0 | 0.8333 |
| train aug | 757 | 0.5376 | 1.0 | 0.8268 |
| eval canonical | 60 | 0.55 | 1.0 | 0.9643 |

#### Label-only delta vs current settled data

| Split | Changed | Changed % |
| --- | ---: | ---: |
| train canonical | 7 / 237 | 2.95% |
| train aug | 74 / 753 | 9.83% |
| combined train | 81 / 990 | 8.18% |
| eval canonical | 0 / 60 | 0.00% |

Key interpretation:

- `lora_system_g7` is now correctly written to all `*_g7.jsonl`.
- `train_aug_g7` has recovered fullaug scale coverage.
- This is strong enough for a performance-oriented `G7` run.
- This is not a strict eval-label ablation, because eval labels are unchanged.

### 5. G7 Evaluation Outcome

| Metric | G7 |
| --- | ---: |
| Track A parse | 100.0 |
| Track A class acc | 100.0 |
| Track A storey acc | 100.0 |
| Track A Hop-1 | 78.3 |
| Track A Pred R | 93.1 |
| Track A Dir | 82.1 |
| Track B-2 GT-in-Pool | 100.0 |
| Track B-2 Top-10 | 23.3 |
| Track B-2 Top-1 | 3.3 |
| Track B-2 MRR@10 | 0.0681 |

Key interpretation:

- The earlier catastrophic `G7` result was an eval-path mismatch artifact, not the final model outcome.
- After fixing the formal Modal eval path to use the G7 eval JSONL and prompt, `G7` recovers to the same regime seen in training-time inference checks.
- `G7` is now:
  - the current `Track A` winner
  - tied with `G1/G4` on `Top-10 = 23.3%`
  - stronger than `G3` on `Top-1` and `MRR@10`
- `G3` still remains the best AP held-out system if the primary objective is `Top-10`.

### 6. Graph-RAG Reranker Outcome

Graph-RAG result set — three experiments, two shortlist sizes:

| Exp | Path | top_k | Mode |
| --- | --- | ---: | --- |
| Exp A (top10) | `legacy/20260405_g7_formal_v3` | 10 | G7 pipeline |
| Exp A-final (top15) | `20260405_top15_g7_v1` (**canonical**) | 15 | G7 pipeline |
| Exp B (P1-only) | `legacy/20260404_p1_formal_v2` | 10 | P1-only |

#### Exp A (top10) — `legacy/20260405_g7_formal_v3`

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| G7 pipeline | 23.3 | 3.3 | 0.0703 |
| G7 + Graph-RAG rerank | 23.3 | 5.0 | 0.0798 |

Target subset: `G7 Top-10 but not Top-1` (n=12)

| Item | Result |
| --- | --- |
| rescued to Top-1 | `1` (AP_SK_259: rank 5→1) |
| subset MRR@10 | `0.1849 -> 0.2325` |
| improved cases | `1` |
| worsened cases | `5` (AP_SK_002, 108, 149, 160, 173) |
| rerank_failed | `0/60` |

#### Exp A-final (top15, canonical) — `20260405_top15_g7_v1`

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| G7 pipeline | 23.3 | 3.3 | 0.0703 |
| G7 + Graph-RAG rerank | 23.3 | 1.7 | 0.0635 |

Target subset: `G7 Top-15 but not Top-1` (n=17)

| Item | Result |
| --- | --- |
| rescued to Top-1 | `0` |
| subset MRR@10 | `0.1305 -> 0.1358` |
| improved cases | `1` (AP_SK_046: rank 5→2) |
| worsened cases | `5` (AP_SK_108, 149, 160, 173, 316; AP_SK_316 knocked from rank 1→2) |
| rerank_failed | `0/60` |

Key interpretation (Exp A vs A-final):

- At top@10 shortlist the reranker finds a marginally positive result (+1.7pp Top-1, +1 rescued).
- At top@15 shortlist the result is negative (−1.6pp Top-1, 0 rescued, MRR degrades).
- The shortlist size sensitivity indicates that graph-context matching is fragile on the current G7 topology-filtered shortlist and should not be treated as a reliable post-hoc add-on.
- **Exp A-final (top15) is the canonical result used in the thesis.**

#### Exp B (P1-only, top10) — `legacy/20260404_p1_formal_v2`

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| P1-only baseline | 20.0 | 0.0 | 0.0321 |
| P1-only + Graph-RAG rerank | 20.0 | 6.7 | 0.0859 |

Case-level outcome:

| Item | Result |
| --- | --- |
| rescued to Top-1 | `4` (AP_SK_046, 160, 314, 233) |
| improved cases | `4` |
| worsened cases | `1` (AP_SK_022) |
| rerank_failed | `0/60` |

Key interpretation (Exp B):

- The P1-only control is clearly stronger: +6.7pp Top-1, MRR nearly triples (0.0321→0.0859).
- `P1-only + rerank` outperforms `G7 + rerank` on Top-1 after reranking (6.7% vs 1.7%).
- This means current evidence does **not** support a claim that structured spatial extraction makes Graph-RAG reranking more effective; the reranker benefits more from a coarser shortlist.
- The safer conclusion is:
  - graph-context matching can help on a coarse P1-only shortlist
  - but it is not yet an effective post-hoc add-on to the current G7 topology-filtered shortlist
  - therefore, the main optimization direction remains planner/extractor co-design rather than a separate last-mile reranker

---

## What To Drop

- Drop the old `30% oracle ceiling` wording from main thesis text.
  - Use the bug-fixed AP held-out oracle only: `40.0% / 5.0% / 0.1279`.
- Drop the idea of rerunning the old agent baseline on the new benchmark.
  - Low ROI for both implementation and thesis value.
- Drop strong shortcut-learning claims that require image-swap reinference.
  - Current evidence supports `shortcut-like` behavior, not full causal proof.
- Drop the earlier `G7` negative-run wording from thesis-facing notes.
  - That result came from a formal eval mismatch and is no longer valid.
- Drop long infrastructure/orchestrator planning from thesis-facing notes.
  - The useful outcome is the result, not the framework design discussion.
- Drop UI/demo brainstorming from the main Phase 5 summary.
  - Keep only if needed for defense prep, not for Chapter 6.
- Treat unified / cross-IFC discussion as secondary.
  - Main result chain should stay on AP held-out.

---

## Current Progress

- Group 4 post-hoc analysis completed.
  - Outputs: `mscd_demo/output/lora6_v2_ap_20260331/group4_post-hoc_analysis/`
- Planner upgraded to:
  - `multi-chain + multi-anchor + fingerprint-aware filter`
- G7 assembly completed with:
  - `lora_system_g7`
  - `g7_position_context`
  - scale augmentation enabled
- Prompt path bug in `6_assemble_lora6.py` fixed.
- Formal Modal eval path bug in `training/eval.py` fixed.
- `G7_position_context` training and evaluation completed.
- Graph-RAG reranker implemented and formally tested on `G7`.
- Graph-RAG control reranker completed on `P1-only`.
- Current result:
  - Track A recovered (`Hop-1 78.3`, `Pred R 93.1`, `Dir 82.1`)
  - Track B-2 recovered (`Top-10 23.3`, `Top-1 3.3`, `MRR@10 0.0681`)
  - Exp A (top10, legacy): G7 + Graph-RAG rerank mildly positive — Top-1 3.3%→5.0%, MRR 0.0703→0.0798, 1/12 rescued
  - Exp A-final (top15, **canonical**): G7 + Graph-RAG rerank **negative** — Top-1 3.3%→1.7%, MRR 0.0703→0.0635, 0/17 rescued, AP_SK_316 knocked from rank 1
  - `P1-only + Graph-RAG rerank` (Exp B, top10): improved `Top-1` to `6.7%` and `MRR@10` to `0.0859` (4 rescued, 1 worsened)
  - Group 4 post-hoc bundle refreshed with `G7`.