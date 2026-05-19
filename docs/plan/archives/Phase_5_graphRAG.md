# Phase 5 — Minimal Graph-RAG Reranker Experiment

## Summary

This phase adds one high-ROI experiment to the current LoRA6/AP pipeline: a **Graph-RAG reranker** that operates only on an existing shortlist, without modifying the planner, retriever, or extractor.

The goal is not to replace the current pipeline. The goal is to test whether **graph fingerprint context can improve last-mile ranking** once the pipeline has already narrowed the candidate space.

Current AP held-out reference points:

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| P1-only upper bound | 16.7% | 0.0% | 0.0392 |
| G3 pipeline | 26.7% | 1.7% | 0.0641 |
| G7 pipeline | 23.3% | 3.3% | 0.0681 |
| Oracle | 40.0% | 5.0% | 0.1279 |

`G7` is used as the default base system because it is currently the best **early-rank** AP model. Although `G3` still has the highest Top-10 coverage, `G7` has the best Top-1 and MRR, and its extracted `direction` / `object_subtype` signals are more aligned with the graph fingerprint view.

For `G7`, the current bottleneck is:

| Rank bucket | Cases | Interpretation |
| --- | ---: | --- |
| Top-1 | 2 | already solved |
| Top-2 to Top-10 | 12 | primary Graph-RAG target |
| Below Top-10 | 46 | reranker alone cannot rescue these |

Therefore, the maximum possible gain of a top-10 reranker is to convert those `12` cases into Top-1. That gives a theoretical reranker-only Top-1 ceiling of `14/60 = 23.3%`.

This phase implements only two experiments:

| Experiment | Pool source | Purpose |
| --- | --- | --- |
| Exp A: `G7 + Graph-RAG reranker` | G7 pipeline top-10 | measure the value of graph-context reranking after structured extraction |
| Exp B: `P1-only + Graph-RAG reranker` | storey+type top-10 | control group for measuring whether spatial extraction is actually necessary |

Key interpretation:

- `A >> B`: spatial extraction shrinks the pool in a way that makes graph reranking effective.
- `A ≈ B`: coarse graph retrieval plus reranking is already enough; spatial extraction contributes less to final ranking than expected.
- `A` improves little over `G7`: graph descriptions are not enough, or Gemini is weak at evidence-to-fingerprint matching.

## Execution Update — 2026-04-05

Result set — two G7 shortlist sizes plus P1-only control:

- `Exp A (top10, legacy)`: `legacy/20260405_g7_formal_v3`
- `Exp A-final (top15, canonical)`: `20260405_top15_g7_v1`
- `Exp B (P1-only, top10)`: `legacy/20260404_p1_formal_v2`
- script: `mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py`

#### Exp A (top@10, legacy) — `legacy/20260405_g7_formal_v3`

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| G7 pipeline | 23.3% | 3.3% | 0.0703 |
| G7 + Graph-RAG rerank | 23.3% | **5.0%** | **0.0798** |

Target subset: `G7 Top-10 but not Top-1` (n=12)

- reranked Top-1: `1/12` (AP_SK_259: rank 5→1)
- baseline MRR@10: `0.1849` → reranked: `0.2325`
- improved: `1`, worsened: `5` (AP_SK_002, 108, 149, 160, 173)
- rerank_failed: `0/60`

#### Exp A-final (top@15, canonical) — `20260405_top15_g7_v1`

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| G7 pipeline | 23.3% | 3.3% | 0.0703 |
| G7 + Graph-RAG rerank | 23.3% | **1.7%** | **0.0635** |

Target subset: `G7 Top-15 but not Top-1` (n=17)

- reranked Top-1: `0/17`
- baseline MRR@10: `0.1305` → reranked: `0.1358`
- improved: `1` (AP_SK_046: rank 5→2), worsened: `5` (AP_SK_108, 149, 160, 173, 316; AP_SK_316 knocked rank 1→2)
- rerank_failed: `0/60`

**Exp A-final (top@15) is the canonical result used in the thesis.**

Interpretation:

- At top@10 shortlist the reranker shows a marginal positive (+1.7pp Top-1, 1 rescued).
- At top@15 shortlist the result is **negative** (−1.6pp Top-1, 0 rescued, AP_SK_316 knocked from rank 1).
- The shortlist sensitivity indicates that graph-context matching is fragile on the current G7 topology-filtered shortlist.
- The thesis conclusion: graph-RAG reranking and topology planning are **complementary but not interchangeable**.

`Exp B: P1-only + Graph-RAG rerank`

- completed after Neo4j recovery
- output: `mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/20260404_p1_formal_v2/`

Formal `Exp B` result:

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| P1-only baseline (generated) | 20.0% | 0.0% | 0.0321 |
| P1-only + Graph-RAG rerank | 20.0% | 6.7% | 0.0859 |

Case-level outcome:

- improved: `4`
- became Top-1: `4`
- worsened: `1`
- rerank_failed: `0/60`

Changed cases:

- `AP_SK_046`: `3 -> 1`
- `AP_SK_160`: `8 -> 1`
- `AP_SK_314`: `6 -> 1`
- `AP_SK_233`: `8 -> 1`
- worsened:
  - `AP_SK_022`: `6 -> 7`

Final interpretation:

- `Exp A` and `Exp B` together show an asymmetric pattern:
  - `G7 + rerank` helps only slightly
  - `P1-only + rerank` helps much more
- Therefore the current evidence does **not** support a strong claim that structured spatial extraction makes graph reranking more effective.
- The stronger interpretation is:
  - Gemini can sometimes use graph fingerprint descriptions to pick the right candidate from a coarse `storey+type` shortlist
  - but the current `G7` shortlist plus candidate-description design exposes only a small amount of additional useful signal for last-mile reranking
- Thesis-safe takeaway:
  - Graph-RAG is not yet a strong post-hoc add-on to the current G7 shortlist
  - however, graph-context matching itself is not useless, because it improved the `P1-only` control

### Thesis-ready interpretation: why `G7` is stronger overall but weaker as a rerank substrate

The results do not imply that `P1-only` is a stronger system than `G7`. On the contrary, the baseline metrics show that `G7` is the stronger retriever before reranking (`Top-10 23.3%`, `Top-1 3.3%`, `MRR@10 0.0703`) than the generated `P1-only` control (`20.0%`, `0.0%`, `0.0321`). The asymmetry appears only after adding the Graph-RAG reranker. A plausible interpretation from direct case inspection is that the two systems expose different reranking problems. `P1-only` produces a coarse shortlist with larger structural variation across candidates, so graph fingerprints such as host wall, slot position, and door/window neighborhood can sharply distinguish the correct element. `G7`, by contrast, already compresses the pool into harder negatives: many top-10 candidates are same-storey, same-type elements with highly similar local topology, especially repeated window triads on the same facade wall. In that setting, the current candidate descriptions do not separate the remaining alternatives strongly enough for Gemini to resolve the final choice reliably.

The overlap case `AP_SK_160` is especially informative. The same target was rescued from rank `8` to rank `1` in the `P1-only` reranker run, but degraded from rank `3` to rank `4` in the `G7` reranker run. This suggests that the current Graph-RAG module is more effective as a coarse-pool disambiguator than as a hard-negative reranker on top of an already compressed learned shortlist. Therefore, the main system conclusion is not that `G7` is weaker overall, but that the present Graph-RAG design is mismatched to the difficulty profile of the `G7` shortlist.

### Scope decision: whether a Graph-RAG-only pipeline is necessary

A pure Graph-RAG-only pipeline is **not necessary in the current phase**. The present two-experiment comparison already answers the main thesis question: graph-context matching can help, but it helps more on a coarse shortlist than on the current `G7` shortlist. Adding a full Graph-RAG-only pipeline would substantially increase implementation and evaluation cost while providing weaker incremental evidence than the existing `P1-only + rerank` control. It may still be useful later as an appendix or defense follow-up baseline, but it is not required for the core Phase 5 claim.

---

## Implementation

Add one standalone script:

`mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py`

This script is **read-only** over existing artifacts plus Neo4j. It does not rerun the AP pipeline and does not change the default planner.

### Inputs

Default inputs:

- G7 AP trace JSONL  
  `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase5_g7/g7_position_context/traces_20260404_132823_v2_lora_p0_union_p1.jsonl`
- AP held-out cases  
  `mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl`
- Gemini model  
  `gemini-2.5-flash`
- `top_k = 10`

Important join rule:

- Case id must be matched by `trace["scenario"]["id"]` or `trace["scenario_id"]` against `cases["case_id"]`.

Important data-source rule:

- Images and chat must come from `cases_ap_heldout_e2e.jsonl`.
- Do **not** rely on `trace["scenario"]["image_paths"]`.

### Experiment modes

The same script supports two modes.

#### Exp A: `g7_pipeline`

- Read the original candidate order from:
  - `trace["internals"]["retrieval_results"][0]["candidates"]`
- Take the first `top_k = 10` candidate GUIDs.
- This preserves the current G7 shortlist exactly.

#### Exp B: `p1_only`

- Do **not** depend on precomputed AP `p1_only` traces.
- Build the control pool on the fly from Neo4j using only:
  - `storey_name`
  - `ifc_class`
- These values should come from the same G7 trace/case context so the comparison remains aligned to the same input case.
- Sort the resulting P1 pool deterministically using the existing structural order proxy, then take the first `10`.

This keeps Exp B fully comparable while avoiding missing artifact dependencies.

### Graph candidate enrichment

For each case, enrich the selected top-10 candidates with a single batch Neo4j query using `UNWIND $guids AS guid`.

Each candidate description must include:

- candidate letter id
- IFC type
- storey
- name / subtype hint
- host wall name and type if available
- `wall_position_index`
- `wall_child_total`
- left neighbor summary
- right neighbor summary

The description should be compact and human-readable, for example:

`A. IfcWindow on 3rd Floor, position 4 of 17 on MockUp Exterior wall; left: IfcWindow (BALANS 10M PRIVATE); right: IfcWindow (BALANS 10M BATHROOM).`

Do not dump raw graph JSON into the prompt.

### Gemini reranker prompt

Use a short comparative reasoning prompt, not a long chain-of-thought prompt.

Prompt ingredients:

- site image
- floorplan patch
- chat/query text
- the 10 lettered candidate descriptions

Prompt objective:

- identify which candidate best matches the evidence
- return a ranking of the 10 candidates

Final implementation contract:

- return only ranked candidate letters from best to worst
- example: `A C B D E F G H I J`
- no JSON
- no long explanation

Fallback rule:

- if parsing fails or the response is empty, fall back to the original order and mark `rerank_failed = true`

### Reranking semantics

This is a **shortlist reranker**, not a retriever.

For each case:

- rerank only the first 10 candidates
- keep everything after rank 10 unchanged
- final candidate list = `reranked top-10 + untouched tail`

This ensures:

- GT-in-pool does not change
- pool size does not change
- Top-10 remains fixed
- changes are reflected only in `Top-1` and `MRR@10`

---

## Outputs

Write all outputs to:

`mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/<date>/`

Required files:

- `graph_rag_rerank_results.csv`
- `graph_rag_rerank_results.jsonl`
- `graph_rag_rerank_summary.json`
- `graph_rag_rerank_summary.md`
- `graph_rag_rerank_comparison.png`

The summary must report:

| System | Top-10 | Top-1 | MRR@10 |
| --- | ---: | ---: | ---: |
| G7 pipeline | 23.3% | 3.3% | 0.0681 |
| Exp A: G7 + Graph-RAG rerank | 23.3% | ? | ? |
| P1-only upper bound | 16.7% | 0.0% | 0.0392 |
| Exp B: P1-only + Graph-RAG rerank | 16.7% | ? | ? |
| Oracle | 40.0% | 5.0% | 0.1279 |

Required subsets in the report:

- all 60 AP held-out cases
- `g7_top10_not_top1_before`
- per-family breakdown if available:
  - `singleton`
  - `paired:FILLS+NEXT_TO`
  - `triad:FILLS+NEXT_TO+NEXT_TO`
  - `mixed-anchor triads`

---

## Test Plan

### Smoke test

Run 3 manual cases first:

- one `paired:FILLS+NEXT_TO`
- one `triad:FILLS+NEXT_TO+NEXT_TO`
- one non-window predicate case

Verify:

- case join works
- images load from `cases_ap_heldout_e2e.jsonl`
- top-10 candidate list is correct
- Neo4j batch enrichment returns usable context
- Gemini returns parseable JSON
- only top-10 ordering changes

### Full run

Run both Exp A and Exp B on all 60 AP held-out cases.

Required checks:

- all 60 cases produce rows in output
- pool size is identical before and after reranking
- GT-in-pool is identical before and after reranking
- Exp A and Exp B both produce valid summary metrics
- parse failures fall back cleanly and are counted

### Acceptance criteria

The experiment is complete when it provides:

- one reproducible reranked result file for `G7 + reranker`
- one reproducible reranked result file for `P1-only + reranker`
- one direct comparison table covering:
  - `G7 baseline`
  - `G7 + rerank`
  - `P1-only upper bound`
  - `P1-only + rerank`
  - `Oracle`

Primary success signal:

- Exp A increases `Top-1` on the `12` `Top-10 but not Top-1` G7 cases

Secondary success signal:

- Exp A clearly outperforms Exp B, showing that spatial extraction provides useful shortlist compression before graph reranking

---

## Assumptions and Defaults

- Default base system is `G7_position_context`, but the script should accept a different trace JSONL so the same reranker can later be reused for another best G-VLM.
- This phase does **not** modify the current planner.
- This phase does **not** rerun extraction or AP retrieval.
- This phase does **not** implement the pure Graph-RAG baseline.
- `top_k` is fixed to `10` in this round because the thesis question is specifically about last-mile reranking inside the current shortlist.
- Gemini is used only as a reranker over graph-enriched candidate descriptions, not as an open-ended retriever.
- `Top-10` is expected to stay constant; the meaningful metrics are `Top-1`, `MRR@10`, and subset conversion counts.
