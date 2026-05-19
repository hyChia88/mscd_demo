# final_plot_generation plan for slides and thesis

## Goal

Generate the final Chapter 7 result figures for both the thesis PDF and presentation slides. The figures must support the thesis narrative, not just reproduce all experiment outputs.

Core narrative:

1. **RQ1, multimodal grounding:** multimodal grounding works as text-grounded topology extraction. Text anchors storey/class and intent; floorplan/site-image evidence improves topology; visual-only evidence collapses.
2. **RQ2, ontology/graph retrieval:** enriched IFC graph retrieval works when supplied with reliable constraints. The graph preserves IFC-valid candidates and compresses the pool, but noisy fields must not be used as destructive hard filters.
3. **Design principle:** stable fields filter; uncertain fields rerank; low-confidence evidence triggers verification or stays auxiliary.

This task should output a small, polished figure set for thesis and slides, plus source CSV/JSON metric tables for reproducibility.

---

## Source references and paths

Use these paths as the codebase reference points. Keep file paths configurable so the plotting task can run even if outputs move.

### Main evaluation data

- AP held-out cases:
  - `mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl`
- Main plot output directory already used in prior work:
  - `mscd_demo/docs/plots/phase4_lora6_main/`
- Recommended new output directory for this final plotting task:
  - `mscd_demo/docs/plots/final/`
- Recommended new data export directory:
  - `mscd_demo/docs/plots/final/data/`

### Phase 6 / final reranking references

- Graph-RAG reranker:
  - `mscd_demo/evaluation/experiments/graph_rag_rerank_ap.py`
- Graph-RAG prompt:
  - `mscd_demo/prompts/graphrag_rerank.yaml`
- Phase 6 rerank artifacts:
  - `mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/phase6_1_g9_resnet_f4_fused/`
  - `graph_rag_rerank_{comparison.png,summary.json,summary.md,results.csv,results.jsonl}`
- Final Graph-RAG figure currently referenced in thesis:
  - `figures/plots/final/fig05_graph_rag_evidence_dependent.png`

### OpenCV / deterministic visual heuristic references

- Counter implementation:
  - `mscd_demo/src/neurosym/floorplan_counter.py`
- Counter validation:
  - `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py`
- Counter overlay annotations:
  - `mscd_demo/evaluation/h2/annotate_phase6_counter.py`
- Post-hoc count injection:
  - `mscd_demo/evaluation/analysis/inject_floorplan_counts.py`
- Full-storey renders:
  - `data_curation/datasets/synth_v0.5_ap/floorplans_full/`
  - `data_curation/scripts/synth/3c_render_full_storeys.py`

### Size-band / ResNet helper references

- Crop dataset script:
  - `data_curation/scripts/synth/10_build_cluster_crops.py`
- ResNet training:
  - `mscd_demo/training/train_cluster_classifier.py`
- Classifier checkpoint:
  - `mscd_demo/models/cluster_classifier_ap/best.pt`
- Inference wrapper:
  - `mscd_demo/src/neurosym/cluster_classifier.py`
- Size-band injection:
  - `mscd_demo/evaluation/analysis/inject_size_band.py`

### Retrieval backend references

- Constraint schema:
  - `mscd_demo/src/neurosym/types.py`
- Planner:
  - `mscd_demo/src/neurosym/constraints_to_query.py`
- Retrieval executor:
  - `mscd_demo/src/neurosym/retrieval_backend.py`
- Metrics:
  - `mscd_demo/src/neurosym/metrics.py`
- Config:
  - `mscd_demo/config.yaml`

---

## Plotting rules

Use `matplotlib` only.

General rules:

- Generate both `.png` and `.pdf` for each figure.
- Use a consistent figure style across all outputs.
- Keep slide figures simpler than thesis figures.
- Avoid dense legends.
- Avoid showing every model if it weakens the story.
- Avoid raw metric tables with small fonts in slides.
- Do not use the word `Phase` in final figure titles or captions. Code paths can still contain `phase` because they are existing paths.
- Do not claim zero hallucination. Use “IFC-valid candidate grounding” or “reduces hallucination risk.”

Recommended sizes:

- Thesis full-width figure: `figsize=(10, 5.5)` or `figsize=(10, 6)`
- Slide figure: `figsize=(11, 5.5)`
- Summary table for slides: `figsize=(11, 6)`

Recommended font sizes:

- Title: 14 to 16
- Axis labels: 12
- Tick labels: 10 to 11
- Bar annotations: 9 to 10
- Caption text is handled in LaTeX/slides, not inside the image.

Output naming convention:

```text
mscd_demo/docs/plots/final/
  fig00_symbolic_reasoning_trace.png
  fig00_symbolic_reasoning_trace.pdf
  fig01_oracle_symbolic_ceiling.png
  fig01_oracle_symbolic_ceiling.pdf
  fig02_fingerprint_ladder.png
  fig02_fingerprint_ladder.pdf
  fig03_lora_vs_gemini.png
  fig03_lora_vs_gemini.pdf
  fig04_multimodal_alignment_gain.png
  fig04_multimodal_alignment_gain.pdf
  fig05_graph_rag_evidence_dependent.png
  fig05_graph_rag_evidence_dependent.pdf
  fig06_summary_findings_table.png
  fig06_summary_findings_table.pdf

mscd_demo/docs/plots/final/data/
  symbolic_reasoning_trace_counts.csv
  symbolic_reasoning_trace_case.json
  oracle_symbolic_ceiling.csv
  fingerprint_ladder.csv
  lora_vs_gemini.csv
  multimodal_alignment_gain.csv
  graph_rag_evidence_dependent.csv
  summary_findings_table.csv
  plot_manifest.json
```

---

## Shared metric definitions

Use these definitions consistently across all plots.

```text
n_cases              = number of evaluated cases
gt_in_pool_rate     = cases where GT is in candidate pool / n_cases
top1                = cases with rank == 1 / n_cases
top5                = cases with rank <= 5 / n_cases
top10               = cases with rank <= 10 / n_cases
mrr10               = mean(1 / rank if rank <= 10 else 0)
median_pool         = median candidate pool size
avg_pool            = mean candidate pool size
coverage            = cases where the field/fingerprint is available / n_cases
field_accuracy      = exact field matches / cases where field is defined
emission_rate       = cases where model emits field / cases where field is expected
hard_filter_loss    = cases where wrong hard field excludes GT
soft_rerank_gain    = improvement in top1/mrr10 after rerank without changing candidate validity
```

---

## Required plots

Generate the following seven outputs.

The new `fig00` is the main slide-first proof for symbolic / graph reasoning.
It is intentionally process-oriented rather than metric-only. `fig01` and `fig02`
remain the quantitative follow-up visuals.

---

# 0. Symbolic / graph reasoning process trace

## Output name

`fig00_symbolic_reasoning_trace`

## Purpose

Show, with one real held-out case, how the symbolic backend turns typed language
constraints into IFC-valid ranked candidates through an inspectable graph-planning
process.

## Thesis / slide claim

> Symbolic retrieval is not a black-box search layer. It performs typed filtering,
> topology-aware traversal, recall-preserving candidate union, and GUID-grounded
> ranking.

## Required data source

Use one representative AP held-out case from:

- `mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl`

Join that case with the latest available symbolic retrieval traces:

- extracted constraints / end-to-end prediction:
  - `mscd_demo/output/lora6_v2_ap_20260331/g9_resnet_band_f4__ap_eval.jsonl`
- internal retrieval trace with planner route and pool counts:
  - `mscd_demo/output/lora6_v2_ap_20260331/ap_e2e_phase6_1_g9_resnet_band_v2/traces/**/*.trace.json`
- optional rerank comparison for final Top-k ordering display:
  - `mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/phase6_1_g9_resnet_f4_fused/graph_rag_rerank_results.jsonl`

If the latest `phase6_1_g9_resnet_band_v2` trace for the chosen case is missing,
fall back to another AP held-out case with:

- parsed constraints present
- `internals.query_plans` populated
- `internals.retrieval_results` populated
- a non-trivial candidate pool reduction

## Metrics / fields to track

For the selected case, capture:

- `case_id`
- `target_guid`
- `input_text`
- `storey_name`
- `ifc_class`
- `spatial_predicate`
- `direction`
- `host_guid` or host type if available
- `planner_route`
- `strategy_actually_used` or `query_plan_used`
- `candidate_count_all`
- `candidate_count_p1`
- `candidate_count_p0`
- `candidate_count_union`
- `candidate_count_top10`
- `gt_in_pool`
- `gt_rank`
- `top5_guid_list`

Export:

- `symbolic_reasoning_trace_counts.csv`
- `symbolic_reasoning_trace_case.json`

## Items to show in the figure

The figure should include four left-to-right reasoning blocks:

1. `Typed Constraints`
   - storey
   - class
   - predicate
   - direction
   - host / subtype if present
2. `Query Planner`
   - `P1: storey + class`
   - `P0: topology relation`
   - `p0 ∪ p1 recall-preserving union`
3. `Graph Traversal`
   - containment
   - type filtering
   - topology edges
   - enriched fingerprint matching
4. `Ranked IFC Candidates`
   - Top-5 GUIDs
   - highlight GT if present

Below the reasoning trace, add a compact candidate-count funnel:

`All IFC elements -> Storey/Class -> Topology -> Fingerprint -> Top-10`

Use real counts from the selected trace. Show counts under each stage.

## Plot design

Preferred design:

- One combined figure for slides and thesis.
- Main body: left-to-right reasoning trace with boxes and arrows.
- Bottom inset: candidate compression funnel or waterfall.
- Optional small thumbnail or short text snippet from the case on the far left.

Do not render this as a plain performance bar chart. The emphasis is:

- inspectable planner logic
- schema-constrained retrieval
- candidate pool compression
- GUID-grounded output

## Annotations

Main annotation:

`Symbolic retrieval constrains output to IFC-valid GUIDs and makes each filtering step inspectable.`

Optional short slide callouts:

- `Typed filters keep schema validity`
- `Topology narrows the pool`
- `Union preserves recall`
- `Final output is an IFC GUID shortlist`

## Logic to encode

- The point is not that the graph alone solves all retrieval.
- The point is that ontology-backed retrieval is inspectable and schema-valid.
- The figure should make the planner route visible enough that a reviewer can
  understand what information each stage adds.
- If rerank output is shown, label it clearly as an optional final ordering step,
  not as part of the symbolic filter core.

## Figure caption draft

```latex
Reasoning trace for one AP held-out case. Typed constraints from the input are converted into query plans over the IFC graph, combining storey/class filtering with topology-aware retrieval and recall-preserving candidate union. Each stage remains inspectable and produces IFC-valid GUID candidates, while enriched graph fingerprints compress the candidate pool before final ranking.
```

## Slide role

This is the strongest single visual proof for RQ2 in the main deck. Use it before
the oracle ceiling and fingerprint compression plots.

---

# 1. Oracle ladder / symbolic ceiling plot

## Output name

`fig01_oracle_symbolic_ceiling`

## Purpose

Prove that the symbolic graph backend can preserve the ground-truth IFC element when supplied with correct structured constraints.

## Thesis / slide claim

> If extraction were perfect, the graph can preserve the correct GUID and strongly compress the candidate pool.

## Required data source

Use the existing oracle / fingerprint ladder output if available. If not available as CSV, create `oracle_symbolic_ceiling.csv` from the existing Chapter 7 numeric table or experiment output.

Expected input schema:

```csv
level,level_name,fields_active,coverage,n_cases_covered,gt_in_pool_rate,top10,top5,top1,mrr10,median_pool,avg_pool
L0,None,none,1.0,60,1.0,,,,,3200,3189.4
L1,Storey + IFC class,storey_name|ifc_class,1.0,60,1.0,,,,,210,245.1
L2,Topology type,topology_type,0.96,58,1.0,,,,,180,201.3
L3,Enriched fingerprints,direction|subtype|distance|connection_degree,0.88,53,1.0,,,,,9,22.4
L4,Exact position,position_slot,0.43,26,1.0,,,,,6,7.2
```

Use real values from the codebase. The numbers above are placeholders except known thesis-facing values such as 100% GT-in-Pool and L3 median compression where confirmed.

## Metrics to track

- `level`
- `level_name`
- `fields_active`
- `coverage`
- `n_cases_covered`
- `gt_in_pool_rate`
- `top10`
- `mrr10` if available
- `median_pool`
- `avg_pool`

## Items to track

- Which fields are activated at each ladder level.
- Whether the field is directly extractable from current input.
- Whether the field exists in the graph for that case.
- Whether the ground truth survives the filter.
- Candidate pool size after each level.

## Plot design

Preferred design:

- Two-panel figure.
- Panel A: line or bar for `gt_in_pool_rate` and `top10`.
- Panel B: bar chart for `median_pool`, log scale if pool sizes differ by orders of magnitude.

Alternative for slides:

- One slide-friendly panel with `median_pool` bars and a top annotation saying `GT-in-Pool = 100% under oracle constraints`.

## Annotations

Add callouts:

- `Oracle preserves GT`
- `L3: dominant compression from enriched fingerprints`
- `Coverage varies by fingerprint availability` if coverage is less than 100%.

## Logic to encode in plot text or subtitle

- Do not say the graph “solves” retrieval.
- Say it establishes the **symbolic ceiling** under correct constraints.
- Use wording: “backend capacity under correct structured constraints.”

## Figure caption draft

```latex
Oracle symbolic ceiling on the AP held-out benchmark. Under correct structured constraints, the enriched graph planner preserves the ground-truth IFC element while progressively compressing the candidate pool. The main compression appears when enriched fingerprints such as direction, subtype, distance, and connection degree are consumed.
```

---

# 2. Fingerprint ladder / pool compression plot or table

## Output name

`fig02_fingerprint_ladder`

## Purpose

Show which graph-derived fingerprints are actually valuable for retrieval.

## Thesis / slide claim

> The enriched IFC graph is not passive storage; it contributes discriminative structure through spatial fingerprints.

## Required data source

Use the same oracle/fingerprint data as Plot 1, but reframe around field bundles.

Expected input schema:

```csv
level,field_bundle,fields_added,coverage,median_pool,avg_pool,top10,mrr10,compression_ratio,pool_reduction_pct
L1,Attribute baseline,storey_name|ifc_class,1.0,210,245.1,0.28,,1.0,0.0
L2,Topology type,topology_type,0.96,180,201.3,0.31,,1.17,0.14
L3,Enriched fingerprints,direction|subtype|distance|connection_degree,0.88,9,22.4,0.79,,23.3,0.96
L4,Exact slot,position_slot,0.43,6,7.2,0.92,,35.0,0.97
```

Use real values from the codebase.

## Metrics to track

- `median_pool`
- `avg_pool`
- `compression_ratio = L1_median_pool / level_median_pool`
- `pool_reduction_pct = 1 - level_median_pool / L1_median_pool`
- `top10`
- `coverage`

## Items to track

For each ladder level:

- Field bundle name.
- Field list.
- Which graph relation/property provides it.
- Whether it is IFC-native or author-added/enriched.
- Whether it is reliable enough for hard filtering or better as soft evidence.

Suggested item-level mapping:

```text
storey_name        -> text cue + IFC containment, stable hard filter
ifc_class          -> text cue + IFC class, stable hard filter
predicate          -> text/floorplan/site image, topology filter/rank
direction          -> text/floorplan/site image, strong fingerprint
subtype            -> graph-derived relation subtype or element class context, strong fingerprint
distance           -> graph-derived relation metric/tolerance, strong fingerprint but coverage-dependent
connection_degree  -> graph neighborhood structure, strong fingerprint
position_context   -> noisy VLM/OpenCV auxiliary cue, soft rerank preferred
size_band          -> specialist helper cue, soft or hard only above precision threshold
```

## Plot design

Preferred thesis figure:

- Horizontal bar chart of `median_pool` by fingerprint bundle.
- Add small text labels for `fields_added` on the y-axis or as annotations.
- Use log-scale x-axis if needed.

Preferred slide figure:

- 3-step compression visual:

```text
Attribute-only -> Topology type -> Enriched fingerprints
large pool        slightly smaller   strongest compression
```

## Annotations

- Mark L3 as `dominant compression gain`.
- If known value is valid, annotate `45 -> 9 median` for subtype/L3 compression.
- If comparing P1-only to full topology, annotate `+13pp Top-10` where this is the exact current value.

## Logic to encode

- Do not overclaim “raw IFC vs enriched graph” unless direct benchmark exists.
- Claim: enriched graph-derived fields add measurable value beyond coarse IFC attributes.

## Figure caption draft

```latex
Fingerprint ladder showing how graph-derived fields compress the retrieval pool. Storey and IFC class provide a coarse attribute baseline, while enriched topology fingerprints provide the main discriminative gain. The result supports the value of the ontology-derived graph as an active retrieval structure rather than a passive IFC storage layer.
```

---

# 3. LoRA vs Gemini extraction / retrieval comparison

## Output name

`fig03_lora_vs_gemini`

## Purpose

Show that domain fine-tuning matters for AEC spatial language and schema mapping.

## Thesis / slide claim

> Zero-shot VLM sees broad objects, but fine-tuning learns the spatial fields required by graph retrieval.

## Required data source

Use existing model-comparison / Track A and Track B outputs. Create `lora_vs_gemini.csv`.

Expected input schema:

```csv
model,model_family,training_condition,ifc_class_acc,storey_acc,predicate_acc,direction_acc,spatial_relation_acc,field_macro_acc,gt_in_pool_rate,top1,top5,top10,mrr10,median_pool,notes
Gemini_v2,zero_shot,zero_shot,,,,,,,,,,,,
G3,lora,finetuned,,,,,,,,,,,,
G7,lora,finetuned,,,,,,,,,,,,
G8,lora,finetuned,,,,,,,,,,,,
```

Include only the models needed for the story in the final plot.

Recommended final plot models:

- `Gemini_v2`
- `G3`
- `G8`

Optional backup plot models:

- `G7`
- `G9`
- `G8 + OpenCV F4`
- `G8 + OpenCV F4 + single-shot rerank`

## Metrics to track

Track A extraction:

- `storey_acc`
- `ifc_class_acc`
- `predicate_acc`
- `direction_acc`
- `spatial_relation_acc`
- `field_macro_acc`

Track B retrieval:

- `top1`
- `top10`
- `mrr10`
- `gt_in_pool_rate`
- `median_pool`

## Items to track

For each model:

- Model name.
- Whether zero-shot or fine-tuned.
- Which input modalities it receives.
- Whether it emits fields that are consumed by graph retrieval.
- Whether high extraction score translates to high retrieval score.

## Plot design

Preferred figure:

- Two panels.
- Panel A: extraction fields as grouped bars.
  - metrics: `ifc_class_acc`, `direction_acc`, `predicate_acc`
- Panel B: downstream retrieval bars.
  - metrics: `top1`, `top10`, `mrr10`

Slide simplification:

- Show only `direction_acc` and `mrr10` or `top10`.
- Emphasize `Gemini direction near zero` and `G8 direction high` if current numbers support this.

## Annotations

- Above Gemini direction bar: `weak spatial direction`.
- Above G8 direction bar: `domain-tuned spatial field`.
- Between panels: `Track A extraction != Track B retrieval`.

## Logic to encode

- Fine-tuning is not claimed as universally better for all tasks.
- Claim it improves the specific structured spatial fields needed for this retrieval pipeline.

## Figure caption draft

```latex
Comparison between zero-shot Gemini and fine-tuned G-series extractors on the AP held-out benchmark. Fine-tuning improves the spatial fields required by graph retrieval, especially direction and relational constraints. The downstream retrieval panel shows that extraction quality and GUID retrieval must be evaluated as coupled but distinct stages.
```

---

# 4. Multimodal alignment learning gain / modality ablation plot

## Output name

`fig04_multimodal_alignment_gain`

This replaces or reframes the TODO figure currently described as:

```latex
% TODO: This figure change to mutlimodal alignenet learning gain performance
```

## Purpose

Show what multimodal grounding actually depends on.

## Thesis / slide claim

> Multimodal grounding works, but it is text-grounded topology extraction rather than pure visual reasoning.

## Required data source

Use modality ablation outputs. Create `multimodal_alignment_gain.csv`.

Expected input schema:

```csv
condition,uses_text,uses_floorplan,uses_site_image,uses_4d_metadata,direction_acc,predicate_acc,spatial_relation_acc,top1,top10,mrr10,notes
text_only,1,0,0,0,,,,,,
text_floorplan,1,1,0,0,,,,,,
text_siteimage,1,0,1,0,,,,,,
full_multimodal,1,1,1,0,,,,,,
visual_only,0,1,1,0,,,,,,
full_plus_4d,1,1,1,1,,,,,,
```

Use current values from experiment outputs. Known thesis-facing values from the current writing can be used if verified:

- Text-only direction around `29%`
- Full multimodal direction around `82%`
- Visual-only collapses

## Metrics to track

Extraction / alignment:

- `direction_acc`
- `predicate_acc`
- `spatial_relation_acc`

Downstream retrieval:

- `top1`
- `top10`
- `mrr10`

Optional:

- `schema_valid_rate`
- `emission_rate` for spatial fields

## Items to track

For each condition:

- Whether text is present.
- Whether floorplan patch is present.
- Whether site photo is present.
- Whether 4D metadata is present.
- Which spatial fields improve or collapse.
- Whether the improvement reaches retrieval metrics or only extraction metrics.

## Plot design

Preferred figure:

- Grouped horizontal bars.
- y-axis: modality condition.
- x-axis: accuracy or retrieval metric.
- Show 2 or 3 metrics only:
  - `direction_acc`
  - `predicate_acc` or `spatial_relation_acc`
  - `top10` or `mrr10`

Slide simplification:

- Use only 3 conditions:
  - `Text only`
  - `Full multimodal`
  - `Visual only`
- Show `direction_acc` as the main metric.
- Add small callout: `text anchors meaning; vision improves topology`.

## Annotations

- On text-only: `semantic anchor`.
- On full multimodal: `best topology extraction`.
- On visual-only: `collapse`.
- If 4D adds no gain, annotate: `4D metadata: no measurable gain in this setting`.

## Logic to encode

- The figure is not meant to prove pure image understanding.
- It proves that spatial grounding is distributed across modalities.
- Text is not a weakness here; it matches real site-reporting workflows where workers verbally specify floor and object type.

## Figure caption draft

```latex
Multimodal alignment gain under modality ablation. Text provides the semantic anchor for storey, class, and intent, while floorplan and site-image evidence improve topology extraction. Visual-only evidence collapses, indicating that the current system performs text-grounded topology extraction rather than pure visual spatial reasoning.
```

---

# 5. Final Graph-RAG reranking plot with OpenCV evidence

## Output name

`fig05_graph_rag_evidence_dependent`

## Purpose

Show that Graph-RAG is evidence-dependent rather than universally beneficial.

## Thesis / slide claim

> Reranking helps only when it receives new discriminative evidence not already consumed by the symbolic planner.

## Required data source

Use the fused Graph-RAG summary at
`mscd_demo/output/lora6_v2_ap_20260331/graph_rag_rerank/phase6_1_g9_resnet_f4_fused/graph_rag_rerank_summary.json`
and create `graph_rag_evidence_dependent.csv`.

Expected input schema:

```json
{
  "modes": {
    "full_topology": {
      "baseline": {"n": 60, "top10_pct": 26.7, "top1_pct": 6.7, "mrr10": 0.1041, "avg_pool": 97.4},
      "reranked": {"n": 60, "top10_pct": 26.7, "top1_pct": 8.3, "mrr10": 0.1244, "avg_pool": 97.4}
    }
  },
  "subsets": {
    "full_topology_topk_not_top1": {
      "n": 12,
      "baseline": {"top1_pct": 0.0, "mrr10": 0.1873},
      "reranked": {"top1_pct": 25.0, "mrr10": 0.4135}
    }
  }
}
```

The current fused trace is the authoritative source for this figure, even though
the embedded mode labels still mention earlier G7 naming.

## Metrics to track

- `top1`
- `top10`
- `mrr10`
- `condition`
- `n_cases`
- `avg_pool`
- `subset_top1`
- `subset_mrr10`

## Items to track

For each condition:

- Whether the row is baseline or fused reranked.
- Whether Top-1 improves.
- Whether MRR@10 improves.
- Whether Top-10 is unchanged, which indicates reranking changes order, not candidate recall.
- How the `Top-k but not Top-1` subset behaves after reranking.

## Plot design

Preferred figure:

- Two panels:
  - Panel A: `Top-10` and `Top-1` for `Baseline` vs `Fused reranked`.
  - Panel B: `MRR@10` for `Baseline` vs `Fused reranked`.
- Add a subset annotation using `subsets.full_topology_topk_not_top1`.

## Annotations

- Callout: `Top-10 unchanged: reranking changes order, not recall`.
- Subset note: `Among 12 cases already in Top-10 but not Top-1, reranking rescues 3 to rank 1`.

## Logic to encode

- Do not say Graph-RAG is always helpful.
- Do not say chain-of-thought is the source of improvement.
- The key insight is evidence availability plus better ordering within an already valid pool.
- The figure supports filter-rerank decoupling.

## Figure caption draft

```latex
Evidence-dependent Graph-RAG reranking on the AP held-out benchmark. On the latest fused rerank trace, Top-10 remains unchanged at 26.7\%, while Top-1 improves from 6.7\% to 8.3\% and MRR@10 rises from 0.1041 to 0.1244. The strongest effect appears on cases where the correct candidate is already in Top-10 but not yet ranked first, where reranking rescues 3 of 12 cases to Top-1. This indicates that the gain comes from better ordering within a valid pool rather than universal recall improvement.
```

## Thesis path replacement

Replace this placeholder:

```latex
figures/plots/[augmented_rerank_comparison].png
```

with:

```latex
figures/plots/final/fig05_graph_rag_evidence_dependent.png
```

---

# 6. Summary findings table

## Output name

`fig06_summary_findings_table`

## Purpose

Provide the final Chapter 7 synthesis for thesis and slides.

## Thesis / slide claim

> The strongest system behavior emerges when perception, symbolic retrieval, and reranking are assigned field-specific responsibilities.

## Required data source

Create `summary_findings_table.csv` manually from final thesis wording.

Recommended thesis CSV:

```csv
section,finding,interpretation,rq_link
System baselines,V2 structured retrieval is more repeatable and efficient than V1,GUID-level grounding requires typed constraints and deterministic retrieval rather than open-ended agent reasoning alone,RQ1|RQ2
Oracle ceiling,The oracle planner preserves the ground-truth element under correct constraints,The symbolic graph layer is viable when extracted constraints are reliable,RQ2
Fingerprint ladder,L3 fingerprints provide the dominant compression gain,Enriched graph fields add active retrieval value beyond storey and IFC class,RQ2
Planner strategy,p0 union p1 preserves recall better than hard intersection,Spatial relations should guide retrieval as recall-preserving evidence rather than brittle exclusion filters,RQ2
LoRA vs Gemini,Fine-tuned G-series models outperform zero-shot Gemini on spatial fields,AEC-specific spatial language benefits from domain supervision,RQ1
Track A / Track B,Higher extraction accuracy does not always produce better retrieval,The pipeline must be evaluated as coupled stages rather than a single model score,RQ1|RQ2
Modality ablation,Text is load-bearing while floorplan and site image improve topology,The system performs text-grounded topology extraction rather than pure visual reasoning,RQ1
Deterministic visual heuristics,OpenCV-derived ordinal cues and preliminary size-band classification supplement weak VLM visual fields,Counting and size estimation are better handled as helper signals than free-form VLM predictions,RQ1
Filter-rerank decoupling,Noisy fields can harm retrieval as hard filters but remain useful as soft evidence,Stable fields should filter; uncertain fields should rank; low-confidence fields should trigger verification,RQ2
Graph-RAG reranking,Reranking improves only when supplied with additional discriminative evidence,Graph-RAG is evidence-dependent rather than universally beneficial,RQ2
Overall,Best behavior emerges from field-specific role assignment,The contribution is a field-routed neuro-symbolic interpreter layer,RQ1|RQ2
```

Recommended slide CSV, shorter:

```csv
finding,meaning
Oracle graph preserves GT,Symbolic backend is viable under correct constraints
Fingerprints compress pools,Enriched graph adds active retrieval value
LoRA beats Gemini,Domain tuning matters for AEC spatial language
Visual-only collapses,Grounding is text-grounded topology extraction
Reranking is evidence-dependent,Soft evidence helps only when it adds new information
```

## Metrics to track

This is a synthesis table, not a metric plot. Track:

- `section`
- `finding`
- `interpretation`
- `rq_link`
- `slide_include` boolean if producing separate slide version

## Items to track

- Ensure every table row links to an existing plot or result section.
- Ensure OpenCV/ResNet are named as deterministic visual heuristics or helper signals, not as a full separate contribution.
- Ensure RQ1 and RQ2 are both explicitly represented.

## Plot/table design

Thesis version:

- 4 columns:
  - Chapter 7 section
  - Evaluation finding
  - Thesis interpretation
  - RQ link
- Render as LaTeX longtable in thesis if possible.
- If using matplotlib rendered table, use small font and wrap text.

Slide version:

- 2 columns only:
  - Result
  - Meaning
- Maximum 5 rows.
- Very large font.

## Logic to encode

- The table is the final synthesis.
- It should not introduce new numbers.
- It should not include weak or speculative results as major claims.

---

## Optional backup figures

These are useful for backup slides or appendix only. Do not include in the main presentation unless needed.

---

# Backup A. OpenCV counting evaluation

## Output name

`backup_opencv_counting_results`

## Purpose

Support the claim that deterministic visual heuristics can recover ordinal cues, but production use depends on wall-frame localization.

## Data source

- `mscd_demo/evaluation/h2/validate_phase6_ap_canonical_counts.py`
- Existing note reports:
  - F3 exact match: `13.3%`
  - F4 exact match: `80.0%`
  - scoped set: 15 cases

## Metrics to track

- `mode`: F3 or F4
- `exact_count_accuracy`
- `n_cases`
- `n_correct`
- `wall_frame_known`
- `vision_only`
- `failure_type`

## Plot design

- Two bars only: F3 vs F4 exact count accuracy.
- Add note: `F4 = oracle wall bounds, not production path`.

## Caption draft

```latex
OpenCV counting evaluation on the scoped AP subset. The production-like F3 counter reaches 13.3\% exact count accuracy, while the oracle-bound F4 counter reaches 80.0\%. The gap shows that ordinal evidence is recoverable, but depends on wall-boundary localization and coordinate-frame alignment.
```

---

# Backup B. Size-band classifier result

## Output name

`backup_size_band_classifier_results`

## Purpose

Support the helper-signal claim for size estimation without overemphasizing it as a main contribution.

## Data source

- `mscd_demo/training/train_cluster_classifier.py`
- `mscd_demo/models/cluster_classifier_ap/best.pt`
- The current diagnostic note reports:
  - crop dataset: 195 crops
  - train/val/test: 155 / 17 / 23
  - ResNet size-band test accuracy: `82.6%`
  - G9 size-band apples-to-apples: `55.3%`
  - improvement: `+27pp`

## Metrics to track

- `model`
- `task`: size-band classification
- `n_classes`: 6
- `n_train`
- `n_val`
- `n_test`
- `test_accuracy`
- `macro_f1` if available
- `confusion_matrix`
- `confused_class_pairs`

## Plot design

- Small bar chart: G9 VLM vs ResNet-18 size-band accuracy.
- Optional confusion matrix as appendix, not main deck.

## Caption draft

```latex
Preliminary size-band classification result on AP element crops. Reframing size estimation as a specialist visual classification task improves coarse size-band accuracy relative to VLM prediction, suggesting that size cues are better treated as auxiliary helper signals than as free-form VLM metric outputs.
```

---

# Backup C. G9 hard-filter regression / filter-rerank decoupling

## Output name

`backup_filter_rerank_decoupling`

## Purpose

Explain why noisy fields must not be placed behind destructive equality filters.

## Data source

Use G9 diagnosis table:

```csv
case_group,count,effect
no_size_cluster_emitted,22,neutral
correct_size_cluster,12,gain
wrong_size_cluster,26,loss_gt_excluded
```

## Metrics to track

- `case_group`
- `count`
- `effect`
- `retrieval_impact`

## Plot design

- Stacked bar or waterfall:
  - neutral: 22
  - gain: +12
  - loss: -26
- Show final message: wrong hard filter excludes GT.

## Caption draft

```latex
Diagnostic breakdown of the size-cluster hard-filter regression. Correct predictions narrow the candidate pool, but wrong predictions remove the ground-truth element entirely. This motivates filter-rerank decoupling: uncertain perceptual fields should be used as soft ranking evidence unless their precision is high enough for hard filtering.
```

---

# Backup D. Field-level extraction heatmap

## Output name

`backup_field_level_accuracy_heatmap`

## Purpose

Show which fields are stable and which fields remain weak.

## Metrics to track

Rows:

- model or condition: Gemini, G3, G7, G8, G9 if relevant

Columns:

- storey
- ifc_class
- target_name_keyword / target_description
- predicate
- direction
- position_context emitted
- position_context exact
- spatial_relations
- size_cluster or size_band
- width / height if still shown as deprecated

## Plot design

- Heatmap with values shown as percentages.
- Use for Q&A only.

---

## Plot manifest

Generate a `plot_manifest.json` with this structure:

```json
{
  "generated_at": "YYYY-MM-DDTHH:MM:SS",
  "output_dir": "mscd_demo/docs/plots/final",
  "source_cases": "mscd_demo/evaluation/cases/cases_ap_heldout_e2e.jsonl",
  "figures": [
    {
      "id": "fig01_oracle_symbolic_ceiling",
      "png": "mscd_demo/docs/plots/final/fig01_oracle_symbolic_ceiling.png",
      "pdf": "mscd_demo/docs/plots/final/fig01_oracle_symbolic_ceiling.pdf",
      "data": "mscd_demo/docs/plots/final/data/oracle_symbolic_ceiling.csv",
      "slide_use": true,
      "thesis_use": true,
      "rq": ["RQ2"],
      "claim": "The symbolic graph can preserve the correct IFC element under correct constraints."
    }
  ]
}
```

Include all seven main figures plus any backup figures generated.
Include all seven main figures plus any backup figures generated.

---

## Suggested script structure

Create one plotting entry point under the existing analysis directory.

```text
mscd_demo/evaluation/analysis/
  generate_final_plot_suite.py
```

Command:

```bash
python mscd_demo/evaluation/analysis/generate_final_plot_suite.py \
  --output-dir mscd_demo/docs/plots/final \
  --data-dir mscd_demo/docs/plots/final/data
```

The script should:

1. Load or assemble CSV inputs.
2. Validate required columns.
3. Generate `.png` and `.pdf` for each figure.
4. Write `plot_manifest.json`.
5. Print a concise summary of generated files.

---

## Validation checks before finalizing

Run these checks after generating figures.

### General checks

- [ ] All seven main plots exist as `.png` and `.pdf`.
- [ ] All data CSVs exist, plus the `fig00` case JSON export.
- [ ] `plot_manifest.json` exists.
- [ ] No figure title contains `Phase 5` or `Phase 6`.
- [ ] No figure claims `zero hallucination`.
- [ ] Figures are readable at slide size.
- [ ] Thesis figures are readable at full-width PDF scale.

### Logic checks

- [ ] Symbolic reasoning trace answers RQ2: inspectable graph-planning process.
- [ ] Oracle plot answers RQ2: graph backend capacity.
- [ ] Fingerprint plot answers RQ2: enriched graph value.
- [ ] LoRA vs Gemini plot answers RQ1: domain tuning for spatial extraction.
- [ ] Modality plot answers RQ1: text-grounded topology extraction.
- [ ] Graph-RAG plot answers RQ2: evidence-dependent reranking.
- [ ] Summary table ties all key findings to RQ1/RQ2.

### Data consistency checks

- [ ] `n_cases` is clearly shown or stored for every metric table.
- [ ] Top-1, Top-10, and MRR@10 are computed from the same scoring definition within each plot.
- [ ] Do not mix legacy full-pool GT-in-Pool with top-10 membership without labeling them separately.
- [ ] If `coverage` differs by level, display it or mention it in caption.
- [ ] If F4 is used, label it as oracle-bound/demo/training ceiling, not production path.

---

## Final slide mapping

Use these outputs in the defense slide result section:

1. **Evaluation setup**: no metric plot; simple pipeline diagram.
2. **Symbolic graph reasoning**: `fig00_symbolic_reasoning_trace`
3. **Symbolic ceiling**: `fig01_oracle_symbolic_ceiling`
4. **Enriched graph value**: `fig02_fingerprint_ladder`
5. **Learned extraction**: `fig03_lora_vs_gemini`
6. **Multimodal grounding**: `fig04_multimodal_alignment_gain`
7. **Evidence-dependent reranking**: `fig05_graph_rag_evidence_dependent`
8. **Key findings**: slide-condensed version of `fig06_summary_findings_table`
9. **Limitations and future work**: no dense plot required; optionally use backup field heatmap in Q&A.

---

## Final thesis mapping

Recommended LaTeX paths:

```latex
figures/plots/final/fig00_symbolic_reasoning_trace.png
figures/plots/final/fig01_oracle_symbolic_ceiling.png
figures/plots/final/fig02_fingerprint_ladder.png
figures/plots/final/fig03_lora_vs_gemini.png
figures/plots/final/fig04_multimodal_alignment_gain.png
figures/plots/final/fig05_graph_rag_evidence_dependent.png
```

For the summary, prefer the LaTeX `longtable` version in Chapter 7 rather than a rendered image. Use the CSV as the source of truth.

---

## Priority order for Codex

1. Build the representative-case exports for `fig00_symbolic_reasoning_trace`.
2. Build data CSVs for the remaining six main figures.
3. Generate `fig05_graph_rag_evidence_dependent` first because it replaces current placeholder paths.
4. Generate `fig04_multimodal_alignment_gain` to satisfy the thesis TODO.
5. Generate `fig00`, `fig01`, and `fig02` as the graph/ontology proof sequence.
6. Generate `fig03` for learned extraction.
7. Generate `fig06` table and slide-condensed version.
8. Generate backup plots only if time remains.

---

## Non-goals

Do not do the following in this task:

- Do not rerun model training.
- Do not change retrieval logic.
- Do not add new thesis claims.
- Do not generate a standalone ResNet main figure unless the user explicitly asks.
- Do not include all intermediate development phases in the main slides.
- Do not use the term `Phase 6` in final thesis/slides figure titles.
