# Final Plot Visual Redesign Notes

## Goal

Make the slide deck feel deliberate, varied, and easier to read at a glance.
The current suite has good data, but too many figures still rely on grouped bar
charts. That makes the visuals feel repetitive and hides the actual story shape.

The redesign should stay:

- tight
- punchy
- slide-first
- low text
- visually differentiated across figures

---

## What the Data Wants

### `fig00_symbolic_reasoning_trace`

Data shape:

- one selected case
- process stages
- candidate-count funnel
- ranked GUID shortlist

Best visual:

- keep as a reasoning trace / flow figure
- this is already the strongest non-bar visual

Do not convert this into a bar chart.

---

### `fig01_oracle_symbolic_ceiling`

Data shape:

- 5 ordered ladder levels
- one pool-size curve
- two retrieval curves
- strong monotonic improvement

Current issue:

- bars + line still reads like a standard benchmark chart

Better visual:

- use a **ladder / step progression** view
- left axis: `Top-10` and `Top-1` as step lines
- right-side annotations at each ladder rung:
  - median pool
  - active fingerprint
- emphasis should be:
  - `76 -> 40 -> 28 -> 6 -> 1`
  - `3.3 -> 8.3 -> 24.2 -> 58.3 -> 100`

Slide message:

`Better fingerprints collapse the pool and lift retrieval together.`

---

### `fig03_lora_vs_gemini`

Data shape:

- 7 systems
- 4 metrics
- only one model is actually the “best learned row”

Current issue:

- grouped bars plus secondary line is dense
- too many colors compete

Better visual:

- use a **horizontal dot-matrix / lollipop matrix**
- one row per model
- three aligned metric columns:
  - `GT-in-pool`
  - `Top-10`
  - `MRR@10`
- optionally drop `Top-1` from the main slide because it is sparse and noisy
- highlight only:
  - `Gemini`
  - `G7`
  - `G9`
  - `Oracle`
- push G3/G4/LoRA5 into lighter neutral styling if retained

Slide message:

`Domain-tuned models beat Gemini; G9 is the strongest learned system.`

---

### `fig04_multimodal_alignment_gain`

Data shape:

- 6 modality slices x 3 models
- extra field-level metrics for G9
- strongest story is directional survival and richer emitted cues

Current issue:

- the full panel still risks feeling like “many bars again”

Better visual:

- main slide should use the tight version only
- use a **two-zone metric strip**
  - left zone: model comparison on `Predicate`, `All SR`, `Direction`
  - right zone: G9-only cues on `Pos emit`, `Pos exact`, `Size exact`
- keep this one compact and punchy

Slide message:

`Direction survives in learned MM models; G9 adds usable position/size cues.`

---

### `fig05_graph_rag_evidence_dependent`

Data shape:

- two before/after pairs
- tiny row count
- the story is change, not absolute magnitude

Current issue:

- bars hide the important thing:
  the rerank delta

Better visual:

- replace with a **dumbbell / before-after slope chart**
- one row for `P1-only`
- one row for `G9 fused`
- use:
  - `Top-1`
  - `MRR@10`
- annotate deltas directly:
  - `0.0 -> 6.7`
  - `6.7 -> 8.3`
  - `0.0321 -> 0.097`
  - `0.1041 -> 0.1244`
- show `Top-10 unchanged` as a side note, not a full bar

Slide message:

`Rerank helps by reordering, not by expanding recall.`

---

### `fig07_retrieval_pipeline_comparison`

Data shape:

- 5 pipelines
- 3 metrics
- one best overall system

Current issue:

- very similar to `fig03`, so it feels duplicative

Better visual:

- use a **ranked podium / ordered dot strip**
- sort by `MRR@10`
- show each pipeline as one horizontal lane
- encode:
  - dot position = `MRR@10`
  - small badges = `Top-10`, `Top-1`
- keep only 5 rows

Alternative:

- convert this into a compact **scorecard table** for slides

Slide message:

`Best overall retrieval comes from the fused G9 + Graph-RAG pipeline.`

---

## Recommended Visual Families

Do not let every slide use a different random chart type.
Use only 4 families across the deck:

1. **Process trace**
   - `fig00`
2. **Ladder / step progression**
   - `fig01`
3. **Dot or lollipop matrix**
   - `fig03`
   - `fig07`
4. **Before/after dumbbell**
   - `fig05`

`fig04` should stay as a compact annotated strip, not a full benchmark chart.

---

## Where an N×N Panel Helps

Use an N×N panel only when the story is **case diversity**, not aggregate scores.

Best use:

- symbolic graph case gallery
- rerank rescue examples
- failure-mode comparison

Good candidate:

### `fig00b_symbolic_case_grid`

- 2x3 panel
- six held-out cases
- each tile shows:
  - short query
  - `All -> P0 -> P0∪P1 -> GT rank`
  - whether GT is rescued by rerank

Why this works:

- one slide can show the symbolic backend is not a one-off anecdote
- avoids repeating the same aggregate metric bars

Not recommended:

- using an N×N panel for model-vs-metric summary tables
- those are better as dot matrices

---

## Slide-First Redesign Proposal

If we want the final deck to feel tight, the best main set is:

1. `fig00_symbolic_reasoning_trace_tight`
2. redesigned `fig01` as ladder / step view
3. redesigned `fig03` as dot matrix
4. `fig04_multimodal_alignment_gain_tight`
5. redesigned `fig05` as dumbbell / slope view
6. redesigned `fig07` as ranked dot strip

This gives six visually distinct slides without becoming chaotic.

---

## Implementation Order

1. Redesign `fig05` first
   - smallest data
   - biggest gain from switching away from bars
2. Redesign `fig03`
   - this is currently the most benchmark-looking slide
3. Redesign `fig01`
   - convert to a proper ladder / ceiling view
4. Redesign `fig07`
   - make it distinct from `fig03`
5. Add optional `fig00b_symbolic_case_grid`

---

## Main Recommendation

Do **not** make every figure more complex.
Instead:

- keep one process slide
- keep one compact metric strip
- turn the benchmark-style comparisons into dot / slope charts
- use an N×N panel only for multi-case qualitative proof

That will make the deck feel tighter, more intentional, and much less like
repeated experiment plots.
