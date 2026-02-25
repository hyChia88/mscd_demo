# Mid-Review Slides Draft
**Cross-Modal Alignment, Schema Mapping, and Compliance**
**MSCD Demo — Multi-modal Site-to-BIM Disambiguation**

> Goals: (1) Progress review (2) Emphasize contribution (3) Method sharing

---

## SLIDE 1 — Title

**Cross-Modal Spatial Mapping and Grounding for AEC Site Inspection**
*VLM-Assisted Constraint Retrieval and Structured Data Management*

- Student: [Your Name]
- Advisor: [Advisor Name]
- Carnegie Mellon University — [Program]
- February 2026 — Mid-Review

---

## SLIDE 2 — Motivation: The "Which Window?" Problem

**The core problem in one sentence:**
> A site inspector sends a photo and a chat message. Which BIM element does it refer to?

**Concrete numbers (AdvancedProject IFC, 10-storey office):**

| Query | Candidates | Precision |
|-------|-----------|-----------|
| "Which window?" (text only) | **263** | 0.38% |
| + floor, task status, images (4D context) | **3** | 33.33% |

→ **98.9% search space reduction** from a single additional context layer

**The gap:** Site language is informal, deictic, and visual. BIM is structured, typed, and geometric.
No existing system bridges the two without expert intervention.

> *Speaker note: Open with this as a live demo anecdote — "An inspector said 'look at this window'. There are 263 windows in the model."*

---

## SLIDE 3 — Research Landscape & Gap

**What exists:**
- BIM authoring tools (Revit, ArchiCAD) — structured, expert-only
- LLM-based NLP for construction (GPT-4, Gemini) — text reasoning, no IFC grounding
- VLM for visual inspection — image understanding, no schema mapping
- BIM-to-NLP (e.g., rule-based IFC querying) — rigid, not multimodal

**The gap:**
- No system takes **informal multimodal site input** (chat + site photos + floorplan + 4D schedule)
- and outputs a **grounded, validated BIM element** (IFC GUID + structured JSON)
- with **explainable, reproducible retrieval**

**Research position:** Interpreter layer between messy site reality and formal BIM model.

---

## SLIDE 4 — Research Questions

| RQ | Focus | Key Challenge |
|----|-------|---------------|
| **RQ1** | Multimodal grounding: can visual + spatial context identify the correct IFC element? | Attribute entropy: 263 windows look the same to a text model |
| **RQ2** | Schema mapping: can the system output a standards-compliant structured report? | CORENET-X submission schema validation (AEC regulatory context) |
| **RQ3** | Abductive reasoning: can the system detect when the element cannot be identified? | Escalation vs false positive — when to say "I don't know" |

**Unified claim:** A constraints-driven, multimodal interpreter layer can close the precision gap
between informal site inspection reports and formal BIM data models.

---

## SLIDE 5 — Research Method: Constructive Design Research

**Methodology:** Constructive Design Research (Koskinen et al.)
- Build a functional prototype as the primary research vehicle
- Evaluate it quantitatively against controlled synthetic benchmarks
- Reflect on system behavior to generate design knowledge

**Prototype:** MSCD Demo
- Two pipelines (V1 agent-driven, V2 constraints-driven)
- Shared IFC retrieval backend, evaluation framework, output contracts
- Controlled synthetic dataset (synth_v0.3, synth_v0.4) for reproducible experiments

**Why synthetic data?**
Real site inspection reports are confidential and unstructured. Synthetic cases give:
- Ground truth IFC GUID labels
- Controlled modality ablation (text/image/floorplan/4D)
- Reproducible difficulty tiers (T1/T2/T3)

---

## SLIDE 6 — System Architecture Overview

**[IMAGE: `docs/diagram/system_architecture_2_simplify.png`]**

**Three layers:**
1. **Input Layer** — chat history, site photos, floorplan patch, 4D project metadata
2. **Pipeline Layer** — V1 (ReAct Agent) or V2 (Constraints-Driven) — both produce `EvalTrace`
3. **Shared Backend** — IFCEngine (IfcOpenShell + spatial index), Neo4j graph, CLIP visual aligner

**Key design principle:** Pipelines are swappable; backend and evaluation contracts are shared.
This enables controlled A/B comparison between V1 and V2.

---

## SLIDE 7 — V1 Pipeline: Agent-Driven Baseline

**[IMAGE: `docs/diagram/sequence_v1_pipeline.png`]**

**Architecture:** LangGraph ReAct Agent + MCP (Model Context Protocol)

```
Input Case → Gemini 2.5 Flash (ReAct agent)
           → calls MCP tools freely (search_by_type, get_by_storey, match_image)
           → IFCEngine + optional CLIP reranker
           → EvalTrace
```

**Strengths:** Flexible, requires no training, handles edge cases via free-form reasoning

**Weaknesses:**
- Non-deterministic — same input can give different retrieval paths
- Hard to ablate individual factors (cannot isolate modality contribution)
- High latency (~8 min / 84 cases vs ~4 min for V2)
- Prompt-sensitive: agent reasoning varies with phrasings

**Role:** Baseline for comparison. V2 is designed to fix these weaknesses.

---

## SLIDE 8 — V2 Pipeline: Constraints-Driven (Key Contribution)

**[IMAGE: `docs/diagram/sequence_v2_pipeline.png`]**

**Pipeline:**
```
Input Case
  → ConditionMask         (apply modality ablation: MA/MB/MC × 4D±)
  → ImageParser (VLM)     (Gemini 2.5 Flash: structured image descriptions, cached)
  → ConstraintsExtractor  (LLM prompt OR LoRA adapter → JSON constraints)
  → QueryPlanner          (5-priority deterministic template: storey+type → storey → type → keyword → fallback)
  → RetrievalBackend      (memory spatial index OR Neo4j Cypher + optional CLIP rerank)
  → EvalTrace + V2Trace
```

**Extracted constraints schema (`src/v2/types.py`):**
```json
{
  "storey_name": "6 - Sixth Floor",
  "ifc_class": "IfcWindow",
  "near_keywords": ["north", "external"],
  "relations": [],
  "space_name": null,
  "target_name_keyword": null,
  "neighbor_type": "IfcColumn"
}
```

**Phase 5 additions:** `space_name`, `target_name_keyword`, `neighbor_type` — for elements that
can't be pinpointed by storey + class alone.

**Two extraction backends:**
- **Prompt-only** (Gemini 2.5 Flash) — zero-shot baseline
- **LoRA adapter** (Qwen2.5-VL-7B, r=16) — fine-tuned on 933 multimodal samples

---

## SLIDE 9 — Evaluation Design: Synthetic Dataset & Ablation

**Dataset overview:**

| Dataset | IFC Models | Cases | Use |
|---------|-----------|-------|-----|
| **synth_v0.3** | AdvancedProject (AP) | 84 | V1 + V2 prompt baseline |
| **synth_v0.4** | AP + BasicHouse (BH) + Duplex_A (DXA) | 361 (933 augmented train + 50 holdout) | LoRA_2 training + eval |

**6-Condition Modality Ablation (synth_v0.4, LoRA_2 evaluation):**

| Condition | Visual Inputs | 4D Context | Purpose |
|-----------|--------------|------------|---------|
| **MA** | Text only | ON | Text + 4D baseline |
| **MB** | Text + Site photos | ON | +Photos vs MA |
| **MC** | Text + Photos + Floorplan | ON | +Floorplan vs MB |
| **MA-** | Text only | OFF | MA without 4D |
| **MB-** | Text + Site photos | OFF | MB without 4D |
| **MC-** | Text + Photos + Floorplan | OFF | MC without 4D |

**Comparing MA vs MA-, MB vs MB-, MC vs MC- isolates the pure 4D contribution.**

**Evaluation metrics:**
- **Top-1 Accuracy** — exact GUID match in top-1 result
- **Search Space Reduction (SSR)** — `(N_initial − N_retrieved) / N_initial` (target: high with GT retained)
- **Field EM F1** — constraint field-level extraction accuracy vs ground truth labels
- **Over-Reduction rate** — SSR where ground truth is *not* in the final candidate set (error case)

---

## SLIDE 10 — Results: Overall Performance (LoRA_2 vs Prompt)

**[IMAGE: `docs/plots/0224_modality_6cond_v3/1_overall_metrics.png`]**

**synth_v0.4, 50-case holdout, 6-condition ablation (300 traces per profile):**

| Metric | V2 LoRA | V2 Prompt | Delta |
|--------|---------|-----------|-------|
| **Top-1 Accuracy** | **35.3%** | 25.7% | **+9.6 pp** |
| Name Match | 67.7% | 60.0% | +7.7 pp |
| Valid SSR (GT retained) | 66.2% | 52.8% | +13.4 pp |
| Over-Reduction (GT lost) | 64.7% | 74.3% | −9.6 pp ✓ |

**Key finding:** LoRA not only improves top-1 accuracy but also *reduces over-aggressive filtering*
— it retrieves with more precision, losing fewer ground-truth elements.

**Context:** Storey Match ≈ 0% for both (known issue: storey name extraction fails on
large multi-floor buildings where storey is ambiguous from chat alone).

---

## SLIDE 11 — Results: Visual Modality Contribution

**[IMAGE: `docs/plots/0224_modality_6cond_v3/9_modality_stack_MA_MB_MC.png`]**

**Key question:** Does adding site photos (MB) or floorplans (MC) actually help?

**Findings (LoRA — 4D ON conditions):**

| Condition | Top-1 | Delta from MA |
|-----------|-------|---------------|
| MA (Text + 4D) | 36% | baseline |
| MB (+Site photos) | 34% | −2% |
| MC (+Floorplan) | **44%** | **+10%** (key gain!) |

**For Prompt baseline:** gains are smaller and less consistent (−2% / +12% / −2%)

**Interpretation:**
- Site photos alone (MB) do not reliably help — noisy, requires spatial grounding
- **Floorplan (MC) gives the clearest boost** — spatial layout is a strong constraint
- LoRA learns to exploit floorplan spatial geometry; Prompt model does not consistently

---

## SLIDE 12 — Results: 4D Project Context Impact

**[IMAGE: `docs/plots/0224_modality_6cond_v3/12_4d_paired_ablation.png`]**

**Key question:** How much does 4D schedule/task metadata contribute?
*(Comparing MA vs MA-, MB vs MB-, MC vs MC- — only 4D differs)*

| Condition Pair | 4D ON | 4D OFF | 4D Gain |
|----------------|-------|--------|---------|
| MA vs MA- | 28% | 25% | **+3%** |
| MB vs MB- | 33% | 30% | **+3%** |
| MC vs MC- | **37%** | 30% | **+7%** |

**Finding:** 4D context provides a **consistent, additive +3–7pp gain** across all modality levels.
Gain is largest when combined with full multimodal input (MC), suggesting 4D context and
visual inputs are complementary, not redundant.

**Implication for RQ1:** Both temporal project context AND spatial visual context are independently
useful for disambiguation — fusing them achieves the best results.

---

## SLIDE 13 — Results: Building Generalization (LoRA_2)

**[IMAGE: `docs/plots/0224_modality_6cond_v3/11_modality_x_building.png`]**

**Key question:** Does the LoRA adapter generalize across different building types?
*(LoRA_2 was trained on 3 IFC models: AP + BH + DXA)*

| Building | MA (Text+4D) | MB (+Photos) | MC (+Floorplan) |
|----------|------------|------------|----------------|
| **AP** AdvancedProject (10-storey office) | 8% | 5% | 10% |
| **BH** BasicHouse (2-storey residential) | 45% | **62%** | 60% |
| **DXA** Duplex_A (split-level) | 35% | 30% | **45%** |

**Key findings:**
- **AP is hardest**: large building, many identical elements, storey ambiguity — Top-1 stays low
- **BH benefits most from site photos (+18%)**: small building, fewer elements, photos disambiguate well
- **DXA benefits from floorplan (+15%)**: split-level geometry is easier to localize spatially
- **Building type determines which modality matters most** — one-size-fits-all retrieval is suboptimal

---

## SLIDE 14 — Key Insights & Failure Analysis

**What works:**
- V2 constraints-driven pipeline achieves **consistent SSR > 80%** — dramatically narrows candidate set
- LoRA_2 outperforms zero-shot Gemini prompt by +9.6pp, confirming value of domain fine-tuning
- 4D context + floorplan = most effective input combination (MC with 4D ON)

**What doesn't work yet:**
1. **Attribute entropy bottleneck** (the core unsolved problem):
   - In large buildings (AP), 10-40 elements of the same type exist per floor
   - Once storey + class filters are applied, remaining elements are indistinguishable by attribute text alone
   - SSR is high, but Top-1 within the reduced set is near-random → need *topological* differentiation

2. **Storey extraction failure on large buildings**:
   - Informal chat rarely says "sixth floor" explicitly — agent must infer from context
   - When storey is wrong, the whole query plan fails → over-reduction

3. **Site photo noise**:
   - Photos add signal for simple buildings (BH +18%) but can hurt for complex ones (AP −2%)
   - VLM descriptions are not spatially anchored to IFC coordinates

**[IMAGE: `docs/plots/0224_modality_6cond_v3/3_search_space_reduction.png`]**
*(Box plots show: LoRA valid SSR median ~85%, but over-reduction in 194/300 cases)*

---

## SLIDE 15 — Progress Summary: What's Done

**Implementation status:**

| Component | Status |
|-----------|--------|
| V1 pipeline (ReAct + MCP) | ✅ Complete, evaluated |
| V2 pipeline (Constraints-Driven) | ✅ Complete, evaluated |
| synth_v0.3 dataset (84 cases, AP) | ✅ Complete |
| synth_v0.4 dataset (361 cases, 3 IFC models) | ✅ Complete |
| LoRA_2 training (Qwen2.5-VL r=16, 933 samples) | ✅ Trained (Modal A100) |
| 6-condition modality ablation (300 traces × 2 profiles) | ✅ Complete |
| Phase 5: fine-grained constraints (space_name, neighbor_type) | ✅ Schema added |
| BCF 2.1 handoff output | ✅ Complete |
| RQ2 CORENET-X schema validation | ✅ Complete |
| Plot generation pipeline | ✅ Complete |

**Key result:** LoRA_2 reaches **35.3% Top-1** on 50-case holdout (3 building types).
Prompt baseline: 25.7%. Gap: **+9.6pp**.

---

## SLIDE 16 — Next Steps: V2.5 Neuro-Symbolic Pipeline

**The bottleneck:** Attribute entropy — elements identical in type/material/size, only differentiable by *topology*

**Solution:** Replace flat attribute matching with **spatial relation graph traversal**

```
Neuro Layer (Perception):
  Input: site photo crop + chat → VLM (LoRA_3)
  Output: LocalSceneGraph (spatial triplets)
    e.g. {"subject": "IfcPipeSegment", "predicate": "INTERSECTS", "object": {"IfcClass": "IfcWall"}}

Symbolic Layer (Execution):
  Input: LocalSceneGraph → Cypher compiler → Neo4j query
    MATCH (p:IfcPipeSegment)-[:INTERSECTS]->(w:IfcWall)
    RETURN p.GlobalId, p.Name
  Zero LLM hallucination in retrieval step
```

**New predicates (long-tail spatial relations):**
- `INTERSECTS` — physical penetration (pipe through wall)
- `ADJACENT_TO` — surface gap < 5cm
- `CANTILEVERED_OVER` — Z-projection overlap, subject above object

**Data plan (synth_v0.5):**
1. Geometric skeleton mining via OCCT (AABB broad phase + exact narrow phase)
2. Object-centric crop rendering (Blender headless — intersection close-up patches)
3. Text augmentation with vague/urgent site jargon
4. Hard negatives: 50 identical elements, only 1 has `INTERSECTS` target topology

**New evaluation metric:** `mR@100` on rare predicates — proves the model learned geometry, not language priors

---

## SLIDE 17 — Next Steps: Timeline

| Phase | Task | Target |
|-------|------|--------|
| **Now** | `synth_v0.5` skeleton mining (OCCT AABB + narrow phase) | Week 1-2 |
| | Object-centric crop generation (Blender) | Week 2-3 |
| | LoRA_3 training on spatial triplet extraction | Week 3-4 |
| **Near** | V2.5 Symbolic Layer: Cypher compiler + Neo4j integration | Week 4-6 |
| | Full evaluation: V2 vs V2.5 on attribute-entropy hard cases | Week 6-7 |
| **Thesis** | Final evaluation chapter, limitation analysis | Month 3 |
| | Thesis writing | Month 3-4 |

**Core thesis argument by end:**
1. Multimodal context narrows BIM search space (RQ1 — shown)
2. LoRA fine-tuning outperforms zero-shot LLM for constraint extraction (RQ1 — shown)
3. Topological spatial relations break the attribute entropy ceiling (RQ1 — upcoming)
4. Structured output validates against AEC schemas (RQ2 — shown)
5. Empty retrieval as escalation signal is reliable (RQ3 — partial)

---

## APPENDIX A — Data Synthetic Pipeline

```
IFC Model (IfcOpenShell)
  ↓ 1_render_wireframes.py        → Blender wireframe renders per element
  ↓ 2_generate_photos.py          → Gemini: photoreal site photos from wireframes
  ↓ 3_generate_floorplans.py      → matplotlib: floorplan patches from IFC geometry
  ↓ 4_generate_cases.py           → LLM: chat history + 4D metadata from GT element
  ↓ 5_filter_cases.py             → quality filter: parse rate, duplicate removal
  ↓ 6_augment_text.py             → 3× augmentation: Original / Vague / Urgent
  ↓ 7_prepare_lora_data.py        → Qwen2.5-VL ChatML format, merge 3 buildings
  →  lora_train.jsonl (933)  +  test_holdout.jsonl (50)
```

**Three IFC buildings in synth_v0.4:**

| Tag | Building | Train | Holdout |
|-----|----------|-------|---------|
| AP | AdvancedProject (10-storey office, 263 windows) | 690 | 20 |
| BH | BasicHouse (2-storey residential) | 33 | 20 |
| DXA | Duplex_A (split-level duplex) | 210 | 10 |

---

## APPENDIX B — LoRA_2 Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` |
| Adapter | LoRA (r=16, alpha=32) |
| Training samples | 933 (AP=690, BH=33, DXA=210) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Effective batch size | 16 (batch=2, grad_accum=8) |
| Hardware | Modal A100 (40GB) |
| Task | Multimodal constraint extraction: [site photo + floorplan + chat] → constraints JSON |

**Adapter location:** `models/adapters/v2_lora_qwen/` (local), `/mscd-lora/final` (Modal volume)

---

## APPENDIX C — Profiles Used in Evaluation

| Profile | Pipeline | Constraints | Retrieval | CLIP |
|---------|----------|-------------|-----------|------|
| `v2_lora` | V2 | LoRA_2 (Qwen2.5-VL) | Neo4j | No |
| `v2_prompt` | V2 | Gemini 2.5 Flash prompt | Neo4j | No |
| `v1_baseline` | V1 | ReAct agent | Memory | No |

All plots in `docs/plots/0224_modality_6cond_v3/` generated by:
```bash
./training/eval.sh --step update-plots --experiment modality_6cond_lora
```
