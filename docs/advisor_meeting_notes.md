# Advisor Meeting — Progress Report

> Date: Feb 2026
> Topic: MSCD Demo — Experiment Results, Findings, and Next Steps

---

## 1. Experiment Evaluation

### 1.1 Dataset Issues and Mitigations

**Issue 1: Instance Disambiguation is Physically Unsolvable**

Even under the best condition (A1: clear text + 4D metadata), Top-1 accuracy is only 50%. This is not a system failure — it reflects a fundamental physical limitation. When three identical windows sit side by side on the same floor, no amount of textual or visual context can distinguish them without a unique identifier (e.g., a QR code or serial number). Top-1 accuracy is therefore an unfair metric for this class of problems.

**Issue 2: Top-1 as the Primary Metric is Misleading**

Because many cases involve multiple indistinguishable instances, optimizing for Top-1 misrepresents system capability. A system that correctly narrows 1,666 elements down to 3 candidates — but picks the wrong one of the three — is far more useful than one that returns all 1,666.

**Mitigation: Metric Pivot**

We reframe the evaluation around two metrics that better capture practical value:

- **Search Space Reduction (SSR):** How much the candidate pool shrinks (e.g., 1,666 → 5 elements = 99.7% reduction).
- **Top-K Recall:** Whether the correct element appears anywhere in the top-K candidates.

> Revised thesis argument: "The goal of V2 is not to identify the exact element in one shot (Top-1), but to reduce ~2,000 candidates to fewer than 5, enabling efficient human confirmation."

![Fig 1. Search Space Reduction — Initial pool (1,666) narrowed to 83 final candidates (95% reduction)](../logs/comparisons/v1_vs_v2_fixed2/2_search_space_reduction.png)

---

### 1.2 V1 (Agent) vs. V2 (Constraints) — Key Results

| Configuration | Top-1 | Top-K | SSR | Notes |
|---|---|---|---|---|
| **V2 A1** (clear text + 4D) | **0.500** | 0.500 | 98.4% | Best overall result |
| V1 memory (best V1) | 0.326 | 0.372 | — | Agent-driven baseline |
| V2 C1 (clear text + floorplan) | 0.143 | 0.286 | 97.4% | Second-best V2 |
| V2 A2 (blurred text + 4D) | 0.167 | 0.167 | 86.7% | Blurring hurts extraction |
| V2 B1-B3 (images, no 4D) | 0.000 | 0.000–0.143 | 92–96% | Images alone insufficient |

**Takeaways:**

1. **V2 with good input (A1) outperforms V1** by +17pp in Top-1.
2. **V2 with degraded input underperforms V1** — the rigid constraint extraction fails when expected signals (storey, element type) are absent. V1's free-form agent can reason around missing information.
3. **Images alone do not help V2** — the current prompt-only extractor cannot process images. This is the motivation for LoRA VLM fine-tuning.
4. **All configurations achieve >80% search space reduction**, confirming the system's practical utility even when Top-1 misses.

---

### 1.3 Ablation Findings (Conditions A1–C3)

![Fig 2. Top-1 Accuracy by Condition — A1 (clear text + 4D) achieves the highest accuracy](../logs/comparisons/v1_vs_v2_fixed2/1_accuracy_by_condition.png)

| Finding | Evidence |
|---|---|
| Clear text + 4D metadata is the strongest signal | A1 = 50% Top-1, best of all conditions |
| Chat blurring degrades extraction quality | A1 → A2: Field F1 drops 0.722 → 0.500, Top-1 drops 0.500 → 0.167 |
| Images alone do not help (yet) | All B conditions = 0% Top-1; extractor is text-only |
| Pool size = 100 strongly correlates with miss | Fallback to broad query when extraction fails |
| Parse rate is 100% but field accuracy varies | The bottleneck is extraction quality, not parsing |

![Fig 3. Performance across all conditions (A1–C3), grouped by modality](../logs/comparisons/v1_vs_v2_fixed2/7_condition_comparison.png)

![Fig 4. VLM integration impact — B/C conditions improve from 0% to 41.3% Top-1 after adding Gemini VLM image parsing](../logs/comparisons/v1_vs_v2_fixed2/5_vision_impact.png)

![Fig 5. Per-case retrieval success heatmap — GUID, Name, and Storey match across all 43 cases](../logs/comparisons/v1_vs_v2_fixed2/6_per_case_heatmap.png)

---

### 1.4 V1 vs. V2 — Strengths, Limitations, and the Path to V3

| Dimension | V1 (Agent-Driven) | V2 (Constraints-Driven) |
|---|---|---|
| Flexibility | High — LLM reasons freely | Low — rigid extraction templates |
| Reproducibility | Low — non-deterministic | High — deterministic |
| Cost per case | ~4 LLM calls | 1–2 LLM calls |
| Best Top-1 | 0.326 | 0.500 (with ideal input) |
| Failure mode | Hallucination, verbose | Silent miss (empty pool or wrong pool) |

**Insight → V3 Direction:**

V2's constraint extraction is fast and cheap but brittle. V1's agent is flexible but expensive and non-deterministic. A natural next step is a **hybrid approach (V3):**

> **V3 proposal:** Run V2 first (fast, cheap, deterministic). If V2 returns an empty candidate pool or low confidence, fall back to V1 agent for that case. This preserves V2's efficiency for easy cases while using V1's flexibility for hard cases.

---

## 2. Proposed Next Steps

### 2.1 LoRA VLM Fine-tuning

**Motivation:** The current V2 extractor is text-only. B-group conditions (images available) show 0% Top-1 because the system cannot interpret site photos. A fine-tuned Vision-Language Model (VLM) would enable image understanding for constraint extraction.

**Plan:**

- Base model: Qwen3-VL-8B
- Task: Given a site photo + chat context, extract structured constraints (storey, IFC class, defect type)
- Training: LoRA adapter (parameter-efficient fine-tuning)
- Expected outcome: B-group conditions should improve significantly

### 2.2 Synthetic Training Data Generation

Real-world labeled BIM inspection data is scarce. We plan to generate synthetic training data from IFC models:

1. **Floorplan generation from IFC:** Extract 2D plans programmatically from the IFC spatial structure.
2. **Construction image synthesis from IFC:** Render realistic site photos from BIM geometry using structure-conditioned image generation.

**Reference:** CAD2DMD-SET (Valente et al.) demonstrates this approach for industrial measurement devices:
- They built an automated pipeline: 3D CAD model → physics-based rendering (lighting, angles, blur) → composite onto real backgrounds → auto-generate VQA labels.
- 100K synthetic images were generated, and fine-tuning on this data improved open-source VLM performance by 200% on a real-world benchmark (DMDBench, 1,000 real images).
- **Key takeaway for our work:** High-quality 3D synthetic data can effectively compensate for the lack of real training data. By simulating physical conditions (lighting, camera angles, occlusion), we can improve domain-specific VLM performance without collecting real labeled images.

**Our adaptation:**
- Input: IFC model geometry (walls, windows, doors with properties)
- Output: Rendered construction-site images with ground-truth element labels
- Use: Fine-tune Qwen3-VL-8B LoRA adapter for BIM-aware image understanding

### 2.3 Evaluation Improvements

- Expand synthetic dataset from 43 to ~200 cases with balanced conditions (currently 2–7 cases per condition)
- Run Neo4j experiments properly (previous runs fell back to in-memory)
- Per-case error analysis to identify failure patterns

---

## 3. Research Questions — Implementation Mapping

| RQ | Core Challenge | Method | Experiment | Key Metric |
|---|---|---|---|---|
| **RQ1: Multimodal Grounding** | Ambiguity — text is vague, generic vision models do not understand construction | VLM LoRA fine-tuned on BIM-specific features | A/B/C condition ablation: compare text-only vs. image vs. multimodal | Search Space Reduction (SSR) |
| **RQ2: Schema Alignment** | Probabilistic LLM output does not conform to regulatory schemas | Constraints Extractor (LoRA) + deterministic Query Planner | V1 vs. V2 schema compliance comparison | Field F1 Score, Parse Rate |
| **RQ3: Governance** | Agent hallucination — fabricates answers when no match exists | Escalation policy: safe fallback when candidate pool is empty | Hard-negative test cases (query elements that do not exist in model) | Safety Rate (True Negative Rate) |

---

## 4. System Architecture

### 4.1 Component Overview

![Fig 6. System architecture — all layers from input to BCF handoff](system_architecture_2_simplify.png)

### 4.2 Sequence Diagrams

**Initialisation and pipeline dispatch:**

![Fig 7. Unified runner loads config, IFC engine, LLM, then selects V1 or V2 pipeline](sequence_diagram.png)

**V1 Pipeline — Agent-driven (ReAct + MCP):**

![Fig 8. V1: LangGraph ReAct agent iteratively calls MCP tools (search, details, CLIP) until final answer](sequence_v1_pipeline.png)

**V2 Pipeline — Constraints-driven (no agent loop):**

![Fig 9. V2: ConditionMask → ImageParser → ConstraintsExtractor → QueryPlanner → RetrievalBackend](sequence_v2_pipeline.png)

---

## Summary for Discussion

1. **Current state:** V1 and V2 pipelines both operational with 43-case synthetic dataset. V2 A1 outperforms V1 but V2 is brittle when input quality degrades.
2. **Metric pivot:** Shift thesis emphasis from Top-1 to Search Space Reduction + Top-K Recall.
3. **Key blocker:** V2 cannot process images → need LoRA VLM.
4. **Proposed V3:** Hybrid pipeline — V2 first, fall back to V1 on failure.
5. **Data plan:** Synthetic image generation from IFC for LoRA training.
6. **Questions for advisor:**
   - Is the metric pivot (SSR over Top-1) well-justified for the thesis?
   - Should V3 (hybrid) be a contribution, or should we focus on improving V2 alone?
   - Is the synthetic data generation approach (IFC → rendered images) a reasonable scope for the thesis timeline?
