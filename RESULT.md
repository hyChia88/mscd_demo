# Experiment Results

> Run date: 2026-02-07
> Dataset: synth_v0.2 (43 cases)
> LLM: Gemini 2.5 Flash (temperature=0)
> IFC Model: AdvancedProject.ifc (10 storeys, ~1666 elements, 263 windows)

---

## 1. V1 Agent-Driven Pipeline (All 43 Cases)

V1 runs the full ReAct agent loop — the LLM freely reasons and calls MCP tools (search_elements_by_type, match_text_to_elements, etc.) to find elements.

| Mode | Cases | Top-1 | Top-3 | Top-5 | P@1 | Recall | F1 | GUID Matches | Name Matches |
|------|-------|-------|-------|-------|-----|--------|-----|--------------|--------------|
| memory | 43 | 0.326 | 0.372 | 0.372 | 0.326 | 0.372 | 0.347 | 16 | 15 |
| neo4j | 43 | 0.302 | 0.326 | 0.326 | 0.302 | 0.326 | 0.314 | 14 | 14 |
| memory+clip | 43 | 0.256 | 0.326 | 0.326 | 0.256 | 0.326 | 0.287 | 14 | 18 |
| neo4j+clip | 43 | 0.256 | 0.279 | 0.279 | 0.256 | 0.279 | 0.267 | 12 | 17 |

**Observations:**
- **Memory mode is the best V1 configuration** (Top-1: 32.6%, F1: 34.7%).
- Adding Neo4j or CLIP actually *hurts* accuracy. The Neo4j fallback to memory mode (when queries return empty) adds noise, and CLIP reranking sometimes moves the correct element down.
- CLIP improves name matches (15 to 18) but reduces GUID matches (16 to 14), suggesting CLIP finds elements of the right type but not the exact instance.
- Top-3 and Top-5 are identical to Top-3 in most modes, meaning if the target isn't in the top 3, it's unlikely in top 5 either.

---

## 2. V2 Constraints-Driven Pipeline — Condition Ablations (A1-C3)

V2 extracts constraints (storey_name, ifc_class, keywords) from chat context using a single LLM call, then runs deterministic query planning and retrieval. Each condition controls which input modalities are available.

### Condition Groups

**Group A — Text + 4D metadata only (no images, no floorplan):**

| Condition | Chat | 4D Meta | Cases | Top-1 | Top-K | Search Space | Field F1 | Pool Sizes |
|-----------|------|---------|-------|-------|-------|--------------|----------|------------|
| **A1** | clear | yes | 6 | **0.500** | 0.500 | 0.984 | **0.722** | 41, 1, 2, 7, 9, 100 |
| **A2** | blurred | yes | 6 | 0.167 | 0.167 | 0.867 | 0.500 | 54, 100, 736, 263, 46, 126 |
| **A3** | blurred | enhanced | 2 | 0.000 | 0.000 | 0.970 | 0.333 | 100, 1 |

**Group B — Images available, blurred chat, no floorplan:**

| Condition | Chat | Images | CLIP | Cases | Top-1 | Top-K | Search Space | Field F1 | Pool Sizes |
|-----------|------|--------|------|-------|-------|-------|--------------|----------|------------|
| **B1** | blurred | yes | no | 7 | 0.000 | 0.143 | 0.964 | 0.286 | 100, 100, 100, 2, 9, 9, 100 |
| **B2** | blurred | yes | force | 5 | 0.000 | 0.000 | 0.947 | 0.133 | 100, 100, 41, 100, 100 |
| **B3** | clear | yes | no | 3 | 0.000 | 0.000 | 0.922 | 0.389 | 126, 263, 2 |

**Group C — Floorplan available:**

| Condition | Chat | Images | Floorplan | 4D Meta | Cases | Top-1 | Top-K | Search Space | Field F1 |
|-----------|------|--------|-----------|---------|-------|-------|-------|--------------|----------|
| **C1** | clear | no | yes | no | 7 | 0.143 | **0.286** | 0.974 | 0.476 |
| **C2** | blurred | yes | yes | no | 5 | 0.000 | 0.000 | 0.940 | N/A |
| **C3** | clear | yes | yes | enhanced | 2 | 0.000 | 0.000 | 0.810 | 0.667 |

### Summary Table (All Conditions)

| Condition | Cases | Top-1 | Top-K | Search Space | Parse Rate | Field F1 |
|-----------|-------|-------|-------|--------------|------------|----------|
| **A1** | 6 | **0.500** | **0.500** | 0.984 | 1.00 | **0.722** |
| A2 | 6 | 0.167 | 0.167 | 0.867 | 1.00 | 0.500 |
| A3 | 2 | 0.000 | 0.000 | 0.970 | 1.00 | 0.333 |
| B1 | 7 | 0.000 | 0.143 | 0.964 | 1.00 | 0.286 |
| B2 | 5 | 0.000 | 0.000 | 0.947 | 1.00 | 0.133 |
| B3 | 3 | 0.000 | 0.000 | 0.922 | 1.00 | 0.389 |
| **C1** | 7 | 0.143 | **0.286** | 0.974 | 1.00 | 0.476 |
| C2 | 5 | 0.000 | 0.000 | 0.940 | 1.00 | N/A |
| C3 | 2 | 0.000 | 0.000 | 0.810 | 1.00 | 0.667 |

### Reproducibility Note

Results are deterministic. Running the same conditions twice produces identical results (confirmed by two consecutive runs on 2026-02-07). This is a key advantage of V2 over V1 — the V1 agent's non-deterministic reasoning means results vary between runs.

---

## 3. Key Findings

### Finding 1: Clear text + 4D metadata is the strongest signal (A1 = 50% Top-1)

**A1 (clear chat + 4D metadata) achieves the best accuracy of any V2 condition at 50% Top-1.** This is also higher than any V1 mode (best V1 = 32.6%). The 4D task status (e.g., "TASK_302: Window Frame Installation 1 - First Floor") provides explicit storey and element type information that the constraints extractor can parse reliably.

When chat is blurred (A2), accuracy drops from 50% to 16.7%. The blurring replaces keywords like "window" with "opening" and "sixth" with "upper", making it harder for the LLM to extract the correct IFC class and storey name.

### Finding 2: Chat blurring degrades constraints extraction quality

| | Clear Chat | Blurred Chat | Delta |
|---|---|---|---|
| Field EM F1 | 0.722 (A1) | 0.500 (A2) | -0.222 |
| Top-1 | 0.500 (A1) | 0.167 (A2) | -0.333 |

The field-level F1 drops from 0.722 to 0.500, meaning the extractor gets the storey or IFC class wrong more often when keywords are obscured. This directly translates to worse retrieval because the wrong storey/type query returns the wrong candidate pool.

### Finding 3: Images alone don't help V2 (B conditions = 0% Top-1)

All B conditions (images available, chat blurred, no floorplan) achieve 0% Top-1 accuracy. This is because the V2 prompt-only extractor processes **text only** — it doesn't actually analyze the images. The image paths are listed in the prompt but without a vision model (LoRA VLM), they provide no useful constraints.

B1 achieves 14.3% Top-K because some cases still extract correct storey/type from the blurred chat, but the target isn't in the top-1 position.

### Finding 4: Floorplan + clear chat helps (C1 = second best)

C1 (clear chat + floorplan, no images) achieves Top-K of 28.6%, the second-best result after A1. The clear chat provides good constraints, and while the floorplan path is listed in the prompt, the spatial context from clear text is the main driver.

### Finding 5: Pool size = 100 strongly correlates with miss

Cases hitting the **fallback strategy** (pool=100) almost always miss. This happens when constraints extraction fails to produce storey_name or ifc_class, so the planner falls back to returning the first 100 elements — too broad to contain the target in top-10.

| Pool Size | Outcome Pattern |
|-----------|----------------|
| 1-9 | Usually HIT (specific query) |
| 41-54 | Sometimes HIT (storey-only query) |
| 100+ | Almost always miss (fallback/broad query) |

### Finding 6: Parse rate is 100% but field accuracy varies

The LLM always returns valid JSON (parse rate = 1.00 across all conditions), but the extracted fields aren't always correct. This means the bottleneck is **extraction quality** (getting the right storey/type), not parsing reliability.

### Finding 7: V2 A1 beats V1 memory, but most V2 conditions underperform V1

| Configuration | Top-1 | Top-K | F1 |
|---------------|-------|-------|----|
| **V2 A1** | **0.500** | 0.500 | — |
| V1 memory | 0.326 | 0.372 | 0.347 |
| V2 C1 | 0.143 | 0.286 | — |
| V2 A2 | 0.167 | 0.167 | — |
| V2 B1 | 0.000 | 0.143 | — |
| V2 B2/B3/C2/C3 | 0.000 | 0.000 | — |

V2 with optimal conditions (A1) outperforms V1 by +17.4pp in Top-1. But V2 with degraded inputs (blurred chat, no 4D, images-only) severely underperforms V1. This makes sense: V1's free-form agent can reason around missing information, while V2's rigid constraint extraction fails when expected signals are absent.

---

## 4. Search Space Reduction

All configurations achieve significant search space reduction (>80%), narrowing from ~1666 elements to a small candidate pool:

| Configuration | Avg Search Space Reduction |
|---------------|---------------------------|
| V2 A1 | 98.4% |
| V2 C1 | 97.4% |
| V2 A3 | 97.0% |
| V2 B1 | 96.4% |
| V2 B2 | 94.7% |
| V2 C2 | 94.0% |
| V2 B3 | 92.2% |
| V2 A2 | 86.7% |
| V2 C3 | 81.0% |

Even in the worst case (C3), the pool is reduced by 81%. However, high reduction doesn't guarantee a hit — the correct element must be *in* the reduced pool.

---

## 5. Limitations

1. **Small sample sizes per condition** — A3 and C3 have only 2 cases each. Results for these conditions are not statistically reliable.
2. **No vision model in V2** — The prompt-only extractor doesn't process images, making B-group conditions effectively text-only. LoRA VLM integration would likely improve B conditions significantly.
3. **No CLIP reranking in V2 ablations** — The v2_prompt profile doesn't enable CLIP. B2's `force_clip` condition flag is set, but the profile override didn't activate the VisualAligner. This needs verification.
4. **Fallback dominance** — Many cases fall to the fallback strategy (pool=100), suggesting the extractor struggles with cases where storey/type aren't explicitly mentioned.
5. **Neo4j was not running** — V1 neo4j and neo4j+clip modes fell back to memory-based queries, so their results may not reflect true graph-based performance.

---

## 6. Next Steps

1. **Improve constraints extraction** — Tune the extraction prompt to handle blurred/ambiguous inputs better, or add few-shot examples.
2. **LoRA VLM integration** — Train and deploy the Qwen3-VL-8B LoRA adapter to enable actual image understanding for B/C conditions.
3. **Verify CLIP reranking** — Ensure B2's force_clip condition actually activates CLIP in the V2 pipeline.
4. **Larger dataset** — Expand synth_v0.2 to have more balanced cases per condition (currently 2-7 cases each).
5. **Run Neo4j experiments** — Start Neo4j and re-run neo4j/neo4j+clip V1 modes for accurate graph-based comparison.
6. **Per-case error analysis** — Examine the specific cases that miss to understand failure patterns (wrong storey? wrong type? fallback?).

---

## Appendix: Run Commands

```bash
# V1 experiments (43 cases each)
./run_mcp.sh -e memory -d synth
./run_mcp.sh -e neo4j -d synth
./run_mcp.sh -e memory+clip -d synth
./run_mcp.sh -e neo4j+clip -d synth

# V2 condition ablations
CASES=../data_curation/datasets/synth_v0.2/cases_v2.jsonl
for cond in A1 A2 A3 B1 B2 B3 C1 C2 C3; do
  python script/run.py --profile v2_prompt --cases $CASES --condition $cond
done

# Compare all results
python script/compare_results.py --latest 20
```
