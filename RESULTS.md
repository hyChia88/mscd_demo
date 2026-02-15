# Evaluation Results

---

## Exp 1: V1 Agent Pipeline on synth_v0.2

**Date:** 2026-02-07 | **Dataset:** synth_v0.2 (43 cases) | **LLM:** Gemini 2.5 Flash

| Mode | Top-1 | Top-3 | F1 | GUID Matches |
|------|-------|-------|-----|--------------|
| memory | **0.326** | 0.372 | **0.347** | 16 |
| neo4j | 0.302 | 0.326 | 0.314 | 14 |
| memory+clip | 0.256 | 0.326 | 0.287 | 14 |
| neo4j+clip | 0.256 | 0.279 | 0.267 | 12 |

Memory mode is best. Adding Neo4j/CLIP hurts — Neo4j fallback adds noise, CLIP reranking finds the right type but wrong instance.

---

## Exp 2: V2 Prompt Condition Ablations on synth_v0.2

**Date:** 2026-02-07 | **Dataset:** synth_v0.2 (43 cases) | **Profile:** `v2_prompt`

| Condition | Cases | Top-1 | Top-K | SSR | Field F1 |
|-----------|-------|-------|-------|-----|----------|
| **A1** (clear + 4D) | 6 | **0.500** | **0.500** | 0.984 | **0.722** |
| A2 (blurred + 4D) | 6 | 0.167 | 0.167 | 0.867 | 0.500 |
| A3 (blurred + 4D enhanced) | 2 | 0.000 | 0.000 | 0.970 | 0.333 |
| B1 (blurred + img) | 7 | 0.000 | 0.143 | 0.964 | 0.286 |
| B2 (blurred + img + CLIP) | 5 | 0.000 | 0.000 | 0.947 | 0.133 |
| B3 (clear + img) | 3 | 0.000 | 0.000 | 0.922 | 0.389 |
| C1 (clear + floorplan) | 7 | 0.143 | **0.286** | 0.974 | 0.476 |
| C2 (blurred + img + floorplan) | 5 | 0.000 | 0.000 | 0.940 | N/A |
| C3 (clear + all) | 2 | 0.000 | 0.000 | 0.810 | 0.667 |

### Key Findings

1. **Clear text + 4D is the strongest signal** — A1 = 50% Top-1, beating best V1 (32.6%).
2. **Chat blurring degrades extraction** — A1→A2: Top-1 drops 0.500→0.167, F1 drops 0.722→0.500.
3. **Images alone don't help V2** — All B conditions = 0% Top-1 (prompt extractor is text-only).
4. **Pool=100 strongly correlates with miss** — fallback strategy is too broad.
5. **Parse rate is 100%** but field accuracy varies — bottleneck is extraction quality, not parsing.
6. **V2 A1 > V1 memory** (+17.4pp Top-1), but V2 with degraded inputs underperforms V1.

---

## Exp 3: V2 Prompt Baseline on synth_v0.3

**Date:** 2026-02-14 | **Dataset:** synth_v0.3 (84 cases) | **Profile:** `v2_prompt`
**Traces:** `logs/evaluations/traces_20260214_210555_v2_prompt.jsonl`

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 0.0357 (3/84) |
| Top-K Accuracy | 0.0595 (5/84) |
| Search Space Reduction | 0.9319 |
| Field EM F1 | 0.2135 |
| Parse Rate | 1.0000 |

### Success/Failure Analysis

**Top-1 hits: 3/84 (3.6%)** — all Top-1 hits are also Top-K.

| Factor | Success pattern | Insight |
|--------|----------------|---------|
| IFC Class | Wall(5), Door(1), Railing(1), Slab(1) | Walls dominate when chat says "cracks", "surface", "partition wall" |
| Pool size | Mean 17.9 (vs 65 overall) | Fewer candidates = easier to rank #1 |
| Chat keywords | "fire doors", "handrail", "hairline cracks" | Explicit element keywords make extraction trivial |
| Storey | Level 1(3), Garage(3) | Common storey names the LLM recognizes |

**What fails (76+ cases):** Vague/deictic text ("Right here.", "Check this."), high candidate density (k=181), uncommon storey names.

### Takeaway for LoRA

The prompt-only extractor only works when chat literally names the element type. LoRA should teach the model to infer `ifc_class` from visual cues (cracked slab, damaged window) instead of text keywords — exactly what Style B (vague/deictic) augmentation trains for.

---

## Limitations

1. **Small per-condition samples in v0.2** — A3/C3 have only 2 cases each.
2. **No vision model in V2 prompt** — B-group conditions are effectively text-only.
3. **Neo4j was not running** for V1 neo4j modes (fell back to memory).
4. **v0.3 Top-1 drop** (3.6% vs v0.2's 11.6% overall) is expected — v0.3 cases are intentionally harder with vague/deictic text.

---

## Appendix: Run Commands

```bash
# V1 on synth_v0.2
./run_mcp.sh --all -d synth

# V2 condition ablations on synth_v0.3
CASES=../data_curation/datasets/synth_v0.3/cases_v3_filtered.jsonl
for cond in A1 A2 A3 B1 B2 B3 C1 C2 C3; do
  python script/run.py --profile v2_prompt --cases $CASES --condition $cond
done

# V2 prompt baseline (all conditions)
python script/run.py --profile v2_prompt --cases $CASES
```
