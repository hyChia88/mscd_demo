# Phase 5 — Mixed-Regime Track B-2 Growth Comparison

This note consolidates the thesis-growth view of end-to-end AP held-out performance.

Evaluation protocol:
- benchmark: `AP held-out` (`n=60`)
- downstream pipeline:
  - `profile=v2_lora`
  - `p0_strategy=p0_union_p1`
- regime rule:
  - only `LoRA5-r32` is kept in the older `Floorplan + Chat` / no-site setting
  - later milestone models keep their canonical multimodal AP held-out setup
  - the table is therefore a growth narrative, not a single strict fairness leaderboard

## Thesis-ready milestone table

| System | Input regime | Family role | Top-10 | Top-1 | MRR@10 | GT-in-Pool | SR extracted | 2-hop SR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `LoRA5-r32 FP` | `Floorplan + Chat` (no site) | pre-LoRA6 baseline | 21.7% | 0.0% | 0.0407 | 100.0% | 100.0% | 40.0% |
| `Gemini AP (MM)` | canonical multimodal | prompt-only baseline | 21.7% | 0.0% | 0.0482 | 91.7% | 66.7% | 30.0% |
| `G3 FullAug r32 (MM)` | canonical multimodal | strongest Top-10 LoRA6 milestone | 26.7% | 1.7% | 0.0641 | 100.0% | 100.0% | 58.3% |
| `G4 Ultimate (MM)` | canonical multimodal | high-Hop-1 but weaker downstream milestone | 23.3% | 0.0% | 0.0324 | 100.0% | 100.0% | 56.7% |
| `G7 Position Context (MM)` | canonical multimodal | final richer-label milestone | 23.3% | 3.3% | 0.0681 | 100.0% | 100.0% | 53.3% |

## Readout

- `LoRA5-r32 FP` remains the historical no-site anchor. It already extracts usable AP topology (`SR 100.0%`, `2-hop 40.0%`) and reaches `Top-10 21.7%`, but it still has `Top-1 0.0%`.
- Among the later canonical multimodal milestones, `G3 FullAug r32` is the strongest strict `Top-10` retriever (`26.7%`).
- `G7 Position Context` is the strongest early-rank system (`Top-1 3.3%`, `MRR@10 0.0681`).
- `Gemini AP (MM)` remains weaker on stable shortlist coverage (`GT-in-Pool 91.7%`) even though it preserves moderate SR coverage.
- The growth trajectory is therefore not monotonic in every metric, but it is structurally progressive: later multimodal LoRA6 variants improve either shortlist coverage (`G3`) or early-rank quality (`G7`) over the older pre-LoRA6 baseline.

## Figure references

- updated main figures:
  - [fig02_v2_extraction_vs_downstream_tradeoff.png](/root/cmu/master_thesis/mscd_demo/docs/plots/phase4_lora6_main/fig02_v2_extraction_vs_downstream_tradeoff.png)
  - [fig03_trackB2_strict_downstream.png](/root/cmu/master_thesis/mscd_demo/docs/plots/phase4_lora6_main/fig03_trackB2_strict_downstream.png)
- appendix companion:
  - [figA9_fair_trackb2_growth.png](/root/cmu/master_thesis/mscd_demo/docs/plots/phase4_lora6_appendix/figA9_fair_trackb2_growth.png)
