# Phase 5 — LoRA2 vs LoRA5-r32 on AP Held-out, Floorplan-only

This note records a fairness check against older LoRA generations on the AP held-out benchmark.

Evaluation protocol:
- dataset: `AP held-out` (`n=60`)
- input slice: `floorplan-only`
- upstream extraction:
  - `LoRA2 FP` uses native `lora2` prompt / output style
  - `LoRA5-r32 FP` uses native `lora5` prompt / output style
- downstream pipeline: identical for both
  - `profile=v2_lora`
  - `p0_strategy=p0_union_p1`

## Thesis-ready table

| System | Native intermediate format | Top-10 | Top-1 | MRR@10 | GT-in-Pool | Parse | SR extracted | 2-hop SR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `LoRA2 FP` | legacy LoRA2 schema, no usable AP `spatial_relations` | 16.7% | 0.0% | 0.0392 | 98.3% | 100.0% | 0.0% | 0.0% |
| `LoRA5-r32 FP` | LoRA5 schema with AP-compatible `spatial_relations` | 21.7% | 0.0% | 0.0407 | 100.0% | 100.0% | 100.0% | 40.0% |

## Readout

- `LoRA5-r32 FP` improves strict AP end-to-end retrieval over `LoRA2 FP` on `Top-10` (`21.7%` vs `16.7%`) and `GT-in-Pool` (`100.0%` vs `98.3%`).
- Neither older LoRA generation solves last-mile ranking under this floorplan-only setup: both remain at `Top-1 = 0.0%`.
- The main structural difference is intermediate-format compatibility. `LoRA2 FP` parsed successfully but produced no usable AP `spatial_relations`, whereas `LoRA5-r32 FP` produced `spatial_relations` in all `60/60` cases and `2-hop` relations in `24/60`.
- This makes the comparison fair and thesis-safe: both models were evaluated on the same AP held-out cases and the same downstream `Track B-2` retrieval stack, with only the upstream LoRA family and its native output format varying.

## Output references

- `LoRA2` summary: [summary_20260405_025930_v2_lora_p0_union_p1.csv](/root/cmu/master_thesis/mscd_demo/output/ap_lora2_vs_lora5_floorplan_only/e2e/lora2_apheldout_FP/summary_20260405_025930_v2_lora_p0_union_p1.csv)
- `LoRA5-r32` summary: [summary_20260405_030157_v2_lora_p0_union_p1.csv](/root/cmu/master_thesis/mscd_demo/output/ap_lora2_vs_lora5_floorplan_only/e2e/lora5r32_apheldout_FP/summary_20260405_030157_v2_lora_p0_union_p1.csv)
- figure bundle:
  - [fig13_lora2_vs_lora5_fp_ap.png](/root/cmu/master_thesis/mscd_demo/docs/plots/phase4_lora6_main/fig13_lora2_vs_lora5_fp_ap.png)
  - [figA8_lora2_vs_lora5_fp_ap.png](/root/cmu/master_thesis/mscd_demo/docs/plots/phase4_lora6_appendix/figA8_lora2_vs_lora5_fp_ap.png)
