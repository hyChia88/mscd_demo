# Evaluation Analysis Scripts

This directory contains scoring, analysis, and thesis-plot utilities. For Phase 4 LoRA6 thesis figures, use the suite runner below as the canonical entry point.

## Canonical Phase 4 Plot Command

From the repository root:

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mscd_demo
python mscd_demo/evaluation/analysis/generate_phase4_plot_suite.py --suite thesis
```

Useful variants:

```bash
python mscd_demo/evaluation/analysis/generate_phase4_plot_suite.py --suite core
python mscd_demo/evaluation/analysis/generate_phase4_plot_suite.py --suite all
python mscd_demo/evaluation/analysis/generate_phase4_plot_suite.py --dry-run
```

- `core`: canonical Phase 4 main/appendix generator plus ablation, Graph-RAG, and retrieval-landscape figures.
- `thesis`: `core` plus summary-backed thesis overrides and the oracle dual-waterfall combo.
- `all`: `thesis` plus slower gallery/diagnostic regeneration.

Outputs are written to:

- `mscd_demo/docs/plots/phase4_lora6_main`
- `mscd_demo/docs/plots/phase4_lora6_appendix`

The suite writes `phase4_plot_suite_manifest.json` into the main plot directory with the commands and expected outputs.

## Scripts Used By The Suite

- `generate_phase4_plot_suite.py`: canonical orchestrator.
- `generate_phase4_plots.py`: core Phase 4 main figures and appendix A1-A5.
- `summarize_ap_modality_ablation.py`: Track A modality-ablation figure and summaries.
- `summarize_ap_text_tier_ablation.py`: Track A text-tier ablation figure and summaries.
- `create_graph_rag_comparison_figure.py`: Graph-RAG rerank main/appendix figure and summaries.
- `create_retrieval_strategy_landscape.py`: retrieval strategy landscape main/appendix figure and summaries.
- `create_fair_trackb2_growth_figures.py`: mixed-regime thesis growth figures and summaries.
- `create_oracle_combo_figure.py`: combined oracle progression/fingerprint waterfall panel.
- `analyze_ap_heldout_topology.py`: only used by `--suite all` for eval gallery and shortcut diagnostic images.
- `phase4_plot_colors.json`: shared source-of-truth palette.
- `phase4_plot_style.py`: palette loader and shared color constants.

## Active Supporting Scripts

These are not directly run by the Phase 4 plot suite, but are still active or useful for scoring, dataset preparation, or current analysis workflows:

- `score_ap_track.py`
- `score_unified_track.py`
- `score_oracle_ap_topology.py`
- `score_opencv_rescored_traces.py`
- `build_ap_modality_slices.py`
- `build_ap_text_tier_slices.py`
- `build_dualtrack_summary.py`
- `build_precomputed.py`
- `build_precomputed_ap_heldout.py`
- `inject_floorplan_counts.py`
- `group4_common.py`
- `group4_minimal_ablation.py`
- `analyze_traces.py`
- `analyze_lora6_ap_augmentation.py`
- `field_level_eval.py`
- `compare_results.py`
- `experiment_plots.py`

## Legacy / One-Off Scripts

Older standalone plotting scripts that are not part of the canonical suite live in `legacy/`:

- `legacy/create_lora2_vs_lora5_fp_ap_figure.py`
- `legacy/diagnose_lora6_collapse_shortcut.py`
- `legacy/diagnose_oracle_phase3_ceiling.py`
- `legacy/plot_modality_ablation.py`

Keep these for provenance and historical figure regeneration. Prefer adding new Phase 4 thesis figures to `generate_phase4_plot_suite.py` instead of adding more top-level one-off plot scripts.
