#!/usr/bin/env python3
"""Generate the Phase 4 LoRA6 thesis plot suite.

Default usage from the repository root:

    python mscd_demo/evaluation/analysis/generate_phase4_plot_suite.py

Run with ``--suite all`` to also refresh the 60-case gallery and diagnostic
figures, which are slower and depend on image-heavy artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent.parent
MAIN_PLOTS_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
APPENDIX_PLOTS_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_appendix"


@dataclass(frozen=True)
class PlotStep:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]


def _script(name: str) -> str:
    return str(ANALYSIS_DIR / name)


def _base_steps() -> list[PlotStep]:
    return [
        PlotStep(
            "core-main-and-appendix",
            (
                sys.executable,
                _script("generate_phase4_plots.py"),
                "--out-dir",
                str(MAIN_PLOTS_DIR),
                "--appendix-out-dir",
                str(APPENDIX_PLOTS_DIR),
            ),
            (
                MAIN_PLOTS_DIR / "phase4_plot_manifest.json",
                APPENDIX_PLOTS_DIR / "phase4_appendix_plot_manifest.json",
            ),
        ),
        PlotStep(
            "trackA-modality-ablation",
            (
                sys.executable,
                _script("summarize_ap_modality_ablation.py"),
                "--out-dir",
                str(MAIN_PLOTS_DIR),
            ),
            (MAIN_PLOTS_DIR / "fig09_trackA_modality_ablation.png",),
        ),
        PlotStep(
            "trackA-text-tier-ablation",
            (
                sys.executable,
                _script("summarize_ap_text_tier_ablation.py"),
                "--out-dir",
                str(MAIN_PLOTS_DIR),
            ),
            (MAIN_PLOTS_DIR / "fig10_text_tier_ablation.png",),
        ),
        PlotStep(
            "graph-rag-rerank",
            (sys.executable, _script("create_graph_rag_comparison_figure.py")),
            (
                MAIN_PLOTS_DIR / "fig11_graph_rag_rerank_comparison.png",
                APPENDIX_PLOTS_DIR / "figA6_graph_rag_rerank_comparison.png",
            ),
        ),
        PlotStep(
            "retrieval-strategy-landscape",
            (sys.executable, _script("create_retrieval_strategy_landscape.py")),
            (
                MAIN_PLOTS_DIR / "fig12_retrieval_strategy_landscape.png",
                APPENDIX_PLOTS_DIR / "figA7_retrieval_strategy_landscape.png",
            ),
        ),
    ]


def _thesis_steps() -> list[PlotStep]:
    return _base_steps() + [
        PlotStep(
            "mixed-regime-growth",
            (sys.executable, _script("create_fair_trackb2_growth_figures.py")),
            (
                MAIN_PLOTS_DIR / "fig02_v2_extraction_vs_downstream_tradeoff.png",
                MAIN_PLOTS_DIR / "fig03_trackB2_strict_downstream.png",
                APPENDIX_PLOTS_DIR / "figA9_fair_trackb2_growth.png",
            ),
        ),
        PlotStep(
            "oracle-dual-waterfall-combo",
            (sys.executable, _script("create_oracle_combo_figure.py")),
            (MAIN_PLOTS_DIR / "fig08_oracle_dual_waterfall_combo.png",),
        ),
    ]


def _gallery_diagnostic_steps() -> list[PlotStep]:
    return [
        PlotStep(
            "eval-gallery",
            (
                sys.executable,
                _script("analyze_ap_heldout_topology.py"),
                "--eval-gallery",
                "--out-dir",
                str(MAIN_PLOTS_DIR),
            ),
            (
                MAIN_PLOTS_DIR / "eval_gallery_60cases.png",
                MAIN_PLOTS_DIR / "eval_gallery_60cases.pdf",
            ),
        ),
        PlotStep(
            "shortcut-diagnostic",
            (
                sys.executable,
                _script("analyze_ap_heldout_topology.py"),
                "--diagnostic",
                "--out-dir",
                str(MAIN_PLOTS_DIR),
            ),
            (
                MAIN_PLOTS_DIR / "diagnostic_shortcut_collapse.png",
                MAIN_PLOTS_DIR / "diagnostic_shortcut_collapse.pdf",
            ),
        ),
    ]


def _steps_for_suite(suite: str) -> list[PlotStep]:
    if suite == "core":
        return _base_steps()
    if suite == "thesis":
        return _thesis_steps()
    if suite == "all":
        return _thesis_steps() + _gallery_diagnostic_steps()
    raise ValueError(f"Unknown suite: {suite}")


def _run_step(step: PlotStep, dry_run: bool) -> int:
    cmd = " ".join(step.argv)
    print(f"\n[{step.name}] {cmd}")
    if dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mscd_demo_matplotlib")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(step.argv, cwd=PROJECT_ROOT.parent, env=env)
    return completed.returncode


def _write_manifest(steps: Iterable[PlotStep], failures: list[str], dry_run: bool) -> None:
    MAIN_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite_manifest": str(MAIN_PLOTS_DIR / "phase4_plot_suite_manifest.json"),
        "main_plots_dir": str(MAIN_PLOTS_DIR),
        "appendix_plots_dir": str(APPENDIX_PLOTS_DIR),
        "dry_run": dry_run,
        "steps": [
            {
                "name": step.name,
                "command": list(step.argv),
                "outputs": [str(path) for path in step.outputs],
            }
            for step in steps
        ],
        "failures": failures,
    }
    out_path = MAIN_PLOTS_DIR / "phase4_plot_suite_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["core", "thesis", "all"],
        default="thesis",
        help="core: canonical figure generator only; thesis: core plus summary-backed thesis figures; all: also gallery/diagnostics.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running later plot steps if an earlier one fails.",
    )
    args = parser.parse_args()

    steps = _steps_for_suite(args.suite)
    failures: list[str] = []
    for step in steps:
        rc = _run_step(step, dry_run=args.dry_run)
        if rc == 0:
            continue
        failures.append(step.name)
        if not args.continue_on_error:
            _write_manifest(steps, failures, args.dry_run)
            raise SystemExit(rc)

    _write_manifest(steps, failures, args.dry_run)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
