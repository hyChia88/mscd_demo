#!/usr/bin/env python3
"""Create a thesis-ready side-by-side oracle waterfall combo figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOCS_PLOTS_DIR = PROJECT_ROOT / "docs" / "plots" / "phase4_lora6_main"
GROUP4_ORACLE_DIR = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "group4_post-hoc_analysis"
    / "oracle_ceiling"
    / "20260404"
)

LEFT_DEFAULT = DOCS_PLOTS_DIR / "fig07_oracle_progression_waterfall.png"
RIGHT_DEFAULT = GROUP4_ORACLE_DIR / "oracle_fingerprint_waterfall.png"
OUT_DEFAULT = DOCS_PLOTS_DIR / "fig08_oracle_dual_waterfall_combo.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=LEFT_DEFAULT)
    parser.add_argument("--right", type=Path, default=RIGHT_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    left_img = mpimg.imread(args.left)
    right_img = mpimg.imread(args.right)

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.5), constrained_layout=True)
    panels = [
        (
            axes[0],
            left_img,
            "A. Realized oracle progression",
            "P1-only -> full topology -> phase3 planner ceiling -> best realized model (G3)",
        ),
        (
            axes[1],
            right_img,
            "B. Fingerprint information-loss waterfall",
            "L-query pool collapse from storey+type to exact slot, with ideal Top-10 / Top-1",
        ),
    ]

    for ax, image, title, subtitle in panels:
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.5,
            1.02,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )
        ax.text(
            0.5,
            -0.06,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )

    fig.suptitle(
        "Oracle ceiling, planner progression, and fingerprint information loss on AP held-out",
        fontsize=17,
        fontweight="bold",
        y=1.03,
    )
    fig.text(
        0.5,
        -0.01,
        "The left panel shows the realized planner progression under the current symbolic backend. "
        "The right panel decomposes the same oracle ceiling into information layers, showing how richer fingerprints collapse the candidate pool and raise ideal Top-10/Top-1.",
        ha="center",
        fontsize=10.5,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Wrote combo figure to {args.out}")


if __name__ == "__main__":
    main()
