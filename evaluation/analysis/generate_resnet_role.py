"""ResNet role diagram — visual size-band classification (slide-ready).

Parallel to OpenCV's counting annotation: shows the actual floorplan with
each window annotated by its predicted size_cluster band, plus a side
panel making the per-element classification flow concrete.

Output: docs/plots/presentation/resnet_role.{png,pdf}
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mscd_demo_matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import (  # noqa: E402
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
FLOORPLAN_PATH = (REPO_ROOT
                  / "data_curation/datasets/synth_v0.5_ap/floorplans/"
                  "AP_SK_228_floorplan.png")
OUT_DIR = PROJECT_ROOT / "docs" / "plots" / "presentation"
OUT_BASE = OUT_DIR / "resnet_role"

# Palette
COL_INK = "#0F172A"
COL_MUTED = "#475569"
COL_RESNET = "#0F766E"
COL_TARGET = "#DC2626"      # red — matches floorplan target colour
COL_HIGHLIGHT = "#EA580C"
COL_BAND_M = "#0EA5E9"
COL_BAND_L = "#0284C7"
COL_BAND_XL = "#1E40AF"

# Per-window annotations (illustrative).
# Coordinates are in floorplan-image pixel space (954×953); each tuple is
# (cx, cy, predicted_band, confidence, label_offset_x, label_offset_y).
WINDOW_ANNOTATIONS = [
    # top-left small window (existing blue mark near top)
    (300, 80,  "window_M", 0.86, 100,  0, False),
    # mid-upper opening
    (510, 230, "window_M", 0.81, 110, -10, False),
    # TARGET (red bar)
    (510, 320, "window_XL", 0.89, 130, -10, True),
    # below target on same wall (anchor area)
    (560, 510, "window_L", 0.74, 140,  20, False),
    # bottom window
    (560, 870, "window_M", 0.83, 130, -20, False),
]

SIZE_BANDS = [
    ("window_S",  600,  900,  "#BAE6FD"),
    ("window_M",  1200, 1500, "#7DD3FC"),
    ("window_L",  1800, 1500, "#38BDF8"),
    ("window_XL", 3000, 2200, "#0EA5E9"),
    ("window_T",  900,  2400, "#0284C7"),
    ("door_S",    900,  2100, "#A7F3D0"),
    ("door_M",    1200, 2100, "#34D399"),
    ("door_D",    1800, 2100, "#10B981"),
]


def _annotate_floorplan(ax, img: any) -> None:
    """Show the floorplan and overlay one size-band tag per opening."""
    h, w = img.shape[:2]
    ax.imshow(img, extent=(0, w, h, 0), zorder=0)
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")

    for cx, cy, band, conf, dx, dy, is_target in WINDOW_ANNOTATIONS:
        # Coloured dot at the window
        if is_target:
            face, edge = "#FCA5A5", COL_TARGET
            text_face, text_edge = "#FEE2E2", COL_TARGET
            text_color = COL_TARGET
            weight = "bold"
        else:
            face, edge = "white", COL_RESNET
            text_face, text_edge = "white", COL_RESNET
            text_color = COL_INK
            weight = "normal"

        ax.add_patch(Circle((cx, cy), 18,
                            facecolor=face, edgecolor=edge,
                            linewidth=2.0, zorder=4))
        # Connector + tag
        tag_x = cx + dx
        tag_y = cy + dy
        ax.plot([cx, tag_x - 8], [cy, tag_y],
                color=edge, lw=1.4, zorder=3, alpha=0.9)
        tag_text = f"{band}\nconf={conf:.2f}"
        if is_target:
            tag_text = f"{band}  ← TARGET\nconf={conf:.2f}"
        ax.text(tag_x, tag_y, tag_text,
                ha="left", va="center",
                fontsize=10, family="monospace",
                color=text_color, fontweight=weight,
                bbox={"boxstyle": "round,pad=0.30",
                      "facecolor": text_face, "edgecolor": text_edge,
                      "linewidth": 1.6})


def _draw_pipeline(ax) -> None:
    """Right-side panel: crop → ResNet → size_cluster output."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Title strip
    ax.text(0.5, 0.96, "Per-window classification flow",
            ha="center", va="top",
            fontsize=13, fontweight="bold", color=COL_INK)
    ax.text(0.5, 0.91,
            "ResNet runs once per opening and outputs a discrete band.",
            ha="center", va="top",
            fontsize=10, color=COL_MUTED)

    # Stage 1 — crop
    ax.add_patch(FancyBboxPatch((0.10, 0.66), 0.28, 0.18,
                                boxstyle="round,pad=0.010,rounding_size=0.020",
                                facecolor="#FEE2E2", edgecolor=COL_TARGET,
                                linewidth=1.8, zorder=2))
    ax.text(0.24, 0.81, "TARGET crop",
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=COL_TARGET)
    ax.text(0.24, 0.74, "224 × 224 px",
            ha="center", va="center",
            fontsize=9.5, color=COL_INK, family="monospace")

    # Arrow 1
    ax.add_patch(FancyArrowPatch((0.40, 0.75), (0.56, 0.75),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.0, color="#94A3B8"))

    # Stage 2 — ResNet
    ax.add_patch(FancyBboxPatch((0.58, 0.66), 0.32, 0.18,
                                boxstyle="round,pad=0.010,rounding_size=0.020",
                                facecolor="#ECFDF5", edgecolor=COL_RESNET,
                                linewidth=1.8, zorder=2))
    ax.text(0.74, 0.81, "ResNet-18",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=COL_RESNET)
    ax.text(0.74, 0.73, "8-class softmax\n(size band)",
            ha="center", va="center",
            fontsize=9.5, color=COL_INK)

    # Arrow 2 (downward)
    ax.add_patch(FancyArrowPatch((0.74, 0.65), (0.74, 0.55),
                                 arrowstyle="-|>", mutation_scale=18,
                                 linewidth=2.0, color="#94A3B8"))

    # Stage 3 — size_cluster output (highlighted)
    ax.add_patch(FancyBboxPatch((0.36, 0.40), 0.55, 0.13,
                                boxstyle="round,pad=0.010,rounding_size=0.020",
                                facecolor="#FFF7ED", edgecolor=COL_HIGHLIGHT,
                                linewidth=2.0, zorder=2))
    ax.text(0.42, 0.49, "size_cluster",
            ha="left", va="top",
            fontsize=10, color=COL_MUTED, fontstyle="italic")
    ax.text(0.42, 0.45,
            "window_XL  (3000 × 2200 mm)   conf = 0.89",
            ha="left", va="top",
            fontsize=11, family="monospace",
            color=COL_HIGHLIGHT, fontweight="bold")

    # Vocabulary cheat sheet — 8 discrete bands
    ax.text(0.5, 0.33, "Output space — 8 discrete bands",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=COL_INK)
    n = len(SIZE_BANDS)
    margin = 0.04
    cell_w = (1 - 2 * margin) / n
    band_y = 0.18
    base_w = cell_w * 0.55
    base_h = 0.07
    max_w_mm = max(b[1] for b in SIZE_BANDS)
    max_h_mm = max(b[2] for b in SIZE_BANDS)
    for i, (name, w_mm, h_mm, color) in enumerate(SIZE_BANDS):
        cx = margin + i * cell_w + cell_w / 2
        rw = base_w * (w_mm / max_w_mm)
        rh = base_h * (h_mm / max_h_mm)
        is_picked = (name == "window_XL")
        ax.add_patch(FancyBboxPatch(
            (cx - cell_w / 2 + 0.005, band_y - 0.005),
            cell_w - 0.010, 0.14,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor="#FEF3C7" if is_picked else "white",
            edgecolor=COL_HIGHLIGHT if is_picked else "#CBD5E1",
            linewidth=1.6 if is_picked else 0.8, zorder=2))
        ax.add_patch(Rectangle(
            (cx - rw / 2, band_y + (base_h - rh) / 2 + 0.020),
            rw, rh,
            facecolor=color, edgecolor=COL_INK,
            linewidth=0.7, zorder=3))
        ax.text(cx, band_y + base_h + 0.038, name.replace("window_", "w_").replace("door_", "d_"),
                ha="center", va="bottom",
                fontsize=8.5, family="monospace",
                fontweight="bold" if is_picked else "normal",
                color=COL_HIGHLIGHT if is_picked else COL_INK)

    # Punchline
    ax.text(0.5, 0.04,
            "Discrete band, not mm regression — that's why it works\n"
            "(32% exact vs G8 LoRA's 5–8% on width / height in mm)",
            ha="center", va="bottom",
            fontsize=10.5, color=COL_HIGHLIGHT, fontweight="bold",
            bbox={"boxstyle": "round,pad=0.34",
                  "facecolor": "#FFF7ED", "edgecolor": COL_HIGHLIGHT,
                  "linewidth": 1.4})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = mpimg.imread(str(FLOORPLAN_PATH))

    fig = plt.figure(figsize=(14.4, 8.1), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0],
                          left=0.03, right=0.985, top=0.86, bottom=0.04,
                          wspace=0.06)
    ax_l = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    fig.suptitle(
        "ResNet — visual size-band classification per opening",
        fontsize=18, fontweight="bold", y=0.965)
    fig.text(0.5, 0.91,
             "Annotate every window crop with a discrete size band — parallel to OpenCV's ordinal counting overlay.",
             ha="center", va="top",
             fontsize=12, color=COL_MUTED)

    _annotate_floorplan(ax_l, img)
    ax_l.set_title("Floorplan with predicted size bands  (AP_SK_228, Level 1)",
                   fontsize=12, loc="left", pad=8, color=COL_INK,
                   fontweight="bold")

    _draw_pipeline(ax_r)

    png = OUT_BASE.with_suffix(".png")
    pdf = OUT_BASE.with_suffix(".pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
