"""
Modality ablation analysis — Track A (AP held-out, 60 cases).
Generates 4 publication-quality figures:

  Fig A: Grouped bar chart — hop1_acc per slice, models side-by-side
  Fig B: Heatmap — hop1_acc (model × slice), masked where no data
  Fig C: Robustness delta — Δhop1 from MC baseline per slice per model
  Fig D: FPSITE spotlight — class / storey / hop1 for G7 vs Gemini visual-only
         + per-predicate breakdown for G7 across slices
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
METRICS_ROOT = Path(
    "output/lora6_v2_ap_20260331/modality_ablation_trackA/metrics"
)
OUT_DIR = Path("output/lora6_v2_ap_20260331/modality_ablation_trackA/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── model / slice config ───────────────────────────────────────────────────────
MODELS = [
    ("g3_fullaug_r32",      "G3",        "#9C27B0"),
    ("g4_ultimate",         "G4",        "#673AB7"),
    ("g7_position_context", "G7",        "#1976D2"),
    ("g8_posctx_dim",       "G8",        "#0D47A1"),
    ("gemini_ap_v2",        "Gemini v2", "#E65100"),
]
SLICES = ["MC", "MC4D", "FP", "SITE", "FPSITE", "MA"]
SLICE_LABELS = {
    "MC":    "MC\n(full context)",
    "MC4D":  "MC4D\n(4-dir chat)",
    "FP":    "FP\n(floorplan only)",
    "SITE":  "SITE\n(site image only)",
    "FPSITE":"FPSITE\n(visual only)",
    "MA":    "MA\n(all images)",
}
FOCUS_MODELS = ["g7_position_context", "g8_posctx_dim", "gemini_ap_v2"]

# ── data loading ───────────────────────────────────────────────────────────────
def load_metric(model_key: str, slice_key: str, field: str = "hop1_acc"):
    p = METRICS_ROOT / slice_key / f"{model_key}__ap_metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(field)

def load_all() -> dict:
    """Returns data[model_key][slice_key] = dict of metrics (or None)."""
    data = {}
    for key, _, _ in MODELS:
        data[key] = {}
        for sl in SLICES:
            p = METRICS_ROOT / sl / f"{key}__ap_metrics.json"
            if p.exists():
                data[key][sl] = json.loads(p.read_text())
            else:
                data[key][sl] = None
    return data

DATA = load_all()

# ── Figure A: Grouped bar chart ────────────────────────────────────────────────
def fig_a_grouped_bar():
    fig, ax = plt.subplots(figsize=(13, 5))
    n_models = len(MODELS)
    n_slices = len(SLICES)
    bar_w = 0.15
    group_gap = 0.05
    x = np.arange(n_slices)

    offsets = np.linspace(
        -(n_models - 1) / 2 * (bar_w + group_gap / n_models),
         (n_models - 1) / 2 * (bar_w + group_gap / n_models),
        n_models,
    )

    for i, (key, label, color) in enumerate(MODELS):
        vals = []
        has_val = []
        for sl in SLICES:
            d = DATA[key][sl]
            if d is not None:
                vals.append(d["hop1_acc"] * 100)
                has_val.append(True)
            else:
                vals.append(0)
                has_val.append(False)

        bars = ax.bar(
            x + offsets[i], vals, bar_w,
            color=color, alpha=0.85, label=label,
            zorder=3,
        )
        # annotate values above bars; grey cross for missing
        for j, (bar, hv, v) in enumerate(zip(bars, has_val, vals)):
            if hv:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 0.8,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=6.5, color=color,
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.5, "✗",
                    ha="center", va="bottom", fontsize=8, color="#AAAAAA",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([SLICE_LABELS[s] for s in SLICES], fontsize=9)
    ax.set_ylabel("Hop-1 Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Track A — Modality Ablation: Hop-1 Accuracy across Input Conditions\n"
        "(n=60 AP held-out cases; ✗ = run pending)",
        fontsize=11,
    )
    ax.axhline(50, color="#CCCCCC", lw=0.8, ls="--", zorder=1)
    ax.axhline(80, color="#CCCCCC", lw=0.8, ls="--", zorder=1)
    ax.yaxis.grid(True, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)

    out = OUT_DIR / "figA_grouped_bar.png"
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ── Figure B: Heatmap ──────────────────────────────────────────────────────────
def fig_b_heatmap():
    # focus on the 5 models × 6 slices
    model_keys = [k for k, _, _ in MODELS]
    model_labels = [lb for _, lb, _ in MODELS]

    matrix = np.full((len(model_keys), len(SLICES)), np.nan)
    for i, key in enumerate(model_keys):
        for j, sl in enumerate(SLICES):
            d = DATA[key][sl]
            if d is not None:
                matrix[i, j] = d["hop1_acc"] * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = matplotlib.cm.RdYlGn.copy()
    cmap.set_bad(color="#F0F0F0")

    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # annotate cells
    for i in range(len(model_keys)):
        for j in range(len(SLICES)):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#AAAAAA")
            else:
                clr = "white" if v < 40 or v > 85 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=clr)

    ax.set_xticks(range(len(SLICES)))
    ax.set_xticklabels([SLICE_LABELS[s] for s in SLICES], fontsize=9)
    ax.set_yticks(range(len(model_keys)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title(
        "Hop-1 Accuracy Heatmap — Modality Ablation (Track A)\n"
        "Green = high accuracy; Red = low; — = run pending",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, shrink=0.8, label="Hop-1 Accuracy (%)")
    fig.tight_layout()
    out = OUT_DIR / "figB_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ── Figure C: Delta from MC baseline ──────────────────────────────────────────
def fig_c_delta():
    """Δhop1 = hop1(slice) - hop1(MC) for each model × slice (skip MC itself)."""
    compare_slices = ["MC4D", "FP", "SITE", "FPSITE", "MA"]
    model_keys = [k for k, _, _ in MODELS]
    model_labels = [lb for _, lb, _ in MODELS]
    colors = [c for _, _, c in MODELS]

    fig, ax = plt.subplots(figsize=(11, 5))
    n_models = len(model_keys)
    n_slices = len(compare_slices)
    bar_w = 0.14
    x = np.arange(n_slices)

    offsets = np.linspace(
        -(n_models - 1) / 2 * bar_w * 1.2,
         (n_models - 1) / 2 * bar_w * 1.2,
        n_models,
    )

    for i, (key, label, color) in enumerate(MODELS):
        mc_d = DATA[key]["MC"]
        if mc_d is None:
            continue
        mc_base = mc_d["hop1_acc"] * 100

        deltas = []
        has_val = []
        for sl in compare_slices:
            d = DATA[key][sl]
            if d is not None:
                deltas.append(d["hop1_acc"] * 100 - mc_base)
                has_val.append(True)
            else:
                deltas.append(0)
                has_val.append(False)

        bars = ax.bar(
            x + offsets[i], deltas, bar_w,
            color=color, alpha=0.85, label=f"{label} (MC={mc_base:.0f}%)",
            zorder=3,
        )
        for j, (bar, hv, dv) in enumerate(zip(bars, has_val, deltas)):
            if hv:
                va = "bottom" if dv >= 0 else "top"
                y_off = 0.3 if dv >= 0 else -0.3
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    dv + y_off,
                    f"{dv:+.1f}",
                    ha="center", va=va, fontsize=6.5, color=color,
                    fontweight="bold",
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    -0.5, "✗",
                    ha="center", va="top", fontsize=8, color="#AAAAAA",
                )

    ax.axhline(0, color="black", lw=1.0, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [SLICE_LABELS[s] for s in compare_slices], fontsize=9
    )
    ax.set_ylabel("Δ Hop-1 vs MC baseline (pp)", fontsize=10)
    ax.set_title(
        "Modality Robustness: Hop-1 Degradation vs Full-Context Baseline (MC)\n"
        "Negative = worse than full-context; ✗ = run pending",
        fontsize=11,
    )
    ax.yaxis.grid(True, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "figC_delta_from_mc.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ── Figure D: FPSITE spotlight + per-predicate breakdown ──────────────────────
def fig_d_fpsite_spotlight():
    """Two-panel figure:
       Left  — class / storey / hop1 bar comparison across slices for G7 vs Gemini
       Right — G7 per-predicate hop1 breakdown: NEXT_TO vs others across slices
    """
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: multi-metric bar (G7 vs Gemini) across key slices ──
    compare = ["MC", "FP", "SITE", "FPSITE"]
    metrics_spec = [
        ("class_acc",  "Class Acc",  "//"),
        ("storey_acc", "Storey Acc", "\\\\"),
        ("hop1_acc",   "Hop-1",      ""),
    ]
    focus = [
        ("g7_position_context", "G7",        "#1976D2"),
        ("gemini_ap_v2",        "Gemini v2", "#E65100"),
    ]
    n_metrics = len(metrics_spec)
    group_w = 0.6
    bar_w = group_w / (len(focus) * n_metrics + 1)
    positions = []
    labels = []
    x_cursor = 0

    for sl in compare:
        group_center = x_cursor + group_w / 2
        positions.append(group_center)
        labels.append(SLICE_LABELS[sl])
        sub_x = x_cursor
        for fi, (key, mlabel, color) in enumerate(focus):
            d = DATA[key][sl]
            for mi, (field, flabel, hatch) in enumerate(metrics_spec):
                bx = sub_x + (fi * n_metrics + mi) * bar_w
                val = (d[field] * 100) if d is not None else 0
                alpha = 0.9 if d is not None else 0.2
                ax_left.bar(
                    bx, val, bar_w * 0.9,
                    color=color, hatch=hatch, alpha=alpha,
                    zorder=3,
                    label=f"{mlabel} {flabel}" if x_cursor == 0 else "_",
                )
                if d is not None:
                    ax_left.text(
                        bx + bar_w * 0.45, val + 1,
                        f"{val:.0f}", ha="center", va="bottom",
                        fontsize=6.5, color=color,
                    )
        x_cursor += group_w + 0.25

    ax_left.set_xticks(positions)
    ax_left.set_xticklabels(labels, fontsize=9)
    ax_left.set_ylabel("Accuracy (%)", fontsize=10)
    ax_left.set_ylim(0, 115)
    ax_left.set_title(
        "G7 vs Gemini: Class / Storey / Hop-1\nacross selected conditions",
        fontsize=10,
    )
    ax_left.yaxis.grid(True, color="#EEEEEE", zorder=0)
    ax_left.set_axisbelow(True)
    ax_left.spines[["top", "right"]].set_visible(False)
    # Legend: model × metric combo via proxy patches + hatches
    legend_items = []
    for key, mlabel, color in focus:
        for field, flabel, hatch in metrics_spec:
            legend_items.append(
                mpatches.Patch(facecolor=color, hatch=hatch, alpha=0.85,
                               label=f"{mlabel} — {flabel}")
            )
    ax_left.legend(handles=legend_items, fontsize=7.5, loc="upper right",
                   framealpha=0.9, ncol=2)

    # ── Right panel: G7 per-predicate hop1 across slices ──
    g7_slices_avail = [sl for sl in SLICES if DATA["g7_position_context"][sl] is not None]
    predicates = ["ADJACENT_TO", "FILLS", "NEXT_TO", "CONNECTS_TO"]
    pred_colors = {
        "ADJACENT_TO": "#42A5F5",
        "FILLS":       "#66BB6A",
        "NEXT_TO":     "#EF5350",
        "CONNECTS_TO": "#FFA726",
    }
    x_g7 = np.arange(len(g7_slices_avail))
    n_pred = len(predicates)
    pw = 0.18
    p_offsets = np.linspace(-(n_pred - 1) / 2 * pw * 1.1,
                             (n_pred - 1) / 2 * pw * 1.1, n_pred)

    for pi, pred in enumerate(predicates):
        vals = []
        for sl in g7_slices_avail:
            d = DATA["g7_position_context"][sl]
            pp = d.get("per_predicate", {}) if d else {}
            pdata = pp.get(pred, None)
            # hop1 variants stored with and without _hop2 suffix
            if pdata:
                vals.append(pdata.get("hop1_acc", 0) * 100)
            else:
                vals.append(np.nan)

        mask = ~np.isnan(vals)
        xs = x_g7 + p_offsets[pi]
        ax_right.bar(
            xs[mask], np.array(vals)[mask], pw,
            color=pred_colors[pred], alpha=0.85, label=pred, zorder=3,
        )
        for xi, (v, m) in enumerate(zip(vals, mask)):
            if m:
                ax_right.text(
                    x_g7[xi] + p_offsets[pi],
                    v + 1,
                    f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7,
                    color=pred_colors[pred], fontweight="bold",
                )

    ax_right.set_xticks(x_g7)
    ax_right.set_xticklabels(
        [SLICE_LABELS[s] for s in g7_slices_avail], fontsize=8
    )
    ax_right.set_ylabel("Hop-1 Accuracy (%)", fontsize=10)
    ax_right.set_ylim(0, 115)
    ax_right.set_title(
        "G7 Per-Predicate Hop-1\nacross Modality Conditions",
        fontsize=10,
    )
    ax_right.yaxis.grid(True, color="#EEEEEE", zorder=0)
    ax_right.set_axisbelow(True)
    ax_right.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax_right.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "FPSITE Spotlight: Visual-Only Interpretation Capability\n"
        "Gemini collapses to 5% hop-1 without text; G7 retains 60%",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = OUT_DIR / "figD_fpsite_spotlight.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

# ── Figure E: Summary table as styled table image ─────────────────────────────
def fig_e_summary_table():
    """Print a concise summary table to terminal and save as text."""
    header = f"{'Model':<12} " + "".join(f"{s:>8}" for s in SLICES)
    print("\n" + "=" * 70)
    print("MODALITY ABLATION — Hop-1 Accuracy Summary")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for key, label, _ in MODELS:
        row = f"{label:<12} "
        for sl in SLICES:
            d = DATA[key][sl]
            if d is not None:
                row += f"{d['hop1_acc']*100:>7.1f}% "
            else:
                row += f"{'—':>7}  "
        print(row)
    print("=" * 70)

    # Also save per-predicate G7 breakdown
    print("\nG7 Per-Predicate Hop-1 Accuracy:")
    print(f"{'Pred':<14} " + "".join(f"{s:>8}" for s in SLICES))
    print("-" * 62)
    predicates = ["ADJACENT_TO", "FILLS", "NEXT_TO", "CONNECTS_TO"]
    for pred in predicates:
        row = f"{pred:<14} "
        for sl in SLICES:
            d = DATA["g7_position_context"][sl]
            if d is None:
                row += f"{'—':>8} "
                continue
            pp = d.get("per_predicate", {}).get(pred)
            if pp:
                row += f"{pp['hop1_acc']*100:>7.1f}% "
            else:
                row += f"{'n/a':>8} "
        print(row)


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent.parent)  # repo root = mscd_demo/

    print("Generating modality ablation plots...")
    fig_a_grouped_bar()
    fig_b_heatmap()
    fig_c_delta()
    fig_d_fpsite_spotlight()
    fig_e_summary_table()
    print(f"\nAll plots saved to: {OUT_DIR}")
