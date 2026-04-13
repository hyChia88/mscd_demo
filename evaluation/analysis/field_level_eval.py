"""
field_level_eval.py
===================
Tests per-field VLM learning across 60 AP held-out cases.

Fields evaluated:
  - target_width_mm / target_height_mm   (numeric, tolerance ±5mm)
  - object_material                      (exact string, case-insensitive, SR[0] only)
  - position_context                     (exact + type-level partial match)
  - predicate                            (SR[0] exact match)
  - storey_name                          (normalized match)
  - ifc_class                            (exact match)

Usage:
  python evaluation/analysis/field_level_eval.py
  python evaluation/analysis/field_level_eval.py --out-dir docs/plots/field_level
"""
import argparse
import json
import re
import collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR     = (PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" /
                "modality_ablation_trackA" / "predictions")
MAIN_DIR     = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
GT_PATH      = (PROJECT_ROOT.parent / "data_curation" / "datasets" /
                "synth_v0.5_ap" / "train" / "lora6_v2_ap_eval_canonical_m_g7.jsonl")
DEFAULT_OUT  = PROJECT_ROOT / "docs" / "plots" / "field_level"

# Models to compare: (display_name, pred_file, color)
# G3/G4/G6: main AP eval files (no modality slice needed — full 60-case MC-equivalent)
# G7/G8/Gemini: MC slice from modality ablation (consistent condition)
MODELS = [
    ("G3",     MAIN_DIR / "g3_fullaug_r32__ap_eval.jsonl",            "#D32F2F"),
    ("G4",     MAIN_DIR / "g4_ultimate__ap_eval.jsonl",               "#B71C1C"),
    ("G7",     PRED_DIR / "g7_position_context__MC__ap_eval.jsonl",   "#6A1B9A"),
    ("G8",     PRED_DIR / "g8_posctx_dim__MC__ap_eval.jsonl",         "#1B5E20"),
    ("Gemini", PRED_DIR / "gemini_ap_v2__MC__ap_eval.jsonl",          "#1565C0"),
]

# ── helpers ───────────────────────────────────────────────────────────────────
def _norm_storey(s: str) -> str:
    s = str(s).lower().strip()
    m = re.search(r"(-?\d+)", s)
    return m.group(1) if m else s

def _pos_ctx_type(ctx: str) -> str:
    """Return coarse type of position context string."""
    ctx = ctx.lower()
    if "junction" in ctx:   return "junction"
    if "corner" in ctx:     return "corner"
    if "end" in ctx:        return "end"
    if "opening" in ctx:    return "opening"
    return "other"

def load_gt() -> dict:
    gt = {}
    with open(GT_PATH) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            msgs = r.get("messages", [])
            asst = next((m for m in msgs if m["role"] == "assistant"), None)
            if asst:
                try:
                    gt[r["id"]] = json.loads(asst["content"])
                except Exception:
                    pass
    return gt

def _parse_raw(raw: str) -> dict | None:
    """Parse JSON output, with fallback for truncated strings."""
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Truncated JSON: try completing it
    for suffix in ["}", "}]}", "}]}", '"}]}', '"]}']:
        try:
            return json.loads(raw + suffix)
        except Exception:
            pass
    # Last resort: regex-extract top-level string fields
    import re
    result = {}
    for key in ["storey_name", "ifc_class", "space_name", "position_context"]:
        pat = '"' + key + '"' + r'\s*:\s*"([^"]*)"'
        m = re.search(pat, raw)
        if m:
            result[key] = m.group(1)
    for key in ["target_width_mm", "target_height_mm"]:
        pat = '"' + key + '"' + r'\s*:\s*(\d+\.?\d*)'
        m = re.search(pat, raw)
        if m:
            result[key] = float(m.group(1))
    sr_pred = re.search(r'"predicate"\s*:\s*"([^"]+)"', raw)
    sr_mat  = re.search(r'"object_material"\s*:\s*"([^"]+)"', raw)
    sr_type = re.search(r'"object_type"\s*:\s*"([^"]+)"', raw)
    if sr_pred or sr_mat or sr_type:
        sr = {}
        if sr_pred: sr["predicate"] = sr_pred.group(1)
        if sr_mat:  sr["object_material"] = sr_mat.group(1)
        if sr_type: sr["object_type"] = sr_type.group(1)
        result["spatial_relations"] = [sr]
    return result if result else None


def load_preds(path: Path) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            preds[r["case_id"]] = _parse_raw(r.get("raw_output", ""))
    return preds

def analyze(gt: dict, preds: dict) -> dict:
    common = sorted(set(gt.keys()) & set(preds.keys()))
    n = len(common)

    # accumulators
    dim_res = {f: dict(gt_n=0, null=0, attempt=0, correct=0, wrong=0)
               for f in ["target_width_mm", "target_height_mm"]}
    mat = dict(attempt=0, correct=0, total=0, confusion=collections.Counter())
    pos = dict(exact=0, type_match=0, no_match=0, total=0)
    pred_r = dict(correct=0, total=0)
    storey = dict(correct=0, total=0)
    cls    = dict(correct=0, total=0)

    per_case = []

    for cid in common:
        g  = gt[cid]
        p  = preds[cid] or {}
        g_srs = g.get("spatial_relations", [])
        p_srs = p.get("spatial_relations", [])
        g_sr0 = g_srs[0] if g_srs else {}
        p_sr0 = p_srs[0] if p_srs else {}

        row = {"case_id": cid}

        # ── dimensions ──────────────────────────────────────────────────────
        for field in ["target_width_mm", "target_height_mm"]:
            gv = g.get(field)
            pv = p.get(field)
            if gv is not None:
                dim_res[field]["gt_n"] += 1
                if pv is not None:
                    dim_res[field]["attempt"] += 1
                    try:
                        if abs(float(pv) - float(gv)) < 5.0:
                            dim_res[field]["correct"] += 1
                        else:
                            dim_res[field]["wrong"] += 1
                    except (TypeError, ValueError):
                        dim_res[field]["wrong"] += 1
                else:
                    dim_res[field]["null"] += 1
            row[field] = dict(gt=gv, pred=pv)

        # ── predicate + material (all SRs, predicate-aligned) ───────────────
        # Group GT and pred SRs by predicate so order in the flat list does
        # not matter.  Within each predicate group, zip by occurrence position
        # to handle duplicate predicates (e.g. two NEXT_TO in a triad).
        g_by_pred: dict = {}
        p_by_pred: dict = {}
        for sr in g_srs:
            k = (sr.get("predicate") or "").upper()
            g_by_pred.setdefault(k, []).append(sr)
        for sr in p_srs:
            k = (sr.get("predicate") or "").upper()
            p_by_pred.setdefault(k, []).append(sr)

        case_mat_pairs = []
        for pred_key, g_list in g_by_pred.items():
            if not pred_key:
                continue
            p_list = p_by_pred.get(pred_key, [])
            for i, g_sr in enumerate(g_list):
                # predicate hit: prediction has at least i+1 SRs of this type
                pred_r["total"] += 1
                pred_r["correct"] += (1 if i < len(p_list) else 0)

                # material: compare against aligned pred SR (same predicate,
                # same occurrence index); empty string if no matching pred SR
                gm = (g_sr.get("object_material") or "").strip()
                pm = (p_list[i].get("object_material") or "").strip() \
                     if i < len(p_list) else ""
                if gm:
                    mat["total"] += 1
                    if pm:
                        mat["attempt"] += 1
                    if gm.lower() == pm.lower():
                        mat["correct"] += 1
                    else:
                        mat["confusion"][(gm, pm)] += 1
                case_mat_pairs.append((gm, pm))

        row["predicate"] = dict(
            gt=[sr.get("predicate", "") for sr in g_srs],
            pred=[sr.get("predicate", "") for sr in p_srs],
        )
        row["material"] = case_mat_pairs

        # ── position_context ─────────────────────────────────────────────────
        gc = (g.get("position_context") or "").strip()
        pc = (p.get("position_context") or "").strip()
        if gc:
            pos["total"] += 1
            if gc.lower() == pc.lower():
                pos["exact"] += 1
            elif _pos_ctx_type(gc) == _pos_ctx_type(pc):
                pos["type_match"] += 1
            else:
                pos["no_match"] += 1
        row["position_context"] = dict(gt=gc, pred=pc)

        # ── storey ───────────────────────────────────────────────────────────
        gs = _norm_storey(g.get("storey_name", ""))
        ps = _norm_storey(p.get("storey_name", ""))
        if gs:
            storey["total"] += 1
            if gs == ps: storey["correct"] += 1
        row["storey"] = dict(gt=gs, pred=ps)

        # ── ifc_class ────────────────────────────────────────────────────────
        gc_ = (g.get("ifc_class") or "").strip()
        pc_ = (p.get("ifc_class") or "").strip()
        if gc_:
            cls["total"] += 1
            if gc_.lower() == pc_.lower(): cls["correct"] += 1
        row["class"] = dict(gt=gc_, pred=pc_)

        per_case.append(row)

    return dict(n=n, dims=dim_res, material=mat, position=pos,
                predicate=pred_r, storey=storey, cls=cls, per_case=per_case)


def _pct(num, den) -> float:
    return 100.0 * num / den if den else 0.0


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_radar(results: dict, out_dir: Path):
    """Spider plot: per-model field accuracy."""
    labels = ["ifc_class", "storey", "predicate\n(SR[0])", "material\n(SR[0])",
              "position\n(exact)", "width_mm\n(attempt)", "height_mm\n(attempt)"]

    def get_scores(r):
        d = r["dims"]
        return [
            _pct(r["cls"]["correct"], r["cls"]["total"]),
            _pct(r["storey"]["correct"], r["storey"]["total"]),
            _pct(r["predicate"]["correct"], r["predicate"]["total"]),
            _pct(r["material"]["correct"], r["material"]["total"]),
            _pct(r["position"]["exact"], r["position"]["total"]),
            _pct(d["target_width_mm"]["attempt"], d["target_width_mm"]["gt_n"]),
            _pct(d["target_height_mm"]["attempt"], d["target_height_mm"]["gt_n"]),
        ]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], size=7, color="grey")
    ax.set_ylim(0, 100)

    for name, _, color in MODELS:
        if name not in results: continue
        vals = get_scores(results[name])
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=name, markersize=4)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title("Field-Level Accuracy (MC, full modality)", pad=18, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "field_radar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_field_bars(results: dict, out_dir: Path):
    """Grouped bar chart for all fields × models."""
    fields = [
        ("ifc_class",      lambda r: _pct(r["cls"]["correct"], r["cls"]["total"])),
        ("storey_name",    lambda r: _pct(r["storey"]["correct"], r["storey"]["total"])),
        ("predicate",      lambda r: _pct(r["predicate"]["correct"], r["predicate"]["total"])),
        ("material",       lambda r: _pct(r["material"]["correct"], r["material"]["total"])),
        ("pos_ctx exact",  lambda r: _pct(r["position"]["exact"], r["position"]["total"])),
        ("pos_ctx type",   lambda r: _pct(r["position"]["exact"] + r["position"]["type_match"], r["position"]["total"])),
        ("width attempt",  lambda r: _pct(r["dims"]["target_width_mm"]["attempt"], r["dims"]["target_width_mm"]["gt_n"])),
        ("width correct",  lambda r: _pct(r["dims"]["target_width_mm"]["correct"], r["dims"]["target_width_mm"]["gt_n"])),
        ("height attempt", lambda r: _pct(r["dims"]["target_height_mm"]["attempt"], r["dims"]["target_height_mm"]["gt_n"])),
    ]
    model_names = [n for n, _, _ in MODELS if n in results]
    colors = {n: c for n, _, c in MODELS}
    x = np.arange(len(fields))
    w = 0.22
    offsets = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2, len(model_names)) * w

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (name, offset) in enumerate(zip(model_names, offsets)):
        vals = [fn(results[name]) for _, fn in fields]
        ax.bar(x + offset, vals, w, color=colors[name], label=name, alpha=0.85, edgecolor="white")
        for xi, v in zip(x + offset, vals):
            if v > 0:
                ax.text(xi, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7, color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels([f for f, _ in fields], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy / Attempt Rate (%)")
    ax.set_ylim(0, 115)
    model_label = " vs ".join(model_names)
    ax.set_title(f"Per-Field Learning: {model_label} (MC condition, n=60)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(100, color="lightgrey", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "field_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_dimension_stacked(results: dict, out_dir: Path):
    """Stacked bar: null / correct / wrong for width & height per model."""
    model_names = [n for n, _, _ in MODELS if n in results]
    colors = {n: c for n, _, c in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    dim_labels = ["target_width_mm", "target_height_mm"]
    titles = ["Width (target_width_mm)", "Height (target_height_mm)"]

    for ax, dim, title in zip(axes, dim_labels, titles):
        x = np.arange(len(model_names))
        null_vals, correct_vals, wrong_vals, gt_ns = [], [], [], []
        for name in model_names:
            d = results[name]["dims"][dim]
            gn = d["gt_n"] or 1
            gt_ns.append(d["gt_n"])
            null_vals.append(100 * d["null"] / gn)
            correct_vals.append(100 * d["correct"] / gn)
            wrong_vals.append(100 * d["wrong"] / gn)

        ax.bar(x, null_vals,    color="#BDBDBD", label="null (no output)")
        ax.bar(x, correct_vals, bottom=null_vals, color="#43A047", label="correct (±5mm)")
        ax.bar(x, wrong_vals,
               bottom=[n+c for n,c in zip(null_vals, correct_vals)],
               color="#E53935", label="wrong value")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n}\n(GT n={gn})" for n, gn in zip(model_names, gt_ns)])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% of GT-present cases")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Dimension Learning: target_width/height_mm\n(null = model outputs null; correct = within ±5mm)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    out = out_dir / "field_dimensions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_material_breakdown(results: dict, out_dir: Path):
    """Material accuracy + top-5 GT materials breakdown."""
    model_names = [n for n, _, _ in MODELS if n in results]
    colors_list = [c for n, _, c in MODELS if n in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: overall material exact-match accuracy
    ax = axes[0]
    accs = [_pct(results[n]["material"]["correct"], results[n]["material"]["total"]) for n in model_names]
    bars = ax.bar(model_names, accs, color=colors_list, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title("object_material Accuracy\n(all SRs, predicate-aligned, case-insensitive)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Right: common GT materials vs prediction distribution
    ax2 = axes[1]
    # Pool all GT materials from all SR pairs (row["material"] is list of (gt, pred) tuples)
    gt_mat_pool = collections.Counter()
    for name in model_names[:1]:  # same GT across models
        for row in results[name]["per_case"]:
            for gm, _ in row["material"]:
                if gm: gt_mat_pool[gm] += 1
    top_gts = [m for m, _ in gt_mat_pool.most_common(6)]
    x = np.arange(len(top_gts))
    w = 0.25

    for i, (name, offset) in enumerate(zip(model_names,
                                            np.linspace(-w, w, len(model_names)))):
        match_rates = []
        for gt_m in top_gts:
            total = gt_mat_pool[gt_m]
            correct = sum(1 for row in results[name]["per_case"]
                          for gm, pm in row["material"]
                          if gm == gt_m and gm.lower() == pm.lower())
            match_rates.append(100 * correct / total if total else 0)
        ax2.bar(x + offset, match_rates, w, color=colors_list[i], label=name, alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{m}\n(n={gt_mat_pool[m]})" for m in top_gts],
                        rotation=20, ha="right", fontsize=8)
    ax2.set_ylabel("Correct Match (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Per-Material Accuracy (top GT materials)", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "field_material.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_position_context(results: dict, out_dir: Path):
    """Position context breakdown: exact / type-match / no-match."""
    model_names = [n for n, _, _ in MODELS if n in results]
    colors_list = [c for n, _, c in MODELS if n in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(model_names))
    w = 0.25

    exact_vals, type_vals, no_vals = [], [], []
    for name in model_names:
        pos = results[name]["position"]
        tot = pos["total"] or 1
        exact_vals.append(100 * pos["exact"] / tot)
        type_vals.append(100 * pos["type_match"] / tot)
        no_vals.append(100 * pos["no_match"] / tot)

    ax.bar(x, exact_vals,  color="#43A047", label="Exact match")
    ax.bar(x, type_vals,   bottom=exact_vals, color="#FFA726", label="Type-match (junction/opening/corner)")
    ax.bar(x, no_vals,
           bottom=[e+t for e,t in zip(exact_vals, type_vals)],
           color="#EF5350", label="No match")
    remaining = [100-(e+t+n) for e,t,n in zip(exact_vals, type_vals, no_vals)]
    ax.bar(x, remaining,
           bottom=[e+t+n for e,t,n in zip(exact_vals, type_vals, no_vals)],
           color="#BDBDBD", label="No output (null/empty)")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of cases (n=59)")
    ax.set_title("position_context Learning\n(exact count match vs type match vs miss)",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Add text labels
    for xi, (e, t, nm) in enumerate(zip(exact_vals, type_vals, no_vals)):
        if e > 2: ax.text(xi, e/2, f"{e:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if t > 5: ax.text(xi, e + t/2, f"{t:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold")

    fig.tight_layout()
    out = out_dir / "field_position_context.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def print_summary(results: dict):
    model_names = [n for n, _, _ in MODELS if n in results]
    col_w = 10
    print("\n" + "="*(28 + col_w * len(model_names)))
    print("  FIELD-LEVEL LEARNING SUMMARY  (MC condition, n=60)")
    print("="*(28 + col_w * len(model_names)))
    header = f"{'Field':<22}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print(header)
    print("-"*(22 + col_w * len(model_names)))

    def row(label, fn):
        vals = []
        for n in model_names:
            try:
                vals.append(f"{fn(results[n]):.1f}%")
            except Exception:
                vals.append("N/A")
        print(f"  {label:<20}" + "".join(f"{v:>{col_w}}" for v in vals))

    # helper: denominator label for a field across models
    def _denom(fn_n):
        """Return the denominator value used (same across all models for GT-derived fields)."""
        for n in model_names:
            try:
                v = fn_n(results[n])
                if v: return v
            except Exception:
                pass
        return "?"

    n_cases   = _denom(lambda r: r["cls"]["total"])
    n_srs     = _denom(lambda r: r["predicate"]["total"])
    n_pos     = _denom(lambda r: r["position"]["total"])
    n_w_gt    = _denom(lambda r: r["dims"]["target_width_mm"]["gt_n"])
    n_h_gt    = _denom(lambda r: r["dims"]["target_height_mm"]["gt_n"])

    row(f"ifc_class  (n={n_cases})",      lambda r: _pct(r["cls"]["correct"],     r["cls"]["total"]))
    row(f"storey_name  (n={n_cases})",    lambda r: _pct(r["storey"]["correct"],   r["storey"]["total"]))
    row(f"predicate  (n={n_srs} SR)",     lambda r: _pct(r["predicate"]["correct"],r["predicate"]["total"]))
    row(f"material   (n={n_srs} SR)",     lambda r: _pct(r["material"]["correct"], r["material"]["total"]))
    row(f"pos_ctx exact (n={n_pos})",     lambda r: _pct(r["position"]["exact"],   r["position"]["total"]))
    row(f"pos_ctx type  (n={n_pos})",     lambda r: _pct(r["position"]["exact"]+r["position"]["type_match"],
                                                          r["position"]["total"]))
    row(f"width attempt (n={n_w_gt} GT)", lambda r: _pct(r["dims"]["target_width_mm"]["attempt"],
                                                          r["dims"]["target_width_mm"]["gt_n"]))
    row(f"width correct (n={n_w_gt} GT)", lambda r: _pct(r["dims"]["target_width_mm"]["correct"],
                                                          r["dims"]["target_width_mm"]["gt_n"]))
    row(f"height attempt(n={n_h_gt} GT)", lambda r: _pct(r["dims"]["target_height_mm"]["attempt"],
                                                          r["dims"]["target_height_mm"]["gt_n"]))
    row(f"height correct(n={n_h_gt} GT)", lambda r: _pct(r["dims"]["target_height_mm"]["correct"],
                                                          r["dims"]["target_height_mm"]["gt_n"]))

    # Dynamic per-model notes
    print("\n  Per-model notes (auto):")
    for n in model_names:
        r = results[n]
        w_att = r["dims"]["target_width_mm"]["attempt"]
        w_cor = r["dims"]["target_width_mm"]["correct"]
        w_gt  = r["dims"]["target_width_mm"]["gt_n"]
        pc_att = r["position"]["total"]
        pc_ex  = r["position"]["exact"]
        mat_tot = r["material"]["total"]
        mat_cor = r["material"]["correct"]
        notes = []
        if w_gt > 0:
            notes.append(f"width: {w_att}/{w_gt} attempt, {w_cor}/{w_gt} correct")
        if pc_att > 0:
            notes.append(f"pos_ctx: {pc_ex}/{pc_att} exact")
        if mat_tot > 0:
            notes.append(f"material: {mat_cor}/{mat_tot} correct")
        print(f"    {n}: " + ("; ".join(notes) if notes else "no new fields"))


# ── main ──────────────────────────────────────────────────────────────────────
def plot_material_frequency_vs_accuracy(results: dict, out_dir: Path):
    """Scatter: per-class GT frequency vs accuracy, + macro vs weighted bar."""
    import matplotlib.gridspec as gridspec

    # Collect per-class stats for each model
    model_names = [n for n, _, _ in MODELS if n in results]
    colors = {n: c for n, _, c in MODELS}

    # Build per-class data — flatten to (name, gm, pm) triples, then aggregate
    # with Counter for a single-pass aggregation instead of 3-loop nesting.
    triples = [
        (name, gm, pm)
        for name in model_names
        for row in results[name]["per_case"]
        for gm, pm in row["material"]
        if gm
    ]
    total_cnt   = collections.Counter((name, gm)          for name, gm, pm in triples)
    correct_cnt = collections.Counter((name, gm)          for name, gm, pm in triples
                                       if gm.lower() == pm.lower())
    gt_classes: dict = {}
    for (name, gm), tot in total_cnt.items():
        gt_classes.setdefault(gm, {})[name] = [correct_cnt.get((name, gm), 0), tot]

    # Frequency (use G7 or first available model)
    ref_model = model_names[0]
    mat_list = sorted(gt_classes.keys(),
                      key=lambda m: gt_classes[m].get(ref_model, [0, 0])[1],
                      reverse=True)

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.6, 1], wspace=0.35)

    # Left: scatter frequency vs accuracy
    ax1 = fig.add_subplot(gs[0])
    n_m = len(model_names)
    offsets_list = np.linspace(-0.12, 0.12, n_m) if n_m > 1 else [0]
    offset = {n: offsets_list[i] for i, n in enumerate(model_names)}
    for name in model_names:
        xs, ys, labels = [], [], []
        for mat in mat_list:
            c, t = gt_classes[mat].get(name, [0, 0])
            if t == 0:
                continue
            xs.append(t + offset.get(name, 0))
            ys.append(100.0 * c / t)
            labels.append(mat)
        ax1.scatter(xs, ys, color=colors[name], s=60, label=name, zorder=3, alpha=0.85)
        # Trend line
        if len(xs) > 2:
            z = np.polyfit(xs, ys, 1)
            p = np.poly1d(z)
            xr = np.linspace(min(xs), max(xs), 50)
            ax1.plot(xr, p(xr), "--", color=colors[name], alpha=0.5, linewidth=1)

    # Label material points (use G7 positions)
    for mat in mat_list:
        c, t = gt_classes[mat].get(ref_model, [0, 0])
        if t == 0:
            continue
        ax1.annotate(mat, (t, 100.0 * c / t),
                     textcoords="offset points", xytext=(5, 3),
                     fontsize=6.5, color="#555555")

    ax1.set_xlabel("GT frequency (n cases)", fontsize=10)
    ax1.set_ylabel("Per-class accuracy (%)", fontsize=10)
    ax1.set_title("Material: Frequency vs Accuracy\n(trend = frequency bias slope)",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 16)
    ax1.set_ylim(-5, 110)
    ax1.grid(alpha=0.3)
    ax1.axhline(100, color="lightgrey", linestyle=":", linewidth=0.8)

    # Right: macro vs weighted accuracy bars
    ax2 = fig.add_subplot(gs[1])
    x = np.arange(len(model_names))
    w = 0.32

    weighted_accs, macro_accs = [], []
    for name in model_names:
        total_c = total_t = 0
        per_class_accs = []
        for mat in gt_classes:
            c, t = gt_classes[mat].get(name, [0, 0])
            if t == 0:
                continue
            total_c += c
            total_t += t
            per_class_accs.append(100.0 * c / t)
        weighted_accs.append(100.0 * total_c / total_t if total_t else 0)
        macro_accs.append(np.mean(per_class_accs) if per_class_accs else 0)

    bars_w = ax2.bar(x - w / 2, weighted_accs, w,
                     color=[colors[n] for n in model_names], alpha=0.85,
                     label="Weighted (freq-biased)", edgecolor="white")
    bars_m = ax2.bar(x + w / 2, macro_accs, w,
                     color=[colors[n] for n in model_names], alpha=0.45,
                     label="Macro (per-class avg)", edgecolor="grey", hatch="//")

    for bar, v in zip(bars_w, weighted_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
                 f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    for bar, v in zip(bars_m, macro_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
                 f"{v:.0f}%", ha="center", fontsize=9, color="grey")

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Weighted vs Macro-Avg\nmaterial accuracy", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "object_material: Frequency Bias Diagnosis\n"
        "Macro-avg reveals minority-class shortcut; weighted accuracy is inflated by top-3 classes",
        fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "field_material_frequency_bias.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth …")
    gt = load_gt()
    print(f"  {len(gt)} cases")

    results = {}
    for name, path, _ in MODELS:
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        print(f"  Loading {name} …")
        preds = load_preds(path)
        results[name] = analyze(gt, preds)

    print_summary(results)

    print("\nGenerating plots …")
    plot_radar(results, out_dir)
    plot_field_bars(results, out_dir)
    plot_dimension_stacked(results, out_dir)
    plot_material_breakdown(results, out_dir)
    plot_position_context(results, out_dir)
    plot_material_frequency_vs_accuracy(results, out_dir)

    print(f"\nDone. All plots → {out_dir}/")


if __name__ == "__main__":
    main()