#!/usr/bin/env python3
"""
Experiment Plots (E-series) for thesis deep-dive findings.

E1 — Shortcut Learning Evidence: modality identity heatmap + SR entropy
E2 — Multi-Hop Extraction & Impact: predicate accuracy + retrieval delta
E3 — Input Pattern Analysis: text feature → accuracy lift + user guide

Usage:
    cd mscd_demo && python evaluation/analysis/experiment_plots.py
"""

import json, math, warnings
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore", message=".*Tight layout not applied.*")
warnings.filterwarnings("ignore", message=".*constrained_layout.*")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (matches compare_results.py)
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "evaluation" / "experiment_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_STYLE = OrderedDict([
    ("lora5r32", ("LoRA₅-r32", "#E65100")),
    ("lora5r16", ("LoRA₅-r16", "#F5A623")),
    ("lora2",    ("LoRA₂",     "#D32F2F")),
    ("gemini",   ("Gemini",    "#1565C0")),
])

UNIFIED_DIR = PROJECT_ROOT / "logs" / "evaluation_output" / "unified"
TRACE_DIR   = UNIFIED_DIR / "strategy_ablation_v3"
CASES_PATH  = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"

TRACE_FILES = {
    "lora5r32": TRACE_DIR / "traces_20260325_170506_v2_lora_p0_union_p1.jsonl",
    "lora5r16": TRACE_DIR / "traces_20260325_170553_v2_lora_p0_union_p1.jsonl",
    "lora2":    TRACE_DIR / "traces_20260325_170638_v2_lora_p0_union_p1.jsonl",
    "gemini":   TRACE_DIR / "traces_20260325_170731_v2_lora_p0_union_p1.jsonl",
}

CONSTRAINT_FILES = {
    "lora5r32_FP": UNIFIED_DIR / "eval_constraints_lora5r32_FP.jsonl",
    "lora5r16_FP": UNIFIED_DIR / "eval_constraints_lora5r16_FP.jsonl",
    "lora2_FP":    UNIFIED_DIR / "eval_constraints_lora2_FP.jsonl",
    "gemini_FP":   UNIFIED_DIR / "eval_constraints_final_FP.jsonl",
    "lora5r32_MC": UNIFIED_DIR / "eval_constraints_lora5r32_MC.jsonl",
    "lora5r16_MC": UNIFIED_DIR / "eval_constraints_lora5r16_MC.jsonl",
    "lora2_MC":    UNIFIED_DIR / "eval_constraints_lora2_MC.jsonl",
    "gemini_MC":   UNIFIED_DIR / "eval_constraints_final_MC.jsonl",
}

MA_PATH = PROJECT_ROOT / "logs" / "evaluation_output" / "synth_v05_lora5" / "eval_constraints_final_MA.jsonl"
FP_V05_PATH = PROJECT_ROOT / "logs" / "evaluation_output" / "synth_v05_lora5" / "eval_constraints_final_FP.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _load_constraints(path):
    return {d["case_id"]: d["constraints"]
            for d in (json.loads(l) for l in open(path))}


def _load_traces(path):
    return [json.loads(l) for l in open(path)]


def _sr_sig(srs):
    if not srs:
        return "EMPTY"
    return " + ".join(
        f"{sr['predicate']}→{sr['object_type']}({sr.get('object_material', '') or ''})"
        for sr in srs
    )


def _sr_pred_seq(srs):
    if not srs:
        return ("EMPTY",)
    return tuple(sr["predicate"] for sr in srs)


def _get_gt_guid(trace):
    return trace.get("scenario", {}).get("ground_truth", {}).get("target_guid", "")


def _gt_in_pool(trace):
    gt = _get_gt_guid(trace)
    rrs = trace.get("internals", {}).get("retrieval_results", [])
    if not rrs:
        return False
    return any(c.get("guid") == gt for c in rrs[0].get("candidates", []))


def _savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# E1 — Shortcut Learning Evidence
# ═════════════════════════════════════════════════════════════════════════════
def plot_e1():
    """Two-panel figure:
    Left:  Heatmap of FP↔MC field identity rate per model
    Right: SR pattern entropy comparison (bar chart)
    """
    print("\n[E1] Shortcut Learning Evidence")

    # ── Compute FP↔MC identity ────────────────────────────────────────
    fields = ["storey_name", "ifc_class", "SR predicate seq", "ALL"]
    models = list(MODEL_STYLE.keys())
    identity_matrix = np.zeros((len(models), len(fields)))

    for mi, model in enumerate(models):
        fp_key = f"{model}_FP"
        mc_key = f"{model}_MC"
        if fp_key not in CONSTRAINT_FILES or mc_key not in CONSTRAINT_FILES:
            continue
        fp_c = _load_constraints(CONSTRAINT_FILES[fp_key])
        mc_c = _load_constraints(CONSTRAINT_FILES[mc_key])
        common = sorted(set(fp_c) & set(mc_c))
        n = len(common)
        if n == 0:
            continue

        st = cl = sr = al = 0
        for cid in common:
            f, m = fp_c[cid], mc_c[cid]
            s_eq = (f.get("storey_name") or "") == (m.get("storey_name") or "")
            c_eq = (f.get("ifc_class") or "") == (m.get("ifc_class") or "")
            sr_eq = _sr_pred_seq(f.get("spatial_relations", [])) == _sr_pred_seq(m.get("spatial_relations", []))
            st += s_eq; cl += c_eq; sr += sr_eq
            al += (s_eq and c_eq and sr_eq)

        identity_matrix[mi] = [st / n * 100, cl / n * 100, sr / n * 100, al / n * 100]

    # ── Compute SR entropy ────────────────────────────────────────────
    entropies = {}
    unique_counts = {}
    for model in models:
        fp_key = f"{model}_FP"
        if fp_key not in CONSTRAINT_FILES:
            continue
        fp_c = _load_constraints(CONSTRAINT_FILES[fp_key])
        patterns = Counter(_sr_sig(c.get("spatial_relations", [])) for c in fp_c.values())
        n = sum(patterns.values())
        if n == 0:
            continue
        entropy = -sum((c / n) * math.log2(c / n) for c in patterns.values())
        max_ent = math.log2(n)
        entropies[model] = (entropy, max_ent, entropy / max_ent * 100)
        unique_counts[model] = len(patterns)

    # ── MA vs FP identity (LoRA5-r32 only, 70 cases) ─────────────────
    ma_identity = {}
    if MA_PATH.exists() and FP_V05_PATH.exists():
        ma_c = _load_constraints(MA_PATH)
        fp_v05 = _load_constraints(FP_V05_PATH)
        common = sorted(set(ma_c) & set(fp_v05))
        n = len(common)
        if n > 0:
            st = cl = sr = al = 0
            for cid in common:
                m, f = ma_c[cid], fp_v05[cid]
                s_eq = (m.get("storey_name") or "") == (f.get("storey_name") or "")
                c_eq = (m.get("ifc_class") or "") == (f.get("ifc_class") or "")
                sr_eq = _sr_pred_seq(m.get("spatial_relations", [])) == _sr_pred_seq(f.get("spatial_relations", []))
                st += s_eq; cl += c_eq; sr += sr_eq
                al += (s_eq and c_eq and sr_eq)
            ma_identity = {
                "storey": st / n * 100, "class": cl / n * 100,
                "sr": sr / n * 100, "all": al / n * 100, "n": n,
            }

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [4, 2.2, 2.2]})

    # Panel A: FP↔MC Identity Heatmap
    ax = axes[0]
    model_labels = [MODEL_STYLE[m][0] for m in models]
    im = ax.imshow(identity_matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_labels, fontsize=11)
    for i in range(len(models)):
        for j in range(len(fields)):
            val = identity_matrix[i, j]
            color = "white" if val < 40 or val > 85 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)
    ax.set_title("(a) FP ↔ MC Field Identity Rate", fontsize=12, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Identity %", fontsize=9)

    # Add MA vs FP annotation for LoRA5-r32
    if ma_identity:
        ax.annotate(
            f"MA↔FP (LoRA₅, n={ma_identity['n']}):\n"
            f"storey={ma_identity['storey']:.0f}%  class={ma_identity['class']:.0f}%\n"
            f"SR={ma_identity['sr']:.0f}%  ALL={ma_identity['all']:.0f}%",
            xy=(0, 0), xytext=(0.02, -0.28),
            xycoords="axes fraction", textcoords="axes fraction",
            fontsize=8.5, fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec="#E65100", alpha=0.9),
        )

    # Panel B: SR Entropy
    ax = axes[1]
    x = range(len(models))
    bars_ent = [entropies.get(m, (0, 0, 0))[0] for m in models]
    max_ents = [entropies.get(m, (0, 1, 0))[1] for m in models]
    colors = [MODEL_STYLE[m][1] for m in models]

    ax.bar(x, max_ents, color="#E0E0E0", edgecolor="#BDBDBD", linewidth=0.5,
           label="Max entropy", zorder=1)
    ax.bar(x, bars_ent, color=colors, edgecolor="white", linewidth=0.5,
           label="Actual entropy", zorder=2)
    for i, m in enumerate(models):
        ent_data = entropies.get(m)
        if ent_data is None or ent_data[1] == 0:
            # LoRA2: no SR extracted → no entropy
            ax.text(i, 0.15, "N/A\n(0% SR)", ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#757575")
        else:
            ax.text(i, ent_data[0] + 0.1, f"{ent_data[2]:.0f}%", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in models], fontsize=9)
    ax.set_ylabel("Shannon Entropy (bits)", fontsize=10)
    ax.set_title("(b) SR Pattern Entropy", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(max_ents) * 1.15)

    # Panel C: Unique SR Pattern Count
    ax = axes[2]
    counts = [unique_counts.get(m, 0) for m in models]
    ax.bar(x, counts, color=colors, edgecolor="white", linewidth=0.5)
    for i, c in enumerate(counts):
        ax.text(i, c + 1, str(c), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in models], fontsize=9)
    ax.set_ylabel("Unique Patterns", fontsize=10)
    ax.set_title("(c) SR Template Diversity\n(out of 116 cases)", fontsize=12, fontweight="bold", pad=10)
    ax.axhline(y=116, color="#9E9E9E", ls="--", lw=0.8, label="max (n=116)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 130)

    fig.suptitle("E1 — Shortcut Learning Evidence: VLM Spatial Extraction Reliability",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "E1_shortcut_learning_evidence.png")


# ═════════════════════════════════════════════════════════════════════════════
# E2 — Multi-Hop Extraction & Impact
# ═════════════════════════════════════════════════════════════════════════════
def plot_e2():
    """Three-panel figure:
    Left:   Hop-1 predicate accuracy (spatial cases) — grouped bar
    Middle: Single-hop vs Multi-hop GIP comparison — grouped bar
    Right:  Hallucination rate — stacked bar
    """
    print("\n[E2] Multi-Hop Extraction & Impact")

    cases = {c["case_id"]: c
             for c in (json.loads(l) for l in open(CASES_PATH))}

    # Classify cases
    gt_preds = {}
    spatial_cases = set()
    for cid, c in cases.items():
        srs = c.get("labels", {}).get("constraints", {}).get("spatial_relations", [])
        if srs:
            spatial_cases.add(cid)
            gt_preds[cid] = srs[0]["predicate"]

    active_models = ["lora5r32", "lora5r16", "gemini"]  # skip lora2 (0% SR)

    # ── Compute metrics ───────────────────────────────────────────────
    pred_acc = {}   # model → correct/total
    hop_gip = {}    # model → {single: (hits, n), multi: (hits, n)}
    halluc = {}     # model → {single_halluc, multi_halluc, correct_no_sr, total_attr}

    for model in active_models:
        traces = _load_traces(TRACE_FILES[model])
        correct = wrong = missed = 0
        sh_hit = sh_n = mh_hit = mh_n = no_hit = no_n = 0
        attr_single_h = attr_multi_h = attr_no = 0
        attr_total = 0

        for t in traces:
            cid = t["scenario_id"]
            ext_srs = t.get("internals", {}).get("constraints", {}).get("spatial_relations", [])
            n_hops = len(ext_srs)
            hit = _gt_in_pool(t)

            # Predicate accuracy (spatial cases only)
            if cid in spatial_cases:
                gt_pred = gt_preds[cid]
                if not ext_srs:
                    missed += 1
                elif ext_srs[0]["predicate"] == gt_pred:
                    correct += 1
                else:
                    wrong += 1

            # GIP by hop count
            if n_hops == 0:
                no_hit += hit; no_n += 1
            elif n_hops == 1:
                sh_hit += hit; sh_n += 1
            else:
                mh_hit += hit; mh_n += 1

            # Hallucination (attribute-only cases)
            if cid not in spatial_cases:
                attr_total += 1
                if n_hops == 0:
                    attr_no += 1
                elif n_hops == 1:
                    attr_single_h += 1
                else:
                    attr_multi_h += 1

        total = correct + wrong + missed
        pred_acc[model] = (correct, total)
        hop_gip[model] = {
            "single": (sh_hit, sh_n),
            "multi": (mh_hit, mh_n),
            "none": (no_hit, no_n),
        }
        halluc[model] = {
            "no_sr": attr_no,
            "single": attr_single_h,
            "multi": attr_multi_h,
            "total": attr_total,
        }

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5),
                             gridspec_kw={"width_ratios": [2, 3, 2.5]})

    # Panel A: Predicate Accuracy
    ax = axes[0]
    x = np.arange(len(active_models))
    w = 0.5
    accs = [pred_acc[m][0] / pred_acc[m][1] * 100 for m in active_models]
    colors = [MODEL_STYLE[m][1] for m in active_models]
    bars = ax.bar(x, accs, w, color=colors, edgecolor="white", linewidth=0.5)
    for i, (a, bar) in enumerate(zip(accs, bars)):
        c, t = pred_acc[active_models[i]]
        ax.text(i, a + 1.5, f"{a:.0f}%\n({c}/{t})", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in active_models], fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("(a) Hop-1 Predicate Accuracy\n(40 spatial cases)", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 70)
    ax.axhline(y=100 / 5, color="#9E9E9E", ls="--", lw=0.8)
    ax.text(len(active_models) - 0.5, 20 + 1, "random (5 classes)", fontsize=7, color="#757575", ha="right")

    # Panel B: Single vs Multi-hop GIP
    ax = axes[1]
    x = np.arange(len(active_models))
    w = 0.28
    groups = ["single", "multi"]
    group_labels = ["Single-hop", "Multi-hop"]
    group_colors = ["#66BB6A", "#FFA726"]  # green, orange
    hatches = [None, "//"]

    for gi, (grp, lbl) in enumerate(zip(groups, group_labels)):
        vals = []
        ns = []
        for m in active_models:
            hits, n = hop_gip[m][grp]
            vals.append(hits / n * 100 if n else 0)
            ns.append(n)
        offset = (gi - 0.5) * w
        b = ax.bar(x + offset, vals, w * 0.9, label=lbl, color=group_colors[gi],
                   hatch=hatches[gi], edgecolor="white", linewidth=0.5)
        for i, (v, n) in enumerate(zip(vals, ns)):
            ax.text(x[i] + offset, v + 1.5, f"{v:.0f}%\n(n={n})", ha="center",
                    va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in active_models], fontsize=10)
    ax.set_ylabel("GT-in-Pool (%)", fontsize=10)
    ax.set_title("(b) Single-Hop vs Multi-Hop\nRetrieval Performance", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, 85)

    # Panel C: Hallucination (attr-only)
    ax = axes[2]
    x = np.arange(len(active_models))
    w = 0.5

    no_vals = [halluc[m]["no_sr"] for m in active_models]
    single_vals = [halluc[m]["single"] for m in active_models]
    multi_vals = [halluc[m]["multi"] for m in active_models]
    totals = [halluc[m]["total"] for m in active_models]

    # Stacked: no_sr (correct) at bottom, then single-halluc, then multi-halluc
    p1 = ax.bar(x, no_vals, w, label="Correct (no SR)", color="#66BB6A", edgecolor="white")
    p2 = ax.bar(x, single_vals, w, bottom=no_vals, label="Halluc: 1-hop SR",
                color="#FFB74D", edgecolor="white")
    bottoms2 = [n + s for n, s in zip(no_vals, single_vals)]
    p3 = ax.bar(x, multi_vals, w, bottom=bottoms2, label="Halluc: 2+ hop SR",
                color="#EF5350", edgecolor="white")

    for i in range(len(active_models)):
        total = totals[i]
        h_rate = (single_vals[i] + multi_vals[i]) / total * 100 if total else 0
        ax.text(i, total + 1, f"{h_rate:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#D32F2F")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in active_models], fontsize=10)
    ax.set_ylabel("Cases (n)", fontsize=10)
    ax.set_title("(c) SR Hallucination\n(76 attribute-only cases)", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, max(totals) * 1.2)

    fig.suptitle("E2 — Multi-Hop Spatial Relation Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "E2_multihop_analysis.png")


# ═════════════════════════════════════════════════════════════════════════════
# E3 — Input Pattern Analysis & User Guidance
# ═════════════════════════════════════════════════════════════════════════════
def plot_e3():
    """Three-panel figure:
    Left:   Text feature → field accuracy lift (grouped bars)
    Middle: Text feature → GIP lift (paired bars: with/without)
    Right:  Information source contribution table/chart
    """
    print("\n[E3] Input Pattern Analysis & User Guidance")

    cases = {c["case_id"]: c
             for c in (json.loads(l) for l in open(CASES_PATH))}

    models_to_check = ["lora5r32", "gemini"]

    # ── Extract text features & compute accuracy per model ────────────
    results_by_model = {}

    for model in models_to_check:
        traces = _load_traces(TRACE_FILES[model])
        results = []

        for t in traces:
            cid = t["scenario_id"]
            s = t["scenario"]
            case = cases.get(cid, {})

            # Gather all text
            all_text = ""
            for msg in s.get("chat_history", []):
                all_text += " " + msg.get("text", "")
            all_text += " " + s.get("query_text", "")
            meta = s.get("context_meta", {})
            task_status = meta.get("task_status", "")

            text_lower = all_text.lower()
            has_floor_chat = any(w in text_lower for w in
                                ["floor", "storey", "level", "ground", "basement", "roof"])
            has_type_chat = any(w in text_lower for w in
                               ["window", "door", "wall", "railing", "stair", "slab", "beam"])
            has_spatial_chat = any(w in text_lower for w in
                                  ["next to", "near", "adjacent", "beside", "fills", "connects"])
            has_floor_task = any(w in task_status.lower() for w in
                                ["floor", "level", "storey", "basement", "ground", "roof",
                                 "carpark", "garage"])
            has_floor_any = has_floor_chat or has_floor_task

            # Extraction accuracy
            ext = t.get("internals", {}).get("constraints", {})
            gt_c = case.get("labels", {}).get("constraints", {})

            gt_storey = (gt_c.get("storey_name") or "").lower()
            gt_class = gt_c.get("ifc_class", "")
            ext_storey = (ext.get("storey_name") or "").lower()
            ext_class = ext.get("ifc_class", "")

            storey_ok = (gt_storey != "" and ext_storey != "" and
                         (gt_storey in ext_storey or ext_storey in gt_storey))
            class_ok = (gt_class != "" and ext_class != "" and
                        (gt_class == ext_class or
                         ext_class.startswith(gt_class) or
                         gt_class.startswith(ext_class)))

            hit = _gt_in_pool(t)

            results.append({
                "has_floor_any": has_floor_any,
                "has_type_chat": has_type_chat,
                "has_spatial_chat": has_spatial_chat,
                "storey_ok": storey_ok,
                "class_ok": class_ok,
                "hit": hit,
            })

        results_by_model[model] = results

    from matplotlib.patches import Patch

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 6),
                             gridspec_kw={"width_ratios": [3.5, 3.5, 2.5]})

    features = [
        ("has_type_chat", "Element type\nmentioned in text"),
        ("has_floor_any", "Floor / storey\nmentioned in text or metadata"),
    ]
    model_colors = [MODEL_STYLE[m][1] for m in models_to_check]
    model_labels = [MODEL_STYLE[m][0] for m in models_to_check]

    # ── Panel A: Class accuracy (with vs without) per feature×model ──
    ax = axes[0]
    # Layout: 2 feature groups × 2 models × 2 conditions (with/without)
    # Simplified: horizontal grouped bars
    group_labels_a = []
    vals_with = []
    vals_without = []
    bar_colors = []

    for feat_key, feat_label in features:
        for mi, model in enumerate(models_to_check):
            results = results_by_model[model]
            w_f = [r for r in results if r[feat_key]]
            wo_f = [r for r in results if not r[feat_key]]
            cl_w = sum(r["class_ok"] for r in w_f) / len(w_f) * 100 if w_f else 0
            cl_wo = sum(r["class_ok"] for r in wo_f) / len(wo_f) * 100 if wo_f else 0
            vals_with.append(cl_w)
            vals_without.append(cl_wo)
            bar_colors.append(model_colors[mi])
            short_feat = "Type" if "type" in feat_key else "Floor"
            group_labels_a.append(f"{model_labels[mi]}\n({short_feat})")

    y = np.arange(len(group_labels_a))
    h = 0.35
    ax.barh(y + h / 2, vals_with, h, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.barh(y - h / 2, vals_without, h, color=bar_colors, alpha=0.35, edgecolor="white", linewidth=0.5)

    for i in range(len(y)):
        ax.text(vals_with[i] + 1, y[i] + h / 2, f"{vals_with[i]:.0f}%",
                va="center", fontsize=9, fontweight="bold")
        ax.text(vals_without[i] + 1, y[i] - h / 2, f"{vals_without[i]:.0f}%",
                va="center", fontsize=9, color="#757575")
        delta = vals_with[i] - vals_without[i]
        ax.text(max(vals_with[i], vals_without[i]) + 8, y[i],
                f"Δ={delta:+.0f}pp",
                va="center", fontsize=8, fontweight="bold",
                color="#2E7D32" if delta > 0 else "#C62828")

    ax.set_yticks(y)
    ax.set_yticklabels(group_labels_a, fontsize=9)
    ax.set_xlabel("ifc_class Accuracy (%)", fontsize=10)
    ax.set_title("(a) Text Feature → Class Extraction Accuracy",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, 110)
    ax.legend(handles=[
        Patch(facecolor="#888", label="With mention"),
        Patch(facecolor="#888", alpha=0.35, label="Without mention"),
    ], fontsize=8.5, loc="lower right")
    ax.invert_yaxis()
    # Separator line between feature groups
    ax.axhline(y=1.5, color="#BDBDBD", ls="--", lw=0.8)

    # ── Panel B: GIP lift per feature×model ──────────────────────────
    ax = axes[1]
    group_labels_b = []
    gip_with = []
    gip_without = []
    ns_w = []
    ns_wo = []
    bar_colors_b = []

    for feat_key, feat_label in features:
        for mi, model in enumerate(models_to_check):
            results = results_by_model[model]
            w_f = [r for r in results if r[feat_key]]
            wo_f = [r for r in results if not r[feat_key]]
            gw = sum(r["hit"] for r in w_f) / len(w_f) * 100 if w_f else 0
            gwo = sum(r["hit"] for r in wo_f) / len(wo_f) * 100 if wo_f else 0
            gip_with.append(gw)
            gip_without.append(gwo)
            ns_w.append(len(w_f))
            ns_wo.append(len(wo_f))
            bar_colors_b.append(model_colors[mi])
            short_feat = "Type" if "type" in feat_key else "Floor"
            group_labels_b.append(f"{model_labels[mi]}\n({short_feat})")

    y = np.arange(len(group_labels_b))
    h = 0.35
    ax.barh(y + h / 2, gip_with, h, color=bar_colors_b, edgecolor="white", linewidth=0.5)
    ax.barh(y - h / 2, gip_without, h, color=bar_colors_b, alpha=0.35, edgecolor="white", linewidth=0.5)

    for i in range(len(y)):
        ax.text(gip_with[i] + 1, y[i] + h / 2,
                f"{gip_with[i]:.0f}%  (n={ns_w[i]})",
                va="center", fontsize=8.5, fontweight="bold")
        ax.text(gip_without[i] + 1, y[i] - h / 2,
                f"{gip_without[i]:.0f}%  (n={ns_wo[i]})",
                va="center", fontsize=8.5, color="#757575")
        delta = gip_with[i] - gip_without[i]
        ax.text(max(gip_with[i], gip_without[i]) + 18, y[i],
                f"Δ={delta:+.0f}pp",
                va="center", fontsize=9, fontweight="bold",
                color="#2E7D32" if delta > 0 else "#C62828")

    ax.set_yticks(y)
    ax.set_yticklabels(group_labels_b, fontsize=9)
    ax.set_xlabel("GT-in-Pool (%)", fontsize=10)
    ax.set_title("(b) Text Feature → Retrieval (GT-in-Pool)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, 110)
    ax.legend(handles=[
        Patch(facecolor="#888", label="With mention"),
        Patch(facecolor="#888", alpha=0.35, label="Without mention"),
    ], fontsize=8.5, loc="lower right")
    ax.invert_yaxis()
    ax.axhline(y=1.5, color="#BDBDBD", ls="--", lw=0.8)

    # Panel C: User Input Guide (summary table as plot)
    ax = axes[2]
    ax.axis("off")

    guide_data = [
        ["Input Field",     "Priority", "Source",       "Impact"],
        ["Element type",    "★★★",     "User text",    "+23pp GIP"],
        ["Floor / storey",  "★★☆",     "Text + meta",  "+5pp storey acc"],
        ["Spatial context", "★☆☆",     "VLM (unreliable)", "±0 (noisy)"],
        ["Material",        "☆☆☆",     "VLM / text",   "Rare signal"],
        ["Multiple photos", "★★☆",     "MC images",    "+3–9pp GIP"],
    ]

    table = ax.table(
        cellText=guide_data[1:],
        colLabels=guide_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(guide_data[0])):
        cell = table[0, j]
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")

    # Color rows by priority
    row_colors = ["#C8E6C9", "#C8E6C9", "#FFF9C4", "#FFECB3", "#C8E6C9"]
    for i, color in enumerate(row_colors):
        for j in range(len(guide_data[0])):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("(c) User Input Priority Guide",
                 fontsize=11, fontweight="bold", pad=15)

    fig.suptitle("E3 — Input Pattern Analysis: What Drives Retrieval Accuracy",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _savefig(fig, "E3_input_analysis_user_guide.png")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating experiment plots...")
    print(f"Output: {OUT_DIR}")
    plot_e1()
    plot_e2()
    plot_e3()
    print("\nDone.")
