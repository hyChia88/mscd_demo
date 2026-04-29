#!/usr/bin/env python3
"""Analyze AP held-out topology structure and generate summary plots.

This script characterizes the AP held-out benchmark as a topology benchmark:
- relation multiplicity
- topology family distribution
- canonical predicate/object signatures
- rare / unique topology cases
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")
import matplotlib.gridspec
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "topology_analysis"
SHORT_OBJ_LABELS = {
    "IfcWallStandardCase": "WallStd",
    "IfcWall": "Wall",
    "IfcDoor": "Door",
    "IfcWindow": "Window",
    "IfcSlab": "Slab",
    "IfcRailing": "Railing",
    "IfcStair": "Stair",
    "IfcColumn": "Column",
}


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _relation_signature(rels: List[dict]) -> Tuple[Tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (
                str(r.get("predicate", "?")),
                str(r.get("object_type", "?")),
                str(r.get("direction", "")),
            )
            for r in rels
        )
    )


def _predicate_hist(rels: List[dict]) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted(Counter(str(r.get("predicate", "?")) for r in rels).items()))


def _topology_family(rels: List[dict]) -> str:
    preds = [str(r.get("predicate", "?")) for r in rels]
    pred_hist = Counter(preds)
    n = len(rels)

    if n == 0:
        return "empty"
    if n == 1:
        return f"singleton:{preds[0]}"
    if n == 2:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 1}):
            return "paired:FILLS+NEXT_TO"
        return "paired:other"
    if n == 3:
        if pred_hist == Counter({"FILLS": 1, "NEXT_TO": 2}):
            objs = [str(r.get("object_type", "?")) for r in rels if r.get("predicate") == "NEXT_TO"]
            if len(set(objs)) > 1:
                return "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)"
            return "triad:FILLS+NEXT_TO+NEXT_TO"
        return "triad:other"
    return f"{n}-rel:other"


def _plot_bar(counter: Counter, title: str, xlabel: str, path: Path, topn: int | None = None) -> None:
    items = counter.most_common(topn)
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fmt_signature(sig: Tuple[Tuple[str, str, str], ...]) -> str:
    parts = []
    for pred, obj, direction in sig:
        if direction:
            parts.append(f"{pred}:{obj}:{direction}")
        else:
            parts.append(f"{pred}:{obj}")
    return " | ".join(parts)


def _short_obj_label(name: str | None) -> str:
    if not name:
        return "?"
    return SHORT_OBJ_LABELS.get(str(name), str(name).replace("Ifc", ""))


def _compact_label_sequence(labels: Iterable[str]) -> str:
    counts = Counter(labels)
    parts = []
    for label, count in sorted(counts.items()):
        parts.append(f"{label}x{count}" if count > 1 else label)
    return "+".join(parts) if parts else "none"


def _anchor_pattern(rels: List[dict]) -> str:
    next_to = [_short_obj_label(r.get("object_type")) for r in rels if r.get("predicate") == "NEXT_TO"]
    if next_to:
        return _compact_label_sequence(next_to)
    others = [_short_obj_label(r.get("object_type")) for r in rels]
    return _compact_label_sequence(others)


def _feature_combo(rels: List[dict]) -> Tuple[str, ...]:
    pred_counts = Counter(str(r.get("predicate", "?")) for r in rels)
    next_anchor_types = {_short_obj_label(r.get("object_type")) for r in rels if r.get("predicate") == "NEXT_TO"}
    feats = []
    for pred in ("CONNECTS_TO", "ADJACENT_TO", "FILLS"):
        if pred_counts[pred] > 0:
            feats.append(pred)
    if pred_counts["NEXT_TO"] > 0:
        feats.append("NEXT_TO")
    if pred_counts["NEXT_TO"] > 1:
        feats.append("NEXT_TOx2")
    if len(next_anchor_types) > 1:
        feats.append("MIXED_ANCHOR")
    return tuple(feats) if feats else ("NONE",)


def _complexity_score(rels: List[dict]) -> float:
    preds = [str(r.get("predicate", "?")) for r in rels]
    obj_types = [str(r.get("object_type", "?")) for r in rels]
    next_objs = {str(r.get("object_type", "?")) for r in rels if r.get("predicate") == "NEXT_TO"}
    repeated_same = 1 if any(v > 1 for v in Counter(preds).values()) else 0
    mixed_anchor = 1 if len(next_objs) > 1 else 0
    directional = len({str(r.get("direction") or "") for r in rels if r.get("direction")})
    return (
        len(rels)
        + 0.45 * max(len(set(preds)) - 1, 0)
        + 0.55 * max(len(set(obj_types)) - 1, 0)
        + 0.35 * mixed_anchor
        + 0.15 * repeated_same
        + 0.1 * max(directional - 1, 0)
    )


def _resolve_asset_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    candidates = [
        REPO_ROOT / path,
        PROJECT_ROOT / path,
    ]
    if raw.startswith("datasets/"):
        candidates.insert(0, REPO_ROOT / "data_curation" / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _top_signature_rows(counter: Counter, topn: int = 10) -> List[dict]:
    return [{"signature": _fmt_signature(sig), "count": count} for sig, count in counter.most_common(topn)]


def _draw_counter_bar(ax, counter: Counter, title: str, xlabel: str, topn: int | None = None) -> None:
    items = counter.most_common(topn)
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    ax.bar(range(len(labels)), values, color="#4f6d7a")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")


def analyze(cases: List[dict]) -> dict:
    relation_count = Counter()
    family_count = Counter()
    predicate_multiset_count = Counter()
    full_signature_count = Counter()
    repeated_same_pred_count = Counter()
    anchor_pattern_count = Counter()
    feature_combo_count = Counter()
    link12 = Counter()
    link23 = Counter()
    examples_by_family: Dict[str, List[dict]] = defaultdict(list)
    unique_cases = []
    per_case_rows = []

    for case in cases:
        constraints = case.get("labels", {}).get("constraints", {})
        rels = list(constraints.get("spatial_relations", []) or [])
        case_id = case["case_id"]
        multiplicity = f"{len(rels)}-rel"
        relation_count[len(rels)] += 1
        family = _topology_family(rels)
        anchor_pattern = _anchor_pattern(rels)
        feature_combo = _feature_combo(rels)
        family_count[family] += 1
        anchor_pattern_count[anchor_pattern] += 1
        feature_combo_count[feature_combo] += 1
        link12[(multiplicity, family)] += 1
        link23[(family, anchor_pattern)] += 1
        pred_hist = _predicate_hist(rels)
        predicate_multiset_count[pred_hist] += 1
        sig = _relation_signature(rels)
        full_signature_count[sig] += 1

        same = tuple(sorted((p, n) for p, n in Counter(r.get("predicate", "?") for r in rels).items() if n > 1))
        if same:
            repeated_same_pred_count[same] += 1

        if len(examples_by_family[family]) < 4:
            examples_by_family[family].append(
                {
                    "case_id": case_id,
                    "relations": rels,
                    "inputs": case.get("inputs", {}),
                    "ifc_class": constraints.get("ifc_class"),
                    "storey_name": constraints.get("storey_name"),
                    "signature": _fmt_signature(sig),
                    "anchor_pattern": anchor_pattern,
                }
            )

        per_case_rows.append(
            {
                "case_id": case_id,
                "family": family,
                "anchor_pattern": anchor_pattern,
                "ifc_class": constraints.get("ifc_class"),
                "storey_name": constraints.get("storey_name"),
                "relation_count": len(rels),
                "complexity": round(_complexity_score(rels), 3),
                "signature": _fmt_signature(sig),
            }
        )

    for case in cases:
        constraints = case.get("labels", {}).get("constraints", {})
        rels = list(constraints.get("spatial_relations", []) or [])
        sig = _relation_signature(rels)
        if full_signature_count[sig] == 1:
            unique_cases.append(
                {
                    "case_id": case["case_id"],
                    "family": _topology_family(rels),
                    "signature": _fmt_signature(sig),
                    "relations": rels,
                    "inputs": case.get("inputs", {}),
                    "ifc_class": constraints.get("ifc_class"),
                    "storey_name": constraints.get("storey_name"),
                    "anchor_pattern": _anchor_pattern(rels),
                }
            )

    triad_hist = Counter(
        _predicate_hist(case.get("labels", {}).get("constraints", {}).get("spatial_relations", []) or [])
        for case in cases
        if len(case.get("labels", {}).get("constraints", {}).get("spatial_relations", []) or []) >= 3
    )
    signature_count_str = Counter({_fmt_signature(sig): count for sig, count in full_signature_count.items()})

    return {
        "n_cases": len(cases),
        "relation_count_distribution": dict(relation_count),
        "topology_family_distribution": dict(family_count),
        "predicate_multiset_distribution": {str(k): v for k, v in predicate_multiset_count.items()},
        "repeated_same_predicate_distribution": {str(k): v for k, v in repeated_same_pred_count.items()},
        "triad_predicate_histograms": {str(k): v for k, v in triad_hist.items()},
        "anchor_pattern_distribution": dict(anchor_pattern_count),
        "multiplicity_family_links": [
            {"from": src, "to": dst, "count": count} for (src, dst), count in sorted(link12.items())
        ],
        "family_anchor_links": [
            {"from": src, "to": dst, "count": count} for (src, dst), count in sorted(link23.items())
        ],
        "feature_combo_distribution": [
            {"features": list(features), "count": count}
            for features, count in sorted(feature_combo_count.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "top_full_signatures": _top_signature_rows(full_signature_count, topn=12),
        "unique_case_count": len(unique_cases),
        "unique_cases": unique_cases,
        "per_case_rows": [{**row, "signature_frequency": signature_count_str.get(row["signature"], 0)} for row in per_case_rows],
        "examples_by_family": examples_by_family,
    }


def _plot_global_dashboard(out_dir: Path, summary: dict) -> None:
    relation_count = Counter({int(k): v for k, v in summary["relation_count_distribution"].items()})
    family_count = Counter(summary["topology_family_distribution"])
    top_sig_count = Counter({row["signature"]: row["count"] for row in summary["top_full_signatures"][:8]})

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _draw_counter_bar(ax1, relation_count, "Relation Multiplicity", "# spatial_relations per case")
    _draw_counter_bar(ax2, family_count, "Topology Families", "Family", topn=8)
    _draw_counter_bar(ax3, top_sig_count, "Top Relation Signatures", "Signature", topn=8)

    ax4.axis("off")
    text = [
        "AP Held-out Topology Readout",
        "",
        f"Cases: {summary['n_cases']}",
        f"Unique signatures: {summary['unique_case_count']}",
        "",
        "Key findings:",
        "- Flat multi-relation benchmark, not a deep multi-hop benchmark",
        "- Dominant triad: FILLS + NEXT_TO + NEXT_TO",
        "- Rare mixed-anchor triads are high-value diagnostics",
        "",
        "Family counts:",
    ]
    for name, count in sorted(summary["topology_family_distribution"].items(), key=lambda kv: (-kv[1], kv[0])):
        text.append(f"- {name}: {count}")
    ax4.text(0.0, 1.0, "\n".join(text), va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("AP Held-out Topology Analysis — Global Overview", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "ap_heldout_topology_dashboard.png", dpi=180)
    plt.close(fig)


def _collapsed_labels(counter: Counter, topn: int) -> List[str]:
    labels = [label for label, _ in counter.most_common(topn)]
    if len(counter) > topn:
        labels.append("other")
    return labels


def _collapse_links(raw_links: List[dict], keep_sources: set[str], keep_targets: set[str]) -> Counter:
    collapsed = Counter()
    for row in raw_links:
        src = row["from"] if row["from"] in keep_sources else "other"
        dst = row["to"] if row["to"] in keep_targets else "other"
        if src == "other" and "other" not in keep_sources:
            continue
        if dst == "other" and "other" not in keep_targets:
            continue
        collapsed[(src, dst)] += int(row["count"])
    return collapsed


def _stack_boxes(items: List[Tuple[str, int]], x: float, width: float, gap: float = 0.025) -> Dict[str, Tuple[float, float, float]]:
    total = sum(v for _, v in items)
    usable = 1.0 - gap * max(len(items) - 1, 0)
    y = 1.0
    boxes: Dict[str, Tuple[float, float, float]] = {}
    for label, value in items:
        h = usable * value / total if total else 0.0
        y0 = y - h
        boxes[label] = (x, y0, h)
        y = y0 - gap
    return boxes


def _draw_flow(ax, x0: float, y0: float, h0: float, x1: float, y1: float, h1: float, color: str, alpha: float = 0.35) -> None:
    c = 0.22 * (x1 - x0)
    verts = [
        (x0, y0),
        (x0 + c, y0),
        (x1 - c, y1),
        (x1, y1),
        (x1, y1 + h1),
        (x1 - c, y1 + h1),
        (x0 + c, y0 + h0),
        (x0, y0 + h0),
        (x0, y0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    patch = PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_patch(patch)


def _plot_topology_flow(out_dir: Path, summary: dict) -> None:
    mult_counter = Counter({f"{int(k)}-rel": v for k, v in summary["relation_count_distribution"].items()})
    family_counter = Counter(summary["topology_family_distribution"])
    anchor_counter = Counter(summary["anchor_pattern_distribution"])

    stage1_labels = [label for label, _ in sorted(mult_counter.items(), key=lambda kv: int(kv[0].split("-")[0]))]
    stage2_labels = [label for label, _ in family_counter.most_common()]
    stage3_labels = _collapsed_labels(anchor_counter, topn=6)

    stage1_keep = set(stage1_labels)
    stage2_keep = set(stage2_labels)
    stage3_keep = set(stage3_labels)

    links12 = _collapse_links(summary["multiplicity_family_links"], stage1_keep, stage2_keep)
    links23 = _collapse_links(summary["family_anchor_links"], stage2_keep, stage3_keep)

    stage3_counter = Counter()
    for label, value in anchor_counter.items():
        stage3_counter[label if label in stage3_keep else "other"] += value

    stage1_items = [(label, mult_counter[label]) for label in stage1_labels]
    stage2_items = [(label, family_counter[label]) for label in stage2_labels]
    stage3_items = [(label, stage3_counter[label]) for label in stage3_labels if stage3_counter[label] > 0]

    x_positions = [0.08, 0.42, 0.76]
    width = 0.14
    box1 = _stack_boxes(stage1_items, x_positions[0], width)
    box2 = _stack_boxes(stage2_items, x_positions[1], width)
    box3 = _stack_boxes(stage3_items, x_positions[2], width)

    # Reserve explicit top/bottom whitespace so the column headers do not
    # collide with the topmost boxes/flows.
    top_margin = 0.14
    bottom_margin = 0.04
    usable = 1.0 - top_margin - bottom_margin

    def _remap_boxes(boxes: Dict[str, Tuple[float, float, float]]) -> Dict[str, Tuple[float, float, float]]:
        return {label: (x, bottom_margin + y * usable, h * usable) for label, (x, y, h) in boxes.items()}

    box1 = _remap_boxes(box1)
    box2 = _remap_boxes(box2)
    box3 = _remap_boxes(box3)

    family_colors = {
        label: color
        for label, color in zip(
            stage2_labels,
            ["#4f6d7a", "#3b8ea5", "#c8553d", "#7a9e7e", "#9d4edd", "#e09f3e", "#6c757d", "#2a9d8f"],
        )
    }

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    for src, dst in sorted(links12.keys(), key=lambda pair: (stage1_labels.index(pair[0]), stage2_labels.index(pair[1]))):
        count = links12[(src, dst)]
        x0, y0, h0 = box1[src]
        x1, y1, h1 = box2[dst]
        seg0 = h0 * count / mult_counter[src]
        seg1 = h1 * count / family_counter[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, family_colors.get(dst, "#999999"))
        if count >= 3:
            mx = (x0 + width + x1) / 2
            my = (y_from + seg0 / 2 + y_to + seg1 / 2) / 2
            ax.text(
                mx,
                my,
                str(count),
                ha="center",
                va="center",
                fontsize=8.5,
                color="#1f2933",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
            )
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    from_offsets = defaultdict(float)
    to_offsets = defaultdict(float)
    stage3_totals = Counter(dict(stage3_items))
    for src, dst in sorted(
        links23.keys(),
        key=lambda pair: (
            stage2_labels.index(pair[0]),
            stage3_labels.index(pair[1]) if pair[1] in stage3_labels else len(stage3_labels),
        ),
    ):
        count = links23[(src, dst)]
        x0, y0, h0 = box2[src]
        x1, y1, h1 = box3[dst]
        seg0 = h0 * count / family_counter[src]
        seg1 = h1 * count / stage3_totals[dst]
        y_from = y0 + from_offsets[src]
        y_to = y1 + to_offsets[dst]
        _draw_flow(ax, x0 + width, y_from, seg0, x1, y_to, seg1, family_colors.get(src, "#999999"))
        if count >= 3:
            mx = (x0 + width + x1) / 2
            my = (y_from + seg0 / 2 + y_to + seg1 / 2) / 2
            ax.text(
                mx,
                my,
                str(count),
                ha="center",
                va="center",
                fontsize=8.5,
                color="#1f2933",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
            )
        from_offsets[src] += seg0
        to_offsets[dst] += seg1

    for label, (x, y, h) in box1.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor="#d9e2ec", edgecolor="#556"))
        ax.text(x - 0.01, y + h / 2, f"{label}\n(n={mult_counter[label]})", ha="right", va="center", fontsize=9.5, weight="bold")
    for label, (x, y, h) in box2.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor=family_colors.get(label, "#cccccc"), edgecolor="#445"))
        ax.text(
            x + width / 2,
            y + h / 2,
            f"{label}\n(n={family_counter[label]})",
            ha="center",
            va="center",
            fontsize=8.3,
            color="white",
            weight="bold",
        )
    for label, (x, y, h) in box3.items():
        ax.add_patch(Rectangle((x, y), width, h, facecolor="#edf2f4", edgecolor="#556"))
        ax.text(
            x + width + 0.01,
            y + h / 2,
            f"{label}\n(n={stage3_counter[label]})",
            ha="left",
            va="center",
            fontsize=9.5,
            weight="bold",
        )

    header_y = 1.0 - top_margin * 0.35
    ax.text(x_positions[0] + width / 2, header_y, "Relation Multiplicity", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(x_positions[1] + width / 2, header_y, "Topology Family", ha="center", va="bottom", fontsize=12, weight="bold")
    ax.text(x_positions[2] + width / 2, header_y, "Anchor Pattern", ha="center", va="bottom", fontsize=12, weight="bold")

    fig.suptitle("AP Held-out Topology Flow", fontsize=16, y=0.995)
    fig.subplots_adjust(top=0.88, left=0.04, right=0.96, bottom=0.06)
    fig.savefig(out_dir / "topology_flow_alluvial.png", dpi=180)
    plt.close(fig)


def _plot_topology_upset(out_dir: Path, summary: dict) -> None:
    combo_rows = summary["feature_combo_distribution"][:8]
    features = ["CONNECTS_TO", "ADJACENT_TO", "FILLS", "NEXT_TO", "NEXT_TOx2", "MIXED_ANCHOR"]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.2], hspace=0.08)
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    counts = [row["count"] for row in combo_rows]
    x = np.arange(len(combo_rows))
    ax_top.bar(x, counts, color="#3b8ea5")
    ax_top.set_ylabel("Cases")
    ax_top.set_title("AP Held-out Topology Combinations (UpSet-style)", fontsize=16)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([])
    ax_top.spines[["top", "right"]].set_visible(False)

    row_y = np.arange(len(features))[::-1]
    ax_bottom.set_yticks(row_y)
    ax_bottom.set_yticklabels(features)
    ax_bottom.set_xlabel("Feature combinations")
    ax_bottom.spines[["top", "right"]].set_visible(False)
    ax_bottom.set_xlim(-0.6, len(combo_rows) - 0.4)

    for idx, row in enumerate(combo_rows):
        active = set(row["features"])
        ys = []
        for y, feature in zip(row_y, features):
            if feature in active:
                ax_bottom.scatter(idx, y, s=85, color="#c8553d", zorder=3)
                ys.append(y)
            else:
                ax_bottom.scatter(idx, y, s=22, color="#d9d9d9", zorder=2)
        if len(ys) >= 2:
            ax_bottom.plot([idx, idx], [min(ys), max(ys)], color="#c8553d", linewidth=2)
        label = " + ".join(row["features"]) if row["features"] else "NONE"
        ax_bottom.text(idx, -0.85, label, rotation=35, ha="right", va="top", fontsize=9)

    ax_bottom.set_ylim(-1.2, len(features) - 0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "topology_upset.png", dpi=180)
    plt.close(fig)


def _family_order(summary: dict) -> List[str]:
    preferred = [
        "singleton:CONNECTS_TO",
        "singleton:ADJACENT_TO",
        "paired:FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
        "singleton:FILLS",
    ]
    counter = Counter(summary["topology_family_distribution"])
    labels = [label for label in preferred if counter.get(label, 0)]
    extras = [label for label, _ in counter.most_common() if label not in labels]
    return labels + extras


def _family_colors(labels: List[str]) -> Dict[str, str]:
    palette = ["#4f6d7a", "#3b8ea5", "#c8553d", "#7a9e7e", "#9d4edd", "#e09f3e", "#6c757d", "#2a9d8f"]
    return {label: palette[i % len(palette)] for i, label in enumerate(labels)}


def _plot_ap_scatter(out_dir: Path, summary: dict) -> None:
    rows = summary["per_case_rows"]
    families = _family_order(summary)
    family_y = {family: idx for idx, family in enumerate(families)}
    colors = _family_colors(families)
    anchor_counter = Counter(summary["anchor_pattern_distribution"])
    anchor_palette = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#84a59d", "#8d99ae"]
    top_anchors = [label for label, _ in anchor_counter.most_common(6)]
    anchor_colors = {label: anchor_palette[i % len(anchor_palette)] for i, label in enumerate(top_anchors)}
    anchor_colors["other"] = "#bbbbbb"

    fig, ax = plt.subplots(figsize=(14, 7))
    used_labels = set()
    for idx, row in enumerate(rows):
        family = row["family"]
        y = family_y[family] + (((idx % 5) - 2) * 0.07)
        anchor = row["anchor_pattern"] if row["anchor_pattern"] in anchor_colors else "other"
        freq = max(int(row.get("signature_frequency", 1)), 1)
        size = 70 + 180 / freq
        label = anchor if anchor not in used_labels else None
        ax.scatter(
            row["complexity"],
            y,
            s=size,
            c=anchor_colors[anchor],
            alpha=0.78,
            edgecolors=colors[family],
            linewidths=1.4,
            label=label,
        )
        if row.get("signature_frequency", 0) == 1 or "mixed-anchor" in family:
            ax.text(row["complexity"] + 0.03, y + 0.02, row["case_id"], fontsize=7.5, color="#333")
        if label:
            used_labels.add(anchor)

    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families)
    ax.set_xlabel("Topology complexity score")
    ax.set_ylabel("Topology family")
    ax.set_title("AP Held-out Case Overview: Type, Complexity, and Rarity")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    ax.text(
        0.99,
        0.02,
        "Bubble size ~ inverse signature frequency\nEdge color = family; fill color = anchor pattern",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        family="monospace",
    )
    ax.legend(title="Anchor pattern", loc="lower right", fontsize=8, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ap_heldout_topology_scatter.png", dpi=180)
    plt.close(fig)


def _plot_topology_bipartite_graph(out_dir: Path, summary: dict) -> None:
    family_counter = Counter(summary["topology_family_distribution"])
    anchor_counter = Counter(summary["anchor_pattern_distribution"])
    families = _family_order(summary)
    anchors = [label for label, _ in anchor_counter.most_common(7)]
    if len(anchor_counter) > len(anchors):
        anchors.append("other")

    edge_counter = Counter()
    for row in summary["family_anchor_links"]:
        src = row["from"]
        dst = row["to"] if row["to"] in anchors else "other"
        edge_counter[(src, dst)] += int(row["count"])
    anchor_totals = Counter()
    for (src, dst), count in edge_counter.items():
        anchor_totals[dst] += count

    family_colors = _family_colors(families)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x, right_x = 0.22, 0.78
    left_y = np.linspace(0.88, 0.12, len(families))
    right_y = np.linspace(0.88, 0.12, len(anchors))
    fam_pos = {family: (left_x, y) for family, y in zip(families, left_y)}
    anc_pos = {anchor: (right_x, y) for anchor, y in zip(anchors, right_y)}
    max_edge = max(edge_counter.values()) if edge_counter else 1

    for (family, anchor), count in sorted(edge_counter.items(), key=lambda kv: (-kv[1], kv[0])):
        x0, y0 = fam_pos[family]
        x1, y1 = anc_pos[anchor]
        lw = 1.0 + 6.0 * count / max_edge
        ax.plot([x0 + 0.06, x1 - 0.06], [y0, y1], color=family_colors[family], alpha=0.35, linewidth=lw)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if count >= 3:
            ax.text(mx, my, str(count), fontsize=8, color="#333", ha="center", va="center")

    for family in families:
        x, y = fam_pos[family]
        ax.add_patch(Circle((x, y), 0.045, facecolor=family_colors[family], edgecolor="white", linewidth=2))
        ax.text(x - 0.07, y, f"{family} ({family_counter[family]})", ha="right", va="center", fontsize=10, weight="bold")

    for anchor in anchors:
        x, y = anc_pos[anchor]
        ax.add_patch(Circle((x, y), 0.04, facecolor="#edf2f4", edgecolor="#495057", linewidth=1.5))
        ax.text(x + 0.07, y, f"{anchor} ({anchor_totals[anchor]})", ha="left", va="center", fontsize=10, weight="bold")

    ax.text(left_x, 0.97, "Topology families", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(right_x, 0.97, "Anchor patterns", ha="center", va="center", fontsize=12, weight="bold")
    fig.suptitle("AP Held-out Topology Bipartite Structure", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
    fig.savefig(out_dir / "topology_bipartite_graph.png", dpi=180)
    plt.close(fig)


def _load_image_or_blank(path: Path | None) -> np.ndarray:
    if path is None or not path.exists():
        return np.ones((128, 128, 3), dtype=np.float32)
    try:
        img = mpimg.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img
    except Exception:
        return np.ones((128, 128, 3), dtype=np.float32)


def _gallery_rows_from_families(summary: dict, max_cases: int = 6) -> List[dict]:
    picked = []
    preferred = [
        "singleton:CONNECTS_TO",
        "singleton:ADJACENT_TO",
        "paired:FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
        "singleton:FILLS",
    ]
    examples = summary["examples_by_family"]
    family_counts = summary["topology_family_distribution"]
    for family in preferred:
        if family in examples and examples[family]:
            row = dict(examples[family][0])
            row["family"] = family
            row["family_count"] = family_counts.get(family, 0)
            picked.append(row)
        if len(picked) >= max_cases:
            break
    return picked[:max_cases]


_PRED_COLORS: Dict[str, str] = {
    "FILLS":       "#D44000",   # burnt orange
    "NEXT_TO":     "#1060C8",   # strong blue
    "CONNECTS_TO": "#7B00CC",   # violet
    "ADJACENT_TO": "#008840",   # green
}
_PRED_NODE_BG: Dict[str, str] = {
    "FILLS":       "#FFE8DC",
    "NEXT_TO":     "#DCE8FF",
    "CONNECTS_TO": "#EEDCFF",
    "ADJACENT_TO": "#DCFFF0",
}


def _draw_topology_glyph(ax, row: dict, fontscale: float = 1.0) -> None:
    """Draw a spatial-relation graph with semantic predicate-based positioning.

    Layout rules
    ────────────
    • FILLS          → bottom centre  (0, -0.85)
    • NEXT_TO:left   → left           (-0.98, 0)
    • NEXT_TO:right  → right          (+0.98, 0)
    • NEXT_TO (none) → top, spread
    • CONNECTS_TO    → upper-right fan
    • ADJACENT_TO    → upper-left fan
    • overflow       → remaining corners

    Edges are colour-coded per predicate.  Arrow heads mark direction.
    Multi-hop chains (A → B already in placed, B → C) are drawn as a chain
    rather than a star when detected.
    """
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.28, 1.28)
    ax.axis("off")

    rels = list(row.get("relations") or [])
    target = _short_obj_label(row.get("ifc_class"))

    # ── target node ──────────────────────────────────────────────────────────
    TARGET_R = 0.21
    ax.add_patch(Circle((0.0, 0.0), TARGET_R,
                         facecolor="#D8E8F0", edgecolor="#2E4F60", linewidth=1.5, zorder=3))
    ax.text(0.0, 0.0, target, ha="center", va="center",
            fontsize=9 * fontscale, color="#1A2E3A", weight="bold", zorder=4)

    if not rels:
        ax.text(0.0, -0.65, "(no relations)", ha="center", va="center",
                fontsize=6 * fontscale, color="#AAAAAA")
        return

    # ── assign anchor positions semantically ─────────────────────────────────
    placed: List[Tuple[dict, Tuple[float, float]]] = []
    used_positions: set = set()

    def _claim(pos, key=None):
        k = key or pos
        used_positions.add(k)
        return pos

    # 1. FILLS → always bottom
    fills_rels = [r for r in rels if r.get("predicate") == "FILLS"]
    for i, r in enumerate(fills_rels):
        offset = (i - (len(fills_rels) - 1) / 2) * 0.42
        placed.append((r, _claim((offset, -0.88), f"fills_{i}")))

    # 2. NEXT_TO:left / right / undirected
    nt_left  = [r for r in rels if r.get("predicate") == "NEXT_TO"
                and str(r.get("direction") or "").lower() == "left"]
    nt_right = [r for r in rels if r.get("predicate") == "NEXT_TO"
                and str(r.get("direction") or "").lower() == "right"]
    nt_other = [r for r in rels if r.get("predicate") == "NEXT_TO"
                and str(r.get("direction") or "").lower() not in ("left", "right")]

    for i, r in enumerate(nt_left):
        off = (i - (len(nt_left) - 1) / 2) * 0.30
        placed.append((r, _claim((-0.98, off), f"nt_left_{i}")))

    for i, r in enumerate(nt_right):
        off = (i - (len(nt_right) - 1) / 2) * 0.30
        placed.append((r, _claim((0.98, off), f"nt_right_{i}")))

    top_xs = np.linspace(-0.40, 0.40, max(len(nt_other), 1))
    for i, r in enumerate(nt_other):
        placed.append((r, _claim((float(top_xs[i]), 0.88), f"nt_top_{i}")))

    # 3. CONNECTS_TO → upper-right fan
    conn_rels = [r for r in rels if r.get("predicate") == "CONNECTS_TO"]
    angles_conn = np.linspace(20, 70, max(len(conn_rels), 1))
    for i, r in enumerate(conn_rels):
        a = np.radians(angles_conn[i])
        placed.append((r, _claim((0.90 * np.cos(a), 0.90 * np.sin(a)), f"conn_{i}")))

    # 4. ADJACENT_TO → upper-left fan
    adj_rels = [r for r in rels if r.get("predicate") == "ADJACENT_TO"]
    angles_adj = np.linspace(110, 160, max(len(adj_rels), 1))
    for i, r in enumerate(adj_rels):
        a = np.radians(angles_adj[i])
        placed.append((r, _claim((0.90 * np.cos(a), 0.90 * np.sin(a)), f"adj_{i}")))

    # 5. Any remaining predicates → overflow corners
    placed_set = {id(r) for r, _ in placed}
    leftover = [r for r in rels if id(r) not in placed_set]
    overflow = [(0.0, 0.88), (-0.88, -0.50), (0.88, -0.50), (-0.60, 0.75), (0.60, 0.75)]
    for r, pos in zip(leftover, overflow):
        placed.append((r, pos))

    # ── detect multi-hop chain ────────────────────────────────────────────────
    # A chain exists if any anchor object_type matches the target ifc_class AND
    # there is another relation whose subject would be that anchor.
    # For a flat SR list this manifests as: same object_type appears in 2+ rels
    # with DIFFERENT predicates (e.g., FILLS:Wall + CONNECTS_TO:Wall means
    # "target fills Wall AND target connects-to Wall", which we annotate as
    # a 2-path: target → Wall → OtherWall via dotted secondary edge).
    obj_pred: Dict[str, List[str]] = defaultdict(list)
    for r in rels:
        obj_pred[_short_obj_label(r.get("object_type"))].append(r.get("predicate", ""))

    chain_anchors = {obj for obj, preds in obj_pred.items() if len(set(preds)) >= 2}

    # ── draw edges and anchor nodes ───────────────────────────────────────────
    NODE_W, NODE_H = 0.54, 0.22

    for rel, (x, y) in placed:
        pred  = str(rel.get("predicate", "?"))
        color = _PRED_COLORS.get(pred, "#6c757d")
        nbg   = _PRED_NODE_BG.get(pred, "#edf2f4")

        # edge: from circle perimeter → anchor box edge
        angle = np.arctan2(y, x)
        ex = TARGET_R * np.cos(angle)
        ey = TARGET_R * np.sin(angle)
        # stop line just outside the box
        bx = x - np.sign(x) * NODE_W / 2 * 0.85 if abs(x) > 0.05 else x
        by = y - np.sign(y) * NODE_H / 2 * 1.1  if abs(y) > 0.05 else y

        ax.annotate(
            "", xy=(bx, by), xytext=(ex, ey),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.5 * fontscale,
                mutation_scale=10 * fontscale,
            ),
            zorder=2,
        )

        # predicate label on edge midpoint
        mx = (ex + bx) / 2
        my = (ey + by) / 2
        ax.text(mx, my, pred,
                ha="center", va="center",
                fontsize=6.5 * fontscale, color=color, weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5),
                zorder=3)

        # anchor node box
        ax.add_patch(FancyBboxPatch(
            (x - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=nbg,
            edgecolor=color,
            linewidth=1.1 * fontscale,
            zorder=3,
        ))

        anchor  = _short_obj_label(rel.get("object_type"))
        dirn    = str(rel.get("direction") or "")
        label   = anchor if not dirn else f"{anchor}\n{dirn}"

        # mark shared anchor (chain indicator) with bold border
        is_chain = anchor in chain_anchors
        ax.text(x, y, label,
                ha="center", va="center",
                fontsize=7.0 * fontscale,
                weight="bold" if is_chain else "normal",
                color="#111111",
                zorder=4)

        if is_chain:
            # draw a secondary dotted arc indicating multi-hop context
            ax.add_patch(Circle(
                (x, y), NODE_H * 0.62,
                facecolor="none",
                edgecolor=color,
                linewidth=0.8 * fontscale,
                linestyle=":",
                zorder=2,
            ))


def _plot_micro_gallery(rows: List[dict], title: str, out_path: Path, subtitle_with_counts: bool = True) -> None:
    if not rows:
        return
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(15, max(3.2 * n, 6)))
    if n == 1:
        axes = np.array([axes])

    for idx, row in enumerate(rows):
        inputs = row.get("inputs", {}) or {}
        site_path = _resolve_asset_path((inputs.get("images") or [None])[0])
        floorplan_path = _resolve_asset_path(inputs.get("floorplan_patch"))
        site = _load_image_or_blank(site_path)
        floor = _load_image_or_blank(floorplan_path)

        axes[idx, 0].imshow(site)
        axes[idx, 1].imshow(floor)
        for ax in axes[idx][:2]:
            ax.set_xticks([])
            ax.set_yticks([])

        _draw_topology_glyph(axes[idx, 2], row)

        rels = row.get("relations", [])
        rel_text = "; ".join(
            f"{r.get('predicate')}:{r.get('object_type')}" + (f":{r.get('direction')}" if r.get("direction") else "")
            for r in rels
        )
        family = row.get("family", "")
        count_note = f" [n={row.get('family_count')}]" if subtitle_with_counts and row.get("family_count") else ""
        axes[idx, 0].set_title(f"{row['case_id']} | {family}{count_note}\nSITE", fontsize=10)
        axes[idx, 1].set_title(f"{row['case_id']}\nFLOORPLAN", fontsize=9)
        axes[idx, 2].set_title(f"{row.get('signature', rel_text)}\n{row.get('storey_name', '')}", fontsize=8.5)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_outputs(out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "ap_heldout_topology_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    relation_count = Counter({int(k): v for k, v in summary["relation_count_distribution"].items()})
    family_count = Counter(summary["topology_family_distribution"])
    top_sig_count = Counter({row["signature"]: row["count"] for row in summary["top_full_signatures"]})

    _plot_bar(
        relation_count,
        "AP Held-out Relation Multiplicity",
        "# spatial_relations per case",
        out_dir / "relation_multiplicity.png",
    )
    _plot_bar(
        family_count,
        "AP Held-out Topology Family Distribution",
        "Topology family",
        out_dir / "topology_family_distribution.png",
        topn=8,
    )
    _plot_bar(
        top_sig_count,
        "Top AP Held-out Relation Signatures",
        "Signature",
        out_dir / "top_relation_signatures.png",
        topn=10,
    )
    _plot_global_dashboard(out_dir, summary)
    _plot_topology_flow(out_dir, summary)
    _plot_topology_upset(out_dir, summary)
    _plot_ap_scatter(out_dir, summary)
    _plot_topology_bipartite_graph(out_dir, summary)

    representative_rows = _gallery_rows_from_families(summary, max_cases=6)
    _plot_micro_gallery(
        representative_rows,
        "AP Held-out Topology — Representative Family Motifs",
        out_dir / "representative_family_gallery.png",
    )
    _plot_micro_gallery(
        summary["unique_cases"][:6],
        "AP Held-out Topology — Rare / Unique Motifs",
        out_dir / "rare_unique_gallery.png",
        subtitle_with_counts=False,
    )

    lines = [
        "# AP Held-out Topology Analysis",
        "",
        f"- Cases: {summary['n_cases']}",
        f"- Unique topology signatures: {summary['unique_case_count']}",
        "",
        "## Key Readout",
        "",
    ]

    dist = summary["relation_count_distribution"]
    lines.extend(
        [
            f"- Relation multiplicity: {dist}",
            f"- Topology families: {summary['topology_family_distribution']}",
            f"- Triad histograms: {summary['triad_predicate_histograms']}",
            "",
            "Interpretation:",
            "- AP held-out behaves more like a flat multi-relation / multi-anchor benchmark than a deep multi-hop benchmark.",
            "- The main recurring 3-relation topology is `FILLS + NEXT_TO + NEXT_TO`, not a labeled 3-hop chain.",
            "- Rare mixed-anchor topologies should be treated as high-value post-hoc cases.",
            "",
            "## Top Signatures",
            "",
        ]
    )

    for row in summary["top_full_signatures"]:
        lines.append(f"- `{row['signature']}`: {row['count']}")

    lines.extend(
        [
            "",
            "## Visualization Outputs",
            "",
            f"- Global dashboard: `{(out_dir / 'ap_heldout_topology_dashboard.png').name}`",
            f"- Topology flow (alluvial): `{(out_dir / 'topology_flow_alluvial.png').name}`",
            f"- Topology combinations (UpSet-style): `{(out_dir / 'topology_upset.png').name}`",
            f"- Case-level complexity scatter: `{(out_dir / 'ap_heldout_topology_scatter.png').name}`",
            f"- Family-anchor bipartite graph: `{(out_dir / 'topology_bipartite_graph.png').name}`",
            f"- Representative family gallery: `{(out_dir / 'representative_family_gallery.png').name}`",
            f"- Rare / unique gallery: `{(out_dir / 'rare_unique_gallery.png').name}`",
            "",
            "## Rare / Unique Cases",
            "",
        ]
    )
    for row in summary["unique_cases"][:12]:
        lines.append(f"- `{row['case_id']}` [{row['family']}]: `{row['signature']}`")

    (out_dir / "ap_heldout_topology_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_EVAL_JSONL = (
    REPO_ROOT / "data_curation/datasets/synth_v0.5_ap/train/lora6_v2_ap_eval_canonical_m.jsonl"
)
DEFAULT_SKINS_JSONL = REPO_ROOT / "data_curation/datasets/synth_v0.5_ap/skins/skins_integrated.jsonl"
DEFAULT_SKELS_JSONL = REPO_ROOT / "data_curation/datasets/synth_v0.5_ap/skeletons/skeletons.jsonl"
DEFAULT_DS_ROOT = REPO_ROOT / "data_curation/datasets/synth_v0.5_ap"

# 10 representative cases covering all topology families present in the held-out set
# (CONNECTS_TO · triad FILLS+NEXT_TO×2 · paired FILLS+NEXT_TO · ADJACENT_TO)
GALLERY_SELECTED: List[Tuple[str, str]] = [
    ("AP_SK_022", "singleton · CONNECTS_TO"),
    ("AP_SK_009", "singleton · CONNECTS_TO"),
    ("AP_SK_346", "triad · FILLS + NEXT_TO×2"),
    ("AP_SK_149", "triad · FILLS + NEXT_TO×2"),
    ("AP_SK_160", "triad · FILLS + NEXT_TO×2"),
    ("AP_SK_082", "paired · FILLS + NEXT_TO"),
    ("AP_SK_107", "paired · FILLS + NEXT_TO"),
    ("AP_SK_217", "paired · FILLS + NEXT_TO"),
    ("AP_SK_279", "singleton · ADJACENT_TO"),
    ("AP_SK_264", "singleton · ADJACENT_TO"),
]

GALLERY_FAMILY_COLORS = {
    "singleton · CONNECTS_TO":   "#9B2BE0",
    "triad · FILLS + NEXT_TO×2": "#E05A2B",
    "paired · FILLS + NEXT_TO":  "#2B7FE0",
    "singleton · ADJACENT_TO":   "#2BB56A",
}

GALLERY_PRED_COLORS = {
    "FILLS":       "#E05A2B",
    "NEXT_TO":     "#2B7FE0",
    "CONNECTS_TO": "#9B2BE0",
    "ADJACENT_TO": "#2BB56A",
}


def _find_dataset_image(case_id: str, kind: str, ds_root: Path) -> Path | None:
    """Locate a site / floorplan / render image for a given case_id."""
    if kind == "site":
        candidates = [ds_root / "imgs" / f"{case_id}_site.png"]
    elif kind == "floorplan":
        candidates = [
            ds_root / "floorplans" / f"{case_id}_floorplan.png",
            ds_root / "floorplans_v2" / f"{case_id}_scale_M.png",
            ds_root / "floorplans_v2" / f"{case_id}_scale_S.png",
        ]
    elif kind == "render":
        candidates = [ds_root / "renders" / "global" / f"{case_id}_global.png"]
    else:
        return None
    for p in candidates:
        if p.exists():
            return p
    return None


def generate_dataset_gallery(
    out_dir: Path,
    eval_jsonl: Path = DEFAULT_EVAL_JSONL,
    skins_jsonl: Path = DEFAULT_SKINS_JSONL,
    skels_jsonl: Path = DEFAULT_SKELS_JSONL,
    ds_root: Path = DEFAULT_DS_ROOT,
    selected: List[Tuple[str, str]] = GALLERY_SELECTED,
) -> None:
    """Generate a 10-row × 5-col full-modality dataset gallery figure.

    Columns: chat query · site photo · floorplan · 3-D render · SR graph.
    Rows: one per selected case, covering all topology families.
    """
    import textwrap

    # ── Load data ────────────────────────────────────────────────────────────
    eval_cases: Dict[str, dict] = {}
    with eval_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            eval_cases[rec["id"]] = rec

    skins_by_id: Dict[str, dict] = {}
    with skins_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            skins_by_id[rec["skeleton_id"]] = rec

    skels_by_id: Dict[str, dict] = {}
    with skels_jsonl.open() as f:
        for line in f:
            rec = json.loads(line)
            skels_by_id[rec["id"]] = rec

    # ── Layout ───────────────────────────────────────────────────────────────
    N = len(selected)
    COL_W = [1.05, 1.55, 1.55, 1.55, 1.3]   # relative column widths
    ROW_H = 1.65                              # inches per data row
    HDR_H = 0.32                             # header row height (inches)
    FIG_W = sum(COL_W) * 1.35 + 0.3
    FIG_H = N * ROW_H + HDR_H + 0.55        # + title band

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")

    gs = matplotlib.gridspec.GridSpec(
        N + 1, 5,
        figure=fig,
        height_ratios=[HDR_H / ROW_H] + [1.0] * N,
        width_ratios=COL_W,
        hspace=0.60,
        wspace=0.10,
        left=0.01,
        right=0.99,
        top=0.955,
        bottom=0.03,
    )

    # ── Master title ─────────────────────────────────────────────────────────
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.72,
        "Dataset Gallery — Full-Modality Paired Samples  (AdvancedProject · n = 10)",
        ha="center", va="center", fontsize=11, fontweight="bold",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.5, 0.12,
        "Columns: Chat Query  ·  Site Photo  ·  Floorplan  ·  3-D Render  ·  Spatial-Relation Graph",
        ha="center", va="center", fontsize=6.8, color="#555555",
        transform=title_ax.transAxes,
    )

    HEADERS = ["Chat Query", "Site Photo", "Floorplan", "3-D Render", "SR Graph"]
    for ci, hdr in enumerate(HEADERS):
        ax = fig.add_subplot(gs[0, ci])
        ax.set_facecolor("#EFEFEF")
        ax.axis("off")
        ax.text(0.5, 0.5, hdr, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="#333333",
                transform=ax.transAxes)

    # ── Rows ─────────────────────────────────────────────────────────────────
    for row_i, (case_id, topo_label) in enumerate(selected):
        gi = row_i + 1
        fam_color = GALLERY_FAMILY_COLORS.get(topo_label, "#888888")

        rec = eval_cases.get(case_id)
        if rec is None:
            print(f"  [WARN] {case_id} not found in eval JSONL — skipping")
            continue

        skel = skels_by_id.get(case_id, {})
        gt = json.loads(rec["messages"][2]["content"])
        srs: List[dict] = gt.get("spatial_relations") or []
        query_text = next(
            (m["text"] for m in rec["messages"][1]["content"] if m.get("type") == "text"),
            "",
        )
        ifc_class  = gt.get("ifc_class", "?")
        storey_nm  = gt.get("storey_name", "?")
        target_name = (skel.get("target_props") or {}).get("Name", "")
        if target_name and len(target_name) > 42:
            target_name = target_name[:42] + "…"
        pool_sz  = skel.get("candidate_pool_size", "?")
        locatab  = skel.get("locatability_score", "?")

        # ── Col 0 : Chat query ────────────────────────────────────────────
        ax_t = fig.add_subplot(gs[gi, 0])
        ax_t.axis("off")
        ax_t.set_facecolor("#FAFAFA")
        rect = matplotlib.patches.FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle="round,pad=0.02", linewidth=0.6,
            edgecolor="#CCCCCC", facecolor="#FAFAFA",
            transform=ax_t.transAxes, zorder=0, clip_on=False,
        )
        ax_t.add_patch(rect)

        ax_t.text(0.5, 0.975, case_id, ha="center", va="top", fontsize=5.0,
                  fontweight="bold", color="white", transform=ax_t.transAxes,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor=fam_color, edgecolor="none"))

        ax_t.text(0.5, 0.875, topo_label, ha="center", va="top", fontsize=4.8,
                  color=fam_color, transform=ax_t.transAxes, style="italic")

        wrapped = "\n".join(textwrap.wrap(f'"{query_text}"', 30))
        ax_t.text(0.5, 0.77, wrapped, ha="center", va="top", fontsize=5.0,
                  color="#333333", transform=ax_t.transAxes, linespacing=1.35)

        meta_lines = [
            f"{ifc_class}",
            f"Floor {storey_nm}",
            f"Pool: {pool_sz}  Loc: {locatab}",
        ]
        if target_name:
            meta_lines.append(target_name)
        ax_t.text(0.5, 0.09, "\n".join(meta_lines), ha="center", va="bottom",
                  fontsize=4.5, color="#555555", transform=ax_t.transAxes,
                  linespacing=1.3,
                  bbox=dict(boxstyle="round,pad=0.18", facecolor="#EFEFEF",
                            edgecolor="#CCCCCC", lw=0.5))

        # ── Col 1 : Site photo ────────────────────────────────────────────
        ax_site = fig.add_subplot(gs[gi, 1])
        img_site = _load_image_or_blank(_find_dataset_image(case_id, "site", ds_root))
        ax_site.imshow(img_site)
        ax_site.axis("off")
        ax_site.set_title("Site", fontsize=5.5, color="#444444", pad=2)

        # ── Col 2 : Floorplan ─────────────────────────────────────────────
        ax_fp = fig.add_subplot(gs[gi, 2])
        img_fp = _load_image_or_blank(_find_dataset_image(case_id, "floorplan", ds_root))
        ax_fp.imshow(img_fp)
        ax_fp.axis("off")
        ax_fp.set_title("Floorplan", fontsize=5.5, color="#444444", pad=2)

        # ── Col 3 : 3-D render ────────────────────────────────────────────
        ax_3d = fig.add_subplot(gs[gi, 3])
        img_3d = _load_image_or_blank(_find_dataset_image(case_id, "render", ds_root))
        ax_3d.imshow(img_3d)
        ax_3d.axis("off")
        ax_3d.set_title("3-D Render", fontsize=5.5, color="#444444", pad=2)

        # ── Col 4 : SR graph ──────────────────────────────────────────────
        ax_sr = fig.add_subplot(gs[gi, 4])
        _draw_topology_glyph(
            ax_sr,
            {
                "ifc_class": ifc_class,
                "relations": srs,
                "storey_name": str(storey_nm),
            },
            fontscale=0.58,
        )
        ax_sr.set_title("SR Graph", fontsize=5.5, color="#444444", pad=2)

    # ── Predicate legend ─────────────────────────────────────────────────────
    pred_patches = [
        matplotlib.patches.Patch(facecolor=c, label=p)
        for p, c in GALLERY_PRED_COLORS.items()
    ]
    fig.legend(
        handles=pred_patches,
        title="Predicate", title_fontsize=6,
        fontsize=6, loc="lower right",
        bbox_to_anchor=(0.998, 0.005),
        ncol=4, framealpha=0.9, edgecolor="#CCCCCC",
    )

    # ── Topology family color legend ──────────────────────────────────────────
    fam_patches = [
        matplotlib.patches.Patch(facecolor=c, label=f)
        for f, c in GALLERY_FAMILY_COLORS.items()
    ]
    fig.legend(
        handles=fam_patches,
        title="Topology family", title_fontsize=6,
        fontsize=6, loc="lower left",
        bbox_to_anchor=(0.002, 0.005),
        ncol=2, framealpha=0.9, edgecolor="#CCCCCC",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset_gallery_10pairs.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Dataset gallery saved → {out_path}")


DEFAULT_HELDOUT_CASES   = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_G8_EVAL_JSONL   = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "g8_posctx_dim__ap_eval.jsonl"
DEFAULT_GEMINI_EVAL_JSONL = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "gemini_ap_v2__ap_eval.jsonl"
DEFAULT_GEMINI_TRACES   = (
    PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g8" / "gemini_ap_v2"
    / "traces_20260407_235044_v2_lora_p0_union_p1.jsonl"
)
# Combined rerank file (phase5, G8): mode='full_topology' = G8 full-topo + GR rerank
#                                     mode='p1_only'     = P1-only coarse + GR rerank
DEFAULT_G8_RERANK_JSONL = (
    PROJECT_ROOT / "output" / "lora6_v2_ap_20260331"
    / "graph_rag_rerank" / "20260407_g8_phase5_v1"
    / "graph_rag_rerank_results.jsonl"
)


def _rank_badge(rank: int | None, pool: int | None = None) -> Tuple[str, str]:
    """Return (label, hex_color) rank badge for display."""
    if rank is None:
        return "MISS", "#CC3333"
    if rank == 1:
        return "TOP-1", "#1A8C3E"
    if rank <= 5:
        return f"Rank {rank}", "#2B7FE0"
    if rank <= 15:
        return f"Rank {rank}", "#E08B2B"
    pool_str = f"/{pool}" if pool else ""
    return f"Rank {rank}{pool_str}", "#CC3333"


def _fmt_constraints(raw: str | None | dict) -> str:
    """Format a raw constraints JSON string for compact display."""
    if raw is None:
        return "—"
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return str(raw)[:120]
    lines = []
    ifc = obj.get("ifc_class") or "?"
    storey = obj.get("storey_name") or "?"
    lines.append(f"{ifc}  |  Floor {storey}")
    space = obj.get("space_name")
    if space:
        lines.append(f"Space: {space}")
    kw = obj.get("target_name_keyword")
    if kw:
        lines.append(f"Keyword: {kw}")
    for sr in (obj.get("spatial_relations") or []):
        pred = sr.get("predicate", "?")
        obj_t = (sr.get("object_type") or "?").replace("IfcWallStandardCase", "IfcWall*")
        mat = sr.get("object_material") or ""
        dirn = sr.get("direction") or ""
        extra = "  ".join(filter(None, [mat, dirn]))
        lines.append(f"  {pred}: {obj_t}" + (f" ({extra})" if extra else ""))
    return "\n".join(lines)


def _compute_gemini_ranks(
    gemini_traces_path: Path,
    cases_gt: Dict[str, str],
) -> Dict[str, Tuple[int | None, int | None]]:
    """Return {case_id: (rank, pool_size)} for gemini traces."""
    result: Dict[str, Tuple[int | None, int | None]] = {}
    if not gemini_traces_path.exists():
        return result
    with gemini_traces_path.open() as f:
        for line in f:
            t = json.loads(line)
            cid = t.get("scenario_id", "")
            gt_guid = cases_gt.get(cid)
            pool_sz = t.get("final_pool_size")
            interns = t.get("internals") or {}
            rr = interns.get("retrieval_results") or []
            cands = rr[0].get("candidates", []) if rr else []
            rank: int | None = None
            if cands and gt_guid:
                for i, c in enumerate(cands):
                    if c.get("guid") == gt_guid:
                        rank = i + 1
                        break
            result[cid] = (rank, pool_sz)
    return result


def _parse_constraints(raw: str | None | dict) -> dict | None:
    """Resolve a constraints value to a dict.

    Handles three storage forms:
    - dict already            → return as-is
    - complete JSON string     → json.loads
    - truncated JSON string    → return None (unparseable)
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


def _fmt_full_json(raw: str | None | dict, max_lines: int = 40, max_line_len: int = 42) -> str:
    """Pretty-print a constraints dict for gallery display.

    Accepts a dict, a complete JSON string, or a truncated JSON string.
    Renders key-value pairs directly (no outer braces) to maximise information
    density.  spatial_relations entries are each shown on their own indented
    block.
    """
    if raw is None:
        return "—"

    obj = _parse_constraints(raw)

    if obj is None:
        # unparseable (e.g. Gemini truncated raw_output) — show raw as-is
        text = str(raw)
        lines = text.split("\n")
        clipped = [
            (ln[:max_line_len] + "…") if len(ln) > max_line_len else ln
            for ln in lines[:max_lines]
        ]
        clipped.append("  [⚠ raw_output truncated in source file]")
        return "\n".join(clipped)

    # ── pretty-format key by key ──────────────────────────────────────────────
    SCALAR_KEYS = ["storey_name", "ifc_class", "space_name",
                   "target_name_keyword", "position_context"]
    lines: List[str] = []

    for key in SCALAR_KEYS:
        if key not in obj:
            continue
        val = obj[key]
        val_str = "null" if val is None else f'"{val}"'
        lines.append(f'{key}: {val_str}')

    # remaining scalar keys not in the priority list
    for key, val in obj.items():
        if key in SCALAR_KEYS or key == "spatial_relations":
            continue
        val_str = json.dumps(val, ensure_ascii=False)
        line = f'{key}: {val_str}'
        lines.append((line[:max_line_len] + "…") if len(line) > max_line_len else line)

    # spatial_relations block
    srs = obj.get("spatial_relations") or []
    if srs:
        lines.append(f'spatial_relations: ({len(srs)} relation{"s" if len(srs) != 1 else ""})')
        for i, sr in enumerate(srs):
            pred  = sr.get("predicate", "?")
            obj_t = sr.get("object_type", "?")
            conf  = sr.get("confidence")
            mat   = sr.get("object_material") or ""
            dirn  = sr.get("direction") or ""
            host  = sr.get("host_name") or ""
            conf_str = f" conf={conf:.2f}" if conf is not None else ""
            extras = "  ".join(filter(None, [dirn, mat, host]))
            sr_line = f'  [{i}] {pred} → {obj_t}{conf_str}'
            lines.append((sr_line[:max_line_len] + "…") if len(sr_line) > max_line_len else sr_line)
            if extras:
                ex_line = f'       {extras}'
                lines.append((ex_line[:max_line_len] + "…") if len(ex_line) > max_line_len else ex_line)
    else:
        lines.append("spatial_relations: []")

    # clip total lines
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append(f"  … +more")

    return "\n".join(lines)


def _srs_from_raw(raw: str | None | dict) -> List[dict]:
    """Extract spatial_relations list from a constraints value."""
    obj = _parse_constraints(raw)
    if obj is None:
        return []
    return list(obj.get("spatial_relations") or [])


def _ifc_from_raw(raw: str | None | dict) -> str:
    obj = _parse_constraints(raw)
    if obj is None:
        return "?"
    return obj.get("ifc_class") or "?"


def _draw_sr_graph_for_model(
    ax,
    raw_output: str | None | dict,
    fontscale: float = 0.55,
    bg: str = "#FFFFFF",
) -> None:
    """Draw a topology glyph from a model's raw_output constraints JSON.

    Shows 'no SR extracted' with ifc/storey info when the model output no
    spatial_relations.  Passes all present SRs to _draw_topology_glyph which
    handles semantic positioning, predicate colouring, and chain detection.
    """
    ax.set_facecolor(bg)
    srs = _srs_from_raw(raw_output)
    ifc = _ifc_from_raw(raw_output)

    if not srs:
        ax.axis("off")
        # Show what the model DID extract even without spatial relations
        try:
            obj = json.loads(raw_output) if isinstance(raw_output, str) else (raw_output or {})
            storey = obj.get("storey_name") or "?"
            space  = obj.get("space_name") or ""
            kw     = obj.get("target_name_keyword") or ""
        except Exception:
            storey = space = kw = ""
        lines = [f"{_short_obj_label(ifc)}", f"Floor {storey}"]
        if space: lines.append(f"Space: {space}")
        if kw:    lines.append(f"kw: {kw}")
        lines.append("─── no SR ───")
        ax.text(0.5, 0.5, "\n".join(lines),
                ha="center", va="center",
                fontsize=5.5 * fontscale, color="#888888",
                transform=ax.transAxes, linespacing=1.35)
        return

    _draw_topology_glyph(ax, {"ifc_class": ifc, "relations": srs}, fontscale=fontscale)


def generate_eval_gallery(
    out_dir: Path,
    heldout_cases_jsonl: Path = DEFAULT_HELDOUT_CASES,
    g8_eval_jsonl: Path = DEFAULT_G8_EVAL_JSONL,
    gemini_eval_jsonl: Path = DEFAULT_GEMINI_EVAL_JSONL,
    gemini_traces_path: Path = DEFAULT_GEMINI_TRACES,
    g8_rerank_jsonl: Path = DEFAULT_G8_RERANK_JSONL,
    ds_root: Path = DEFAULT_DS_ROOT,
) -> None:
    """Generate a 60-row × 10-col eval gallery saved as PDF + PNG.

    Columns
    ──────────────────────────────────────────────────────────────────
    0   Case label + chat query + GT metadata
    1   Site photo          ┐ tight left group
    2   Floorplan           │
    3   GT SR graph         ┘
    4   G8 full JSON        ┐ G8 group (blue tint)
    5   G8 SR graph         ┘
    6   G8-P1 rerank info   ┐ P1 group (green tint)
    7   P1 SR graph (= G8)  ┘
    8   Gemini v2 full JSON ┐ Gemini group (orange tint)
    9   Gemini v2 SR graph  ┘
    """
    import textwrap

    # ── Load all data ──────────────────────────────────────────────────────────
    heldout_cases: List[dict] = _load_jsonl(heldout_cases_jsonl)

    def _load_by_id(path: Path, key: str = "case_id") -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    out[r[key]] = r
        return out

    def _load_by_id_mode(path: Path, mode_val: str, key: str = "case_id") -> Dict[str, dict]:
        """Load JSONL filtered to a specific 'mode' field value."""
        out: Dict[str, dict] = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("mode") == mode_val:
                        out[r[key]] = r
        return out

    g8_eval     = _load_by_id(g8_eval_jsonl)
    gemini_eval = _load_by_id(gemini_eval_jsonl)
    # G8 rerank JSONL contains both modes in one file
    g8_rerank   = _load_by_id_mode(g8_rerank_jsonl, mode_val="full_topology")
    p1_rerank   = _load_by_id_mode(g8_rerank_jsonl, mode_val="p1_only")
    eval_msgs   = _load_by_id(DEFAULT_EVAL_JSONL, key="id")

    cases_gt_guid = {
        c["case_id"]: (c.get("ground_truth") or {}).get("target_guid")
        for c in heldout_cases
    }
    gem_ranks = _compute_gemini_ranks(gemini_traces_path, cases_gt_guid)

    FAMILY_COLORS_EVAL = {
        "singleton:CONNECTS_TO":   "#9B2BE0",
        "singleton:ADJACENT_TO":   "#2BB56A",
        "paired:FILLS+NEXT_TO":    "#2B7FE0",
        "triad:FILLS+NEXT_TO+NEXT_TO": "#E05A2B",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": "#E0A02B",
    }

    def _case_family(c: dict) -> str:
        rels = list((c.get("labels", {}).get("constraints", {}).get("spatial_relations") or []))
        return _topology_family(rels)

    # ── Layout: 10 columns ────────────────────────────────────────────────────
    #   col  0    1    2    3    4     5    6     7    8     9
    #   role case site  fp  gtSR g7J  g7SR p1inf p1SR gemJ gemSR
    COL_W = [1.35, 1.00, 1.00, 1.00, 1.70, 1.00, 1.70, 1.00, 1.70, 1.00]
    N     = len(heldout_cases)
    ROW_H = 2.2    # inches — enough vertical room for JSON text
    HDR_H = 0.35

    FIG_W = sum(COL_W) * 1.30           # ≈ 18.9 inches
    FIG_H = N * ROW_H + HDR_H + 0.5    # ≈ 133 inches for 60 rows

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="#FFFFFF")

    gs = matplotlib.gridspec.GridSpec(
        N + 1, 10,
        figure=fig,
        height_ratios=[HDR_H / ROW_H] + [1.0] * N,
        width_ratios=COL_W,
        hspace=0.38,
        wspace=0.04,   # tight globally; visual separation via background color
        left=0.004,
        right=0.996,
        top=0.993,
        bottom=0.003,
    )

    # ── Header row ────────────────────────────────────────────────────────────
    HDR_SPEC = [
        # (col_span_start, col_span_end, label, bg)
        (0, 1, "Case / Query",          "#DCDCDC"),
        (1, 2, "Site",                  "#DCDCDC"),
        (2, 3, "Floorplan",             "#DCDCDC"),
        (3, 4, "GT SR Graph",           "#DCDCDC"),
        (4, 5, "G8 JSON",               "#C8DCEF"),
        (5, 6, "G8 SR",                 "#C8DCEF"),
        (6, 7, "G8-P1+GR\nRerank",      "#C8EFD8"),
        (7, 8, "P1 SR\n(G8 query)",     "#C8EFD8"),
        (8, 9, "Gemini JSON",           "#EFD8C8"),
        (9, 10, "Gemini SR",            "#EFD8C8"),
    ]
    for c_start, c_end, label, bg in HDR_SPEC:
        ax = fig.add_subplot(gs[0, c_start:c_end])
        ax.set_facecolor(bg)
        ax.axis("off")
        ax.text(0.5, 0.5, label, ha="center", va="center",
                fontsize=6.0, fontweight="bold", color="#1A1A1A",
                transform=ax.transAxes, linespacing=1.25)

    # ── Helper: JSON text panel ───────────────────────────────────────────────
    def _json_panel(ax, raw_output, rank_val, pool_sz, bg):
        ax.axis("off")
        ax.set_facecolor(bg)
        # rank badge
        label, badge_color = _rank_badge(rank_val, pool_sz)
        ax.text(0.97, 0.98, label,
                ha="right", va="top", fontsize=5.2, fontweight="bold",
                color="white", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.20", facecolor=badge_color, edgecolor="none", zorder=5))
        # full JSON
        text = _fmt_full_json(raw_output, max_lines=34, max_line_len=38)
        ax.text(0.03, 0.94, text,
                ha="left", va="top", fontsize=3.8,
                color="#1A1A1A", transform=ax.transAxes,
                linespacing=1.25, family="monospace")

    # ── Helper: P1 rerank info panel ──────────────────────────────────────────
    def _p1_panel(ax, p1r_rec, g8_raw, rank_val, pool_sz, bg):
        ax.axis("off")
        ax.set_facecolor(bg)
        label, badge_color = _rank_badge(rank_val, pool_sz)
        ax.text(0.97, 0.98, label,
                ha="right", va="top", fontsize=5.2, fontweight="bold",
                color="white", transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.20", facecolor=badge_color, edgecolor="none"))
        p1_base    = p1r_rec.get("base_rank")
        p1_winner  = p1r_rec.get("raw_output") or "—"
        p1_pool    = p1r_rec.get("pool_size")
        lines = [
            f"Strategy: P1-only",
            f"Base rank : {p1_base}",
            f"Reranked  : {rank_val}",
            f"Pool size : {p1_pool}",
            f"Winner    : {p1_winner}",
            "",
        ]
        # Append the underlying G8 JSON (truncated) for reference
        lines.append("── G8 constraints used ──")
        lines.append(_fmt_full_json(g8_raw, max_lines=22, max_line_len=38))
        text = "\n".join(lines)
        ax.text(0.03, 0.94, text,
                ha="left", va="top", fontsize=3.8,
                color="#1A1A1A", transform=ax.transAxes,
                linespacing=1.25, family="monospace")

    # ── Data rows ─────────────────────────────────────────────────────────────
    for row_i, case in enumerate(heldout_cases):
        gi  = row_i + 1
        cid = case["case_id"]

        gt       = (case.get("labels") or {}).get("constraints") or {}
        gt_srs   = list(gt.get("spatial_relations") or [])
        ifc_class = gt.get("ifc_class", "?")
        storey_nm = gt.get("storey_name", "?")
        fam       = _case_family(case)
        fam_color = FAMILY_COLORS_EVAL.get(fam, "#777777")

        # Query text (prefer heldout, fall back to eval_msgs)
        query_text = case.get("query_text") or ""
        if not query_text:
            emsg = eval_msgs.get(cid)
            if emsg:
                query_text = next(
                    (m["text"] for m in emsg["messages"][1]["content"] if m.get("type") == "text"),
                    "",
                )

        # Rank data
        g8r_rec  = g8_rerank.get(cid, {})
        p1r_rec  = p1_rerank.get(cid, {})
        gem_rank, gem_pool = gem_ranks.get(cid, (None, None))
        g8_base_rank  = g8r_rec.get("base_rank")
        g8_pool_size  = g8r_rec.get("pool_size")
        p1_final_rank = p1r_rec.get("reranked_rank")
        p1_pool_size  = p1r_rec.get("pool_size")

        # G8 raw_output: prefer the complete parsed constraints dict.
        g8_raw  = (g8_eval.get(cid) or {}).get("constraints") \
                  or (g8_eval.get(cid) or {}).get("raw_output")
        # Gemini raw_output is hard-truncated at ~67 chars in the precomputed
        # file (logging bug); use the complete parsed constraints dict instead.
        gem_raw = (gemini_eval.get(cid) or {}).get("constraints") \
                  or (gemini_eval.get(cid) or {}).get("raw_output")

        # ── Col 0: Case info + query ───────────────────────────────────────
        ax_t = fig.add_subplot(gs[gi, 0])
        ax_t.axis("off")
        ax_t.set_facecolor("#F8F8F8")
        ax_t.text(0.5, 0.985, cid,
                  ha="center", va="top", fontsize=5.8, fontweight="bold",
                  color="white", transform=ax_t.transAxes,
                  bbox=dict(boxstyle="round,pad=0.22", facecolor=fam_color, edgecolor="none"))
        fam_short = (fam.replace("singleton:", "sngl:")
                        .replace("triad:", "triad:")
                        .replace("paired:", "pair:"))
        ax_t.text(0.5, 0.880, fam_short,
                  ha="center", va="top", fontsize=4.2, color=fam_color,
                  transform=ax_t.transAxes, style="italic")
        wrapped = "\n".join(textwrap.wrap(f'"{query_text}"', 30))
        ax_t.text(0.5, 0.810, wrapped,
                  ha="center", va="top", fontsize=4.2,
                  color="#333333", transform=ax_t.transAxes, linespacing=1.3)
        tag = f"{ifc_class} | Fl {storey_nm}"
        ax_t.text(0.5, 0.06, tag,
                  ha="center", va="bottom", fontsize=4.0, color="#555555",
                  transform=ax_t.transAxes,
                  bbox=dict(boxstyle="round,pad=0.12", facecolor="#EBEBEB",
                            edgecolor="#CCCCCC", lw=0.4))

        # ── Col 1: Site photo ──────────────────────────────────────────────
        ax_s = fig.add_subplot(gs[gi, 1])
        ax_s.imshow(
            _load_image_or_blank(_find_dataset_image(cid, "site", ds_root)),
            aspect="auto",
        )
        ax_s.axis("off")

        # ── Col 2: Floorplan ───────────────────────────────────────────────
        ax_f = fig.add_subplot(gs[gi, 2])
        ax_f.imshow(
            _load_image_or_blank(_find_dataset_image(cid, "floorplan", ds_root)),
            aspect="auto",
        )
        ax_f.axis("off")

        # ── Col 3: GT SR graph ─────────────────────────────────────────────
        ax_gtsr = fig.add_subplot(gs[gi, 3])
        ax_gtsr.set_facecolor("#F6F6F6")
        _draw_topology_glyph(ax_gtsr, {
            "ifc_class": ifc_class,
            "relations": gt_srs,
        }, fontscale=0.62)

        # ── Col 4: G7 full JSON + rank badge ──────────────────────────────
        ax_g8j = fig.add_subplot(gs[gi, 4])
        _json_panel(ax_g8j, g8_raw, g8_base_rank, g8_pool_size, bg="#EFF5FF")

        # ── Col 5: G7 SR graph ─────────────────────────────────────────────
        ax_g8sr = fig.add_subplot(gs[gi, 5])
        _draw_sr_graph_for_model(ax_g8sr, g8_raw, fontscale=0.52, bg="#EFF5FF")

        # ── Col 6: P1-only rerank info ─────────────────────────────────────
        ax_p1j = fig.add_subplot(gs[gi, 6])
        _p1_panel(ax_p1j, p1r_rec, g8_raw, p1_final_rank, p1_pool_size, bg="#EFFFEF")

        # ── Col 7: P1 SR graph (G7 query graph — same constraints) ─────────
        ax_p1sr = fig.add_subplot(gs[gi, 7])
        _draw_sr_graph_for_model(ax_p1sr, g8_raw, fontscale=0.52, bg="#EFFFEF")

        # ── Col 8: Gemini full JSON + rank badge ───────────────────────────
        ax_gemj = fig.add_subplot(gs[gi, 8])
        _json_panel(ax_gemj, gem_raw, gem_rank, gem_pool, bg="#FFF5EE")

        # ── Col 9: Gemini SR graph ─────────────────────────────────────────
        ax_gemsr = fig.add_subplot(gs[gi, 9])
        _draw_sr_graph_for_model(ax_gemsr, gem_raw, fontscale=0.52, bg="#FFF5EE")

    # ── Rank tier legend ──────────────────────────────────────────────────────
    rank_legend = [
        matplotlib.patches.Patch(facecolor="#1A8C3E", label="TOP-1"),
        matplotlib.patches.Patch(facecolor="#2B7FE0", label="Rank 2–5"),
        matplotlib.patches.Patch(facecolor="#E08B2B", label="Rank 6–15"),
        matplotlib.patches.Patch(facecolor="#CC3333", label="MISS (>15 or not found)"),
    ]
    fig.legend(handles=rank_legend, title="Rank tier", title_fontsize=6.5,
               fontsize=6.5, loc="lower right", bbox_to_anchor=(0.999, 0.0005),
               ncol=4, framealpha=0.92, edgecolor="#BBBBBB")

    fam_patches = [
        matplotlib.patches.Patch(
            facecolor=c,
            label=(f.replace("singleton:", "sngl:").replace("triad:", "triad:").replace("paired:", "pair:"))
        )
        for f, c in FAMILY_COLORS_EVAL.items()
    ]
    fig.legend(handles=fam_patches, title="Topology family", title_fontsize=6.5,
               fontsize=6.5, loc="lower left", bbox_to_anchor=(0.001, 0.0005),
               ncol=3, framealpha=0.92, edgecolor="#BBBBBB")

    # ── Save PDF + PNG ────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "eval_gallery_60cases.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight",
                facecolor="white", dpi=150)
    print(f"Eval gallery PDF → {pdf_path}")

    png_path = out_dir / "eval_gallery_60cases.png"
    fig.savefig(png_path, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"Eval gallery PNG → {png_path}")

    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Diagnostic: shortcut learning & template collapse analysis
# ════════════════════════════════════════════════════════════════════════════

def generate_diagnostic_figure(out_dir: Path) -> None:
    """Generate a multi-panel diagnostic figure exposing shortcut / collapse risks.

    Panels
    ──────
    A  Confidence collapse — all G7 SRs output conf=1.0 (bar chart)
    B  Output fingerprint uniqueness — 46.7% unique vs 53.3% duplicates (pie/donut)
    C  Duplicate-fingerprint cases: each shares distinct GT targets (scatter)
    D  NEXT_TO count confusion matrix (GT 0/1/2 vs pred 0/1/2)
    E  Material hallucination: predicted material vs GT material (horizontal bar)
    F  Rank distribution per topology family (violin/strip)
    G  Storey accuracy by floor — 100% across all 7 levels (perfect bar)
    H  Predicate set match breakdown (stacked bar)
    """
    cases_path = DEFAULT_HELDOUT_CASES
    g7_path    = DEFAULT_G7_EVAL_JSONL
    rerank_path = DEFAULT_G7_RERANK_JSONL

    cases  = {json.loads(l)['case_id']: json.loads(l)
              for l in cases_path.open() if l.strip()}
    g7     = {json.loads(l)['case_id']: json.loads(l)
              for l in g7_path.open() if l.strip()}
    rerank = {json.loads(l)['case_id']: json.loads(l)
              for l in rerank_path.open() if l.strip()}

    def gt_c(row):
        return row.get('labels', {}).get('constraints', {})

    # ── Pre-compute metrics ──────────────────────────────────────────────────

    # Confidence values
    g7_confs = []
    for row in g7.values():
        for sr in (row.get('constraints', {}).get('spatial_relations') or []):
            g7_confs.append(sr.get('confidence'))
    conf_ctr = Counter(g7_confs)

    # Fingerprint uniqueness
    fingerprints = []
    fingerprint_map: Dict[tuple, List] = defaultdict(list)
    for cid, row in g7.items():
        c = row.get('constraints', {})
        fp = (c.get('ifc_class'), c.get('storey_name'),
              tuple(sorted((sr.get('predicate', ''), sr.get('object_type', ''))
                           for sr in (c.get('spatial_relations') or []))))
        fingerprints.append(fp)
        fingerprint_map[fp].append((cid, cases[cid]['ground_truth']['target_guid']))
    n_unique = len(set(fingerprints))
    n_dup    = len(fingerprints) - n_unique

    # NEXT_TO confusion
    nt_confusion: Dict[Tuple[int,int], int] = defaultdict(int)
    for cid, row in cases.items():
        gt_srs = gt_c(row).get('spatial_relations') or []
        pr_srs = (g7.get(cid) or {}).get('constraints', {}).get('spatial_relations') or []
        gt_nt  = sum(1 for s in gt_srs if s.get('predicate') == 'NEXT_TO')
        pr_nt  = sum(1 for s in pr_srs if s.get('predicate') == 'NEXT_TO')
        nt_confusion[(gt_nt, pr_nt)] += 1

    # Material hallucination
    mat_correct = mat_halluc = mat_null = 0
    halluc_mats: List[Tuple[str, str]] = []   # (gt_mat, pred_mat)
    for cid, row in cases.items():
        gt_srs = gt_c(row).get('spatial_relations') or []
        pr_srs = (g7.get(cid) or {}).get('constraints', {}).get('spatial_relations') or []
        gt_mats = set(s.get('object_material') for s in gt_srs if s.get('object_material'))
        pr_mats = set(s.get('object_material') for s in pr_srs if s.get('object_material'))
        for m in pr_mats:
            if m in gt_mats:
                mat_correct += 1
            else:
                mat_halluc += 1
                gt_label = list(gt_mats)[0] if gt_mats else "—"
                halluc_mats.append((gt_label, m))
        if not pr_mats:
            mat_null += 1

    # Predicate match breakdown
    pred_correct = pred_over = pred_under = pred_other = 0
    for cid, row in cases.items():
        gt_srs = gt_c(row).get('spatial_relations') or []
        pr_srs = (g7.get(cid) or {}).get('constraints', {}).get('spatial_relations') or []
        gt_p = tuple(sorted(s.get('predicate','') for s in gt_srs))
        pr_p = tuple(sorted(s.get('predicate','') for s in pr_srs))
        if gt_p == pr_p:
            pred_correct += 1
        elif len(pr_p) > len(gt_p):
            pred_over += 1
        elif len(pr_p) < len(gt_p):
            pred_under += 1
        else:
            pred_other += 1

    # Rank by family
    family_ranks: Dict[str, List[int]] = defaultdict(list)
    for cid, rrow in rerank.items():
        fam = rrow.get('family', '?')
        r   = rrow.get('base_rank')
        if r is not None:
            family_ranks[fam].append(r)

    # Storey accuracy
    storey_acc: Dict[str, Dict] = defaultdict(lambda: {'correct': 0, 'total': 0})
    for cid, row in cases.items():
        gt_s   = str(gt_c(row).get('storey_name', ''))
        pred_s = str((g7.get(cid) or {}).get('constraints', {}).get('storey_name', ''))
        storey_acc[gt_s]['total'] += 1
        if gt_s == pred_s:
            storey_acc[gt_s]['correct'] += 1

    # ── Layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs_top = matplotlib.gridspec.GridSpec(
        2, 4, figure=fig,
        top=0.93, bottom=0.54,
        left=0.06, right=0.97,
        wspace=0.38, hspace=0.55,
    )
    gs_bot = matplotlib.gridspec.GridSpec(
        2, 4, figure=fig,
        top=0.48, bottom=0.07,
        left=0.06, right=0.97,
        wspace=0.38, hspace=0.55,
    )

    PANEL_LABEL_KW = dict(fontsize=11, fontweight="bold", color="#222")
    SUBTITLE_KW    = dict(fontsize=8.5, color="#555")
    GRID_KW        = dict(color="#DDDDDD", linewidth=0.6, zorder=0)
    ACCENT         = "#2E4F60"
    RED            = "#C0392B"
    GREEN          = "#1A8C3E"
    AMBER          = "#D4850A"

    fig.text(0.5, 0.97,
             "G7 Extractor Diagnostic: Shortcut Learning & Template Collapse",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#111")
    fig.text(0.5, 0.945,
             "AP held-out benchmark  •  n=60 cases  •  G7 position_context model",
             ha="center", va="top", fontsize=9, color="#666")

    # ── A: Confidence collapse ────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_a.set_facecolor("white")
    conf_keys = sorted(conf_ctr.keys(), key=lambda x: x if x is not None else -1)
    conf_vals = [conf_ctr[k] for k in conf_keys]
    conf_labels = [str(k) for k in conf_keys]
    bars = ax_a.bar(conf_labels, conf_vals, color=[RED if k == 1.0 else ACCENT for k in conf_keys],
                    width=0.55, edgecolor="white", linewidth=0.8)
    ax_a.bar_label(bars, fmt="%d", fontsize=8, padding=2)
    ax_a.set_xlabel("Confidence value", fontsize=8)
    ax_a.set_ylabel("Count", fontsize=8)
    ax_a.set_title("A  Confidence Collapse", **PANEL_LABEL_KW)
    ax_a.text(0.5, 0.72, "100% of SRs output conf=1.0\n(training data all conf=1.0 → collapse)",
              transform=ax_a.transAxes, ha="center", va="top", **SUBTITLE_KW,
              bbox=dict(facecolor="#FFF3CD", edgecolor="#D4850A", boxstyle="round,pad=0.3"))
    ax_a.grid(axis="y", **GRID_KW)
    ax_a.set_axisbelow(True)
    ax_a.tick_params(labelsize=8)

    # ── B: Fingerprint uniqueness donut ──────────────────────────────────────
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_b.set_facecolor("white")
    wedge_colors = [GREEN, RED]
    wedges, texts, autotexts = ax_b.pie(
        [n_unique, n_dup],
        labels=["Unique\noutputs", "Duplicate\noutputs"],
        colors=wedge_colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=8),
        pctdistance=0.72,
    )
    for at, c in zip(autotexts, ["white", "white"]):
        at.set_color(c)
        at.set_fontsize(9)
        at.set_fontweight("bold")
    ax_b.text(0, 0, f"{n_unique}/{len(fingerprints)}\nunique", ha="center", va="center",
              fontsize=9, fontweight="bold", color="#222")
    ax_b.set_title("B  Output Fingerprints", **PANEL_LABEL_KW)
    ax_b.text(0.5, -0.08,
              "Duplicate fingerprints → same query sent for n different GT targets\n"
              "Max collapse: 8 cases share one fingerprint (all distinct GTs)",
              transform=ax_b.transAxes, ha="center", va="top", **SUBTITLE_KW,
              bbox=dict(facecolor="#FFF3CD", edgecolor=AMBER, boxstyle="round,pad=0.3"))

    # ── C: Duplicate fingerprint collapse scatter ─────────────────────────────
    ax_c = fig.add_subplot(gs_top[0, 2])
    ax_c.set_facecolor("white")
    dup_sizes = sorted(
        [(fp, len(items)) for fp, items in fingerprint_map.items() if len(items) >= 2],
        key=lambda x: -x[1]
    )
    ys     = list(range(len(dup_sizes)))
    counts = [cnt for _, cnt in dup_sizes]
    labels = [f"{fp[0].replace('Ifc','')}/F{fp[1]}" for fp, _ in dup_sizes]
    ax_c.barh(ys, counts, color=RED, alpha=0.75, edgecolor="white")
    ax_c.set_yticks(ys)
    ax_c.set_yticklabels(labels, fontsize=6.5)
    ax_c.set_xlabel("# cases sharing fingerprint", fontsize=8)
    ax_c.set_title("C  Collapse Severity per Fingerprint", **PANEL_LABEL_KW)
    ax_c.axvline(1, color="#AAAAAA", lw=0.8, ls="--")
    ax_c.grid(axis="x", **GRID_KW)
    ax_c.set_axisbelow(True)
    ax_c.tick_params(labelsize=7)
    # annotation: each cluster has distinct GT
    ax_c.text(0.98, 0.02,
              "Each cluster → distinct GT targets\n(retrieval must separate within pool)",
              transform=ax_c.transAxes, ha="right", va="bottom", fontsize=7,
              color="#666", style="italic")

    # ── D: NEXT_TO count confusion matrix ────────────────────────────────────
    ax_d = fig.add_subplot(gs_top[0, 3])
    ax_d.set_facecolor("white")
    vals = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for (gt_n, pr_n), cnt in nt_confusion.items():
        if gt_n <= 2 and pr_n <= 2:
            vals[gt_n, pr_n] += cnt
    im = ax_d.imshow(vals, cmap="RdYlGn", vmin=0, vmax=max(vals.max(), 1))
    for i in range(3):
        for j in range(3):
            v = vals[i, j]
            col = "white" if v > vals.max() * 0.6 else "#222"
            ax_d.text(j, i, str(v), ha="center", va="center", fontsize=11,
                      fontweight="bold", color=col)
    ax_d.set_xticks([0, 1, 2])
    ax_d.set_yticks([0, 1, 2])
    ax_d.set_xticklabels(["pred 0", "pred 1", "pred 2"], fontsize=8)
    ax_d.set_yticklabels(["GT 0", "GT 1", "GT 2"], fontsize=8)
    ax_d.set_title("D  NEXT_TO Count Confusion", **PANEL_LABEL_KW)
    ax_d.set_xlabel("Predicted count", fontsize=8)
    ax_d.set_ylabel("GT count", fontsize=8)
    off_diag = int(vals.sum()) - int(np.trace(vals))
    ax_d.text(0.5, -0.22,
              f"Off-diagonal: {off_diag}/60 = {100*off_diag/60:.0f}% error "
              f"(paired↔triad confusion)",
              transform=ax_d.transAxes, ha="center", va="top", **SUBTITLE_KW)

    # ── E: Material hallucination ─────────────────────────────────────────────
    ax_e = fig.add_subplot(gs_bot[0, 0])
    ax_e.set_facecolor("white")
    mat_groups = {"Correct\nmaterial": mat_correct,
                  "Hallucinated\nmaterial": mat_halluc,
                  "No material\noutput": mat_null}
    colors_e = [GREEN, RED, "#AAAAAA"]
    bars_e = ax_e.bar(list(mat_groups.keys()), list(mat_groups.values()),
                      color=colors_e, edgecolor="white", linewidth=0.8)
    ax_e.bar_label(bars_e, fmt="%d", fontsize=9, padding=2)
    ax_e.set_ylabel("Count", fontsize=8)
    ax_e.set_title("E  Material Output Accuracy", **PANEL_LABEL_KW)
    ax_e.grid(axis="y", **GRID_KW)
    ax_e.set_axisbelow(True)
    ax_e.tick_params(labelsize=8)
    # List top hallucination example
    if halluc_mats:
        hall_ctr = Counter(pr for _, pr in halluc_mats)
        top_hall = hall_ctr.most_common(1)[0]
        ax_e.text(0.5, 0.95,
                  f"Top hallucination: '{top_hall[0]}'\n"
                  f"({top_hall[1]}× output when GT differs) → frequency bias",
                  transform=ax_e.transAxes, ha="center", va="top", fontsize=7,
                  color=RED, style="italic",
                  bbox=dict(facecolor="#FDECEA", edgecolor=RED, boxstyle="round,pad=0.3", alpha=0.9))

    # ── F: Rank distribution by family (strip + mean) ────────────────────────
    ax_f = fig.add_subplot(gs_bot[0, 1:3])
    ax_f.set_facecolor("white")
    fam_order = [
        "singleton:CONNECTS_TO",
        "singleton:ADJACENT_TO",
        "singleton:FILLS",
        "paired:FILLS+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
    ]
    fam_short = {
        "singleton:CONNECTS_TO":                  "CONNECTS_TO\n(n=14)",
        "singleton:ADJACENT_TO":                  "ADJACENT_TO\n(n=12)",
        "singleton:FILLS":                        "FILLS\n(n=1)",
        "paired:FILLS+NEXT_TO":                   "FILLS+NEXT_TO\n(n=10)",
        "triad:FILLS+NEXT_TO+NEXT_TO":            "FILLS+NEXT_TO×2\n(n=21)",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": "FILLS+NTx2(mix)\n(n=2)",
    }
    fam_colors = {
        "singleton:CONNECTS_TO":   "#7B00CC",
        "singleton:ADJACENT_TO":   "#008840",
        "singleton:FILLS":         "#D44000",
        "paired:FILLS+NEXT_TO":    "#1060C8",
        "triad:FILLS+NEXT_TO+NEXT_TO": "#1060C8",
        "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)": "#E08B2B",
    }
    np.random.seed(42)
    for xi, fam in enumerate(fam_order):
        ranks = family_ranks.get(fam, [])
        if not ranks:
            continue
        col = fam_colors.get(fam, ACCENT)
        jitter = np.random.uniform(-0.15, 0.15, len(ranks))
        ax_f.scatter(np.full(len(ranks), xi) + jitter, ranks,
                     color=col, alpha=0.6, s=28, zorder=3)
        ax_f.hlines(np.mean(ranks), xi - 0.3, xi + 0.3,
                    colors=col, linewidth=2.5, zorder=4, label=f"mean={np.mean(ranks):.0f}")
        ax_f.text(xi, np.mean(ranks) + 8, f"μ={np.mean(ranks):.0f}",
                  ha="center", fontsize=7.5, color=col, fontweight="bold")
    ax_f.set_xticks(range(len(fam_order)))
    ax_f.set_xticklabels([fam_short[f] for f in fam_order], fontsize=7.5)
    ax_f.set_ylabel("Base rank (G7, before rerank)", fontsize=8.5)
    ax_f.set_title("F  Rank Distribution by Topology Family", **PANEL_LABEL_KW)
    ax_f.grid(axis="y", **GRID_KW)
    ax_f.set_axisbelow(True)
    ax_f.axhline(1, color=GREEN, lw=1.2, ls="--", alpha=0.5, label="Top-1")
    ax_f.axhline(15, color=AMBER, lw=1.2, ls=":", alpha=0.5, label="Top-15")
    ax_f.set_ylim(bottom=-5)
    ax_f.tick_params(labelsize=8)

    # ── G: Storey accuracy by floor ──────────────────────────────────────────
    ax_g = fig.add_subplot(gs_bot[0, 3])
    ax_g.set_facecolor("white")
    storeys_sorted = sorted(storey_acc.keys(), key=lambda x: int(x) if x.lstrip('-').isdigit() else 0)
    accs = [100 * storey_acc[s]['correct'] / storey_acc[s]['total'] for s in storeys_sorted]
    ns   = [storey_acc[s]['total'] for s in storeys_sorted]
    labels_g = [f"F{s}\n(n={n})" for s, n in zip(storeys_sorted, ns)]
    bars_g = ax_g.bar(labels_g, accs, color=GREEN, alpha=0.85, edgecolor="white")
    ax_g.set_ylim(0, 115)
    ax_g.axhline(100, color="#AAAAAA", lw=0.8, ls="--")
    ax_g.bar_label(bars_g, fmt="%.0f%%", fontsize=8, padding=2)
    ax_g.set_ylabel("Accuracy (%)", fontsize=8)
    ax_g.set_title("G  Storey Accuracy by Floor", **PANEL_LABEL_KW)
    ax_g.text(0.5, 0.42, "100% across all 7 floors\n→ genuine floor reading, not shortcut",
              transform=ax_g.transAxes, ha="center", va="top", **SUBTITLE_KW,
              bbox=dict(facecolor="#DCFFF0", edgecolor=GREEN, boxstyle="round,pad=0.3"))
    ax_g.tick_params(labelsize=7.5)
    ax_g.grid(axis="y", **GRID_KW)
    ax_g.set_axisbelow(True)

    # ── H: Predicate match breakdown stacked bar ─────────────────────────────
    ax_h = fig.add_subplot(gs_bot[1, :2])
    ax_h.set_facecolor("white")
    categories  = ["Exact match", "Over-predict\n(extra rels)", "Under-predict\n(missing rels)", "Wrong predicate"]
    values_h    = [pred_correct, pred_over, pred_under, pred_other]
    colors_h    = [GREEN, AMBER, RED, "#9B59B6"]
    bars_h = ax_h.barh(categories, values_h, color=colors_h, edgecolor="white", height=0.55)
    ax_h.bar_label(bars_h, fmt="%d", fontsize=9, padding=3)
    ax_h.set_xlabel("# cases", fontsize=8.5)
    ax_h.set_title("H  Predicate-Set Match Breakdown  (n=60)", **PANEL_LABEL_KW)
    ax_h.set_xlim(0, 55)
    ax_h.grid(axis="x", **GRID_KW)
    ax_h.set_axisbelow(True)
    ax_h.tick_params(labelsize=9)
    # Annotation
    ax_h.text(0.98, 0.15,
              f"73.3% exact  •  Main error: paired↔triad confusion\n"
              f"Over-predict: {pred_over} cases (+1 NEXT_TO)  "
              f"Under-predict: {pred_under} cases (−1 NEXT_TO)",
              transform=ax_h.transAxes, ha="right", va="bottom", fontsize=7.5, color="#444",
              bbox=dict(facecolor="#F5F5F5", edgecolor="#CCCCCC", boxstyle="round,pad=0.3"))

    # ── I: Summary text panel ─────────────────────────────────────────────────
    ax_i = fig.add_subplot(gs_bot[1, 2:])
    ax_i.axis("off")
    ax_i.set_facecolor("#F0F4F8")
    summary_lines = [
        ("FINDINGS SUMMARY", True, "#1A2E3A"),
        ("", False, "#444"),
        ("✓  IFC class: 100% correct (60/60)", False, GREEN),
        ("✓  Storey name: 100% correct (60/60)", False, GREEN),
        ("✓  Object type: 94.5% correct (103/109 aligned pairs)", False, GREEN),
        ("", False, "#444"),
        ("⚠  Confidence: collapsed to 1.0 in ALL 117 SRs", False, AMBER),
        ("⚠  Fingerprints: 53.3% duplicate (28/60 unique)", False, AMBER),
        ("   → same SR query for n≤8 distinct GT targets", False, "#777"),
        ("   → retrieval must separate within the symbolic pool", False, "#777"),
        ("", False, "#444"),
        ("✗  NEXT_TO count: 14/60 (23%) wrong (paired↔triad)", False, RED),
        ("✗  Material: 5 hallucinations; top='Leather, weathered'", False, RED),
        ("", False, "#444"),
        ("Root causes:", True, "#1A2E3A"),
        ("  1. Confidence collapse → training labels all conf=1.0", False, "#444"),
        ("  2. Attribute shortcut risk: storey/ifc_class so strong", False, "#444"),
        ("     the symbolic layer may over-rely on them alone", False, "#444"),
        ("  3. Triad ambiguity: 1 vs 2 NEXT_TO not visually distinct", False, "#444"),
        ("  4. Material frequency bias: 'Leather, weathered' dominant", False, "#444"),
        ("     in CONNECTS_TO training cases → bleeds cross-family", False, "#444"),
    ]
    y = 0.97
    for text, bold, color in summary_lines:
        ax_i.text(0.04, y, text, transform=ax_i.transAxes,
                  ha="left", va="top", fontsize=8,
                  fontweight="bold" if bold else "normal",
                  color=color)
        y -= 0.046 if text else 0.022

    ax_i.set_title("I  Summary of Findings", **PANEL_LABEL_KW)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "diagnostic_shortcut_collapse.png"
    out_pdf = out_dir / "diagnostic_shortcut_collapse.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Diagnostic PNG → {out_png}")
    print(f"Diagnostic PDF → {out_pdf}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--gallery",
        action="store_true",
        help="Generate full-modality dataset gallery (10 representative pairs).",
    )
    parser.add_argument(
        "--eval-gallery",
        action="store_true",
        help="Generate 60-case eval gallery with G8 / G8-P1-rerank / Gemini v2 model outputs.",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Generate shortcut-learning & template-collapse diagnostic figure.",
    )
    args = parser.parse_args()

    if args.gallery:
        generate_dataset_gallery(args.out_dir)
        return

    if args.eval_gallery:
        generate_eval_gallery(args.out_dir)
        return

    if args.diagnostic:
        generate_diagnostic_figure(args.out_dir)
        return

    cases = _load_jsonl(args.cases)
    summary = analyze(cases)
    write_outputs(args.out_dir, summary)
    print(f"Wrote topology analysis to {args.out_dir}")

"""
conda run -n mscd_demo python mscd_demo/evaluation/analysis/analyze_ap_heldout_topology.py --eval-gallery --out-dir mscd_demo/output/lora6_v2_ap_20260331/topology_analysis/ap_held_out 2>&1
"""

if __name__ == "__main__":
    main()
