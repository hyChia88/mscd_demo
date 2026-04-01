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


def _draw_topology_glyph(ax, row: dict) -> None:
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    target = _short_obj_label(row.get("ifc_class"))
    center = (0.0, 0.0)
    ax.add_patch(Circle(center, 0.24, facecolor="#4f6d7a", edgecolor="none"))
    ax.text(0.0, 0.0, target, ha="center", va="center", fontsize=10, color="white", weight="bold")

    rels = row.get("relations", [])
    by_dir = defaultdict(list)
    auto_rels = []
    for rel in rels:
        direction = str(rel.get("direction") or "").lower()
        if direction in {"left", "right", "up", "down"}:
            by_dir[direction].append(rel)
        else:
            auto_rels.append(rel)

    base = {"left": (-0.98, 0.0), "right": (0.98, 0.0), "up": (0.0, 0.85), "down": (0.0, -0.85)}
    fallback = [(-0.85, 0.68), (0.85, 0.68), (-0.85, -0.68), (0.85, -0.68)]
    placed = []

    for direction, rel_group in by_dir.items():
        n = len(rel_group)
        offsets = np.linspace(-0.28, 0.28, n) if n > 1 else np.array([0.0])
        for rel, offset in zip(rel_group, offsets):
            x, y = base[direction]
            if direction in {"left", "right"}:
                y += float(offset)
            else:
                x += float(offset)
            placed.append((rel, (x, y)))

    for rel, pos in zip(auto_rels, fallback):
        placed.append((rel, pos))

    for rel, (x, y) in placed:
        ax.plot([0.0, x], [0.0, y], color="#6c757d", linewidth=1.8)
        mx, my = x * 0.55, y * 0.55
        ax.text(mx, my, str(rel.get("predicate", "?")), fontsize=8, ha="center", va="center", color="#2b2d42")
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.27, y - 0.12),
                0.54,
                0.24,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#edf2f4",
                edgecolor="#8d99ae",
                linewidth=1.2,
            )
        )
        anchor = _short_obj_label(rel.get("object_type"))
        direction = str(rel.get("direction") or "")
        label = anchor if not direction else f"{anchor}\n{direction}"
        ax.text(x, y, label, ha="center", va="center", fontsize=8.5, weight="bold")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    cases = _load_jsonl(args.cases)
    summary = analyze(cases)
    write_outputs(args.out_dir, summary)
    print(f"Wrote topology analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
