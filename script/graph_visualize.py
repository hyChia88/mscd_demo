#!/usr/bin/env python3
"""
graph_visualize.py

Generate two thesis-ready vector figures that compare:
1. a base IFC graph view (closer to IFC-native structure)
2. an enriched IFC graph view (retrieval-oriented relations and properties)

The script avoids external plotting libraries and writes PDF files directly,
so it can run in constrained environments.

Recommended usage:
    conda run -n mscd_demo python mscd_demo/script/graph_visualize.py

Outputs by default:
    thesis_overleaf/15_ifc_base_graph.pdf
    thesis_overleaf/16_ifc_enriched_graph.pdf
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import ifcopenshell
import ifcopenshell.util.element as ifc_element
import ifcopenshell.util.placement as ifc_placement
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IFC = ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc"
DEFAULT_INDEX = ROOT / "data_curation" / "references" / "element_index.jsonl"
DEFAULT_OUT_DIR = ROOT / "thesis_overleaf"


@dataclass
class Node:
    guid: str
    label: str
    kind: str
    subtitle: str = ""
    attrs: list[str] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    w: float = 146.0
    h: float = 52.0


@dataclass
class Edge:
    source: str
    target: str
    rel: str
    color: Tuple[float, float, float]
    dashed: bool = False


class PdfCanvas:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.parts: list[str] = []
        self.set_fill_rgb(1, 1, 1)
        self.rect(0, 0, width, height, fill=True, stroke=False)

    @staticmethod
    def _esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    def set_stroke_rgb(self, r: float, g: float, b: float):
        self.parts.append(f"{r:.3f} {g:.3f} {b:.3f} RG")

    def set_fill_rgb(self, r: float, g: float, b: float):
        self.parts.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def set_line_width(self, width: float):
        self.parts.append(f"{width:.2f} w")

    def set_dash(self, on: float = 0.0, off: float = 0.0):
        if on > 0 and off > 0:
            self.parts.append(f"[{on:.2f} {off:.2f}] 0 d")
        else:
            self.parts.append("[] 0 d")

    def rect(self, x: float, y: float, w: float, h: float, *, fill: bool, stroke: bool):
        op = "B" if fill and stroke else ("f" if fill else "S")
        self.parts.append(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def line(self, x1: float, y1: float, x2: float, y2: float):
        self.parts.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def circle(self, x: float, y: float, r: float, *, fill: bool, stroke: bool):
        k = 0.5522847498
        ox = r * k
        oy = r * k
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        cmds = [
            f"{x:.2f} {y1:.2f} m",
            f"{x+ox:.2f} {y1:.2f} {x1:.2f} {y+oy:.2f} {x1:.2f} {y:.2f} c",
            f"{x1:.2f} {y-oy:.2f} {x+ox:.2f} {y0:.2f} {x:.2f} {y0:.2f} c",
            f"{x-ox:.2f} {y0:.2f} {x0:.2f} {y-oy:.2f} {x0:.2f} {y:.2f} c",
            f"{x0:.2f} {y+oy:.2f} {x-ox:.2f} {y1:.2f} {x:.2f} {y1:.2f} c",
        ]
        op = "B" if fill and stroke else ("f" if fill else "S")
        self.parts.append(" ".join(cmds) + f" {op}")

    def text(self, x: float, y: float, text: str, size: int = 10):
        self.parts.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({self._esc(text)}) Tj ET")

    def write_pdf(self, path: Path):
        content = "\n".join(self.parts).encode("latin-1", errors="replace")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width} {self.height}] "
                f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ).encode("latin-1"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream",
        ]

        pdf = bytearray()
        pdf.extend(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, obj in enumerate(objs, start=1):
            offsets.append(len(pdf))
            pdf.extend(f"{i} 0 obj\n".encode("latin-1"))
            pdf.extend(obj)
            pdf.extend(b"\nendobj\n")

        startxref = len(pdf)
        pdf.extend(f"xref\n0 {len(objs)+1}\n".encode("latin-1"))
        pdf.extend(b"0000000000 65535 f \n")
        for off in offsets[1:]:
            pdf.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
        pdf.extend(
            f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n".encode(
                "latin-1"
            )
        )
        path.write_bytes(pdf)


def load_index(path: Path) -> dict[str, dict]:
    by_guid: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            guid = row.get("global_id")
            if guid:
                by_guid[guid] = row
    return by_guid


def get_storey_name(entity) -> str:
    container = ifc_element.get_container(entity)
    if container and container.is_a("IfcBuildingStorey"):
        return container.Name or "Unknown Storey"
    return "Unknown Storey"


def extract_continuous_attrs(index_row: dict) -> tuple[bool, str]:
    psets = index_row.get("psets", {})
    constraints = psets.get("Constraints", {})
    base = str(constraints.get("Base Constraint", "")).replace("Level: ", "").strip()
    top = str(constraints.get("Top Constraint", "")).replace("Level: ", "").strip()
    is_cont = bool(base and top and base != top and "Unconnected" not in top)
    if is_cont:
        return True, f"{base} -> {top}"
    return False, ""


def build_graph_data(ifc_path: Path, index_path: Path) -> dict[str, Any]:
    ifc = ifcopenshell.open(str(ifc_path))
    index = load_index(index_path)

    guid_to_entity: dict[str, Any] = {}
    for cls in [
        "IfcWall",
        "IfcWallStandardCase",
        "IfcDoor",
        "IfcWindow",
        "IfcRailing",
        "IfcColumn",
        "IfcBeam",
        "IfcSlab",
        "IfcBuildingStorey",
        "IfcSpace",
        "IfcOpeningElement",
    ]:
        try:
            for ent in ifc.by_type(cls):
                guid_to_entity[ent.GlobalId] = ent
        except RuntimeError:
            continue

    opening_to_host: dict[str, str] = {}
    wall_to_openings: dict[str, list[str]] = defaultdict(list)
    for rel in ifc.by_type("IfcRelVoidsElement"):
        host = rel.RelatingBuildingElement
        opening = rel.RelatedOpeningElement
        if host and opening:
            opening_to_host[opening.GlobalId] = host.GlobalId
            wall_to_openings[host.GlobalId].append(opening.GlobalId)

    opening_to_filler: dict[str, str] = {}
    wall_to_fillers: dict[str, list[str]] = defaultdict(list)
    for rel in ifc.by_type("IfcRelFillsElement"):
        filler = rel.RelatedBuildingElement
        opening = rel.RelatingOpeningElement
        if filler and opening and opening.GlobalId in opening_to_host:
            opening_to_filler[opening.GlobalId] = filler.GlobalId
            wall_to_fillers[opening_to_host[opening.GlobalId]].append(filler.GlobalId)

    wall_connections: dict[str, set[str]] = defaultdict(set)
    for rel in ifc.by_type("IfcRelConnectsPathElements"):
        a = rel.RelatingElement
        b = rel.RelatedElement
        if a and b:
            wall_connections[a.GlobalId].add(b.GlobalId)
            wall_connections[b.GlobalId].add(a.GlobalId)

    def nearest_adjacent(guid: str, limit: int = 3) -> list[str]:
        anchor = index.get(guid)
        if not anchor:
            return []
        c0 = anchor.get("centroid") or {}
        storey = anchor.get("storey_name")
        if not c0 or not storey:
            return []
        out: list[tuple[float, str]] = []
        for other_guid, row in index.items():
            if other_guid == guid:
                continue
            if row.get("storey_name") != storey:
                continue
            if row.get("ifc_class") == anchor.get("ifc_class"):
                continue
            c1 = row.get("centroid") or {}
            if not c1:
                continue
            dx = c0.get("x", 0.0) - c1.get("x", 0.0)
            dy = c0.get("y", 0.0) - c1.get("y", 0.0)
            dz = c0.get("z", 0.0) - c1.get("z", 0.0)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if 100.0 < dist <= 1500.0:
                out.append((dist, other_guid))
        out.sort()
        return [g for _, g in out[:limit]]

    def next_to_pairs(wall_guid: str) -> list[tuple[str, str]]:
        wall = guid_to_entity.get(wall_guid)
        fillers = [guid_to_entity[g] for g in wall_to_fillers.get(wall_guid, []) if g in guid_to_entity]
        if not wall or len(fillers) < 2:
            return []
        try:
            wall_mat = ifc_placement.get_local_placement(wall.ObjectPlacement)
            wall_dir = np.array([wall_mat[0][0], wall_mat[1][0], wall_mat[2][0]])
            wall_origin = np.array([wall_mat[0][3], wall_mat[1][3], wall_mat[2][3]])
        except Exception:
            return []

        storey_groups: dict[str, list[tuple[float, Any]]] = defaultdict(list)
        for filler in fillers:
            try:
                mat = ifc_placement.get_local_placement(filler.ObjectPlacement)
                centroid = np.array([mat[0][3], mat[1][3], mat[2][3]])
                proj = float(np.dot(centroid - wall_origin, wall_dir))
                storey_groups[get_storey_name(filler)].append((proj, filler))
            except Exception:
                continue

        pairs: list[tuple[str, str]] = []
        for group in storey_groups.values():
            if len(group) < 2:
                continue
            group.sort(key=lambda x: x[0])
            for i in range(len(group) - 1):
                pairs.append((group[i][1].GlobalId, group[i + 1][1].GlobalId))
        return pairs

    candidates: list[tuple[int, int, int, str]] = []
    for wall_guid, fillers in wall_to_fillers.items():
        n_fill = len(fillers)
        n_conn = len(wall_connections.get(wall_guid, set()))
        n_adj = len(nearest_adjacent(wall_guid))
        score = n_fill * 12 + n_conn * 4 + n_adj
        candidates.append((score, n_fill, n_conn, wall_guid))
    candidates.sort(reverse=True)
    anchor_guid = candidates[0][3] if candidates else next(iter(wall_to_fillers.keys()))

    return {
        "ifc": ifc,
        "index": index,
        "guid_to_entity": guid_to_entity,
        "anchor_guid": anchor_guid,
        "wall_to_openings": wall_to_openings,
        "opening_to_filler": opening_to_filler,
        "wall_to_fillers": wall_to_fillers,
        "wall_connections": wall_connections,
        "nearest_adjacent": nearest_adjacent,
        "next_to_pairs": next_to_pairs,
    }


def node_title(entity, fallback_guid: str) -> str:
    if entity is None:
        return fallback_guid[:8]
    name = getattr(entity, "Name", None) or getattr(entity, "LongName", None)
    return str(name) if name else entity.is_a()


def short_type(entity_or_row: Any) -> str:
    if hasattr(entity_or_row, "is_a"):
        return entity_or_row.is_a().replace("Ifc", "")
    return str(entity_or_row.get("ifc_class", "Element")).replace("Ifc", "")


def build_base_graph(ctx: dict[str, Any]) -> tuple[list[Node], list[Edge], str]:
    anchor_guid = ctx["anchor_guid"]
    guid_to_entity = ctx["guid_to_entity"]
    index = ctx["index"]
    wall_to_openings = ctx["wall_to_openings"]
    opening_to_filler = ctx["opening_to_filler"]
    wall_connections = ctx["wall_connections"]

    anchor = guid_to_entity[anchor_guid]
    storey_name = get_storey_name(anchor)
    storey_guid = f"storey::{storey_name}"

    nodes: dict[str, Node] = {
        storey_guid: Node(storey_guid, storey_name, "storey", subtitle="IfcBuildingStorey", w=170),
        anchor_guid: Node(
            anchor_guid,
            node_title(anchor, anchor_guid),
            "anchor_wall",
            subtitle=short_type(anchor),
            attrs=[f"GUID: {anchor_guid[:8]}"],
            w=178,
            h=60,
        ),
    }
    edges: list[Edge] = [
        Edge(storey_guid, anchor_guid, "CONTAINS", (0.45, 0.45, 0.45)),
    ]

    conn_guids = list(sorted(wall_connections.get(anchor_guid, set())))[:2]
    opening_guids = list(sorted(wall_to_openings.get(anchor_guid, [])))[:3]

    for guid in conn_guids:
        ent = guid_to_entity.get(guid)
        nodes[guid] = Node(guid, node_title(ent, guid), "wall", subtitle=short_type(ent))
        edges.append(Edge(anchor_guid, guid, "IFC_PATH_CONNECT", (0.31, 0.44, 0.75), dashed=True))
        edges.append(Edge(storey_guid, guid, "CONTAINS", (0.45, 0.45, 0.45)))

    for op_guid in opening_guids:
        op = guid_to_entity.get(op_guid)
        nodes[op_guid] = Node(op_guid, node_title(op, op_guid), "opening", subtitle="Opening", w=132, h=46)
        edges.append(Edge(anchor_guid, op_guid, "VOIDS", (0.75, 0.42, 0.12)))
        filler_guid = opening_to_filler.get(op_guid)
        if filler_guid and filler_guid in guid_to_entity:
            filler = guid_to_entity[filler_guid]
            nodes[filler_guid] = Node(
                filler_guid,
                node_title(filler, filler_guid),
                "filler",
                subtitle=short_type(filler),
                w=132,
                h=46,
            )
            edges.append(Edge(filler_guid, op_guid, "FILLS_OPENING", (0.15, 0.50, 0.78)))
            edges.append(Edge(storey_guid, filler_guid, "CONTAINS", (0.45, 0.45, 0.45)))

    # Layout
    nodes[storey_guid].x, nodes[storey_guid].y = 420, 470
    nodes[anchor_guid].x, nodes[anchor_guid].y = 420, 340

    wall_xs = [200, 640]
    for i, guid in enumerate(conn_guids):
        nodes[guid].x = wall_xs[i % len(wall_xs)]
        nodes[guid].y = 340

    opening_xs = [240, 420, 600]
    for i, guid in enumerate(opening_guids):
        nodes[guid].x = opening_xs[i % len(opening_xs)]
        nodes[guid].y = 210
        filler_guid = opening_to_filler.get(guid)
        if filler_guid and filler_guid in nodes:
            nodes[filler_guid].x = nodes[guid].x
            nodes[filler_guid].y = 105

    subtitle = (
        "Base IFC graph view: IFC-native spatial structure with explicit opening nodes "
        "and schema-level relation chains."
    )
    return list(nodes.values()), edges, subtitle


def build_enriched_graph(ctx: dict[str, Any]) -> tuple[list[Node], list[Edge], str]:
    anchor_guid = ctx["anchor_guid"]
    guid_to_entity = ctx["guid_to_entity"]
    index = ctx["index"]
    wall_connections = ctx["wall_connections"]
    wall_to_fillers = ctx["wall_to_fillers"]
    nearest_adjacent = ctx["nearest_adjacent"]
    next_to_pairs_fn = ctx["next_to_pairs"]

    anchor = guid_to_entity[anchor_guid]
    row = index.get(anchor_guid, {})
    storey_name = row.get("storey_name") or get_storey_name(anchor)
    storey_guid = f"storey::{storey_name}"
    is_cont, cont_span = extract_continuous_attrs(row)
    material = row.get("material") or "n/a"
    fillers = list(sorted(wall_to_fillers.get(anchor_guid, [])))[:4]
    conn_guids = list(sorted(wall_connections.get(anchor_guid, set())))[:3]
    adj_guids = nearest_adjacent(anchor_guid, limit=3)
    next_pairs = next_to_pairs_fn(anchor_guid)

    nodes: dict[str, Node] = {
        storey_guid: Node(storey_guid, storey_name, "storey", subtitle="Canonical storey", w=174),
        anchor_guid: Node(
            anchor_guid,
            node_title(anchor, anchor_guid),
            "anchor_wall",
            subtitle=short_type(anchor),
            attrs=[
                f"material: {material[:28]}",
                f"wall_child_count: {len(fillers)}",
                "is_continuous: true" if is_cont else "is_continuous: false",
                f"span: {cont_span}" if cont_span else "",
            ],
            w=210,
            h=82,
        ),
    }

    edges: list[Edge] = [
        Edge(storey_guid, anchor_guid, "CONTAINS", (0.45, 0.45, 0.45)),
    ]

    for guid in fillers:
        ent = guid_to_entity.get(guid)
        nodes[guid] = Node(
            guid,
            node_title(ent, guid),
            "filler",
            subtitle=short_type(ent),
            attrs=[f"storey: {storey_name}"],
            w=128,
            h=46,
        )
        edges.append(Edge(guid, anchor_guid, "FILLS", (0.12, 0.47, 0.71)))
        edges.append(Edge(storey_guid, guid, "CONTAINS", (0.45, 0.45, 0.45)))

    for a, b in next_pairs:
        if a in nodes and b in nodes:
            edges.append(Edge(a, b, "NEXT_TO", (0.54, 0.17, 0.89)))
            edges.append(Edge(b, a, "NEXT_TO", (0.54, 0.17, 0.89)))

    for guid in conn_guids:
        ent = guid_to_entity.get(guid)
        ent_row = index.get(guid, {})
        ent_cont, _ = extract_continuous_attrs(ent_row) if ent_row else (False, "")
        nodes[guid] = Node(
            guid,
            node_title(ent, guid),
            "wall",
            subtitle=short_type(ent),
            attrs=[
                f"material: {(ent_row.get('material') or 'n/a')[:24]}",
                "is_continuous: true" if ent_cont else "is_continuous: false",
            ],
            w=164,
            h=56,
        )
        edges.append(Edge(anchor_guid, guid, "CONNECTS_TO", (0.00, 0.55, 0.31)))
        edges.append(Edge(storey_guid, guid, "CONTAINS", (0.45, 0.45, 0.45)))

    for guid in adj_guids:
        ent_row = index.get(guid, {})
        label = ent_row.get("name") or guid[:8]
        nodes[guid] = Node(
            guid,
            label,
            "adjacent",
            subtitle=short_type(ent_row),
            attrs=[f"material: {(ent_row.get('material') or 'n/a')[:24]}"],
            w=140,
            h=48,
        )
        edges.append(Edge(anchor_guid, guid, "ADJACENT_TO", (0.82, 0.24, 0.23), dashed=True))

    # Layout
    nodes[storey_guid].x, nodes[storey_guid].y = 430, 480
    nodes[anchor_guid].x, nodes[anchor_guid].y = 430, 310

    filler_positions = [(200, 180), (350, 180), (510, 180), (660, 180)]
    for i, guid in enumerate(fillers):
        if guid in nodes:
            nodes[guid].x, nodes[guid].y = filler_positions[i]

    conn_positions = [(180, 345), (690, 345), (430, 90)]
    for i, guid in enumerate(conn_guids):
        if guid in nodes:
            nodes[guid].x, nodes[guid].y = conn_positions[i]

    adj_positions = [(150, 90), (680, 100), (760, 255)]
    for i, guid in enumerate(adj_guids):
        if guid in nodes:
            nodes[guid].x, nodes[guid].y = adj_positions[i]

    subtitle = (
        "Enriched IFC graph view: retrieval-oriented compression, derived topology, "
        "and normalized properties used by the interpreter pipeline."
    )
    return list(nodes.values()), edges, subtitle


def node_style(kind: str) -> tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    styles = {
        "storey": ((0.90, 0.94, 1.00), (0.35, 0.48, 0.75)),
        "anchor_wall": ((1.00, 0.94, 0.78), (0.72, 0.51, 0.12)),
        "wall": ((0.95, 0.90, 0.82), (0.58, 0.42, 0.19)),
        "opening": ((0.96, 0.96, 0.96), (0.45, 0.45, 0.45)),
        "filler": ((0.86, 0.95, 1.00), (0.12, 0.47, 0.71)),
        "adjacent": ((1.00, 0.89, 0.89), (0.82, 0.24, 0.23)),
    }
    return styles.get(kind, ((0.97, 0.97, 0.97), (0.4, 0.4, 0.4)))


def draw_edge(canvas: PdfCanvas, nodes: dict[str, Node], edge: Edge):
    src = nodes[edge.source]
    dst = nodes[edge.target]
    x1, y1 = src.x, src.y
    x2, y2 = dst.x, dst.y
    dx, dy = x2 - x1, y2 - y1
    dist = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / dist, dy / dist
    start_x = x1 + ux * (src.w * 0.34)
    start_y = y1 + uy * (src.h * 0.34)
    end_x = x2 - ux * (dst.w * 0.34)
    end_y = y2 - uy * (dst.h * 0.34)

    canvas.set_stroke_rgb(*edge.color)
    canvas.set_line_width(1.5)
    canvas.set_dash(5.0, 3.0 if edge.dashed else 0.0)
    canvas.line(start_x, start_y, end_x, end_y)
    canvas.set_dash()

    # Arrow head
    arrow_len = 7.5
    arrow_w = 4.0
    bx = end_x - ux * arrow_len
    by = end_y - uy * arrow_len
    px = -uy
    py = ux
    canvas.set_fill_rgb(*edge.color)
    canvas.parts.append(
        f"{end_x:.2f} {end_y:.2f} m "
        f"{bx + px * arrow_w:.2f} {by + py * arrow_w:.2f} l "
        f"{bx - px * arrow_w:.2f} {by - py * arrow_w:.2f} l f"
    )

    lx = (start_x + end_x) / 2 + px * 8
    ly = (start_y + end_y) / 2 + py * 8
    canvas.set_fill_rgb(0.1, 0.1, 0.1)
    canvas.text(lx - 18, ly - 3, edge.rel, size=8)


def draw_node(canvas: PdfCanvas, node: Node):
    fill, stroke = node_style(node.kind)
    canvas.set_fill_rgb(*fill)
    canvas.set_stroke_rgb(*stroke)
    canvas.set_line_width(1.1)
    canvas.rect(node.x - node.w / 2, node.y - node.h / 2, node.w, node.h, fill=True, stroke=True)
    canvas.set_fill_rgb(0.08, 0.08, 0.08)
    canvas.text(node.x - node.w / 2 + 8, node.y + node.h / 2 - 16, node.label[:34], size=10)
    if node.subtitle:
        canvas.set_fill_rgb(0.28, 0.28, 0.28)
        canvas.text(node.x - node.w / 2 + 8, node.y + node.h / 2 - 30, node.subtitle[:34], size=8)
    if node.attrs:
        base_y = node.y + node.h / 2 - 42
        for i, attr in enumerate([a for a in node.attrs if a][:2]):
            canvas.set_fill_rgb(0.22, 0.22, 0.22)
            canvas.text(node.x - node.w / 2 + 8, base_y - i * 11, attr[:34], size=7)


def render_graph(title: str, subtitle: str, nodes: list[Node], edges: list[Edge], out_path: Path):
    canvas = PdfCanvas(920, 560)
    canvas.set_fill_rgb(0.08, 0.10, 0.14)
    canvas.text(44, 524, title, size=18)
    canvas.set_fill_rgb(0.28, 0.28, 0.28)
    canvas.text(44, 506, subtitle, size=10)

    # Legend
    legend_items = [
        ("CONTAINS", (0.45, 0.45, 0.45)),
        ("VOIDS / FILLS_OPENING", (0.75, 0.42, 0.12)),
        ("FILLS", (0.12, 0.47, 0.71)),
        ("CONNECTS_TO", (0.00, 0.55, 0.31)),
        ("NEXT_TO", (0.54, 0.17, 0.89)),
        ("ADJACENT_TO", (0.82, 0.24, 0.23)),
    ]
    lx = 640
    ly = 522
    for i, (name, color) in enumerate(legend_items):
        x = lx
        y = ly - i * 15
        canvas.set_stroke_rgb(*color)
        canvas.set_line_width(2.0)
        canvas.line(x, y, x + 18, y)
        canvas.set_fill_rgb(0.1, 0.1, 0.1)
        canvas.text(x + 24, y - 4, name, size=8)

    node_map = {n.guid: n for n in nodes}
    for edge in edges:
        if edge.source in node_map and edge.target in node_map:
            draw_edge(canvas, node_map, edge)
    for node in nodes:
        draw_node(canvas, node)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.write_pdf(out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate thesis-ready base and enriched IFC graph figures.")
    parser.add_argument("--ifc", type=Path, default=DEFAULT_IFC, help="Path to IFC model")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to element_index.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for output images")
    parser.add_argument("--anchor-guid", type=str, default="", help="Optional anchor wall GUID")
    args = parser.parse_args()

    ctx = build_graph_data(args.ifc, args.index)
    if args.anchor_guid:
        ctx["anchor_guid"] = args.anchor_guid

    base_nodes, base_edges, base_subtitle = build_base_graph(ctx)
    enriched_nodes, enriched_edges, enriched_subtitle = build_enriched_graph(ctx)

    base_out = args.output_dir / "15_ifc_base_graph.pdf"
    enriched_out = args.output_dir / "16_ifc_enriched_graph.pdf"

    render_graph("Base IFC Graph", base_subtitle, base_nodes, base_edges, base_out)
    render_graph("Enriched IFC Graph", enriched_subtitle, enriched_nodes, enriched_edges, enriched_out)

    print(f"Anchor GUID: {ctx['anchor_guid']}")
    print(f"Wrote: {base_out}")
    print(f"Wrote: {enriched_out}")


if __name__ == "__main__":
    main()
