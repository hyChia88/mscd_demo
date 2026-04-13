#!/usr/bin/env python3
"""AP held-out oracle waterfall — aligned with live retrieval system.

Uses ground-truth constraints (perfect extraction) to run the same Cypher
logic as retrieval_backend.py and shows pool-size reduction at each
enrichment layer.

Waterfall layers:
  L0  Full model                 (no filter)
  L1  P1 storey + ifc_type       (IFC attribute baseline, no topology)
  L2  Topology — type only       (1 edge: pred + obj_type + storey)
  L3  Topology + fingerprint     (+ object_subtype / material / direction /
                                   connection_degree / distance_mm)
  L4  Topology + position_index  (exact wall slot; FILLS / NEXT_TO only)
  L5  Topology + dimensions      (width/height ±50 mm; IfcWindow/Door only)
  L6  Multi-anchor AND           (2+ SRs AND-intersected; mirrors
                                   _execute_multi_anchor)
  L7  p0 ∪ p1                    (L2 ∪ P1; live default strategy)

Edge-type taxonomy:
  IFC-native  : FILLS        (IfcRelFillsElement — IFC schema)
                CONNECTS_TO  (IfcRelConnectsPathElements — IFC schema)
  Author-added: NEXT_TO      (computed: filler projection on wall axis)
                ADJACENT_TO  (computed: centroid distance 100–1500 mm)

Usage:
    cd /root/cmu/master_thesis/mscd_demo
    conda run -n mscd_demo python evaluation/oracle_ap_heldout.py \\
        --cases evaluation/cases/cases_ap_heldout_e2e.jsonl \\
        --out-dir output/lora6_v2_ap_20260331/oracle_live_aligned
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mscd")

import yaml
from py2neo import Graph  # type: ignore

from evaluation.analysis.compare_results import compute_metrics  # type: ignore
from evaluation.track_registry import GROUP_DISPLAY  # type: ignore


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CASES     = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_TRACE_ROOT = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "ap_e2e_phase5_g8"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "oracle_live_aligned"
DEFAULT_GROUPS    = ["g7_position_context", "g8_posctx_dim", "gemini_ap"]

IFC_NATIVE_EDGES  = {"FILLS", "CONNECTS_TO"}
AUTHOR_ADDED_EDGES = {"NEXT_TO", "ADJACENT_TO"}
ALL_EDGE_TYPES    = sorted(IFC_NATIVE_EDGES | AUTHOR_ADDED_EDGES)

DIM_TOL_MM  = 50.0   # ±50 mm tolerance for width/height (L5)
DIST_TOL_MM = 200.0  # ±200 mm tolerance for ADJACENT_TO edge distance (L3)


# ── Neo4j helpers ─────────────────────────────────────────────────────────────

def _connect(cfg_path: Path = PROJECT_ROOT / "config.yaml") -> Graph:
    cfg = yaml.safe_load(cfg_path.read_text())
    n = cfg.get("neo4j", {})
    return Graph(n.get("uri", "bolt://localhost:7687"),
                 auth=(n.get("user", "neo4j"), n.get("password", "password")))


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _guids(result) -> set:
    return {r["guid"] for r in result.data()}


def _fmt_num(v: Optional[float], pct: bool = False) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    if isinstance(v, float) and v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def _pool_stats(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 1),
        "median": round(statistics.median(vals), 1),
        "min": min(vals),
        "max": max(vals),
    }


# ── Layer queries (mirror live Cypher exactly) ────────────────────────────────

def _q_total(g: Graph, model: str) -> int:
    return int(g.run(
        "MATCH (n:IFCElement) WHERE n.ifc_model=$m RETURN count(n) AS cnt",
        m=model).evaluate())


def _q_l1_p1(g: Graph, ifc_type: str, storey: str, model: str) -> set:
    """P1 — storey + ifc_type (IFC attribute baseline, no topology)."""
    return _guids(g.run("""
        MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
        WHERE e.ifc_model = $m
          AND (e.ifc_type = $t OR e.ifc_type STARTS WITH $t)
          AND toLower(s.name) CONTAINS toLower($s)
        RETURN DISTINCT e.guid AS guid
    """, m=model, t=ifc_type, s=storey))


def _q_l2_type_only(g: Graph, ifc_type: str, storey: str, model: str,
                    pred: str, obj_type: str) -> set:
    """L2 — single-hop topology, type + storey only.
    Mirrors live single-hop Cypher at its minimum (no fingerprint).
    """
    cypher = f"""
        MATCH (target:IFCElement)-[:{pred}]->(ref:IFCElement)
        WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
          AND (ref.ifc_type = $ot OR ref.ifc_type STARTS WITH $ot)
          AND target.ifc_model = $m
          AND toLower(target.storey) CONTAINS toLower($s)
        RETURN DISTINCT target.guid AS guid
    """
    return _guids(g.run(cypher, t=ifc_type, ot=obj_type, m=model, s=storey))


def _q_l3_fingerprint(g: Graph, ifc_type: str, storey: str, model: str,
                      pred: str, obj_type: str,
                      obj_subtype: str = "", obj_material: str = "",
                      direction: Optional[str] = None,
                      connection_degree: Optional[int] = None,
                      distance_mm: Optional[float] = None) -> set:
    """L3 — topology + fingerprint signals.
    Adds object_subtype CONTAINS, material CONTAINS, direction comparison,
    COUNT{CONNECTS_TO} = degree, and ADJACENT_TO edge distance ±200mm.
    """
    params: Dict[str, Any] = {
        "t": ifc_type, "ot": obj_type, "m": model, "s": storey,
        "sub": (obj_subtype or "").lower(),
        "mat": (obj_material or "").lower(),
    }
    extra = ""
    if obj_subtype:
        extra += ("\n  AND (toLower(coalesce(ref.object_type,'')) CONTAINS $sub"
                  " OR toLower(coalesce(ref.name,'')) CONTAINS $sub)")
    if obj_material:
        extra += "\n  AND ($mat = '' OR toLower(coalesce(ref.material,'')) CONTAINS $mat)"
    if pred == "NEXT_TO":
        if direction == "left":
            extra += ("\n  AND coalesce(ref.wall_position_index,999999) < "
                      "coalesce(target.wall_position_index,-1)")
        elif direction == "right":
            extra += ("\n  AND coalesce(ref.wall_position_index,-1) > "
                      "coalesce(target.wall_position_index,999999)")
    if pred == "CONNECTS_TO" and connection_degree is not None:
        params["deg"] = connection_degree
        extra += "\n  AND COUNT { (target)-[:CONNECTS_TO]-() } = $deg"

    if pred == "ADJACENT_TO" and distance_mm is not None:
        params["dist"] = distance_mm
        params["dist_tol"] = DIST_TOL_MM
        cypher = f"""
            MATCH (target:IFCElement)-[r_adj:ADJACENT_TO]->(ref:IFCElement)
            WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
              AND (ref.ifc_type = $ot OR ref.ifc_type STARTS WITH $ot)
              AND target.ifc_model = $m
              AND toLower(target.storey) CONTAINS toLower($s)
              AND abs(coalesce(r_adj.distance_mm,-9999) - $dist) <= $dist_tol
              {extra}
            RETURN DISTINCT target.guid AS guid
        """
        return _guids(g.run(cypher, **params))

    cypher = f"""
        MATCH (target:IFCElement)-[:{pred}]->(ref:IFCElement)
        WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
          AND (ref.ifc_type = $ot OR ref.ifc_type STARTS WITH $ot)
          AND target.ifc_model = $m
          AND toLower(target.storey) CONTAINS toLower($s)
          {extra}
        RETURN DISTINCT target.guid AS guid
    """
    return _guids(g.run(cypher, **params))


def _q_l4_position(g: Graph, ifc_type: str, storey: str, model: str,
                   pred: str, obj_type: str,
                   position_index: int,
                   position_total: Optional[int] = None) -> set:
    """L4 — topology + exact wall slot (position_index, position_total).
    Mirrors live exact_slot fingerprint for FILLS / NEXT_TO.
    """
    params: Dict[str, Any] = {
        "t": ifc_type, "ot": obj_type, "m": model, "s": storey,
        "pos": max(position_index - 1, 0),
    }
    pos_clause = "\n  AND coalesce(target.wall_position_index,-1) = $pos"
    if position_total is not None:
        params["tot"] = position_total
        pos_clause += "\n  AND coalesce(target.wall_child_total,-1) = $tot"
    cypher = f"""
        MATCH (target:IFCElement)-[:{pred}]->(ref:IFCElement)
        WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
          AND (ref.ifc_type = $ot OR ref.ifc_type STARTS WITH $ot)
          AND target.ifc_model = $m
          AND toLower(target.storey) CONTAINS toLower($s)
          {pos_clause}
        RETURN DISTINCT target.guid AS guid
    """
    return _guids(g.run(cypher, **params))


def _q_l5_dimensions(g: Graph, ifc_type: str, storey: str, model: str,
                     pred: str, obj_type: str,
                     width_mm: Optional[float],
                     height_mm: Optional[float]) -> Optional[set]:
    """L5 — topology + physical dimensions ±50 mm (IfcWindow/Door).
    Author-contributed: dimensions stored on Neo4j nodes from IFC psets.
    Returns None if neither width nor height is available.
    """
    if width_mm is None and height_mm is None:
        return None
    params: Dict[str, Any] = {"t": ifc_type, "ot": obj_type, "m": model, "s": storey}
    dim_clause = ""
    if width_mm is not None:
        params["w"] = width_mm
        dim_clause += f"\n  AND abs(coalesce(target.width_mm,-9999) - $w) <= {DIM_TOL_MM}"
    if height_mm is not None:
        params["h"] = height_mm
        dim_clause += f"\n  AND abs(coalesce(target.height_mm,-9999) - $h) <= {DIM_TOL_MM}"
    cypher = f"""
        MATCH (target:IFCElement)-[:{pred}]->(ref:IFCElement)
        WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
          AND (ref.ifc_type = $ot OR ref.ifc_type STARTS WITH $ot)
          AND target.ifc_model = $m
          AND toLower(target.storey) CONTAINS toLower($s)
          {dim_clause}
        RETURN DISTINCT target.guid AS guid
    """
    return _guids(g.run(cypher, **params))


def _q_l6_multi_anchor(g: Graph, ifc_type: str, storey: str, model: str,
                       spatial_relations: List[dict]) -> Optional[set]:
    """L6 — multi-anchor AND intersection.
    Mirrors _execute_multi_anchor(): all SRs must hold simultaneously
    from the same target (star pattern, not chain).
    """
    if len(spatial_relations) < 2:
        return None
    params: Dict[str, Any] = {"t": ifc_type, "m": model, "s": storey}
    exists_clauses = []
    for i, sr in enumerate(spatial_relations):
        pred = sr.get("predicate", "")
        ot_key = f"ot{i}"
        params[ot_key] = sr.get("object_type", "")
        exists_clauses.append(
            f"EXISTS {{ (target)-[:{pred}]->(r{i}:IFCElement) "
            f"WHERE (r{i}.ifc_type = ${ot_key} OR r{i}.ifc_type STARTS WITH ${ot_key}) "
            f"AND r{i}.ifc_model = $m }}"
        )
    cypher = f"""
        MATCH (target:IFCElement)
        WHERE (target.ifc_type = $t OR target.ifc_type STARTS WITH $t)
          AND target.ifc_model = $m
          AND toLower(target.storey) CONTAINS toLower($s)
          AND {" AND ".join(exists_clauses)}
        RETURN DISTINCT target.guid AS guid
    """
    return _guids(g.run(cypher, **params))


# ── GT introspection ──────────────────────────────────────────────────────────

def _gt_node(g: Graph, guid: str, model: str) -> Optional[dict]:
    rows = g.run("""
        MATCH (e:IFCElement)
        WHERE e.guid = $guid AND e.ifc_model = $model
        RETURN e.guid AS guid, e.ifc_type AS ifc_type, e.storey AS storey,
               e.name AS name,
               e.wall_position_index AS pos, e.wall_child_total AS total,
               e.width_mm AS width_mm, e.height_mm AS height_mm
    """, guid=guid, model=model).data()
    return rows[0] if rows else None


def _discover_gt_edges(g: Graph, guid: str, model: str) -> Dict[str, List[dict]]:
    """Return all outgoing topology edges from GT element."""
    result: Dict[str, List[dict]] = {}
    for et in ALL_EDGE_TYPES:
        if et == "ADJACENT_TO":
            rows = g.run(f"""
                MATCH (src:IFCElement)-[r:{et}]->(dst:IFCElement)
                WHERE src.guid = $guid AND src.ifc_model = $model
                RETURN dst.guid AS guid, dst.ifc_type AS ifc_type,
                       dst.name AS name, dst.material AS material,
                       dst.object_type AS object_type,
                       r.distance_mm AS distance_mm
            """, guid=guid, model=model).data()
        else:
            rows = g.run(f"""
                MATCH (src:IFCElement)-[:{et}]->(dst:IFCElement)
                WHERE src.guid = $guid AND src.ifc_model = $model
                RETURN dst.guid AS guid, dst.ifc_type AS ifc_type,
                       dst.name AS name, dst.material AS material,
                       dst.object_type AS object_type
            """, guid=guid, model=model).data()
        if rows:
            result[et] = rows
    return result


# ── Per-case oracle ───────────────────────────────────────────────────────────

def _is_better(best_pool: Optional[int], candidate_pool: int,
               candidate_gt_in: bool) -> bool:
    if not candidate_gt_in:
        return False
    return best_pool is None or candidate_pool < best_pool


def run_oracle(cases: List[dict]) -> List[dict]:
    g = _connect()
    model_key = "AP"
    l0_total = _q_total(g, model_key)

    results: List[dict] = []
    for case in cases:
        case_id = case["case_id"]
        gt = case.get("ground_truth", {})
        gt_guid = gt.get("target_guid", "")

        node = _gt_node(g, gt_guid, model_key)
        if not node:
            results.append({"case_id": case_id, "error": "gt_not_in_neo4j"})
            continue

        gt_type   = node["ifc_type"]
        gt_storey = node["storey"] or ""
        gt_pos    = node.get("pos")
        gt_total  = node.get("total")
        gt_w      = node.get("width_mm")
        gt_h      = node.get("height_mm")

        # L1 — P1 (storey + type)
        l1 = _q_l1_p1(g, gt_type, gt_storey, model_key)
        l1_pool = len(l1)
        l1_gt_in = gt_guid in l1

        # Discover GT's outgoing edges
        gt_edges = _discover_gt_edges(g, gt_guid, model_key)

        # Per edge-type results at L2–L5
        edge_results: Dict[str, dict] = {}
        best_l2_pool: Optional[int] = None
        best_l2_et: Optional[str] = None
        best_l2_guids: Optional[set] = None
        best_l3_pool: Optional[int] = None
        best_l4_pool: Optional[int] = None
        best_l5_pool: Optional[int] = None

        for et, neighbors in gt_edges.items():
            nb        = neighbors[0]
            obj_type  = nb["ifc_type"]
            obj_mat   = (nb.get("material") or "").lower()
            obj_sub   = (nb.get("object_type") or "").lower()
            dist      = nb.get("distance_mm")

            # L2 — type only
            l2g = _q_l2_type_only(g, gt_type, gt_storey, model_key, et, obj_type)
            l2_in = gt_guid in l2g

            # L3 — + fingerprint
            l3g = _q_l3_fingerprint(
                g, gt_type, gt_storey, model_key, et, obj_type,
                obj_subtype=obj_sub, obj_material=obj_mat,
                distance_mm=dist if et == "ADJACENT_TO" else None,
            )
            l3_in = gt_guid in l3g

            # L4 — + exact slot (FILLS / NEXT_TO only)
            l4_pool_et = None
            l4_in_et   = False
            if et in ("FILLS", "NEXT_TO") and gt_pos is not None:
                l4g = _q_l4_position(
                    g, gt_type, gt_storey, model_key, et, obj_type,
                    position_index=int(gt_pos) + 1,
                    position_total=int(gt_total) if gt_total is not None else None,
                )
                l4_pool_et = len(l4g)
                l4_in_et   = gt_guid in l4g

            # L5 — + dimensions (IfcWindow/Door only)
            l5_pool_et = None
            l5_in_et   = False
            if gt_type in ("IfcWindow", "IfcDoor"):
                l5g = _q_l5_dimensions(
                    g, gt_type, gt_storey, model_key, et, obj_type, gt_w, gt_h
                )
                if l5g is not None:
                    l5_pool_et = len(l5g)
                    l5_in_et   = gt_guid in l5g

            edge_results[et] = {
                "obj_type":    obj_type,
                "edge_origin": "ifc_native" if et in IFC_NATIVE_EDGES else "author_added",
                "l2_pool":     len(l2g),  "l2_gt_in": l2_in,
                "l3_pool":     len(l3g),  "l3_gt_in": l3_in,
                "l4_pool":     l4_pool_et, "l4_gt_in": l4_in_et,
                "l5_pool":     l5_pool_et, "l5_gt_in": l5_in_et,
            }

            if _is_better(best_l2_pool, len(l2g), l2_in):
                best_l2_pool  = len(l2g)
                best_l2_et    = et
                best_l2_guids = l2g
            if _is_better(best_l3_pool, len(l3g), l3_in):
                best_l3_pool = len(l3g)
            if l4_in_et and _is_better(best_l4_pool, l4_pool_et, True):
                best_l4_pool = l4_pool_et
            if l5_in_et and _is_better(best_l5_pool, l5_pool_et, True):
                best_l5_pool = l5_pool_et

        # L6 — multi-anchor AND (uses GT labels' spatial_relations)
        gt_srs = case.get("labels", {}).get("constraints", {}).get("spatial_relations", [])
        l6_pool = None
        l6_gt_in = False
        if len(gt_srs) >= 2:
            l6g = _q_l6_multi_anchor(g, gt_type, gt_storey, model_key, gt_srs)
            if l6g is not None:
                l6_pool  = len(l6g)
                l6_gt_in = gt_guid in l6g

        # L7 — p0 ∪ p1 (live default: best L2 ∪ P1, P0 elements ranked first)
        l7_pool = None
        l7_gt_in = False
        if best_l2_guids is not None:
            union    = best_l2_guids | l1
            l7_pool  = len(union)
            l7_gt_in = gt_guid in union

        results.append({
            "case_id":          case_id,
            "gt_guid":          gt_guid,
            "gt_type":          gt_type,
            "gt_storey":        gt_storey,
            "has_dims":         gt_w is not None or gt_h is not None,
            "n_gt_srs":         len(gt_srs),
            "l0_pool":          l0_total,
            "l1_pool":          l1_pool,
            "l1_gt_in":         l1_gt_in,
            "edge_types_found": list(gt_edges.keys()),
            "edge_types_native": [et for et in gt_edges if et in IFC_NATIVE_EDGES],
            "edge_types_author": [et for et in gt_edges if et in AUTHOR_ADDED_EDGES],
            "edge_results":     edge_results,
            "best_l2_edge":     best_l2_et,
            "best_l2_pool":     best_l2_pool,
            "best_l3_pool":     best_l3_pool,
            "best_l4_pool":     best_l4_pool,
            "best_l5_pool":     best_l5_pool,
            "l6_pool":          l6_pool,
            "l6_gt_in":         l6_gt_in,
            "l7_pool":          l7_pool,
            "l7_gt_in":         l7_gt_in,
        })

    return results


# ── Summary ───────────────────────────────────────────────────────────────────

def build_summary(records: List[dict]) -> dict:
    valid = [r for r in records if "error" not in r]

    def _sizes(key: str, gt_key: Optional[str] = None) -> List[float]:
        out = []
        for r in valid:
            v = r.get(key)
            if v is None:
                continue
            if gt_key and not r.get(gt_key):
                continue
            out.append(float(v))
        return out

    # per-predicate breakdown
    by_pred_l2: Dict[str, List[float]] = defaultdict(list)
    by_pred_l3: Dict[str, List[float]] = defaultdict(list)
    native_l2:  List[float] = []
    author_l2:  List[float] = []

    for r in valid:
        for et, er in r.get("edge_results", {}).items():
            if er.get("l2_gt_in"):
                by_pred_l2[et].append(er["l2_pool"])
                (native_l2 if et in IFC_NATIVE_EDGES else author_l2).append(er["l2_pool"])
            if er.get("l3_gt_in"):
                by_pred_l3[et].append(er["l3_pool"])

    return {
        "n":                    len(valid),
        "n_with_any_edge":      sum(1 for r in valid if r.get("edge_types_found")),
        "n_with_native_edge":   sum(1 for r in valid if r.get("edge_types_native")),
        "n_with_author_edge":   sum(1 for r in valid if r.get("edge_types_author")),
        "n_with_dims":          sum(1 for r in valid if r.get("has_dims")),
        "n_with_multi_sr":      sum(1 for r in valid if r.get("n_gt_srs", 0) >= 2),

        # Waterfall pool stats (only cases where GT is in pool)
        "L0":  _pool_stats(_sizes("l0_pool")),
        "L1":  _pool_stats(_sizes("l1_pool", "l1_gt_in")),
        "L2":  _pool_stats([r["best_l2_pool"] for r in valid if r.get("best_l2_pool") is not None]),
        "L3":  _pool_stats([r["best_l3_pool"] for r in valid if r.get("best_l3_pool") is not None]),
        "L4":  _pool_stats([r["best_l4_pool"] for r in valid if r.get("best_l4_pool") is not None]),
        "L5":  _pool_stats([r["best_l5_pool"] for r in valid if r.get("best_l5_pool") is not None]),
        "L6":  _pool_stats([r["l6_pool"]  for r in valid if r.get("l6_gt_in")]),
        "L7":  _pool_stats([r["l7_pool"]  for r in valid if r.get("l7_gt_in")]),

        # GT-in-pool rates
        "l1_gt_in_rate": sum(1 for r in valid if r.get("l1_gt_in")) / len(valid) if valid else 0,
        "l2_gt_in_rate": sum(1 for r in valid if r.get("best_l2_pool") is not None) / len(valid) if valid else 0,

        # Per-predicate breakdown
        "by_predicate_L2": {et: _pool_stats(v) for et, v in by_pred_l2.items()},
        "by_predicate_L3": {et: _pool_stats(v) for et, v in by_pred_l3.items()},
        "native_edges_L2":      _pool_stats(native_l2),
        "author_added_edges_L2": _pool_stats(author_l2),
    }


# ── Comparison to actual model traces ────────────────────────────────────────

def _find_latest_trace(root: Path, group: str) -> Optional[Path]:
    d = root / group
    if not d.exists():
        return None
    traces = sorted(d.glob("traces_*.jsonl"))
    return traces[-1] if traces else None


def _actual_metrics(trace_path: Path) -> dict:
    traces = _load_jsonl(trace_path)
    overall = compute_metrics(traces)
    finals   = [t.get("final_pool_size")   for t in traces if t.get("final_pool_size")   is not None]
    initials = [t.get("initial_pool_size") for t in traces if t.get("initial_pool_size") is not None]
    reds = [1 - f / i for i, f in zip(initials, finals) if i > 0]
    mf = statistics.median(finals) if finals else None
    return {
        "overall":   overall,
        "med_final": mf,
        "avg_red":   round(sum(reds) / len(reds), 4) if reds else None,
    }


def compare_to_actual(records: List[dict], summary: dict,
                      trace_root: Path, groups: List[str]) -> List[dict]:
    oracle_l2_med = summary["L2"]["median"]
    oracle_l7_med = summary["L7"]["median"]
    rows = []
    for group in groups:
        tp = _find_latest_trace(trace_root, group)
        if tp is None:
            continue
        am = _actual_metrics(tp)
        med = am["med_final"]
        rows.append({
            "group":          group,
            "display_name":   GROUP_DISPLAY.get(group, group),
            "gt_in_pct":      am["overall"].get("gt_in_pct"),
            "top10_pct":      am["overall"].get("top10_pct"),
            "top1_pct":       am["overall"].get("top1_pct"),
            "mrr":            am["overall"].get("mrr"),
            "median_final_pool": med,
            "avg_reduction":  am["avg_red"],
            "gap_vs_L2":      (med - oracle_l2_med) if med is not None and oracle_l2_med else None,
            "gap_vs_L7":      (med - oracle_l7_med) if med is not None and oracle_l7_med else None,
        })
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

ORIGIN_NOTE = {
    "FILLS":       "IFC-native (IfcRelFillsElement)",
    "CONNECTS_TO": "IFC-native (IfcRelConnectsPathElements)",
    "NEXT_TO":     "Author-added (filler projection on wall axis)",
    "ADJACENT_TO": "Author-added (centroid distance 100–1500 mm)",
}


def write_outputs(out_dir: Path, records: List[dict],
                  summary: dict, comparisons: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "oracle_ap_heldout_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "oracle_ap_heldout_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "oracle_ap_heldout_vs_models.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False), encoding="utf-8")

    S = summary
    lines = [
        "# AP Held-out Oracle — Live System Alignment",
        "",
        "Pool sizes use **ground-truth constraints** (perfect extraction).",
        "Layers mirror `retrieval_backend.py` Cypher exactly.",
        "",
        "## Coverage",
        f"- Total cases: {S['n']}",
        f"- Cases with any topology edge: {S['n_with_any_edge']}",
        f"  - IFC-native (FILLS / CONNECTS_TO): {S['n_with_native_edge']}",
        f"  - Author-added (NEXT_TO / ADJACENT_TO): {S['n_with_author_edge']}",
        f"- Cases with IfcWindow/Door dimensions: {S['n_with_dims']}",
        f"- Cases with 2+ spatial relations (multi-anchor): {S['n_with_multi_sr']}",
        "",
        "## Waterfall — Median Pool Size",
        "(only cases where GT is in pool at that layer)",
        "",
        "| Layer | What it adds | Median | n |",
        "| --- | --- | ---: | ---: |",
        f"| L0 Full model      | no filter                                  | {_fmt_num(S['L0']['median'])} | {S['L0']['n']} |",
        f"| L1 P1 storey+type  | IFC attribute baseline — no topology        | {_fmt_num(S['L1']['median'])} | {S['L1']['n']} |",
        f"| L2 Topology (type) | 1 edge pred+obj_type+storey                | {_fmt_num(S['L2']['median'])} | {S['L2']['n']} |",
        f"| L3 + Fingerprint   | subtype/material/direction/degree/dist      | {_fmt_num(S['L3']['median'])} | {S['L3']['n']} |",
        f"| L4 + Position      | exact wall slot (FILLS/NEXT_TO)             | {_fmt_num(S['L4']['median'])} | {S['L4']['n']} |",
        f"| L5 + Dimensions    | width/height ±50 mm (Window/Door)           | {_fmt_num(S['L5']['median'])} | {S['L5']['n']} |",
        f"| L6 Multi-anchor    | 2+ SRs AND-intersected (star, not chain)    | {_fmt_num(S['L6']['median'])} | {S['L6']['n']} |",
        f"| L7 p0 ∪ p1         | L2 topology ∪ P1 (live default strategy)    | {_fmt_num(S['L7']['median'])} | {S['L7']['n']} |",
        "",
        "## L2 Pool by Predicate and Edge Origin",
        "",
        "| Predicate | Edge origin | L2 median | L3 median (+ fingerprint) | n |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for et in ["FILLS", "CONNECTS_TO", "NEXT_TO", "ADJACENT_TO"]:
        s2 = S["by_predicate_L2"].get(et, {})
        s3 = S["by_predicate_L3"].get(et, {})
        if s2.get("n", 0) == 0:
            continue
        lines.append(
            f"| {et} | {ORIGIN_NOTE[et]} "
            f"| {_fmt_num(s2.get('median'))} "
            f"| {_fmt_num(s3.get('median'))} "
            f"| {s2.get('n', 0)} |"
        )

    lines += [
        "",
        f"- IFC-native edges L2 median pool: {_fmt_num(S['native_edges_L2']['median'])} "
        f"(n={S['native_edges_L2']['n']})",
        f"- Author-added edges L2 median pool: {_fmt_num(S['author_added_edges_L2']['median'])} "
        f"(n={S['author_added_edges_L2']['n']})",
    ]

    if comparisons:
        lines += [
            "",
            "## Oracle vs Track B-2 Models",
            "(Gap = model median pool − oracle median pool; lower = closer to oracle)",
            "",
            "| Group | GT-in-Pool | Top-10 | Top-1 | MRR@10 | Med Pool | Reduction | Gap vs L2 | Gap vs L7 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in comparisons:
            lines.append(
                f"| {row['display_name']} | "
                f"{_fmt_num(row['gt_in_pct'])}% | "
                f"{_fmt_num(row['top10_pct'])}% | "
                f"{_fmt_num(row['top1_pct'])}% | "
                f"{row['mrr']:.4f} | "
                f"{_fmt_num(row['median_final_pool'])} | "
                f"{_fmt_num(row['avg_reduction'], pct=True)} | "
                f"{_fmt_num(row['gap_vs_L2'])} | "
                f"{_fmt_num(row['gap_vs_L7'])} |"
            )

    report = "\n".join(lines) + "\n"
    (out_dir / "oracle_ap_heldout_report.md").write_text(report, encoding="utf-8")
    print(report)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",      type=Path, default=DEFAULT_CASES)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--groups",     nargs="*", default=DEFAULT_GROUPS)
    args = parser.parse_args()

    cases = _load_jsonl(args.cases)
    print(f"Loaded {len(cases)} cases")

    records     = run_oracle(cases)
    summary     = build_summary(records)
    comparisons = compare_to_actual(records, summary, args.trace_root, args.groups)
    write_outputs(args.out_dir, records, summary, comparisons)
    print(f"\nWrote oracle outputs to {args.out_dir}")

    S = summary
    print(
        f"Waterfall medians: "
        f"L0={_fmt_num(S['L0']['median'])}  "
        f"L1={_fmt_num(S['L1']['median'])}  "
        f"L2={_fmt_num(S['L2']['median'])}  "
        f"L3={_fmt_num(S['L3']['median'])}  "
        f"L4={_fmt_num(S['L4']['median'])}  "
        f"L5={_fmt_num(S['L5']['median'])}  "
        f"L6={_fmt_num(S['L6']['median'])}  "
        f"L7={_fmt_num(S['L7']['median'])}"
    )


if __name__ == "__main__":
    main()
