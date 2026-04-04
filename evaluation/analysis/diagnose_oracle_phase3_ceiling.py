#!/usr/bin/env python3
"""LoRA6 Group 4 oracle ceiling + unique topology diagnosis."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analysis.compare_results import compute_metrics, load_traces  # type: ignore
from evaluation.analysis.group4_common import (
    AP_IFC_PATH,
    CASES_PATH,
    DEFAULT_DATE_TAG,
    ELEMENT_INDEX_PATH,
    EXPERIMENT_ROOT,
    GT_EVAL_PATH,
    ORACLE_PHASE3_DIR,
    WALL_REGION_INDEX_PATH,
    build_next_to_context,
    ensure_dir,
    label_signature,
    load_cases_map,
    load_element_index,
    load_gt_eval_labels,
    load_jsonl,
    markdown_table,
    normalize_storey,
    ordered_unique,
    subtype_keyword_from_row,
    topology_family,
    type_matches,
    universe_key,
    write_csv,
    write_json,
)


LEVELS = [
    "L0_p1_only",
    "L1_pred_obj",
    "L2_pred_obj_dir",
    "L3_pred_obj_dir_sub",
    "L4_full_fingerprint",
]
LEVEL_DISPLAY = {
    "L0_p1_only": "L0\nstorey+type",
    "L1_pred_obj": "L1\n+pred+obj",
    "L2_pred_obj_dir": "L2\n+direction",
    "L3_pred_obj_dir_sub": "L3\n+subtype",
    "L4_full_fingerprint": "L4\n+slot",
}
LEVEL_ORDER = {name: idx for idx, name in enumerate(LEVELS)}
POSITION_SENSITIVE_FAMILIES = {
    "singleton:FILLS",
    "paired:FILLS+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO",
    "triad:FILLS+NEXT_TO+NEXT_TO(mixed-anchor)",
}
ROOT_CAUSES = {
    "ground_truth_not_collected",
    "query_not_using_available_info",
    "fallback_pool_inflation",
    "ranking_proxy_insufficient",
    "true_graph_ambiguity",
    "none_top1_success",
}
RAW_TRACE_JSONL = ORACLE_PHASE3_DIR / "traces_20260401_191910_v2_lora_p0_union_p1.jsonl"
SUMMARY_CSV = ORACLE_PHASE3_DIR / "summary_20260401_191910_v2_lora_p0_union_p1.csv"
DEFAULT_OUT_DIR = (
    EXPERIMENT_ROOT / "group4_post-hoc_analysis" / "oracle_ceiling" / DEFAULT_DATE_TAG
)
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TRACE_DIR_HINT = ORACLE_PHASE3_DIR / "traces"


@dataclass
class CandidateOption:
    predicate: str
    object_guid: str
    object_type: str
    direction: str = ""
    object_subtype: str = ""
    host_guid: str = ""
    wall_position_index: Optional[int] = None
    wall_child_total: Optional[int] = None


def _safe_int(value: Any) -> Optional[int]:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_summary_csv(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    rows = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, float] = {}
    for idx, line in enumerate(rows):
        if line.strip() != "=== OVERALL METRICS ===":
            continue
        for metric_line in rows[idx + 2 :]:
            if not metric_line.strip():
                break
            metric, value = metric_line.split(",", 1)
            key = metric.strip()
            raw = value.strip()
            if raw.endswith("%"):
                out[key] = float(raw.rstrip("%")) / 100.0
            elif raw.startswith("60/60"):
                continue
            else:
                try:
                    out[key] = float(raw)
                except ValueError:
                    continue
        break
    return out


def _root_storey(case: dict, gt_label: dict) -> str:
    return normalize_storey(
        gt_label.get("storey_name")
        or case.get("labels", {}).get("constraints", {}).get("storey_name")
        or case.get("ground_truth", {}).get("target_storey")
    )


def _primary_predicate(case: dict, gt_label: dict) -> str:
    return (
        case.get("difficulty_tags", {}).get("spatial_predicate")
        or (
            (gt_label.get("spatial_relations") or [{}])[0].get("predicate")
            if gt_label.get("spatial_relations")
            else "NONE"
        )
        or "NONE"
    )


def _load_trace_rows(path: Path) -> Dict[str, dict]:
    rows = load_traces(str(path))
    return {row.get("scenario_id", row.get("scenario", {}).get("id", "")): row for row in rows}


def _load_wall_region_index(path: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = defaultdict(list)
    if not path.exists():
        return out
    for row in load_jsonl(path):
        owner = row.get("owner_guid")
        if owner:
            out[owner].append(row)
    return out


def _connect_neo4j(config_path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from py2neo import Graph
    except Exception as exc:
        return None, f"py2neo unavailable: {exc}"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        neo4j_cfg = config.get("neo4j", {})
        graph = Graph(
            neo4j_cfg.get("uri", "bolt://localhost:7687"),
            auth=(neo4j_cfg.get("user", "neo4j"), neo4j_cfg.get("password", "password")),
        )
        graph.run("RETURN 1").data()
        return graph, None
    except Exception as exc:
        return None, str(exc)


def _infer_direction(source_node: dict, target_node: dict) -> str:
    src_pos = _safe_int(source_node.get("wall_position_index"))
    tgt_pos = _safe_int(target_node.get("wall_position_index"))
    if src_pos is None or tgt_pos is None:
        return ""
    if tgt_pos < src_pos:
        return "left"
    if tgt_pos > src_pos:
        return "right"
    return ""


def _relation_matches(option: CandidateOption, requirement: dict, level: str) -> bool:
    if option.predicate != str(requirement.get("predicate") or ""):
        return False
    if not type_matches(option.object_type, requirement.get("object_type")):
        return False
    if level in {"L2_pred_obj_dir", "L3_pred_obj_dir_sub"}:
        expected_dir = str(requirement.get("direction") or "")
        if expected_dir and option.direction != expected_dir:
            return False
    if level == "L3_pred_obj_dir_sub":
        expected_sub = str(requirement.get("object_subtype") or "").strip().lower()
        if expected_sub and option.object_subtype != expected_sub:
            return False
    return True


def _multiset_match(options: List[CandidateOption], requirements: List[dict], level: str) -> bool:
    if not requirements:
        return True
    used = [False] * len(options)
    for req in requirements:
        matched = False
        for idx, option in enumerate(options):
            if used[idx]:
                continue
            if _relation_matches(option, req, level):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _unique_ranked_guids(trace: dict) -> List[str]:
    ranked = [
        row.get("guid", "")
        for row in trace.get("interpreter_output", {}).get("candidates", [])
        if row.get("guid")
    ]
    if not ranked:
        rr = trace.get("internals", {}).get("retrieval_results", [])
        if rr:
            ranked = [row.get("guid", "") for row in rr[0].get("candidates", []) if row.get("guid")]
    return ordered_unique(ranked)


class GraphSnapshot:
    def __init__(self, element_index: Dict[str, dict], next_to_context: Dict[str, dict]):
        self.element_index = element_index
        self.next_to_context = next_to_context
        self.nodes: Dict[str, dict] = {}
        self.edges: Dict[str, Dict[str, List[dict]]] = {
            "FILLS": defaultdict(list),
            "NEXT_TO": defaultdict(list),
            "CONNECTS_TO": defaultdict(list),
            "ADJACENT_TO": defaultdict(list),
        }
        self.source = "local_partial"

    def _ensure_node(self, guid: str, ifc_type: Optional[str] = None, storey: Optional[str] = None) -> dict:
        node = self.nodes.setdefault(guid, {})
        row = self.element_index.get(guid, {})
        if "guid" not in node:
            node.update(
                {
                    "guid": guid,
                    "ifc_type": ifc_type or row.get("ifc_class") or "",
                    "storey": storey or row.get("storey_name") or "",
                    "name": row.get("name") or "",
                    "material": row.get("material") or "",
                    "wall_position_index": None,
                    "wall_child_total": None,
                    "target_name_keyword": row.get("target_name_keyword"),
                    "object_type": row.get("object_type"),
                }
            )
        if ifc_type and not node.get("ifc_type"):
            node["ifc_type"] = ifc_type
        if storey and not node.get("storey"):
            node["storey"] = storey
        if guid in self.next_to_context:
            ctx = self.next_to_context[guid]
            node["host_guid"] = ctx.get("host_guid") or node.get("host_guid")
            node["wall_position_index"] = (
                node.get("wall_position_index") or ctx.get("position_index")
            )
            node["wall_child_total"] = node.get("wall_child_total") or ctx.get("position_total")
        return node

    def load_from_neo4j(self, graph: Any) -> None:
        self.source = "neo4j"
        nodes_query = """
        MATCH (e:IFCElement)
        WHERE e.ifc_model = 'AP'
        RETURN e.guid AS guid,
               e.ifc_type AS ifc_type,
               e.storey AS storey,
               e.name AS name,
               e.material AS material,
               e.wall_position_index AS wall_position_index,
               e.wall_child_total AS wall_child_total
        """
        for row in graph.run(nodes_query).data():
            guid = row["guid"]
            node = self._ensure_node(guid, row.get("ifc_type"), row.get("storey"))
            node["name"] = row.get("name") or node.get("name") or ""
            node["material"] = row.get("material") or node.get("material") or ""
            node["wall_position_index"] = (
                row.get("wall_position_index")
                if row.get("wall_position_index") is not None
                else node.get("wall_position_index")
            )
            node["wall_child_total"] = (
                row.get("wall_child_total")
                if row.get("wall_child_total") is not None
                else node.get("wall_child_total")
            )

        edge_specs = {
            "FILLS": """
                MATCH (a:IFCElement)-[:FILLS]->(b:IFCElement)
                WHERE a.ifc_model = 'AP' AND b.ifc_model = 'AP'
                RETURN a.guid AS source_guid, b.guid AS target_guid, '' AS wall_guid
            """,
            "NEXT_TO": """
                MATCH (a:IFCElement)-[r:NEXT_TO]->(b:IFCElement)
                WHERE a.ifc_model = 'AP' AND b.ifc_model = 'AP'
                RETURN a.guid AS source_guid, b.guid AS target_guid, r.wall_guid AS wall_guid
            """,
            "CONNECTS_TO": """
                MATCH (a:IFCElement)-[:CONNECTS_TO]->(b:IFCElement)
                WHERE a.ifc_model = 'AP' AND b.ifc_model = 'AP'
                RETURN a.guid AS source_guid, b.guid AS target_guid, '' AS wall_guid
            """,
            "ADJACENT_TO": """
                MATCH (a:IFCElement)-[:ADJACENT_TO]->(b:IFCElement)
                WHERE a.ifc_model = 'AP' AND b.ifc_model = 'AP'
                RETURN a.guid AS source_guid, b.guid AS target_guid, '' AS wall_guid
            """,
        }
        for pred, query in edge_specs.items():
            for row in graph.run(query).data():
                src = row["source_guid"]
                tgt = row["target_guid"]
                src_node = self._ensure_node(src)
                tgt_node = self._ensure_node(tgt)
                self.edges[pred][src].append(
                    {
                        "target_guid": tgt,
                        "wall_guid": row.get("wall_guid") or "",
                        "target_type": tgt_node.get("ifc_type") or "",
                    }
                )
                if pred == "FILLS":
                    src_node["host_guid"] = tgt

    def load_local_fallback(self, relevant_guids: Iterable[str]) -> None:
        for guid in relevant_guids:
            self._ensure_node(guid)
        for guid, ctx in self.next_to_context.items():
            if guid not in self.nodes:
                continue
            host_guid = ctx.get("host_guid")
            if host_guid:
                self._ensure_node(host_guid)
                self.edges["FILLS"][guid].append(
                    {
                        "target_guid": host_guid,
                        "wall_guid": host_guid,
                        "target_type": self.nodes[host_guid].get("ifc_type") or "",
                    }
                )
                self.nodes[guid]["host_guid"] = host_guid
            for direction in ("left", "right"):
                nb_guid = ctx.get(f"{direction}_neighbor_guid")
                nb_type = ctx.get(f"{direction}_neighbor_type")
                if not nb_guid:
                    continue
                self._ensure_node(nb_guid, nb_type)
                self.edges["NEXT_TO"][guid].append(
                    {
                        "target_guid": nb_guid,
                        "wall_guid": host_guid or "",
                        "target_type": self.nodes[nb_guid].get("ifc_type") or nb_type or "",
                    }
                )

    def subtype_for_guid(self, guid: str) -> str:
        row = self.element_index.get(guid, {})
        node = self.nodes.get(guid, {})
        subtype = subtype_keyword_from_row(row) or subtype_keyword_from_row(node)
        return subtype or ""

    def relation_options(self, guid: str) -> List[CandidateOption]:
        node = self.nodes.get(guid, {})
        options: List[CandidateOption] = []
        for pred in ("FILLS", "NEXT_TO", "CONNECTS_TO", "ADJACENT_TO"):
            for edge in self.edges[pred].get(guid, []):
                tgt_guid = edge.get("target_guid") or ""
                tgt_node = self.nodes.get(tgt_guid, {})
                direction = ""
                host_guid = edge.get("wall_guid") or node.get("host_guid") or ""
                if pred == "NEXT_TO":
                    direction = _infer_direction(node, tgt_node)
                option = CandidateOption(
                    predicate=pred,
                    object_guid=tgt_guid,
                    object_type=str(tgt_node.get("ifc_type") or edge.get("target_type") or ""),
                    direction=direction,
                    object_subtype=self.subtype_for_guid(tgt_guid),
                    host_guid=host_guid,
                    wall_position_index=_safe_int(node.get("wall_position_index")),
                    wall_child_total=_safe_int(node.get("wall_child_total")),
                )
                options.append(option)
        return options

    def slot_fingerprint(self, guid: str) -> Dict[str, Any]:
        node = self.nodes.get(guid, {})
        return {
            "host_guid": node.get("host_guid") or "",
            "wall_position_index": _safe_int(node.get("wall_position_index")),
            "wall_child_total": _safe_int(node.get("wall_child_total")),
        }


def _enrich_gt_label(label: dict, target_guid: str, snapshot: GraphSnapshot) -> dict:
    rels = label.get("spatial_relations", []) or []
    options = snapshot.relation_options(target_guid)
    used = [False] * len(options)
    enriched_rels: List[dict] = []
    for rel in rels:
        candidate = dict(rel)
        for idx, option in enumerate(options):
            if used[idx]:
                continue
            if option.predicate != str(rel.get("predicate") or ""):
                continue
            if not type_matches(option.object_type, rel.get("object_type")):
                continue
            used[idx] = True
            if option.direction:
                candidate["direction"] = option.direction
            if option.object_subtype:
                candidate["object_subtype"] = option.object_subtype
            break
        enriched_rels.append(candidate)
    enriched = dict(label)
    enriched["spatial_relations"] = enriched_rels
    return enriched


def _build_gt_requirements(enriched_label: dict, level: str) -> List[dict]:
    out = []
    for rel in enriched_label.get("spatial_relations", []) or []:
        item = {
            "predicate": rel.get("predicate"),
            "object_type": rel.get("object_type"),
        }
        if level in {"L2_pred_obj_dir", "L3_pred_obj_dir_sub"} and rel.get("direction"):
            item["direction"] = rel.get("direction")
        if level == "L3_pred_obj_dir_sub" and rel.get("object_subtype"):
            item["object_subtype"] = str(rel.get("object_subtype")).strip().lower()
        out.append(item)
    return out


def _match_level_pool(
    candidate_guids: List[str],
    requirements: List[dict],
    snapshot: GraphSnapshot,
    level: str,
) -> List[str]:
    out = []
    for guid in candidate_guids:
        options = snapshot.relation_options(guid)
        if _multiset_match(options, requirements, level):
            out.append(guid)
    return out


def _match_slot_pool(candidate_guids: List[str], target_guid: str, snapshot: GraphSnapshot) -> Tuple[List[str], bool]:
    gt_slot = snapshot.slot_fingerprint(target_guid)
    if not gt_slot.get("host_guid") or gt_slot.get("wall_position_index") is None:
        return candidate_guids, False
    out = []
    for guid in candidate_guids:
        slot = snapshot.slot_fingerprint(guid)
        if slot.get("host_guid") != gt_slot["host_guid"]:
            continue
        if slot.get("wall_position_index") != gt_slot["wall_position_index"]:
            continue
        if gt_slot.get("wall_child_total") is not None and slot.get("wall_child_total") not in {
            None,
            gt_slot["wall_child_total"],
        }:
            continue
        out.append(guid)
    return out, True


def _infer_used_level(constraints: dict) -> str:
    rels = constraints.get("spatial_relations", []) or []
    if not rels:
        return "L0_p1_only"
    if any(rel.get("direction") for rel in rels):
        if any(rel.get("object_subtype") for rel in rels):
            return "L3_pred_obj_dir_sub"
        return "L2_pred_obj_dir"
    return "L1_pred_obj"


def _rank_bucket(rank: int) -> str:
    if rank == 1:
        return "1"
    if 2 <= rank <= 5:
        return "2-5"
    if 6 <= rank <= 10:
        return "6-10"
    if 11 <= rank <= 20:
        return "11-20"
    if 21 <= rank <= 50:
        return "21-50"
    return "51+"


def _first_unique_level(level_pools: Dict[str, List[str]], level_coverage: Dict[str, bool]) -> Tuple[str, bool]:
    for level in LEVELS:
        if not level_coverage.get(level, True):
            continue
        if len(level_pools[level]) == 1:
            return level, True
    return "never_unique_even_at_L4", False


def _classify_uniqueness(first_unique_level: str, is_unique_by_l4: bool) -> str:
    if not is_unique_by_l4:
        return "never_unique_even_at_L4"
    if first_unique_level == "L1_pred_obj":
        return "unique_at_L1"
    if first_unique_level == "L2_pred_obj_dir":
        return "unique_at_L2"
    if first_unique_level == "L3_pred_obj_dir_sub":
        return "unique_at_L3"
    if first_unique_level == "L4_full_fingerprint":
        return "unique_at_L4_only"
    return "unique_at_L1"


def _bool_pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def _median(values: List[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _avg(values: List[int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _root_cause_for_case(
    gt_rank: int,
    used_level: str,
    first_unique_level: str,
    is_unique_by_l4: bool,
    fallback_triggered: bool,
    local_label: dict,
    enriched_label: dict,
    level_pools: Dict[str, List[str]],
    final_pool_unique: int,
) -> str:
    if gt_rank == 1:
        return "none_top1_success"
    if not is_unique_by_l4:
        return "true_graph_ambiguity"
    if first_unique_level in LEVEL_ORDER and LEVEL_ORDER[first_unique_level] > LEVEL_ORDER[used_level]:
        if first_unique_level == "L2_pred_obj_dir" and not any(rel.get("direction") for rel in local_label.get("spatial_relations", []) or []) and any(rel.get("direction") for rel in enriched_label.get("spatial_relations", []) or []):
            return "ground_truth_not_collected"
        if first_unique_level == "L3_pred_obj_dir_sub" and not any(rel.get("object_subtype") for rel in local_label.get("spatial_relations", []) or []) and any(rel.get("object_subtype") for rel in enriched_label.get("spatial_relations", []) or []):
            return "ground_truth_not_collected"
        if first_unique_level == "L4_full_fingerprint" and not local_label.get("position_context"):
            return "ground_truth_not_collected"
        return "query_not_using_available_info"
    used_pool = len(level_pools.get(used_level, []))
    if fallback_triggered or final_pool_unique > max(used_pool, 1):
        return "fallback_pool_inflation"
    return "ranking_proxy_insufficient"


def _design_implication_rows(root_counter: Counter) -> List[List[str]]:
    return [
        [
            "ground_truth_not_collected",
            str(root_counter.get("ground_truth_not_collected", 0)),
            "Enrich canonical GT with direction/subtype/position-aware labels.",
        ],
        [
            "query_not_using_available_info",
            str(root_counter.get("query_not_using_available_info", 0)),
            "Extend planner/schema to use direction, subtype, and exact slot filters.",
        ],
        [
            "fallback_pool_inflation",
            str(root_counter.get("fallback_pool_inflation", 0)),
            "Tighten p0∪p1 union / fallback policy instead of always inflating to storey+type.",
        ],
        [
            "ranking_proxy_insufficient",
            str(root_counter.get("ranking_proxy_insufficient", 0)),
            "Add dense reranker, anchor-aware reranker, or exact-slot ordering after retrieval.",
        ],
        [
            "true_graph_ambiguity",
            str(root_counter.get("true_graph_ambiguity", 0)),
            "Treat these as non-unique retrieval tasks rather than strict Top-1 localization.",
        ],
    ]


def _label_enrichment_target(
    first_unique_level: str,
    is_unique_by_l4: bool,
    loss_root: str,
) -> Tuple[str, str, str]:
    if loss_root == "none_top1_success":
        return (
            "current_label_sufficient",
            "Current extraction/planner path already reaches Top-1 for this case.",
            "No enrichment needed for this case under the current planner.",
        )
    if not is_unique_by_l4:
        return (
            "not_fixable_by_label_only",
            "Label enrichment alone is insufficient; use retrieval+report semantics.",
            "Relax strict Top-1 and report small candidate sets for graph-ambiguous cases.",
        )
    if first_unique_level == "L1_pred_obj":
        return (
            "current_label_sufficient",
            "Current predicate+object label is already sufficient.",
            "No extra label field required; focus on planner/ranker.",
        )
    if first_unique_level == "L2_pred_obj_dir":
        return (
            "add_direction",
            "Add explicit direction as the minimal discriminative field.",
            "Planner should expose a direction-aware filter in Cypher.",
        )
    if first_unique_level == "L3_pred_obj_dir_sub":
        return (
            "add_direction_and_object_subtype",
            "Add subtype-bearing labels; keep direction whenever available.",
            "Planner should support direction+subtype filters before fallback.",
        )
    if first_unique_level == "L4_full_fingerprint":
        return (
            "add_position_context",
            "Add slot-aware position_context as the minimal unique fingerprint.",
            "Planner should support exact slot / wall-position filtering.",
        )
    return (
        "current_label_sufficient",
        "Current predicate+object label is already sufficient.",
        "No extra label field required; focus on planner/ranker.",
    )


def _plot_fingerprint_waterfall(
    fingerprint_summary_rows: List[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    if plt is None:
        return "matplotlib unavailable"

    slice_order = ["all_cases", "position_sensitive_subset"]
    slice_titles = {
        "all_cases": "All AP held-out cases",
        "position_sensitive_subset": "Position-sensitive subset",
    }
    level_rows = {
        slice_name: {
            row["level"]: row
            for row in fingerprint_summary_rows
            if row["slice"] == slice_name
        }
        for slice_name in slice_order
    }

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    pool_color = "#1565C0"
    top1_color = "#D32F2F"

    for ax, slice_name in zip(axes, slice_order):
        rows = [level_rows[slice_name][level] for level in LEVELS if level in level_rows[slice_name]]
        x = list(range(len(rows)))
        avg_pool = [row["avg_pool"] for row in rows]
        top1 = [row["top1_rate"] * 100.0 for row in rows]
        top10 = [row["top10_rate"] * 100.0 for row in rows]
        coverage = [row["coverage"] * 100.0 for row in rows]

        bars = ax.bar(x, avg_pool, color=pool_color, alpha=0.88, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, [LEVEL_DISPLAY[row["level"]] for row in rows])
        ax.set_yscale("log")
        ax.set_ylabel("Average pool size (log scale)")
        ax.set_title(slice_titles[slice_name])
        ax.grid(axis="y", linestyle="--", alpha=0.28)

        ax2 = ax.twinx()
        ax2.plot(x, top1, color=top1_color, marker="o", linewidth=2.0)
        ax2.plot(x, top10, color="#F57C00", marker="s", linewidth=2.0, linestyle="--")
        ax2.set_ylabel("Ideal Top-10 / Top-1 (%)")
        ax2.tick_params(axis="y", colors=top1_color)
        ax2.set_ylim(0, max(top1 + top10 + [5.0]) * 1.2 if (top1 or top10) else 100)

        for idx, (bar, pool, top1_val, top10_val, cov) in enumerate(zip(bars, avg_pool, top1, top10, coverage)):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(pool, 0.8) * 1.1,
                f"{pool:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=pool_color,
                fontweight="semibold",
            )
            ax2.text(
                idx,
                top1_val + max(1.2, ax2.get_ylim()[1] * 0.02),
                f"{top1_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color=top1_color,
                fontweight="semibold",
            )
            ax2.text(
                idx,
                top10_val + max(1.2, ax2.get_ylim()[1] * 0.06),
                f"{top10_val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#F57C00",
                fontweight="semibold",
            )
            ax.text(
                idx,
                ax.get_ylim()[0] * 1.25,
                f"cov {cov:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    fig.suptitle(
        "Oracle fingerprint waterfall: pool collapse vs ideal Top-1 by L-query level",
        fontsize=15,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Bars show average candidate pool size after each hypothetical fingerprint level. "
        "Orange dashed line shows ideal Top-10 and red line shows ideal Top-1 under perfect filtering, not realized MRR.",
        ha="center",
        fontsize=10,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, default=RAW_TRACE_JSONL)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--cases", type=Path, default=CASES_PATH)
    parser.add_argument("--gt-eval", type=Path, default=GT_EVAL_PATH)
    parser.add_argument("--element-index", type=Path, default=ELEMENT_INDEX_PATH)
    parser.add_argument("--wall-region-index", type=Path, default=WALL_REGION_INDEX_PATH)
    parser.add_argument("--ifc", type=Path, default=AP_IFC_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-neo4j", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    cases = load_cases_map(args.cases)
    gt_labels = load_gt_eval_labels(args.gt_eval)
    traces = _load_trace_rows(args.trace_jsonl)
    element_index = load_element_index(args.element_index)
    wall_region_index = _load_wall_region_index(args.wall_region_index)
    next_to_context = build_next_to_context(args.ifc, element_index)

    relevant_guids = set()
    for case in cases.values():
        relevant_guids.add(case.get("ground_truth", {}).get("target_guid", ""))
    for trace in traces.values():
        relevant_guids.update(_unique_ranked_guids(trace))
    relevant_guids = {guid for guid in relevant_guids if guid}

    snapshot = GraphSnapshot(element_index, next_to_context)
    neo4j_error = None
    if not args.skip_neo4j:
        graph, neo4j_error = _connect_neo4j(args.config)
        if graph is not None:
            snapshot.load_from_neo4j(graph)
        else:
            snapshot.load_local_fallback(relevant_guids)
    else:
        snapshot.load_local_fallback(relevant_guids)

    for guid in relevant_guids:
        snapshot._ensure_node(guid)

    overall_metrics = compute_metrics(list(traces.values()))
    summary_csv_metrics = _parse_summary_csv(args.summary_csv)

    case_rows: List[Dict[str, Any]] = []
    uniqueness_case_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    position_test_rows: List[Dict[str, Any]] = []
    enrichment_rows: List[Dict[str, Any]] = []

    rank_counter = Counter()
    pred_rank_counter: Dict[str, Counter] = defaultdict(Counter)
    family_rank_counter: Dict[str, Counter] = defaultdict(Counter)
    universe_rank_counter: Dict[str, Counter] = defaultdict(Counter)

    uniqueness_summary_counter = Counter()
    root_counter = Counter()

    fingerprint_case_rows: List[Dict[str, Any]] = []

    for case_id, case in sorted(cases.items()):
        if case_id not in traces or case_id not in gt_labels:
            continue
        trace = traces[case_id]
        gt_label = gt_labels[case_id]
        enriched_gt = _enrich_gt_label(gt_label, case["ground_truth"]["target_guid"], snapshot)
        rels = gt_label.get("spatial_relations", []) or []
        family = topology_family(rels)
        universe = universe_key(rels)
        predicate = _primary_predicate(case, gt_label)
        ranked_guids = _unique_ranked_guids(trace)
        gt_guid = case["ground_truth"]["target_guid"]
        gt_rank = ranked_guids.index(gt_guid) + 1 if gt_guid in ranked_guids else 999
        bucket = _rank_bucket(gt_rank)
        raw_ranked_len = len(trace.get("interpreter_output", {}).get("candidates", []))
        final_pool_unique = len(ranked_guids)

        storey = _root_storey(case, gt_label)
        ifc_class = str(gt_label.get("ifc_class") or case["ground_truth"].get("target_ifc_class") or "")

        l0_pool = [
            guid
            for guid, node in snapshot.nodes.items()
            if normalize_storey(node.get("storey")) == storey and type_matches(node.get("ifc_type"), ifc_class)
        ]
        l1_requirements = _build_gt_requirements(enriched_gt, "L1_pred_obj")
        l2_requirements = _build_gt_requirements(enriched_gt, "L2_pred_obj_dir")
        l3_requirements = _build_gt_requirements(enriched_gt, "L3_pred_obj_dir_sub")

        l1_pool = _match_level_pool(l0_pool, l1_requirements, snapshot, "L1_pred_obj")
        l2_applicable = any(rel.get("direction") for rel in enriched_gt.get("spatial_relations", []) or [])
        l2_pool = _match_level_pool(l0_pool, l2_requirements, snapshot, "L2_pred_obj_dir") if l2_applicable else list(l1_pool)
        l3_applicable = any(rel.get("object_subtype") for rel in enriched_gt.get("spatial_relations", []) or [])
        l3_pool = _match_level_pool(l0_pool, l3_requirements, snapshot, "L3_pred_obj_dir_sub") if l3_applicable else list(l2_pool)
        l4_pool, l4_applicable = _match_slot_pool(l3_pool, gt_guid, snapshot)

        level_pools = {
            "L0_p1_only": l0_pool,
            "L1_pred_obj": l1_pool,
            "L2_pred_obj_dir": l2_pool,
            "L3_pred_obj_dir_sub": l3_pool,
            "L4_full_fingerprint": l4_pool,
        }
        level_coverage = {
            "L0_p1_only": True,
            "L1_pred_obj": True,
            "L2_pred_obj_dir": l2_applicable,
            "L3_pred_obj_dir_sub": l3_applicable,
            "L4_full_fingerprint": l4_applicable,
        }
        first_unique_level, is_unique_by_l4 = _first_unique_level(level_pools, level_coverage)
        uniqueness_class = _classify_uniqueness(first_unique_level, is_unique_by_l4)
        used_level = _infer_used_level(trace.get("internals", {}).get("constraints", {}) or {})
        rr = trace.get("internals", {}).get("retrieval_results", [])
        rr0 = rr[0] if rr else {}
        expected_pool_size = (rr0.get("query_plan_used", {}) or {}).get("expected_pool_size")
        fallback_triggered = bool(rr0.get("fallback_triggered"))
        strategy_actually_used = rr0.get("strategy_actually_used", "")

        loss_root = _root_cause_for_case(
            gt_rank=gt_rank,
            used_level=used_level,
            first_unique_level=first_unique_level,
            is_unique_by_l4=is_unique_by_l4,
            fallback_triggered=fallback_triggered,
            local_label=gt_label,
            enriched_label=enriched_gt,
            level_pools=level_pools,
            final_pool_unique=final_pool_unique,
        )
        if loss_root not in ROOT_CAUSES:
            loss_root = "ranking_proxy_insufficient"
        enrichment_target, enrichment_rationale, planner_target = _label_enrichment_target(
            first_unique_level=first_unique_level,
            is_unique_by_l4=is_unique_by_l4,
            loss_root=loss_root,
        )

        rank_counter[bucket] += 1
        pred_rank_counter[predicate][bucket] += 1
        family_rank_counter[family][bucket] += 1
        universe_rank_counter[universe][bucket] += 1
        uniqueness_summary_counter[uniqueness_class] += 1
        root_counter[loss_root] += 1

        has_local_direction = any(rel.get("direction") for rel in gt_label.get("spatial_relations", []) or [])
        has_graph_direction = any(rel.get("direction") for rel in enriched_gt.get("spatial_relations", []) or [])
        has_local_subtype = any(rel.get("object_subtype") for rel in gt_label.get("spatial_relations", []) or [])
        has_graph_subtype = any(rel.get("object_subtype") for rel in enriched_gt.get("spatial_relations", []) or [])
        gt_slot = snapshot.slot_fingerprint(gt_guid)

        case_row = {
            "case_id": case_id,
            "predicate": predicate,
            "family": family,
            "universe": universe,
            "storey_name": storey,
            "ifc_class": ifc_class,
            "gt_rank": gt_rank,
            "rank_bucket": bucket,
            "top10_hit": gt_rank <= 10,
            "top1_hit": gt_rank == 1,
            "final_pool_size_raw": trace.get("final_pool_size"),
            "final_pool_size_unique": final_pool_unique,
            "expected_pool_size": expected_pool_size,
            "fallback_triggered": fallback_triggered,
            "strategy_actually_used": strategy_actually_used,
            "used_level": used_level,
            "first_unique_level": first_unique_level,
            "is_unique_by_L4": is_unique_by_l4,
            "uniqueness_class": uniqueness_class,
            "loss_point_root_cause": loss_root,
            "minimal_label_enrichment_target": enrichment_target,
            "minimal_label_enrichment_rationale": enrichment_rationale,
            "planner_query_target": planner_target,
            "snapshot_source": snapshot.source,
            "graph_has_direction": has_graph_direction,
            "graph_has_subtype": has_graph_subtype,
            "graph_has_position": bool(gt_slot.get("host_guid") and gt_slot.get("wall_position_index") is not None),
            "local_has_direction": has_local_direction,
            "local_has_subtype": has_local_subtype,
            "local_has_position_context": bool(gt_label.get("position_context")),
            "wall_region_patch_count": len(wall_region_index.get(gt_guid, [])),
            "L0_pool": len(l0_pool),
            "L1_pool": len(l1_pool),
            "L2_pool": len(l2_pool),
            "L3_pool": len(l3_pool),
            "L4_pool": len(l4_pool),
        }
        case_rows.append(case_row)
        uniqueness_case_rows.append(case_row.copy())
        enrichment_rows.append(
            {
                "case_id": case_id,
                "family": family,
                "universe": universe,
                "first_unique_level": first_unique_level,
                "is_unique_by_L4": is_unique_by_l4,
                "loss_point_root_cause": loss_root,
                "minimal_label_enrichment_target": enrichment_target,
                "minimal_label_enrichment_rationale": enrichment_rationale,
                "planner_query_target": planner_target,
                "local_has_direction": has_local_direction,
                "graph_has_direction": has_graph_direction,
                "local_has_subtype": has_local_subtype,
                "graph_has_subtype": has_graph_subtype,
                "local_has_position_context": bool(gt_label.get("position_context")),
                "graph_has_position": bool(gt_slot.get("host_guid") and gt_slot.get("wall_position_index") is not None),
            }
        )
        loss_rows.append(
            {
                "case_id": case_id,
                "gt_rank": gt_rank,
                "used_level": used_level,
                "first_unique_level": first_unique_level,
                "loss_point_root_cause": loss_root,
                "fallback_triggered": fallback_triggered,
                "strategy_actually_used": strategy_actually_used,
                "local_has_direction": has_local_direction,
                "graph_has_direction": has_graph_direction,
                "local_has_subtype": has_local_subtype,
                "graph_has_subtype": has_graph_subtype,
                "local_has_position_context": bool(gt_label.get("position_context")),
                "graph_has_position": bool(gt_slot.get("host_guid") and gt_slot.get("wall_position_index") is not None),
            }
        )

        for level in LEVELS:
            fingerprint_case_rows.append(
                {
                    "case_id": case_id,
                    "slice": "all_cases",
                    "family": family,
                    "level": level,
                    "coverage": level_coverage[level],
                    "pool_size": len(level_pools[level]),
                    "becomes_unique": len(level_pools[level]) == 1 if level_coverage[level] else False,
                }
            )
            if family in POSITION_SENSITIVE_FAMILIES:
                fingerprint_case_rows.append(
                    {
                        "case_id": case_id,
                        "slice": "position_sensitive_subset",
                        "family": family,
                        "level": level,
                        "coverage": level_coverage[level],
                        "pool_size": len(level_pools[level]),
                        "becomes_unique": len(level_pools[level]) == 1 if level_coverage[level] else False,
                    }
                )

        if 2 <= gt_rank <= 10:
            position_filtered = ranked_guids
            position_applicable = False
            if gt_slot.get("host_guid") and gt_slot.get("wall_position_index") is not None:
                position_filtered = []
                position_applicable = True
                for guid in ranked_guids:
                    slot = snapshot.slot_fingerprint(guid)
                    if slot.get("host_guid") != gt_slot["host_guid"]:
                        continue
                    if slot.get("wall_position_index") != gt_slot["wall_position_index"]:
                        continue
                    if gt_slot.get("wall_child_total") is not None and slot.get("wall_child_total") not in {
                        None,
                        gt_slot["wall_child_total"],
                    }:
                        continue
                    position_filtered.append(guid)
            position_test_rows.append(
                {
                    "case_id": case_id,
                    "gt_rank_before": gt_rank,
                    "current_pool_unique": final_pool_unique,
                    "position_filter_applicable": position_applicable,
                    "position_filter_mode": "host_guid+wall_position_index(+wall_child_total)" if position_applicable else "not_applicable",
                    "pool_after_position_unique": len(position_filtered),
                    "top1_after_position": bool(position_filtered and position_filtered[0] == gt_guid),
                    "family": family,
                    "universe": universe,
                    "predicate": predicate,
                }
            )

    # Summaries
    oracle_rank_distribution = [
        {"bucket": bucket, "count": rank_counter.get(bucket, 0)}
        for bucket in ["1", "2-5", "6-10", "11-20", "21-50", "51+"]
    ]

    def _slice_rank_rows(counter_map: Dict[str, Counter], label_key: str) -> List[Dict[str, Any]]:
        rows = []
        for label, counter in sorted(counter_map.items()):
            row = {label_key: label}
            for bucket in ["1", "2-5", "6-10", "11-20", "21-50", "51+"]:
                row[bucket] = counter.get(bucket, 0)
            rows.append(row)
        return rows

    fingerprint_summary_rows: List[Dict[str, Any]] = []
    for slice_name in ("all_cases", "position_sensitive_subset"):
        slice_rows = [row for row in fingerprint_case_rows if row["slice"] == slice_name]
        total_cases = len({row["case_id"] for row in slice_rows})
        for level in LEVELS:
            level_rows = [row for row in slice_rows if row["level"] == level]
            applicable = [row for row in level_rows if row["coverage"]]
            pool_values = [int(row["pool_size"]) for row in applicable]
            unique_hits = sum(1 for row in applicable if row["becomes_unique"])
            top10_hits = sum(1 for row in applicable if int(row["pool_size"]) <= 10)
            fingerprint_summary_rows.append(
                {
                    "slice": slice_name,
                    "level": level,
                    "coverage": round(_bool_pct(len(applicable), total_cases), 4),
                    "n_total_cases": total_cases,
                    "n_applicable_cases": len(applicable),
                    "avg_pool": round(_avg(pool_values), 3),
                    "median_pool": round(_median(pool_values), 3),
                    "top10_rate": round(_bool_pct(top10_hits, len(applicable)), 4) if applicable else 0.0,
                    "top1_rate": round(_bool_pct(unique_hits, len(applicable)), 4) if applicable else 0.0,
                }
            )

    # Output files
    write_csv(
        out_dir / "oracle_case_table.csv",
        case_rows,
        [
            "case_id",
            "predicate",
            "family",
            "universe",
            "storey_name",
            "ifc_class",
            "gt_rank",
            "rank_bucket",
            "top10_hit",
            "top1_hit",
            "final_pool_size_raw",
            "final_pool_size_unique",
            "expected_pool_size",
            "fallback_triggered",
            "strategy_actually_used",
            "used_level",
            "first_unique_level",
            "is_unique_by_L4",
            "uniqueness_class",
            "loss_point_root_cause",
            "minimal_label_enrichment_target",
            "minimal_label_enrichment_rationale",
            "planner_query_target",
            "snapshot_source",
            "graph_has_direction",
            "graph_has_subtype",
            "graph_has_position",
            "local_has_direction",
            "local_has_subtype",
            "local_has_position_context",
            "wall_region_patch_count",
            "L0_pool",
            "L1_pool",
            "L2_pool",
            "L3_pool",
            "L4_pool",
        ],
    )
    write_csv(
        out_dir / "topology_uniqueness_case_table.csv",
        uniqueness_case_rows,
        [
            "case_id",
            "family",
            "universe",
            "ifc_class",
            "storey_name",
            "first_unique_level",
            "is_unique_by_L4",
            "uniqueness_class",
            "L0_pool",
            "L1_pool",
            "L2_pool",
            "L3_pool",
            "L4_pool",
        ],
    )
    write_csv(
        out_dir / "topology_uniqueness_by_level.csv",
        [
            {
                "class": key,
                "count": value,
            }
            for key, value in sorted(uniqueness_summary_counter.items())
        ],
        ["class", "count"],
    )
    write_csv(
        out_dir / "label_enrichment_requirements.csv",
        enrichment_rows,
        [
            "case_id",
            "family",
            "universe",
            "first_unique_level",
            "is_unique_by_L4",
            "loss_point_root_cause",
            "minimal_label_enrichment_target",
            "minimal_label_enrichment_rationale",
            "planner_query_target",
            "local_has_direction",
            "graph_has_direction",
            "local_has_subtype",
            "graph_has_subtype",
            "local_has_position_context",
            "graph_has_position",
        ],
    )
    write_csv(out_dir / "oracle_rank_distribution.csv", oracle_rank_distribution, ["bucket", "count"])
    write_csv(
        out_dir / "oracle_rank_distribution_by_predicate.csv",
        _slice_rank_rows(pred_rank_counter, "predicate"),
        ["predicate", "1", "2-5", "6-10", "11-20", "21-50", "51+"],
    )
    write_csv(
        out_dir / "oracle_rank_distribution_by_family.csv",
        _slice_rank_rows(family_rank_counter, "family"),
        ["family", "1", "2-5", "6-10", "11-20", "21-50", "51+"],
    )
    write_csv(
        out_dir / "oracle_rank_distribution_by_universe.csv",
        _slice_rank_rows(universe_rank_counter, "universe"),
        ["universe", "1", "2-5", "6-10", "11-20", "21-50", "51+"],
    )
    write_csv(
        out_dir / "oracle_pool_vs_rank.csv",
        [
            {
                "case_id": row["case_id"],
                "gt_rank": row["gt_rank"],
                "final_pool_size_raw": row["final_pool_size_raw"],
                "final_pool_size_unique": row["final_pool_size_unique"],
                "expected_pool_size": row["expected_pool_size"],
                "rank_bucket": row["rank_bucket"],
            }
            for row in case_rows
        ],
        ["case_id", "gt_rank", "final_pool_size_raw", "final_pool_size_unique", "expected_pool_size", "rank_bucket"],
    )
    write_csv(
        out_dir / "oracle_failure_buckets.csv",
        [
            {"bucket": key, "count": root_counter.get(key, 0)}
            for key in sorted(ROOT_CAUSES)
        ],
        ["bucket", "count"],
    )
    write_csv(
        out_dir / "loss_point_attribution.csv",
        loss_rows,
        [
            "case_id",
            "gt_rank",
            "used_level",
            "first_unique_level",
            "loss_point_root_cause",
            "fallback_triggered",
            "strategy_actually_used",
            "local_has_direction",
            "graph_has_direction",
            "local_has_subtype",
            "graph_has_subtype",
            "local_has_position_context",
            "graph_has_position",
        ],
    )
    write_csv(
        out_dir / "oracle_position_test.csv",
        position_test_rows,
        [
            "case_id",
            "gt_rank_before",
            "current_pool_unique",
            "position_filter_applicable",
            "position_filter_mode",
            "pool_after_position_unique",
            "top1_after_position",
            "family",
            "universe",
            "predicate",
        ],
    )
    write_csv(
        out_dir / "fingerprint_loss_case_table.csv",
        fingerprint_case_rows,
        ["case_id", "slice", "family", "level", "coverage", "pool_size", "becomes_unique"],
    )
    write_csv(
        out_dir / "fingerprint_loss_by_level.csv",
        fingerprint_summary_rows,
        ["slice", "level", "coverage", "n_total_cases", "n_applicable_cases", "avg_pool", "median_pool", "top10_rate", "top1_rate"],
    )

    summary_payload = {
        "snapshot_source": snapshot.source,
        "neo4j_error": neo4j_error,
        "overall_metrics_from_traces": overall_metrics,
        "overall_metrics_from_summary_csv": summary_csv_metrics,
        "uniqueness_classes": dict(uniqueness_summary_counter),
        "root_causes": dict(root_counter),
        "label_enrichment_targets": dict(
            Counter(row["minimal_label_enrichment_target"] for row in enrichment_rows)
        ),
    }
    write_json(out_dir / "oracle_ceiling_summary.json", summary_payload)

    plot_error = _plot_fingerprint_waterfall(
        fingerprint_summary_rows,
        out_dir / "oracle_fingerprint_waterfall.png",
    )

    position_applicable = [row for row in position_test_rows if row["position_filter_applicable"]]
    position_top1_after = sum(1 for row in position_applicable if row["top1_after_position"])
    info_loss_rows = [
        [
            row["level"],
            row["slice"],
            f"{row['coverage'] * 100:.1f}%",
            f"{row['avg_pool']:.2f}",
            f"{row['median_pool']:.2f}",
            f"{row['top10_rate'] * 100:.1f}%",
            f"{row['top1_rate'] * 100:.1f}%",
        ]
        for row in fingerprint_summary_rows
    ]

    oracle_md = [
        "# Oracle Ceiling Summary",
        "",
        f"- Snapshot source: `{snapshot.source}`",
        f"- Neo4j status: {'connected' if snapshot.source == 'neo4j' else 'unavailable (' + str(neo4j_error or 'unknown') + ')'}",
        f"- Trace-derived overall: Top-10 `{overall_metrics['top10_pct']:.1f}%`, Top-1 `{overall_metrics['top1_pct']:.1f}%`, MRR@10 `{overall_metrics['mrr']:.4f}`",
        f"- Fingerprint waterfall plot: `{'oracle_fingerprint_waterfall.png' if plot_error is None else 'not generated (' + plot_error + ')'}`",
        "",
        "## Rank Distribution",
        "",
        markdown_table(
            ["Bucket", "Count"],
            [[row["bucket"], row["count"]] for row in oracle_rank_distribution],
        ),
        "",
        "## Topology Uniqueness",
        "",
        markdown_table(
            ["Class", "Count"],
            [[key, value] for key, value in sorted(uniqueness_summary_counter.items())],
        ),
        "",
        "## Oracle Position Test",
        "",
        f"- Subset size (`Top-10=YES and Top-1=NO`): `{len(position_test_rows)}`",
        f"- Position-filter applicable: `{len(position_applicable)}`",
        f"- Top-1 after exact slot filter: `{position_top1_after}/{len(position_applicable)}`"
        if position_applicable
        else "- Position-filter applicable: `0`",
        "",
        "## Information Loss Chain",
        "",
        markdown_table(
            ["Level", "Slice", "Coverage", "Avg Pool", "Median Pool", "Top-10", "Top-1"],
            info_loss_rows,
        ),
        "",
        "## Query Semantics",
        "",
        "- The L-query audit is a target-rooted multi-anchor neighborhood filter, not a chained A->B->C multi-hop traversal.",
        "- A candidate survives a level only if its local neighborhood jointly satisfies all required relations at that fingerprint level.",
        "- Therefore, L1-L4 quantify how much discriminative information is available around the target, not how many hops the current planner explicitly traverses.",
        "",
        "## Label Enrichment Targets",
        "",
        markdown_table(
            ["Target", "Count"],
            [
                [key, value]
                for key, value in sorted(
                    Counter(row["minimal_label_enrichment_target"] for row in enrichment_rows).items()
                )
            ],
        ),
        "",
        "## Design Implications",
        "",
        markdown_table(
            ["Root Cause", "Count", "Recommended Optimization"],
            _design_implication_rows(root_counter),
        ),
        "",
    ]
    (out_dir / "oracle_ceiling_summary.md").write_text("\n".join(oracle_md), encoding="utf-8")

    uniqueness_md = [
        "# Topology Uniqueness Summary",
        "",
        markdown_table(
            ["Class", "Count"],
            [[key, value] for key, value in sorted(uniqueness_summary_counter.items())],
        ),
        "",
    ]
    (out_dir / "topology_uniqueness_summary.md").write_text("\n".join(uniqueness_md), encoding="utf-8")

    label_target_counter = Counter(row["minimal_label_enrichment_target"] for row in enrichment_rows)
    enrichment_md = [
        "# Label Enrichment Requirements",
        "",
        markdown_table(
            ["Target", "Count"],
            [[key, value] for key, value in sorted(label_target_counter.items())],
        ),
        "",
        "These targets answer the reverse-GT question: if a target is unique in the graph, what is the minimal label enrichment needed for extraction to recover that uniqueness?",
        "",
    ]
    (out_dir / "label_enrichment_requirements.md").write_text("\n".join(enrichment_md), encoding="utf-8")

    loss_md = [
        "# Loss Point Summary",
        "",
        markdown_table(
            ["Root Cause", "Count"],
            [[key, root_counter.get(key, 0)] for key in sorted(ROOT_CAUSES)],
        ),
        "",
    ]
    (out_dir / "loss_point_summary.md").write_text("\n".join(loss_md), encoding="utf-8")

    pos_md = [
        "# Oracle Position Test Summary",
        "",
        f"- Subset size: `{len(position_test_rows)}`",
        f"- Applicable cases: `{len(position_applicable)}`",
        f"- Top-1 after exact slot filter: `{position_top1_after}/{len(position_applicable)}`"
        if position_applicable
        else "- Top-1 after exact slot filter: `0/0`",
        "",
    ]
    (out_dir / "oracle_position_test_summary.md").write_text("\n".join(pos_md), encoding="utf-8")

    loss_chain_md = [
        "# Fingerprint Information Loss Summary",
        "",
        markdown_table(
            ["Level", "Slice", "Coverage", "Avg Pool", "Median Pool", "Top-10", "Top-1"],
            info_loss_rows,
        ),
        "",
    ]
    (out_dir / "fingerprint_loss_summary.md").write_text("\n".join(loss_chain_md), encoding="utf-8")

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
