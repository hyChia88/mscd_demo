#!/usr/bin/env python3
"""
Prepare a minimal Q4 verification bundle from unified evaluation traces.

This script:
1. Loads the main unified trace.
2. Finds cases where GT is in the retrieval pool but not ranked Top-1.
3. Samples N cases deterministically.
4. Exports the current top-K candidates for each case.
5. Builds lightweight Graph-to-Language (G2L) candidate description scaffolds.
6. Writes two prompt variants per case under a floorplan+text-only setup:
   - chat + candidate descriptions
   - chat + floorplan + candidate descriptions

No on-site image is used here. The output is intended for quick verification
experiments with Gemini or other multimodal VLMs before building a full
retrieve-then-verify system.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ifcopenshell
import ifcopenshell.util.placement


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent

DEFAULT_TRACE = (
    PROJECT_ROOT
    / "output"
    / "unified"
    / "strategy_ablation_v3"
    / "traces_20260324_191220_v2_lora_p0_union_p1.jsonl"
)
DEFAULT_CASES = PROJECT_ROOT / "evaluation" / "cases" / "cases_unified_test.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "q4_verification"

MODEL_PATHS = {
    "AP": REPO_ROOT / "data_curation" / "ifc_models" / "AdvancedProject.ifc",
    "BH": REPO_ROOT / "data_curation" / "ifc_models" / "BasicHouse.ifc",
    "DXA": REPO_ROOT / "data_curation" / "ifc_models" / "Duplex_A_20110505.ifc",
}


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_name(value: Optional[str]) -> str:
    if not value:
        return "unnamed"
    return str(value)


def short_guid(guid: Optional[str]) -> str:
    if not guid:
        return "unknown"
    return guid[:8]


def first_spatial_relation(constraints: Dict[str, Any]) -> Dict[str, Any]:
    rels = constraints.get("spatial_relations") or []
    if rels:
        return rels[0]
    return {}


def resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for root in (REPO_ROOT / "data_curation", PROJECT_ROOT):
        resolved = root / path_str
        if resolved.exists():
            return resolved
    return None


@dataclass
class FillerTopology:
    host_wall_guid: str
    host_wall_name: Optional[str]
    host_wall_type: Optional[str]
    position_index: int
    wall_child_total: int
    left_neighbor_type: Optional[str]
    right_neighbor_type: Optional[str]


class ModelContext:
    def __init__(self, model_key: str, ifc_path: Path):
        self.model_key = model_key
        self.ifc_path = ifc_path
        self.file = ifcopenshell.open(str(ifc_path))
        self.storey_of_guid = self._build_storey_map()
        self.filler_topology = self._build_filler_topology()
        self.wall_facts = self._build_wall_facts()

    def _build_storey_map(self) -> Dict[str, str]:
        storey_of_guid: Dict[str, str] = {}
        for rel in self.file.by_type("IfcRelContainedInSpatialStructure"):
            structure = rel.RelatingStructure
            if not structure or not structure.is_a("IfcBuildingStorey"):
                continue
            storey_name = safe_name(structure.Name)
            for element in rel.RelatedElements:
                if not element.is_a("IfcOpeningElement"):
                    storey_of_guid[element.GlobalId] = storey_name
        return storey_of_guid

    def _get_centroid(self, element) -> Optional[Tuple[float, float, float]]:
        try:
            matrix = ifcopenshell.util.placement.get_local_placement(element.ObjectPlacement)
            return (matrix[0][3], matrix[1][3], matrix[2][3])
        except Exception:
            return None

    def _build_filler_topology(self) -> Dict[str, FillerTopology]:
        opening_to_host: Dict[str, Any] = {}
        for rel in self.file.by_type("IfcRelVoidsElement"):
            opening_to_host[rel.RelatedOpeningElement.GlobalId] = rel.RelatingBuildingElement

        wall_children: Dict[str, List[Any]] = defaultdict(list)
        for rel in self.file.by_type("IfcRelFillsElement"):
            filler = rel.RelatedBuildingElement
            opening = rel.RelatingOpeningElement
            host = opening_to_host.get(opening.GlobalId)
            if host is not None:
                wall_children[host.GlobalId].append(filler)

        topology: Dict[str, FillerTopology] = {}
        for wall_guid, children in wall_children.items():
            ordered_children = []
            for child in children:
                pos = self._get_centroid(child)
                if pos is not None:
                    ordered_children.append((child, pos))
            if not ordered_children:
                continue

            xs = [p[1][0] for p in ordered_children]
            ys = [p[1][1] for p in ordered_children]
            sort_axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
            ordered_children.sort(key=lambda item: item[1][sort_axis])

            host_wall = self.file.by_guid(wall_guid)
            host_name = safe_name(getattr(host_wall, "Name", None)) if host_wall else None
            host_type = host_wall.is_a() if host_wall else None
            wall_child_total = len(ordered_children)

            for index, (child, _) in enumerate(ordered_children):
                left_type = ordered_children[index - 1][0].is_a() if index > 0 else None
                right_type = ordered_children[index + 1][0].is_a() if index + 1 < wall_child_total else None
                topology[child.GlobalId] = FillerTopology(
                    host_wall_guid=wall_guid,
                    host_wall_name=host_name,
                    host_wall_type=host_type,
                    position_index=index,
                    wall_child_total=wall_child_total,
                    left_neighbor_type=left_type,
                    right_neighbor_type=right_type,
                )
        return topology

    def _build_wall_facts(self) -> Dict[str, Dict[str, Any]]:
        facts: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "connected_wall_count": 0,
                "connected_wall_types": Counter(),
                "filled_by_types": Counter(),
                "filled_by_count": 0,
            }
        )

        for filler_guid, topo in self.filler_topology.items():
            facts[topo.host_wall_guid]["filled_by_count"] += 1
            filler = self.file.by_guid(filler_guid)
            if filler is not None:
                facts[topo.host_wall_guid]["filled_by_types"][filler.is_a()] += 1

        for rel in self.file.by_type("IfcRelConnectsPathElements"):
            left = rel.RelatingElement
            right = rel.RelatedElement
            if left is None or right is None:
                continue
            if left.is_a("IfcWall") or left.is_a("IfcWallStandardCase"):
                facts[left.GlobalId]["connected_wall_count"] += 1
                facts[left.GlobalId]["connected_wall_types"][right.is_a()] += 1
            if right.is_a("IfcWall") or right.is_a("IfcWallStandardCase"):
                facts[right.GlobalId]["connected_wall_count"] += 1
                facts[right.GlobalId]["connected_wall_types"][left.is_a()] += 1

        return facts

    def get_element(self, guid: str):
        return self.file.by_guid(guid)

    def get_element_props(self, guid: str) -> Dict[str, Any]:
        element = self.get_element(guid)
        if element is None:
            return {}

        props = {
            "guid": guid,
            "name": safe_name(getattr(element, "Name", None)),
            "type": element.is_a(),
            "object_type": getattr(element, "ObjectType", None),
            "description": getattr(element, "Description", None),
            "storey": self.storey_of_guid.get(guid),
        }
        return props

    def build_candidate_scaffold(
        self,
        candidate: Dict[str, Any],
        query_relation: Dict[str, Any],
        is_gt: bool,
        pool_rank: int,
    ) -> Dict[str, Any]:
        guid = candidate.get("guid", "")
        props = self.get_element_props(guid)
        elem_type = props.get("type") or candidate.get("type") or candidate.get("element_type")
        storey = props.get("storey") or candidate.get("ref_storey")
        name = props.get("name") or candidate.get("name")
        query_pred = query_relation.get("predicate")
        query_obj_type = query_relation.get("object_type")

        draft_lines = [
            f"Candidate C{pool_rank} is a {elem_type} named \"{name}\".",
            f"It is located on {storey or 'unknown storey'}.",
        ]

        scaffold = {
            "guid": guid,
            "name": name,
            "type": elem_type,
            "storey": storey,
            "pool_rank": pool_rank,
            "is_ground_truth": is_gt,
            "object_type": props.get("object_type"),
            "retrieval_match": {
                "ref_type": candidate.get("ref_type"),
                "ref_storey": candidate.get("ref_storey"),
                "has_hop2": candidate.get("has_hop2"),
                "query_predicate": query_pred,
                "query_object_type": query_obj_type,
            },
            "graph_facts": {},
            "draft_description": "",
            "slots": [],
        }

        filler_topo = self.filler_topology.get(guid)
        if filler_topo is not None:
            scaffold["graph_facts"] = {
                "host_wall_guid": filler_topo.host_wall_guid,
                "host_wall_name": filler_topo.host_wall_name,
                "host_wall_type": filler_topo.host_wall_type,
                "position_index": filler_topo.position_index,
                "wall_child_total": filler_topo.wall_child_total,
                "left_neighbor_type": filler_topo.left_neighbor_type,
                "right_neighbor_type": filler_topo.right_neighbor_type,
            }
            draft_lines.extend(
                [
                    f"It is hosted by {filler_topo.host_wall_type} \"{filler_topo.host_wall_name}\" ({short_guid(filler_topo.host_wall_guid)}).",
                    f"On that wall it is position {filler_topo.position_index + 1} of {filler_topo.wall_child_total}.",
                    f"Its left neighbor type is {filler_topo.left_neighbor_type or 'none'} and its right neighbor type is {filler_topo.right_neighbor_type or 'none'}.",
                ]
            )
            scaffold["slots"] = [
                "host wall name / type",
                "position on wall",
                "left/right neighbor cue",
                "whether that local sequence matches the floorplan evidence",
            ]
        elif elem_type in ("IfcWall", "IfcWallStandardCase"):
            wall_fact = self.wall_facts.get(guid, {})
            filled_by_types = wall_fact.get("filled_by_types", Counter())
            filled_by_summary = ", ".join(
                f"{count}x {kind}" for kind, count in filled_by_types.items()
            ) or "none"
            scaffold["graph_facts"] = {
                "connected_wall_count": wall_fact.get("connected_wall_count", 0),
                "filled_by_count": wall_fact.get("filled_by_count", 0),
                "filled_by_types": dict(filled_by_types),
            }
            draft_lines.extend(
                [
                    f"It is path-connected to {wall_fact.get('connected_wall_count', 0)} other wall segments.",
                    f"It is filled by {wall_fact.get('filled_by_count', 0)} opening elements ({filled_by_summary}).",
                ]
            )
            scaffold["slots"] = [
                "opening pattern on wall",
                "connected wall context",
                "whether this wall is the one implied by the evidence",
            ]
        else:
            scaffold["slots"] = [
                "storey and name cue",
                "object type / subtype cue",
                "any additional topology facts to add manually if needed",
            ]

        if candidate.get("ref_type"):
            draft_lines.append(
                f"This candidate entered the pool under the cue {query_pred or 'UNKNOWN'} -> {candidate.get('ref_type')}."
            )
        elif query_pred and query_obj_type:
            draft_lines.append(
                f"This candidate should be checked against the cue {query_pred} {query_obj_type}."
            )

        scaffold["draft_description"] = " ".join(draft_lines)
        return scaffold


def build_verification_prompt(
    case_bundle: Dict[str, Any],
    include_floorplan: bool,
) -> str:
    evidence = case_bundle["evidence"]
    mode_label = "chat + floorplan + descriptions" if include_floorplan else "chat + descriptions"
    lines = [
        "# Verification Task",
        "",
        f"You are running a floorplan+text-only verification baseline: {mode_label}.",
        "No on-site photo is available in this experiment.",
        "Verify which BIM candidate best matches the provided evidence.",
        "Pick exactly one candidate ID.",
        "Return JSON: {\"best_candidate_id\": \"C?\", \"reason\": \"...\"}",
        "",
        "## Evidence",
        f"- Case ID: {case_bundle['case_id']}",
        f"- Query text: {evidence.get('query_text') or 'N/A'}",
        f"- 4D task status: {evidence.get('task_status') or 'N/A'}",
        f"- Project phase: {evidence.get('project_phase') or 'N/A'}",
    ]

    if include_floorplan:
        lines.append(
            f"- Floorplan image path: {evidence.get('floorplan_path') or 'N/A'}"
        )
    else:
        lines.append("- Floorplan image: not provided in this baseline")

    lines.extend(["", "### Chat log"])
    chat_history = evidence.get("chat_history") or []
    if chat_history:
        for item in chat_history:
            lines.append(f"- {item.get('role', 'Unknown')}: {item.get('text', '')}")
    else:
        lines.append("- N/A")

    lines.extend(
        [
            "",
            "## Candidates",
            "Compare the candidate descriptions against the evidence. Use only the provided candidates.",
            "",
        ]
    )

    for candidate in case_bundle["top_candidates"]:
        retrieval = candidate["retrieval_match"]
        lines.extend(
            [
                f"### {candidate['candidate_id']}",
                f"- GUID: {candidate['guid']}",
                f"- Draft description: {candidate['draft_description']}",
                f"- Retrieval cue: predicate={retrieval.get('query_predicate')} ref_type={retrieval.get('ref_type')} ref_storey={retrieval.get('ref_storey')}",
                f"- Slots to pay attention to: {', '.join(candidate.get('slots') or [])}",
                "",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def choose_sample(
    eligible: List[dict],
    sample_size: int,
    seed: int,
) -> List[dict]:
    rng = random.Random(seed)
    if sample_size >= len(eligible):
        return list(eligible)
    return rng.sample(eligible, sample_size)


def case_lookup(cases_path: Path) -> Dict[str, dict]:
    return {row["case_id"]: row for row in load_jsonl(cases_path)}


def load_model_contexts() -> Dict[str, ModelContext]:
    return {
        model_key: ModelContext(model_key, ifc_path)
        for model_key, ifc_path in MODEL_PATHS.items()
    }


def gather_eligible_cases(traces: List[dict]) -> List[dict]:
    eligible = []
    for trace in traces:
        gt_guid = trace.get("scenario", {}).get("ground_truth", {}).get("target_guid")
        if not gt_guid:
            continue
        rr_list = trace.get("internals", {}).get("retrieval_results", [])
        if not rr_list:
            continue

        pool = rr_list[0].get("candidates", [])
        if not pool:
            continue

        pool_guids = [cand.get("guid") for cand in pool]
        if gt_guid not in pool_guids:
            continue
        if pool_guids[:1] == [gt_guid]:
            continue

        eligible.append(trace)
    return eligible


def gt_pool_rank(trace: dict) -> Optional[int]:
    gt_guid = trace.get("scenario", {}).get("ground_truth", {}).get("target_guid")
    rr_list = trace.get("internals", {}).get("retrieval_results", [])
    if not gt_guid or not rr_list:
        return None
    pool_guids = [cand.get("guid") for cand in rr_list[0].get("candidates", [])]
    if gt_guid not in pool_guids:
        return None
    return pool_guids.index(gt_guid) + 1


def build_case_bundle(
    trace: dict,
    case_row: dict,
    model_contexts: Dict[str, ModelContext],
    top_k: int,
) -> Dict[str, Any]:
    case_id = trace["scenario_id"]
    model_key = case_row["ifc_model"]
    model_ctx = model_contexts[model_key]
    gt_guid = trace["scenario"]["ground_truth"]["target_guid"]

    constraints = trace.get("internals", {}).get("constraints", {})
    query_relation = first_spatial_relation(constraints)

    pool = trace["internals"]["retrieval_results"][0]["candidates"]
    pool_guids = [cand.get("guid") for cand in pool]
    gt_pool_rank = pool_guids.index(gt_guid) + 1

    top_candidates = []
    for index, candidate in enumerate(pool[:top_k], start=1):
        scaffold = model_ctx.build_candidate_scaffold(
            candidate=candidate,
            query_relation=query_relation,
            is_gt=(candidate.get("guid") == gt_guid),
            pool_rank=index,
        )
        scaffold["candidate_id"] = f"C{index}"
        top_candidates.append(scaffold)

    gt_candidate = None
    if gt_pool_rank > top_k:
        gt_candidate_raw = pool[gt_pool_rank - 1]
        gt_candidate = model_ctx.build_candidate_scaffold(
            candidate=gt_candidate_raw,
            query_relation=query_relation,
            is_gt=True,
            pool_rank=gt_pool_rank,
        )
        gt_candidate["candidate_id"] = f"GT@{gt_pool_rank}"

    floorplan_path = resolve_path(case_row.get("inputs", {}).get("floorplan_patch"))
    evidence = {
        "query_text": case_row.get("query_text"),
        "task_status": case_row.get("inputs", {}).get("project_context", {}).get("4d_task_status"),
        "project_phase": case_row.get("inputs", {}).get("project_context", {}).get("project_phase"),
        "chat_history": case_row.get("inputs", {}).get("chat_history", []),
        "floorplan_path": str(floorplan_path) if floorplan_path else None,
        "floorplan_path_relative": case_row.get("inputs", {}).get("floorplan_patch"),
    }

    bundle = {
        "case_id": case_id,
        "ifc_model": model_key,
        "ground_truth": trace["scenario"]["ground_truth"],
        "constraints": constraints,
        "pool_size": len(pool),
        "gt_pool_rank": gt_pool_rank,
        "gt_in_topk": gt_pool_rank <= top_k,
        "ground_truth_candidate_id": (
            f"C{gt_pool_rank}" if gt_pool_rank <= top_k else None
        ),
        "evidence": evidence,
        "top_candidates": top_candidates,
        "gt_candidate_if_outside_topk": gt_candidate,
    }
    bundle["prompts"] = {
        "chat_desc": build_verification_prompt(bundle, include_floorplan=False),
        "chat_floorplan_desc": build_verification_prompt(bundle, include_floorplan=True),
    }
    return bundle


def write_case_bundle(case_dir: Path, bundle: Dict[str, Any]) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_dir / "chat_desc_prompt.md").write_text(
        bundle["prompts"]["chat_desc"], encoding="utf-8"
    )
    (case_dir / "chat_floorplan_desc_prompt.md").write_text(
        bundle["prompts"]["chat_floorplan_desc"], encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--require-gt-in-topk",
        action="store_true",
        help="Only sample cases where GT is already within the raw top-K pool order.",
    )
    args = parser.parse_args()

    traces = load_jsonl(args.trace)
    cases = case_lookup(args.cases)
    eligible = gather_eligible_cases(traces)
    candidate_pool = eligible
    if args.require_gt_in_topk:
        candidate_pool = [
            trace for trace in eligible
            if (gt_pool_rank(trace) or 10**9) <= args.top_k
        ]
    sampled = choose_sample(candidate_pool, args.sample_size, args.seed)
    model_contexts = load_model_contexts()

    run_dir = (
        args.output_dir
        / (
            f"{args.trace.stem}_sample{args.sample_size}_seed{args.seed}_top{args.top_k}"
            + ("_gtin" if args.require_gt_in_topk else "")
        )
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    bundles = []
    for trace in sampled:
        case_id = trace["scenario_id"]
        case_row = cases[case_id]
        bundle = build_case_bundle(
            trace=trace,
            case_row=case_row,
            model_contexts=model_contexts,
            top_k=args.top_k,
        )
        bundles.append(bundle)
        write_case_bundle(run_dir / case_id, bundle)

    summary = {
        "trace_path": str(args.trace),
        "cases_path": str(args.cases),
        "eligible_case_count": len(eligible),
        "candidate_pool_count": len(candidate_pool),
        "sample_size": len(sampled),
        "seed": args.seed,
        "top_k": args.top_k,
        "require_gt_in_topk": args.require_gt_in_topk,
        "selected_case_ids": [bundle["case_id"] for bundle in bundles],
        "selected_predicates": Counter(
            (first_spatial_relation(bundle["constraints"]).get("predicate") or "NONE")
            for bundle in bundles
        ),
        "gt_in_topk_count": sum(1 for bundle in bundles if bundle["gt_in_topk"]),
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "bundles.json").write_text(
        json.dumps(bundles, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Eligible GT-in-pool-but-not-top1 cases: {len(eligible)}")
    if args.require_gt_in_topk:
        print(f"Filtered to GT-in-top-{args.top_k} cases: {len(candidate_pool)}")
    print(f"Selected sample: {len(sampled)} cases")
    print(f"GT in top-{args.top_k} among sampled cases: {summary['gt_in_topk_count']}/{len(sampled)}")
    print(f"Output written to: {run_dir}")


if __name__ == "__main__":
    main()
