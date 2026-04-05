#!/usr/bin/env python3
"""Graph-RAG shortlist reranking for AP held-out traces.

This experiment is intentionally read-only over existing AP traces plus Neo4j.
It does not rerun the extractor or planner; it only reranks an existing
shortlist with Gemini using graph-enriched candidate descriptions.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_graph_rag")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.analysis.group4_common import topology_family, universe_key

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None


DEFAULT_TRACE_JSONL = (
    PROJECT_ROOT
    / "output"
    / "lora6_v2_ap_20260331"
    / "ap_e2e_phase5_g7"
    / "g7_position_context"
    / "traces_20260404_132823_v2_lora_p0_union_p1.jsonl"
)
DEFAULT_CASES_JSONL = PROJECT_ROOT / "evaluation" / "cases" / "cases_ap_heldout_e2e.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TOP_K = 10

MODE_BASE_LABELS = {
    "g7_pipeline": "Full-topology (G7)",
    "p1_only": "P1-only (G7 coarse)",
}
MODE_RERANK_LABELS = {
    "g7_pipeline": "Full-topology (G7) + Graph-RAG rerank",
    "p1_only": "P1-only (G7 coarse) + Graph-RAG rerank",
}

REFERENCE_ROWS = [
    {"system": "G7 Position Context", "top10": 23.3, "top1": 3.3, "mrr10": 0.0681},
    {"system": "P1-only upper bound", "top10": 16.7, "top1": 0.0, "mrr10": 0.0392},
    {"system": "Oracle", "top10": 40.0, "top1": 5.0, "mrr10": 0.1279},
]


def _date_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _case_id_from_trace(trace: dict) -> str:
    return (
        trace.get("scenario", {}).get("id")
        or trace.get("scenario_id")
        or trace.get("case_id")
        or ""
    )


def _ordered_unique_candidates(candidates: Sequence[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for row in candidates:
        guid = row.get("guid")
        if not guid or guid in seen:
            continue
        seen.add(guid)
        out.append(dict(row))
    return out


def _trace_candidates(trace: dict) -> List[dict]:
    rr = trace.get("internals", {}).get("retrieval_results", [])
    if not rr:
        return []
    return _ordered_unique_candidates(rr[0].get("candidates", []) or [])


def _trace_constraints(trace: dict) -> dict:
    return trace.get("internals", {}).get("constraints", {}) or {}


def _flatten_chat(inputs: dict) -> str:
    query_text = inputs.get("query_text")
    if isinstance(query_text, str) and query_text.strip():
        return query_text.strip()
    chat = inputs.get("chat_history") or []
    if isinstance(chat, str):
        return chat.strip()
    lines: List[str] = []
    for msg in chat:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                lines.append(f"{role}: {content}")
        elif isinstance(msg, str):
            lines.append(msg)
    return "\n".join(lines).strip()


def _resolve_storey_aliases(storey_name: Optional[str]) -> List[str]:
    if not storey_name:
        return []
    raw = str(storey_name).strip().lower()
    aliases = [raw]
    m = re.match(r"^\s*(-?\d+)\s*$", raw)
    if m:
        num = m.group(1)
        aliases.extend([f"{num} -", f"level {num}"])
    return list(dict.fromkeys(a for a in aliases if a))


def _resolve_asset_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(str(raw))
    if p.is_absolute() and p.exists():
        return p
    candidates = [
        REPO_ROOT / p,
        PROJECT_ROOT / p,
    ]
    if str(p).startswith("datasets/"):
        candidates.append(REPO_ROOT / "data_curation" / p)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _encode_image(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")
    return {"inline_data": {"mime_type": mime, "data": data}}


def _init_gemini(model_name: str) -> Any:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(REPO_ROOT / ".env")
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "google-generativeai is required for Graph-RAG reranking."
        ) from exc
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GOOGLE_API_KEY or GEMINI_API_KEY before running Graph-RAG rerank.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=model_name,
        system_instruction=(
            "You are a careful BIM graph reranker. "
            "Use the site evidence and candidate graph fingerprints to rank candidates. "
            "Return only the ordered candidate letters from best to worst."
        ),
        generation_config=genai.GenerationConfig(
            temperature=0.0,
            max_output_tokens=64,
        ),
    )


def _complete_prefix_order(order: Sequence[str], valid_ids: Sequence[str]) -> List[str]:
    valid = [str(x) for x in valid_ids]
    seen = set()
    prefix: List[str] = []
    for item in order:
        key = str(item).strip()
        if not key or key not in valid or key in seen:
            continue
        seen.add(key)
        prefix.append(key)
    return prefix + [item for item in valid if item not in seen]


def _extract_letter_ids(text: str, valid_ids: Sequence[str]) -> List[str]:
    if not text:
        return []
    pattern = r"\b(" + "|".join(re.escape(item) for item in valid_ids) + r")\b"
    return re.findall(pattern, text)


def _parse_rerank_response(raw_text: str, valid_ids: Sequence[str]) -> Tuple[List[str], str, str]:
    text = (raw_text or "").strip()
    candidates: List[Any] = []
    if text:
        candidates.append(text)
        fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        candidates.extend(chunk.strip() for chunk in fenced if chunk.strip())

    parsed_objects: List[Any] = []
    for candidate in candidates:
        try:
            parsed_objects.append(json.loads(candidate))
        except Exception:
            pass

    for obj in parsed_objects:
        if isinstance(obj, dict):
            ordered = obj.get("ordered_ids")
            winner = obj.get("winner")
            reason = str(obj.get("reason") or "")
            if isinstance(ordered, list):
                order = [str(x).strip() for x in ordered if str(x).strip()]
                if _is_valid_order(order, valid_ids):
                    return order, str(winner or order[0]), reason
                completed = _complete_prefix_order(order, valid_ids)
                if completed and completed[0] in valid_ids:
                    return completed, str(winner or completed[0]), reason
        if isinstance(obj, list):
            ranked = []
            for item in obj:
                if isinstance(item, dict) and item.get("id"):
                    ranked.append(str(item["id"]).strip())
            if _is_valid_order(ranked, valid_ids):
                return ranked, ranked[0], ""
            completed = _complete_prefix_order(ranked, valid_ids)
            if completed and completed[0] in valid_ids:
                return completed, completed[0], ""

    regex_ids = _extract_letter_ids(text, valid_ids)
    if _is_valid_order(regex_ids, valid_ids):
        return regex_ids, regex_ids[0], ""
    completed = _complete_prefix_order(regex_ids, valid_ids)
    if completed and completed[0] in valid_ids:
        return completed, completed[0], ""
    return [], "", ""


def _is_valid_order(order: Sequence[str], valid_ids: Sequence[str]) -> bool:
    valid = list(valid_ids)
    if len(order) != len(valid):
        return False
    return sorted(order) == sorted(valid)


def _rank_of(guid: str, ordered_guids: Sequence[str]) -> int:
    try:
        return ordered_guids.index(guid) + 1
    except ValueError:
        return -1


def _mrr10(rank: int) -> float:
    return 1.0 / rank if 1 <= rank <= 10 else 0.0


def _label_constraints(case: dict) -> dict:
    return (case.get("labels") or {}).get("constraints") or {}


def _family_from_case(case: dict) -> str:
    rels = _label_constraints(case).get("spatial_relations", []) or []
    return topology_family(rels)


def _universe_from_case(case: dict) -> str:
    rels = _label_constraints(case).get("spatial_relations", []) or []
    return universe_key(rels)


def _coarse_fields(trace: dict, case: dict) -> Tuple[str, str, str]:
    constraints = _trace_constraints(trace)
    params = (
        trace.get("internals", {})
        .get("retrieval_results", [{}])[0]
        .get("query_plan_used", {})
        .get("params", {})
    )
    storey = (
        constraints.get("storey_name")
        or params.get("storey")
        or case.get("ground_truth", {}).get("target_storey")
        or ""
    )
    ifc_class = (
        constraints.get("ifc_class")
        or params.get("type")
        or params.get("subject_type")
        or case.get("ground_truth", {}).get("target_ifc_class")
        or ""
    )
    source = "constraints"
    if not constraints.get("storey_name") or not constraints.get("ifc_class"):
        source = "query_plan"
    if not ifc_class or not storey:
        source = "ground_truth_fallback"
    return str(storey), str(ifc_class), source


def _connect_graph(config_path: Path) -> Any:
    try:
        from py2neo import Graph  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit("py2neo is required for Graph-RAG reranking.") from exc

    cfg = _load_yaml(config_path)
    neo4j_cfg = cfg.get("neo4j", {}) or {}
    return Graph(
        neo4j_cfg.get("uri", "bolt://localhost:7687"),
        auth=(neo4j_cfg.get("user", "neo4j"), neo4j_cfg.get("password", "password")),
    )


def _query_candidate_contexts(graph: Any, guids: Sequence[str]) -> Dict[str, dict]:
    if not guids:
        return {}
    query = """
    UNWIND $guids AS guid
    MATCH (t:IFCElement {guid: guid})
    WHERE t.ifc_model = 'AP'
    OPTIONAL MATCH (t)-[:FILLS]->(host:IFCElement)
    WITH guid, t, head(collect(DISTINCT {
        guid: host.guid,
        ifc_type: host.ifc_type,
        name: host.name
    })) AS host
    OPTIONAL MATCH (t)-[r:NEXT_TO]->(nb:IFCElement)
    WITH guid, t, host, collect(DISTINCT {
        guid: nb.guid,
        ifc_type: nb.ifc_type,
        name: nb.name,
        direction: CASE
            WHEN t.wall_position_index IS NOT NULL
             AND nb.wall_position_index IS NOT NULL
             AND nb.wall_position_index < t.wall_position_index THEN 'left'
            WHEN t.wall_position_index IS NOT NULL
             AND nb.wall_position_index IS NOT NULL
             AND nb.wall_position_index > t.wall_position_index THEN 'right'
            ELSE ''
        END
    }) AS next_neighbors
    OPTIONAL MATCH (t)-[:CONNECTS_TO]->(ct:IFCElement)
    WITH guid, t, host, next_neighbors, collect(DISTINCT {
        guid: ct.guid,
        ifc_type: ct.ifc_type,
        name: ct.name
    }) AS connects_to
    OPTIONAL MATCH (t)-[:ADJACENT_TO]->(ad:IFCElement)
    RETURN guid,
           t.ifc_type AS ifc_type,
           t.storey AS storey,
           t.name AS name,
           t.object_type AS object_type,
           t.wall_position_index AS wall_position_index,
           t.wall_child_total AS wall_child_total,
           host,
           next_neighbors,
           connects_to,
           collect(DISTINCT {
               guid: ad.guid,
               ifc_type: ad.ifc_type,
               name: ad.name
           }) AS adjacent_to
    """
    out: Dict[str, dict] = {}
    for row in graph.run(query, guids=list(guids)).data():
        out[row["guid"]] = row
    return out


def _query_p1_pool(graph: Any, storey_name: str, ifc_class: str) -> List[dict]:
    aliases = _resolve_storey_aliases(storey_name)
    query = """
    MATCH (s:IFCStorey)-[:CONTAINS]->(e:IFCElement)
    WHERE e.ifc_model = 'AP'
      AND (e.ifc_type = $ifc_class OR e.ifc_type STARTS WITH $ifc_class)
      AND (
        size($storey_aliases) = 0
        OR ANY(alias IN $storey_aliases WHERE toLower(s.name) STARTS WITH alias)
      )
    RETURN e.guid AS guid,
           e.name AS name,
           e.ifc_type AS type,
           s.name AS storey,
           e.wall_position_index AS pos,
           e.wall_child_total AS wall_child_total
    ORDER BY coalesce(e.wall_position_index, 999999), guid
    """
    return [dict(row) for row in graph.run(query, storey_aliases=aliases, ifc_class=ifc_class).data()]


def _name_hint(name: str) -> str:
    if not name:
        return ""
    parts = [part.strip() for part in str(name).split(":") if part.strip()]
    if len(parts) >= 2:
        return parts[1]
    return str(name).strip()


def _slot_text(ctx: dict) -> str:
    pos = ctx.get("wall_position_index")
    total = ctx.get("wall_child_total")
    if pos is None:
        return ""
    if total is not None:
        return f"position {int(pos) + 1} of {int(total)}"
    return f"position {int(pos) + 1}"


def _compact_neighbor(label: str, neighbors: Sequence[dict]) -> str:
    if not neighbors:
        return ""
    names = []
    for row in neighbors[:2]:
        type_name = str(row.get("ifc_type") or "?")
        hint = _name_hint(str(row.get("name") or ""))
        names.append(f"{type_name} ({hint})" if hint else type_name)
    return f"{label}: " + "; ".join(names)


def _format_candidate_description(letter: str, ctx: dict, fallback: dict) -> str:
    ifc_type = str(ctx.get("ifc_type") or fallback.get("type") or fallback.get("ref_type") or "?")
    storey = str(ctx.get("storey") or fallback.get("storey") or fallback.get("ref_storey") or "?")
    hint = _name_hint(str(ctx.get("name") or fallback.get("name") or ""))
    host = ctx.get("host") or {}
    host_hint = _name_hint(str(host.get("name") or ""))
    slot = _slot_text(ctx)

    left = [row for row in (ctx.get("next_neighbors") or []) if row.get("direction") == "left"]
    right = [row for row in (ctx.get("next_neighbors") or []) if row.get("direction") == "right"]
    connects = list(ctx.get("connects_to") or [])[:2]
    adjacent = list(ctx.get("adjacent_to") or [])[:2]

    fields: List[str] = [f"{letter}. {ifc_type} on {storey}"]
    if hint:
        fields.append(hint)
    if host_hint:
        host_type = str(host.get("ifc_type") or "")
        host_text = f"{host_type} ({host_hint})" if host_type else host_hint
        fields.append(f"host: {host_text}")
    if slot:
        fields.append(slot)
    if left:
        fields.append(_compact_neighbor("left", left))
    if right:
        fields.append(_compact_neighbor("right", right))
    if not left and not right and connects:
        fields.append(_compact_neighbor("connects", connects))
    if not left and not right and not connects and adjacent:
        fields.append(_compact_neighbor("adjacent", adjacent))
    return "; ".join(x for x in fields if x)


def _build_prompt(case: dict, descriptions: Sequence[str], letters: Sequence[str]) -> str:
    query_text = case.get("query_text") or _flatten_chat(case.get("inputs") or {})
    example = " ".join(letters[: min(len(letters), 8)])
    return (
        "Match the construction-site evidence to the best BIM candidate.\n\n"
        "Use the site photo, floorplan patch, and query text. "
        "Prefer candidates whose type, storey, host wall, slot position, and left/right neighbors best match the evidence.\n\n"
        f"Query:\n{query_text}\n\n"
        "Candidates:\n"
        + "\n".join(descriptions)
        + "\n\nReturn only the ranked candidate letters from best to worst, separated by spaces.\n"
        f"Example: {example}\n"
        "Do not return JSON. Do not explain."
    )


def _call_gemini(
    model: Any,
    prompt_text: str,
    site_images: Sequence[Path],
    floorplan_path: Optional[Path],
) -> str:
    parts: List[dict] = []
    for image_path in site_images:
        encoded = _encode_image(image_path)
        if encoded is not None:
            parts.append(encoded)
    if floorplan_path is not None:
        encoded = _encode_image(floorplan_path)
        if encoded is not None:
            parts.append(encoded)
    parts.append({"text": prompt_text})
    response = model.generate_content([{"role": "user", "parts": parts}])
    try:
        text = (response.text or "").strip()
    except Exception:
        text = ""
    if text:
        return text
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            value = getattr(part, "text", None)
            if value:
                return str(value).strip()
    return ""


def _metrics(rows: Sequence[Dict[str, Any]], rank_key: str) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "gt_in_pool_pct": 0.0,
            "top10_pct": 0.0,
            "top1_pct": 0.0,
            "mrr10": 0.0,
            "avg_pool": 0.0,
        }
    gt_in_pool = sum(1 for row in rows if row.get("gt_in_pool"))
    top10 = sum(1 for row in rows if 1 <= int(row.get(rank_key, -1)) <= 10)
    top1 = sum(1 for row in rows if int(row.get(rank_key, -1)) == 1)
    mrr10 = sum(_mrr10(int(row.get(rank_key, -1))) for row in rows) / n
    avg_pool = mean(float(row.get("pool_size", 0)) for row in rows)
    return {
        "n": n,
        "gt_in_pool_pct": round(gt_in_pool / n * 100, 1),
        "top10_pct": round(top10 / n * 100, 1),
        "top1_pct": round(top1 / n * 100, 1),
        "mrr10": round(mrr10, 4),
        "avg_pool": round(avg_pool, 1),
    }


def _subset_rows(rows: Sequence[Dict[str, Any]], mode: str, subset: str, top_k: int) -> List[Dict[str, Any]]:
    target = [row for row in rows if row.get("mode") == mode]
    if subset == "all":
        return target
    if subset == "topk_not_top1_before":
        return [row for row in target if 2 <= int(row.get("base_rank", -1)) <= top_k]
    return target


def _family_rows(rows: Sequence[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    target = [row for row in rows if row.get("mode") == mode]
    families = sorted({str(row.get("family") or "") for row in target})
    out = []
    for family in families:
        fam_rows = [row for row in target if row.get("family") == family]
        before = _metrics(fam_rows, "base_rank")
        after = _metrics(fam_rows, "reranked_rank")
        out.append(
            {
                "mode": mode,
                "family": family,
                "n": before["n"],
                "base_top10": before["top10_pct"],
                "base_top1": before["top1_pct"],
                "base_mrr10": before["mrr10"],
                "reranked_top10": after["top10_pct"],
                "reranked_top1": after["top1_pct"],
                "reranked_mrr10": after["mrr10"],
            }
        )
    return out


def _plot_comparison(path: Path, rows: Sequence[Dict[str, Any]], top_k: int) -> None:
    systems = [row["system"] for row in rows]
    top10 = [row["top10"] for row in rows]
    top1 = [row["top1"] for row in rows]
    mrr = [row["mrr10"] for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ["#0F766E", "#14B8A6", "#7C3AED", "#A78BFA", "#D97706", "#F59E0B"]

    for ax, values, title, fmt in (
        (axes[0], top10, "Top-10", "{:.1f}%"),
        (axes[1], top1, "Top-1", "{:.1f}%"),
        (axes[2], mrr, "MRR@10", "{:.4f}"),
    ):
        bars = ax.bar(range(len(systems)), values, color=colors[: len(systems)])
        ax.set_title(title)
        ax.set_xticks(range(len(systems)))
        ax.set_xticklabels(systems, rotation=24, ha="right")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Graph-RAG reranking at top-{top_k} vs G7 / P1 / Oracle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_mode(
    *,
    mode: str,
    trace_rows: Sequence[dict],
    cases_map: Dict[str, dict],
    graph: Any,
    gemini_model: Optional[Any],
    top_k: int,
    sleep_seconds: float,
    skip_gemini: bool,
    limit: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rows = list(trace_rows[:limit] if limit > 0 else trace_rows)

    for idx, trace in enumerate(rows, start=1):
        case_id = _case_id_from_trace(trace)
        case = cases_map.get(case_id)
        if not case:
            continue
        gt_guid = str(case.get("ground_truth", {}).get("target_guid") or "")
        base_pool = _trace_candidates(trace)
        storey_name, ifc_class, coarse_source = _coarse_fields(trace, case)
        if mode == "g7_pipeline":
            pool = list(base_pool)
        elif mode == "p1_only":
            pool = _query_p1_pool(graph, storey_name, ifc_class)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ordered_guids = [str(row.get("guid") or "") for row in pool if row.get("guid")]
        pool_size = len(ordered_guids)
        gt_in_pool = gt_guid in ordered_guids
        base_rank = _rank_of(gt_guid, ordered_guids)

        topk_guids = ordered_guids[:top_k]
        contexts = _query_candidate_contexts(graph, topk_guids)
        letters = [chr(ord("A") + i) for i in range(len(topk_guids))]
        letter_to_guid = dict(zip(letters, topk_guids))
        guid_to_letter = {guid: letter for letter, guid in letter_to_guid.items()}
        descriptions = []
        for guid in topk_guids:
            letter = guid_to_letter[guid]
            ctx = contexts.get(guid, {})
            fallback = next((row for row in pool if row.get("guid") == guid), {})
            descriptions.append(_format_candidate_description(letter, ctx, fallback))

        query_text = case.get("query_text") or _flatten_chat(case.get("inputs") or {})
        input_images = [
            path
            for path in (
                _resolve_asset_path(raw)
                for raw in (case.get("inputs") or {}).get("images", []) or []
            )
            if path is not None
        ]
        floorplan_path = _resolve_asset_path((case.get("inputs") or {}).get("floorplan_patch"))
        prompt_text = _build_prompt(
            {
                "query_text": query_text,
                "inputs": case.get("inputs") or {},
            },
            descriptions,
            letters,
        )

        raw_output = ""
        rerank_failed = False
        reason = ""
        winner = ""
        ordered_ids: List[str] = []

        if skip_gemini:
            ordered_ids = list(letters)
            rerank_failed = True
            reason = "SKIPPED_GEMINI"
            winner = ordered_ids[0] if ordered_ids else ""
        else:
            try:
                raw_output = _call_gemini(gemini_model, prompt_text, input_images, floorplan_path)
                ordered_ids, winner, reason = _parse_rerank_response(raw_output, letters)
                if not ordered_ids:
                    ordered_ids = list(letters)
                    rerank_failed = True
            except Exception as exc:  # pragma: no cover - runtime/network dependent
                raw_output = f"ERROR: {exc}"
                ordered_ids = list(letters)
                rerank_failed = True
            if idx < len(rows) and sleep_seconds > 0:
                time.sleep(sleep_seconds)

        reranked_topk = [letter_to_guid[item] for item in ordered_ids]
        reranked_guids = reranked_topk + ordered_guids[top_k:]
        reranked_rank = _rank_of(gt_guid, reranked_guids)

        family = _family_from_case(case)
        universe = _universe_from_case(case)
        out.append(
            {
                "mode": mode,
                "case_id": case_id,
                "family": family,
                "universe": universe,
                "gt_guid": gt_guid,
                "storey_name": storey_name,
                "ifc_class": ifc_class,
                "coarse_source": coarse_source,
                "pool_size": pool_size,
                "gt_in_pool": gt_in_pool,
                "base_rank": base_rank,
                "reranked_rank": reranked_rank,
                "improved": (reranked_rank > 0 and base_rank > 0 and reranked_rank < base_rank),
                "became_top1": (base_rank != 1 and reranked_rank == 1),
                "rerank_failed": rerank_failed,
                "winner": winner,
                "reason": reason,
                "top_k": top_k,
                "topk_letters": letters,
                "topk_guids": topk_guids,
                "reranked_topk_guids": reranked_topk,
                "query_text": query_text,
                "raw_output": raw_output,
            }
        )
        print(
            f"[{mode} {idx:02d}/{len(rows)}] {case_id} "
            f"base={base_rank:>3} rerank={reranked_rank:>3} pool={pool_size:>3} "
            f"top1={reranked_rank == 1}"
        )

    return out


def _build_summary(rows: Sequence[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "meta": {
            "top_k": top_k,
            "mode_base_labels": MODE_BASE_LABELS,
            "mode_rerank_labels": MODE_RERANK_LABELS,
        },
        "modes": {},
        "subsets": {},
        "families": {},
    }
    for mode in ("g7_pipeline", "p1_only"):
        mode_rows = [row for row in rows if row.get("mode") == mode]
        summary["modes"][mode] = {
            "baseline": _metrics(mode_rows, "base_rank"),
            "reranked": _metrics(mode_rows, "reranked_rank"),
            "rerank_failed_cases": sum(1 for row in mode_rows if row.get("rerank_failed")),
            "improved_cases": sum(1 for row in mode_rows if row.get("improved")),
            "became_top1_cases": sum(1 for row in mode_rows if row.get("became_top1")),
        }
        if mode == "g7_pipeline":
            subset_rows = _subset_rows(rows, mode, "topk_not_top1_before", top_k)
            summary["subsets"]["g7_topk_not_top1_before"] = {
                "baseline": _metrics(subset_rows, "base_rank"),
                "reranked": _metrics(subset_rows, "reranked_rank"),
                "n": len(subset_rows),
            }
        if mode == "p1_only":
            subset_rows = _subset_rows(rows, mode, "topk_not_top1_before", top_k)
            summary["subsets"]["p1_topk_not_top1_before"] = {
                "baseline": _metrics(subset_rows, "base_rank"),
                "reranked": _metrics(subset_rows, "reranked_rank"),
                "n": len(subset_rows),
            }
        summary["families"][mode] = _family_rows(rows, mode)
    return summary


def _write_summary_md(path: Path, summary: Dict[str, Any], comparison_rows: Sequence[Dict[str, Any]], top_k: int) -> None:
    lines = [
        "# Graph-RAG Rerank Summary",
        "",
        "## Main Comparison",
        "",
        "| System | Top-10 | Top-1 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['system']} | {row['top10']:.1f}% | {row['top1']:.1f}% | {row['mrr10']:.4f} |"
        )

    subset = summary.get("subsets", {}).get("g7_topk_not_top1_before", {})
    lines.extend(
        [
            "",
            f"## Target Subset: Full-topology (G7) Top-{top_k} But Not Top-1",
            "",
            f"- Cases: {subset.get('n', 0)}",
            f"- Baseline Top-1: {subset.get('baseline', {}).get('top1_pct', 0.0):.1f}%",
            f"- Reranked Top-1: {subset.get('reranked', {}).get('top1_pct', 0.0):.1f}%",
            f"- Baseline MRR@10: {subset.get('baseline', {}).get('mrr10', 0.0):.4f}",
            f"- Reranked MRR@10: {subset.get('reranked', {}).get('mrr10', 0.0):.4f}",
            "",
            f"## Target Subset: P1-only (G7 coarse) Top-{top_k} But Not Top-1",
            "",
            f"- Cases: {summary.get('subsets', {}).get('p1_topk_not_top1_before', {}).get('n', 0)}",
            f"- Baseline Top-1: {summary.get('subsets', {}).get('p1_topk_not_top1_before', {}).get('baseline', {}).get('top1_pct', 0.0):.1f}%",
            f"- Reranked Top-1: {summary.get('subsets', {}).get('p1_topk_not_top1_before', {}).get('reranked', {}).get('top1_pct', 0.0):.1f}%",
            f"- Baseline MRR@10: {summary.get('subsets', {}).get('p1_topk_not_top1_before', {}).get('baseline', {}).get('mrr10', 0.0):.4f}",
            f"- Reranked MRR@10: {summary.get('subsets', {}).get('p1_topk_not_top1_before', {}).get('reranked', {}).get('mrr10', 0.0):.4f}",
            "",
            "## Per-Family Breakdown",
            "",
            "| Mode | Family | n | Base Top-1 | Reranked Top-1 | Base MRR@10 | Reranked MRR@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, rows in summary.get("families", {}).items():
        for row in rows:
            lines.append(
                f"| {mode} | {row['family']} | {row['n']} | {row['base_top1']:.1f}% | "
                f"{row['reranked_top1']:.1f}% | {row['base_mrr10']:.4f} | {row['reranked_mrr10']:.4f} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACE_JSONL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--mode", choices=["all", "g7_pipeline", "p1_only"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "lora6_v2_ap_20260331" / "graph_rag_rerank" / _date_tag(),
    )
    args = parser.parse_args()

    trace_rows = _load_jsonl(args.trace_jsonl)
    trace_rows = [row for row in trace_rows if _case_id_from_trace(row)]
    cases_map = {row["case_id"]: row for row in _load_jsonl(args.cases)}
    graph = _connect_graph(args.config)
    gemini_model = None if args.skip_gemini else _init_gemini(args.model)

    modes = ["g7_pipeline", "p1_only"] if args.mode == "all" else [args.mode]
    all_rows: List[Dict[str, Any]] = []
    for mode in modes:
        all_rows.extend(
            _run_mode(
                mode=mode,
                trace_rows=trace_rows,
                cases_map=cases_map,
                graph=graph,
                gemini_model=gemini_model,
                top_k=args.top_k,
                sleep_seconds=args.sleep_seconds,
                skip_gemini=args.skip_gemini,
                limit=args.limit,
            )
        )

    summary = _build_summary(all_rows, args.top_k)
    comparison_rows = [
        {
            "system": MODE_BASE_LABELS["g7_pipeline"],
            "top10": summary.get("modes", {}).get("g7_pipeline", {}).get("baseline", {}).get("top10_pct", 0.0),
            "top1": summary.get("modes", {}).get("g7_pipeline", {}).get("baseline", {}).get("top1_pct", 0.0),
            "mrr10": summary.get("modes", {}).get("g7_pipeline", {}).get("baseline", {}).get("mrr10", 0.0),
        },
        {
            "system": MODE_RERANK_LABELS["g7_pipeline"],
            "top10": summary.get("modes", {}).get("g7_pipeline", {}).get("reranked", {}).get("top10_pct", 0.0),
            "top1": summary.get("modes", {}).get("g7_pipeline", {}).get("reranked", {}).get("top1_pct", 0.0),
            "mrr10": summary.get("modes", {}).get("g7_pipeline", {}).get("reranked", {}).get("mrr10", 0.0),
        },
        {
            "system": MODE_BASE_LABELS["p1_only"],
            "top10": summary.get("modes", {}).get("p1_only", {}).get("baseline", {}).get("top10_pct", 0.0),
            "top1": summary.get("modes", {}).get("p1_only", {}).get("baseline", {}).get("top1_pct", 0.0),
            "mrr10": summary.get("modes", {}).get("p1_only", {}).get("baseline", {}).get("mrr10", 0.0),
        },
        {
            "system": MODE_RERANK_LABELS["p1_only"],
            "top10": summary.get("modes", {}).get("p1_only", {}).get("reranked", {}).get("top10_pct", 0.0),
            "top1": summary.get("modes", {}).get("p1_only", {}).get("reranked", {}).get("top1_pct", 0.0),
            "mrr10": summary.get("modes", {}).get("p1_only", {}).get("reranked", {}).get("mrr10", 0.0),
        },
        *REFERENCE_ROWS,
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "graph_rag_rerank_results.jsonl", all_rows)
    csv_fields = [
        "mode",
        "case_id",
        "family",
        "universe",
        "gt_guid",
        "storey_name",
        "ifc_class",
        "coarse_source",
        "pool_size",
        "gt_in_pool",
        "base_rank",
        "reranked_rank",
        "improved",
        "became_top1",
        "rerank_failed",
        "winner",
        "reason",
    ]
    _write_csv(args.out_dir / "graph_rag_rerank_results.csv", all_rows, csv_fields)
    _write_json(args.out_dir / "graph_rag_rerank_summary.json", summary)
    _write_summary_md(args.out_dir / "graph_rag_rerank_summary.md", summary, comparison_rows, args.top_k)
    _plot_comparison(args.out_dir / "graph_rag_rerank_comparison.png", comparison_rows, args.top_k)

    print(f"\nSaved results to: {args.out_dir}")


if __name__ == "__main__":
    main()
