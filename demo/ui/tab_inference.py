"""
Tab 1 — Live Inference: upload images + enter chat → G4-Ultimate VLM extracts constraints
→ QueryPlanner builds plans → RetrievalBackend executes → 3D viewer shows pool.

Full neuro-symbolic pipeline: VLM (Modal GPU) → Constraints → Cypher → GUIDs → 3D.
Two modes: Full Topology (P0 spatial + P1 fallback) and P1+Rerank (storey+type first).
"""
import asyncio
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st
import streamlit.components.v1 as components

# IFC model code -> static path used by demo static server (for 3D viewer URL).
_IFC_VIEWER_REL_PATH = {
    "AP": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc",
    "BH": "data/ifc/BasicHouse.ifc",
    "DXA": "data/ifc/Duplex_A_20110505.ifc",
}


def _kv_pill(key: str, value: str, color: str = "#3b82f6") -> str:
    """Render a key-value pill."""
    v_display = value or '<span style="color:#475569;">null</span>'
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;'
        f'padding:3px 10px;margin:2px 4px 2px 0;background:rgba(30,41,59,0.8);'
        f'border:1px solid #334155;border-radius:6px;font-size:0.82em;font-family:monospace;">'
        f'<span style="color:#94a3b8;">{key}:</span>'
        f'<span style="color:{color};">{v_display}</span></span>'
    )


def _modality_badge(label: str, active: bool, color: str) -> str:
    """Render a compact modality badge for the live demo input summary."""
    bg = color if active else "#e2e8f0"
    fg = "white" if active else "#94a3b8"
    strike = "text-decoration:line-through;" if not active else ""
    return (
        f"<span style='background:{bg};color:{fg};padding:3px 11px;"
        f"border-radius:12px;font-size:0.82em;font-weight:600;"
        f"margin-right:6px;{strike}'>{label}</span>"
    )


def _render_live_modalities(live_inputs: dict | None = None) -> None:
    """Show a compact badge row for the inputs used in the current run."""
    live_inputs = live_inputs or st.session_state.get("last_live_inputs") or {}
    has_chat = bool((live_inputs.get("chat_text") or "").strip())
    n_images = len(live_inputs.get("photo_paths") or [])
    if live_inputs.get("floorplan_path"):
        n_images += 1
    has_4d = bool((live_inputs.get("task_status") or "").strip())
    badges = (
        _modality_badge("Chat", has_chat, "#22c55e")
        + _modality_badge(f"{n_images} image{'s' if n_images != 1 else ''}", n_images > 0, "#3b82f6")
        + _modality_badge("4D", has_4d, "#8b5cf6")
    )
    st.markdown(
        f"<div style='margin-bottom:8px;'>{badges}</div>",
        unsafe_allow_html=True,
    )


def _build_vlm_relation_graph_dot(parsed: dict) -> str:
    """Return a compact Graphviz DOT view of the VLM spatial relations."""
    target_type = parsed.get("ifc_class") or "Target"
    rels = parsed.get("spatial_relations") or []

    def _node_id(prefix: str, idx: int) -> str:
        return f"{prefix}_{idx}"

    lines = [
        "digraph VLMRelations {",
        '  rankdir=LR;',
        '  graph [bgcolor="transparent", pad="0.2", nodesep="0.5", ranksep="0.8"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#334155"];',
        '  edge [fontname="Helvetica", fontsize=10, color="#64748b"];',
        f'  target [label="{target_type}", fillcolor="#dbeafe", color="#2563eb", penwidth=1.6];',
    ]

    if not rels:
        lines.append('  empty [label="No spatial relations", fillcolor="#f8fafc", color="#cbd5e1"];')
        lines.append('  target -> empty [label="attribute-only", style="dashed"];')
        lines.append("}")
        return "\n".join(lines)

    for idx, rel in enumerate(rels):
        obj_label = rel.get("object_type") or "Object"
        predicate = rel.get("predicate", "?")
        extras: list[str] = []
        if rel.get("direction"):
            extras.append(str(rel["direction"]))
        if rel.get("object_material"):
            extras.append(str(rel["object_material"]))
        if rel.get("object_subtype"):
            extras.append(str(rel["object_subtype"]))
        edge_label = predicate
        if extras:
            edge_label += "\\n" + "\\n".join(extras[:2])
        obj_id = _node_id("obj", idx)
        fill = "#fef3c7" if predicate == "FILLS" else "#f8fafc"
        border = "#f59e0b" if predicate == "FILLS" else "#94a3b8"
        lines.append(
            f'  {obj_id} [label="{obj_label}", fillcolor="{fill}", color="{border}"];'
        )
        lines.append(f'  target -> {obj_id} [label="{edge_label}"];')

    lines.append("}")
    return "\n".join(lines)


def _collect_backend_queries(retrieval_result: dict | None) -> list[dict]:
    """Flatten executed Cypher queries across plans for backend inspection."""
    plans = (retrieval_result or {}).get("plans") or []
    results = (retrieval_result or {}).get("results") or []
    payloads: list[dict] = []
    for idx, plan in enumerate(plans):
        result = results[idx] if idx < len(results) else None
        queries = list((result or {}).get("executed_queries") or [])
        if not queries:
            continue
        payloads.append(
            {
                "priority": plan.get("priority", "?"),
                "strategy": plan.get("strategy", "?"),
                "queries": queries,
                "pool_size": (result or {}).get("pool_size", 0),
                "is_winner": idx == (retrieval_result or {}).get("winning_index"),
            }
        )
    return payloads


def _step_header(num: int, title: str, hint: str = "") -> None:
    """Render a numbered step header with strong visual hierarchy."""
    hint_html = (
        f'<span style="color:#94a3b8;font-weight:400;font-size:0.78em;margin-left:10px;">{hint}</span>'
        if hint else ""
    )
    st.markdown(
        f'<div style="margin:18px 0 8px 0;display:flex;align-items:center;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:22px;height:22px;border-radius:50%;background:#3b82f6;color:#fff;'
        f'font-size:0.75em;font-weight:700;margin-right:10px;">{num}</span>'
        f'<span style="font-size:1.02em;font-weight:600;color:#0f172a;">{title}</span>'
        f'{hint_html}</div>',
        unsafe_allow_html=True,
    )


def _render_backend_panel(vlm_result: dict, retrieval_result: dict | None) -> None:
    """Inspector panel: raw parsed JSON, executed Cypher, and full retrieval payload."""
    parsed = vlm_result.get("parsed") or {}
    query_payloads = _collect_backend_queries(retrieval_result)

    with st.container(border=True):
        st.markdown(
            '<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.5px;color:#64748b;margin:0 0 8px 0;">Inspect</p>',
            unsafe_allow_html=True,
        )
        backend_tabs = st.tabs(["Parsed JSON", "Cypher", "Retrieval JSON"])
        with backend_tabs[0]:
            st.code(json.dumps(parsed, indent=2, ensure_ascii=True), language="json")
        with backend_tabs[1]:
            if not query_payloads:
                st.info("No captured Cypher text yet. Live runs with recorded execution details will appear here.")
            else:
                for payload in query_payloads:
                    winner_suffix = " · winner" if payload["is_winner"] else ""
                    with st.expander(
                        f"P{payload['priority']} · {payload['strategy']} · {payload['pool_size']} candidates{winner_suffix}",
                        expanded=payload["is_winner"],
                    ):
                        for q_idx, query in enumerate(payload["queries"], 1):
                            if len(payload["queries"]) > 1:
                                st.caption(f"Query {q_idx}")
                            st.code(query, language="cypher")
        with backend_tabs[2]:
            if retrieval_result is None:
                st.info("Run retrieval to inspect the backend response payload.")
            else:
                st.code(json.dumps(retrieval_result, indent=2, ensure_ascii=True), language="json")


# ══════════════════════════════════════════════════════════════════════════════
# Live inference layout — chat thread (left) · stepper + canvas + drawer (right)
# ══════════════════════════════════════════════════════════════════════════════

def _stepper_chip(num: int, title: str, value_html: str, status: str = "idle") -> str:
    """Single chip in the top stepper strip."""
    palette = {
        "idle":   {"border": "#e2e8f0", "bg": "#f8fafc", "title": "#94a3b8"},
        "active": {"border": "#3b82f6", "bg": "#eff6ff", "title": "#0f172a"},
        "winner": {"border": "#15803d", "bg": "#f0fdf4", "title": "#0f172a"},
        "empty":  {"border": "#dc2626", "bg": "#fef2f2", "title": "#0f172a"},
    }.get(status, {"border": "#e2e8f0", "bg": "#f8fafc", "title": "#94a3b8"})
    return (
        f"<div style='flex:1;border:1px solid {palette['border']};background:{palette['bg']};"
        f"border-radius:8px;padding:10px 12px;min-height:78px;'>"
        f"<div style='font-size:11px;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.4px;color:#64748b;margin-bottom:6px;display:flex;align-items:center;'>"
        f"<span style='display:inline-flex;align-items:center;justify-content:center;"
        f"width:18px;height:18px;border-radius:50%;background:{palette['border']};color:#fff;"
        f"font-size:0.7em;font-weight:700;margin-right:8px;'>{num}</span>{title}</div>"
        f"<div style='font-size:0.88em;color:{palette['title']};line-height:1.4;'>{value_html}</div>"
        f"</div>"
    )


def _render_stepper(vlm_result: dict | None, retrieval_result: dict | None) -> None:
    """Top horizontal stepper: Constraints → Cypher → Pool → Rerank. Always visible."""
    parsed = (vlm_result or {}).get("parsed") or {}
    plans = (retrieval_result or {}).get("plans") or []
    results = (retrieval_result or {}).get("results") or []
    winning_idx = (retrieval_result or {}).get("winning_index")
    perception = (vlm_result or {}).get("perception") or {}
    rerank = (retrieval_result or {}).get("rerank") or None

    # Step 1 — Constraints (now also shows perception signals when present)
    if vlm_result is None:
        c1 = _stepper_chip(1, "Constraints", "<span style='color:#94a3b8;'>—</span>", "idle")
    else:
        rels = parsed.get("spatial_relations") or []
        ifc_class = parsed.get("ifc_class") or "?"
        storey = parsed.get("storey_name") or "?"
        rel_summary = (
            f"<code>{rels[0].get('predicate', '?')}</code> {rels[0].get('object_type', '?')}"
            if rels else "<i style='color:#94a3b8;'>attribute-only</i>"
        )
        extra_bits: list[str] = []
        if parsed.get("size_band"):
            extra_bits.append(f"<span style='color:#0f766e;'>band {parsed['size_band']}</span>")
        if parsed.get("position_context"):
            pc = str(parsed["position_context"])
            extra_bits.append(f"<span style='color:#7c3aed;'>{pc[:24]}{'…' if len(pc) > 24 else ''}</span>")
        extra_html = (" · ".join(extra_bits)) if extra_bits else ""
        c1 = _stepper_chip(
            1, "Constraints",
            f"<code>{ifc_class}</code> · {storey}<br>{rel_summary}"
            + (f"<br>{extra_html}" if extra_html else ""),
            "active",
        )

    # Step 2 — Cypher
    if not plans:
        c2 = _stepper_chip(2, "Cypher", "<span style='color:#94a3b8;'>—</span>", "idle")
    elif winning_idx is not None and winning_idx < len(plans):
        win_plan = plans[winning_idx]
        c2 = _stepper_chip(
            2, "Cypher",
            f"<code>P{win_plan.get('priority', '?')}</code> · {win_plan.get('strategy', '?')}<br>"
            f"<span style='color:#15803d;font-weight:600;'>winner</span> · {len(plans)} plans tried",
            "winner",
        )
    else:
        c2 = _stepper_chip(
            2, "Cypher",
            f"{len(plans)} plans · <span style='color:#dc2626;'>no winner</span>",
            "empty",
        )

    # Step 3 — Pool: show the relaxation ladder of plan pool sizes, ending
    # with the winning plan's count and a P0∪P1 annotation when the backend
    # appended P1-only candidates onto a strict P0 hit (raw < pool_size).
    if winning_idx is not None and winning_idx < len(results):
        win = results[winning_idx]
        pool_size = win.get("pool_size", 0)
        raw = win.get("raw_pool_size")
        latency = (retrieval_result or {}).get("_latency_ms", 0)

        # Build ladder up to the winner. Each rung = "P{priority}:{count}".
        ladder_bits: list[str] = []
        for i in range(winning_idx + 1):
            plan_i = plans[i] if i < len(plans) else {}
            res_i = results[i] if i < len(results) else {}
            prio = plan_i.get("priority", "?")
            count = (res_i or {}).get("pool_size", 0)
            if i == winning_idx:
                ladder_bits.append(
                    f"<code style='color:#15803d;font-weight:700;'>P{prio}:{count}</code>"
                )
            else:
                ladder_bits.append(
                    f"<span style='color:#94a3b8;'><code>P{prio}:{count}</code></span>"
                )
        ladder_html = " → ".join(ladder_bits)

        # Suffix: "(pool_size) P0∪P1" when the winner is a P0 strategy that
        # was union-merged with P1; otherwise "(pool_size)".
        win_strategy = (plans[winning_idx] or {}).get("strategy", "")
        is_p0 = win_strategy in ("spatial_triplet", "continuous_span")
        if is_p0 and raw is not None and raw != pool_size:
            suffix_html = (
                f"<span style='color:#0f172a;font-weight:600;'>({pool_size})</span> "
                f"<span style='color:#7c3aed;font-weight:600;'>P0∪P1</span>"
            )
        else:
            suffix_html = f"<span style='color:#0f172a;font-weight:600;'>({pool_size})</span>"

        c3 = _stepper_chip(
            3, "Pool",
            f"{ladder_html}<br>{suffix_html} · "
            f"<span style='color:#64748b;'>{latency} ms</span>",
            "winner",
        )
    elif retrieval_result is not None:
        c3 = _stepper_chip(3, "Pool", "<span style='color:#dc2626;'>0 candidates</span>", "empty")
    else:
        c3 = _stepper_chip(3, "Pool", "<span style='color:#94a3b8;'>—</span>", "idle")

    # Step 4 — Rerank
    if rerank is None:
        if retrieval_result is not None:
            c4 = _stepper_chip(
                4, "Rerank",
                "<span style='color:#94a3b8;'>off</span>",
                "idle",
            )
        else:
            c4 = _stepper_chip(4, "Rerank", "<span style='color:#94a3b8;'>—</span>", "idle")
    elif rerank.get("failed"):
        reason = str(rerank.get("reason") or "failed")[:32]
        c4 = _stepper_chip(
            4, "Rerank",
            f"<span style='color:#dc2626;font-weight:600;'>failed</span><br>"
            f"<span style='color:#64748b;'>{reason}</span>",
            "empty",
        )
    else:
        fusions = list((rerank.get("fusion_scores") or {}).values())
        top_fusion = max(fusions) if fusions else 0.0
        winner_guid = (rerank.get("winner_guid") or "")
        winner_short = winner_guid[:10] + "…" if winner_guid else "—"
        c4 = _stepper_chip(
            4, "Rerank",
            f"<span style='color:#15803d;font-weight:600;'>{winner_short}</span><br>"
            f"fusion <code>{top_fusion:.2f}</code>",
            "winner",
        )

    st.markdown(
        f"<div style='display:flex;gap:10px;margin-bottom:14px;'>{c1}{c2}{c3}{c4}</div>",
        unsafe_allow_html=True,
    )


def _render_canvas(
    vlm_result: dict | None,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
) -> None:
    """Hero 3D canvas with the grounded element + pool highlighted."""
    approved_guid = st.session_state.get("approved_guid", "")
    if not approved_guid:
        approved_guid = _top_candidate_guid(retrieval_result)

    if not approved_guid:
        st.markdown(
            "<div style='border:1px dashed #cbd5e1;border-radius:8px;"
            "padding:140px 20px;text-align:center;color:#94a3b8;background:#f8fafc;"
            "margin-bottom:14px;'>"
            "<div style='font-size:0.95em;font-weight:600;color:#64748b;'>3D Canvas</div>"
            "<div style='font-size:0.82em;margin-top:8px;'>Send a message to ground a result.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    pool_guids = list((retrieval_result or {}).get("pool_guids") or [])
    viewer_pool = [g for g in pool_guids if g and g != approved_guid]
    gt_guid = (retrieval_result or {}).get("gt_guid", "")
    viewer_url = _build_viewer_url(
        static_base_url,
        viewer_pool,
        target_guid=approved_guid,
        gt_guid=gt_guid,
        guid_match=bool(gt_guid and approved_guid == gt_guid) if gt_guid else True,
        ifc_model_code=(retrieval_result or {}).get("ifc_model_code", "AP"),
    )
    components.iframe(viewer_url, height=480, scrolling=False)


def _render_symbolic_graph_panel(
    vlm_result: dict | None, retrieval_result: dict | None
) -> None:
    """Live Cytoscape.js panel: 4-stage JSON → Cypher → Graph → Answer animation."""
    if vlm_result is None:
        st.caption("Run inference to populate the graph animation.")
        return

    parsed = (vlm_result or {}).get("parsed") or {}
    plans = (retrieval_result or {}).get("plans") or []
    results = (retrieval_result or {}).get("results") or []
    winning_idx = (retrieval_result or {}).get("winning_index")
    snapshot = (retrieval_result or {}).get("graph_snapshot") or {}
    subgraph = (retrieval_result or {}).get("subgraph") or {}
    pool_guids = (retrieval_result or {}).get("pool_guids") or []


    # Pull the winning Cypher for stage 2.
    cypher_text = ""
    win_priority = ""
    win_strategy = ""
    if (
        isinstance(winning_idx, int)
        and 0 <= winning_idx < len(results)
        and 0 <= winning_idx < len(plans)
    ):
        win_result = results[winning_idx] or {}
        win_plan = plans[winning_idx] or {}
        queries = win_result.get("executed_queries") or []
        cypher_text = queries[0] if queries else ""
        win_priority = win_plan.get("priority", "")
        win_strategy = win_plan.get("strategy", "")

    # Per-stage candidate guids for the 6-stage animation.
    _ATTR_STRATS = ("storey+type", "space+type", "storey_only", "type_only")
    _TOPO_STRATS = ("spatial_triplet", "continuous_span")

    def _result_guids(strats: tuple[str, ...], strict_topology: bool = False) -> list[str]:
        """Return candidate guids for the first plan whose strategy matches.

        When `strict_topology` is True and the matched plan is P0
        (spatial_triplet/continuous_span), only the strict P0 hits are
        returned — i.e. `candidates[:raw_pool_size]`. The retrieval backend
        appends P1-only candidates *after* the strict P0 set when
        `p0_strategy="p0_union_p1"` (the default), so this slice cleanly
        isolates the exact-topology subset.
        """
        for idx, plan in enumerate(plans):
            if plan.get("strategy") in strats:
                res = results[idx] if idx < len(results) else None
                if not res:
                    continue
                cands = res.get("candidates") or []
                if strict_topology:
                    raw = res.get("raw_pool_size")
                    if isinstance(raw, int) and raw >= 0:
                        cands = cands[:raw]
                return [c.get("guid", "") for c in cands if c.get("guid")]
        return []

    attribute_pool_guids = _result_guids(_ATTR_STRATS)
    # Stage 5 should show ONLY exact-topology matches, not P0∪P1.
    topology_pool_guids = _result_guids(_TOPO_STRATS, strict_topology=True)

    payload = {
        "parsed": parsed,
        "cypher": cypher_text,
        "winningPlan": {"priority": win_priority, "strategy": win_strategy},
        "snapshot": {
            "nodes": snapshot.get("nodes") or [],
            "edges": snapshot.get("edges") or [],
        },
        "subgraph": {
            "anchor_guid": subgraph.get("anchor_guid", ""),
            "nodes": subgraph.get("nodes") or [],
            "edges": subgraph.get("edges") or [],
        },
        "anchor_guid": pool_guids[0] if pool_guids else "",
        "pool_guids": pool_guids[:30],
        "ifc_class": parsed.get("ifc_class") or "",
        "storey_name": parsed.get("storey_name") or "",
        "attribute_pool_guids": attribute_pool_guids[:60],
        "attribute_pool_size": len(attribute_pool_guids),
        "topology_pool_guids": topology_pool_guids[:30],
        "topology_pool_size": len(topology_pool_guids),
    }

    html = _SYMBOLIC_GRAPH_HTML_TEMPLATE.replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=True)
    )
    components.html(html, height=560, scrolling=False)


def _render_perception_pill(vlm_result: dict | None) -> None:
    """Single-glance pill: did OpenCV + ResNet run, what did they see?

    Renders silently (returns) when the run was clearly not G9 — i.e. no
    `perception` key in the VLM payload at all — so non-G9 sessions don't
    see misleading 'off' badges.
    """
    if vlm_result is None or "perception" not in vlm_result:
        return

    perception = vlm_result.get("perception")

    base_style = (
        "display:inline-flex;align-items:center;gap:8px;"
        "padding:6px 12px;border-radius:8px;font-size:0.84em;"
        "font-family:-apple-system,sans-serif;margin:0 0 12px 0;"
        "border:1px solid;"
    )

    if perception is None:
        # G9 was selected but no floorplan was attached (predictor returned None).
        st.markdown(
            f"<div style='{base_style}background:#fef2f2;border-color:#fca5a5;color:#7f1d1d;'>"
            f"<strong>Perception off</strong>"
            f"<span style='color:#991b1b;'>· attach a floorplan via the “＋ plan” slot to enable OpenCV + ResNet</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    if perception.get("warning") and not perception.get("position_context"):
        st.markdown(
            f"<div style='{base_style}background:#fef3c7;border-color:#fcd34d;color:#78350f;'>"
            f"<strong>Perception failed</strong>"
            f"<span>· {perception['warning']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    chips: list[str] = []
    storey = perception.get("storey_name") or "?"
    scan = perception.get("scan_mode") or ""
    scan_suffix = " · auto" if scan == "auto" else ""
    chips.append(
        f"<span style='color:#475569;'>storey</span> "
        f"<code style='background:#f1f5f9;padding:1px 6px;border-radius:4px;'>{storey}</code>{scan_suffix}"
    )

    pc = perception.get("position_context")
    if pc:
        pc_conf = perception.get("position_context_confidence", 0.0)
        match = perception.get("match_score")
        match_suffix = f" / m={match:.2f}" if isinstance(match, (int, float)) else ""
        chips.append(
            f"<span style='color:#475569;'>OpenCV</span> "
            f"<code style='background:#ecfdf5;color:#065f46;padding:1px 6px;border-radius:4px;'>"
            f"{pc} · {pc_conf:.2f}{match_suffix}</code>"
        )
    else:
        chips.append(
            "<span style='color:#475569;'>OpenCV</span> "
            "<span style='color:#dc2626;font-weight:600;'>n/a</span>"
        )

    band = perception.get("size_band")
    if band:
        band_conf = perception.get("size_band_confidence", 0.0)
        chips.append(
            f"<span style='color:#475569;'>ResNet</span> "
            f"<code style='background:#eff6ff;color:#1e3a8a;padding:1px 6px;border-radius:4px;'>"
            f"{band} · {band_conf:.2f}</code>"
        )
    elif perception.get("resnet_error"):
        chips.append(
            f"<span style='color:#475569;'>ResNet</span> "
            f"<span style='color:#dc2626;font-weight:600;'>err</span>"
        )
    else:
        chips.append(
            "<span style='color:#475569;'>ResNet</span> "
            "<span style='color:#94a3b8;'>—</span>"
        )

    body = " · ".join(chips)
    st.markdown(
        f"<div style='{base_style}background:#f8fafc;border-color:#e2e8f0;color:#0f172a;'>"
        f"<strong style='color:#15803d;'>Perception</strong> · {body}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_rerank_panel(retrieval_result: dict | None) -> None:
    """Live Graph-RAG rerank panel: fusion scores, prompt, Gemini ranking."""
    if retrieval_result is None:
        st.caption("Run inference with a rerank-enabled pipeline to populate this panel.")
        return
    rerank = (retrieval_result or {}).get("rerank")
    if rerank is None:
        st.info(
            "Rerank is **off** for this run. Switch the pipeline to a "
            "`+ Graph-RAG rerank` variant in **Settings** to enable it."
        )
        return

    if rerank.get("failed"):
        st.error(f"Rerank failed: {rerank.get('reason') or 'unknown'}")
    else:
        st.success(
            f"Reranked top-{len(rerank.get('ordered_guids') or [])}. "
            f"Winner: `{rerank.get('winner_guid') or '?'}`"
        )

    fusion_scores = rerank.get("fusion_scores") or {}
    letter_to_guid = rerank.get("letter_to_guid") or {}
    descriptions = rerank.get("descriptions") or []
    ordered_guids = rerank.get("ordered_guids") or []

    if descriptions:
        st.markdown("**Candidate fingerprints (pre-rerank fusion order)**")
        for desc in descriptions[:10]:
            st.markdown(f"- {desc}")

    if ordered_guids and fusion_scores:
        st.markdown("**Reranked order (Gemini)**")
        rows: list[str] = []
        guid_to_letter = {g: l for l, g in letter_to_guid.items()}
        for rank, guid in enumerate(ordered_guids[:10], 1):
            letter = guid_to_letter.get(guid, "?")
            fusion = fusion_scores.get(guid, 0.0)
            rows.append(
                f"{rank}. **{letter}** · `{guid}` · fusion `{fusion:.2f}`"
            )
        st.markdown("\n".join(rows))

    cot = rerank.get("cot_reasoning")
    if cot:
        with st.expander("CoT reasoning", expanded=False):
            st.markdown(cot)
    raw = rerank.get("raw_output")
    if raw:
        with st.expander("Raw Gemini output", expanded=False):
            st.code(raw)
    prompt = rerank.get("prompt_text")
    if prompt:
        with st.expander("Final prompt", expanded=False):
            st.code(prompt)


def _render_drawer(vlm_result: dict | None, retrieval_result: dict | None) -> None:
    """Bottom drawer: raw JSON · Cypher · Perception. Collapsed by default."""
    label = "Process details — JSON · Cypher · Perception"
    with st.expander(label, expanded=False):
        if vlm_result is None:
            st.caption("Run inference to populate details.")
            return

        tabs = st.tabs(["Parsed JSON", "Cypher", "Perception"])
        parsed = (vlm_result or {}).get("parsed") or {}
        query_payloads = _collect_backend_queries(retrieval_result)

        with tabs[0]:
            st.code(json.dumps(parsed, indent=2, ensure_ascii=True), language="json")

        with tabs[1]:
            if not query_payloads:
                st.info("No captured Cypher yet.")
            else:
                for payload in query_payloads:
                    winner_suffix = " · winner" if payload["is_winner"] else ""
                    with st.expander(
                        f"P{payload['priority']} · {payload['strategy']} · {payload['pool_size']} candidates{winner_suffix}",
                        expanded=payload["is_winner"],
                    ):
                        for q_idx, query in enumerate(payload["queries"], 1):
                            if len(payload["queries"]) > 1:
                                st.caption(f"Query {q_idx}")
                            st.code(query, language="cypher")

        with tabs[2]:
            perception = (vlm_result or {}).get("perception")
            if not perception:
                st.caption("No server-side perception (G9 Modal predictor not used, "
                           "or no floorplan patch / storey provided).")
            else:
                st.code(json.dumps(perception, indent=2, ensure_ascii=True), language="json")


def _render_user_bubble(live_inputs: dict) -> None:
    """User chat bubble: query text + attached images shown large for visibility."""
    photo_paths = live_inputs.get("photo_paths") or []
    chat_text = (live_inputs.get("chat_text") or "").strip()
    with st.chat_message("user", avatar="👷"):
        if photo_paths:
            visible = [p for p in photo_paths if Path(p).exists()][:4]
            if len(visible) == 1:
                st.image(visible[0], use_container_width=True)
            elif visible:
                cols = st.columns(2)
                for i, p in enumerate(visible):
                    cols[i % 2].image(p, use_container_width=True)
        if chat_text:
            st.markdown(chat_text)


def _render_assistant_bubble(
    vlm_result: dict, retrieval_result: dict | None, *, case_id: str
) -> None:
    """Assistant chat bubble: top match + alternatives + see-process pointer."""
    rows = _candidate_rows(retrieval_result, limit=None)
    with st.chat_message("assistant"):
        if not vlm_result.get("valid_json", False):
            st.error("VLM returned invalid JSON. Try rephrasing the message.")
            return
        if not rows:
            st.warning("No candidates returned. Try adjusting the query or settings.")
            return

        top = rows[0]
        approved_guid = st.session_state.get("approved_guid", top["guid"])
        is_top_approved = approved_guid == top["guid"]

        approved_row = next((r for r in rows if r["guid"] == approved_guid), top)
        st.markdown(
            f"**Suggested item:** `{approved_row['guid']}`  \n"
            f"<span style='color:#475569;font-size:0.9em;'>"
            f"{approved_row.get('name') or 'Unnamed'} · {approved_row.get('type') or '—'} · {approved_row.get('storey') or '—'}"
            f"</span>",
            unsafe_allow_html=True,
        )
        if approved_row.get("gt"):
            st.success("Matches ground truth.")
        elif (retrieval_result or {}).get("gt_guid"):
            st.caption("Does not match ground truth — pick an alternative below.")

        if len(rows) > 1:
            visible_count_key = _live_flow_state_key("live", case_id, "candidate_visible_count")
            visible_count = int(st.session_state.get(visible_count_key, 5) or 5)
            visible_count = max(5, min(visible_count, len(rows)))
            st.session_state[visible_count_key] = visible_count

            with st.expander(f"{len(rows) - 1} alternatives", expanded=False):
                approve_prefix = _live_flow_state_key("live", case_id, "approve")
                for row in rows[1:visible_count]:
                    guid = row["guid"]
                    cols = st.columns([0.4, 2.4, 1.2, 1.2, 0.9], gap="small")
                    cols[0].markdown(f"**{row['rank']}**")
                    cols[1].markdown(
                        (row.get("name") or "Unnamed")
                        + (" · ✓ GT" if row.get("gt") else "")
                    )
                    cols[2].markdown(row.get("type") or "—")
                    cols[3].markdown(row.get("storey") or "—")
                    if guid == approved_guid:
                        cols[4].markdown(
                            "<span style='color:#15803d;font-weight:600;'>✓</span>",
                            unsafe_allow_html=True,
                        )
                    elif cols[4].button("–", key=f"{approve_prefix}_{guid}", use_container_width=True, help="Select this candidate"):
                        st.session_state["approved_guid"] = guid
                        st.session_state["approval_candidate_guid"] = guid
                        st.rerun()

                if visible_count < len(rows):
                    if st.button(
                        f"Show next 5 ({visible_count}/{len(rows)})",
                        key=_live_flow_state_key("live", case_id, "show_next_candidates"),
                    ):
                        st.session_state[visible_count_key] = min(visible_count + 5, len(rows))
                        st.rerun()

        st.caption("See process ▾ — open the *Process details* drawer below for parsed JSON, Cypher, and the symbolic graph.")


def _render_chat_history(*, case_id: str) -> None:
    """Render the most recent user→assistant exchange (or an empty hint)."""
    vlm_result = st.session_state.get("last_inference")
    retrieval_result = st.session_state.get("last_retrieval")
    live_inputs = st.session_state.get("last_live_inputs") or {}

    if vlm_result is None:
        st.markdown(
            "<div style='border:1px dashed #cbd5e1;border-radius:8px;"
            "padding:32px 16px;text-align:center;color:#94a3b8;background:#f8fafc;"
            "margin-bottom:10px;'>"
            "Send a message to start a query."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    _render_user_bubble(live_inputs)
    _render_assistant_bubble(vlm_result, retrieval_result, case_id=case_id)


_MODEL_VARIANTS = {
    "G9 — OpenCV + ResNet · full topology": {
        "predictor": "G9Predictor", "skip_p0": False, "rerank": False,
    },
    "G9 — OpenCV + ResNet · full topology + rerank": {
        "predictor": "G9Predictor", "skip_p0": False, "rerank": True,
    },
    "G9 — OpenCV + ResNet · P1 + rerank": {
        "predictor": "G9Predictor", "skip_p0": True, "rerank": True,
    },
    "G8 — full topology": {
        "predictor": "G8ModelPredictor", "skip_p0": False, "rerank": False,
    },
    "G8 — P1 + rerank": {
        "predictor": "G8ModelPredictor", "skip_p0": True, "rerank": True,
    },
    "G4-Ultimate — full topology": {
        "predictor": "G8Predictor", "skip_p0": False, "rerank": False,
    },
    "G4-Ultimate — P1 + rerank": {
        "predictor": "G8Predictor", "skip_p0": True, "rerank": True,
    },
}


def _render_composer(*, case_id: str, trace: dict | None) -> None:
    """Bottom composer: large attached thumbnails · text · attach (+) · send · settings."""
    # Nonce-rotated keys so the composer renders truly fresh after each submit
    # — popping or assigning '' to widget state is unreliable for text_area and
    # outright forbidden for file_uploader. Rotating the key forces Streamlit
    # to mount brand-new widgets with no carried-over input.
    nonce = st.session_state.get("composer_nonce", 0)
    chat_key = f"inference_chat_{nonce}"
    images_key = f"inference_images_{nonce}"
    floorplan_key = f"inference_floorplan_{nonce}"
    send_key = f"run_pipeline_{nonce}"

    # Show the floorplan slot only when the selected pipeline actually uses it
    # (G9 → server-side OpenCV + ResNet). For G8 / G4-Ultimate the slot is
    # hidden so the composer collapses back to the original one-uploader UX.
    selected_variant = st.session_state.get("inf_model_variant", "") or next(iter(_MODEL_VARIANTS))
    show_floorplan_slot = (
        _MODEL_VARIANTS.get(selected_variant, {}).get("predictor") == "G9Predictor"
    )

    with st.container(border=True):
        # Show currently-attached thumbnails large, above the text area.
        existing_files = list(st.session_state.get(images_key) or [])[:4]
        if existing_files:
            if len(existing_files) == 1:
                st.image(existing_files[0], use_container_width=True)
            else:
                cols = st.columns(2)
                for i, f in enumerate(existing_files):
                    cols[i % 2].image(f, use_container_width=True)

        # accept_multiple_files=False returns a single UploadedFile (or None),
        # not a list — iterating it as a list feeds PIL the raw bytes.
        floorplan_upload = st.session_state.get(floorplan_key)
        if isinstance(floorplan_upload, list):
            floorplan_upload = floorplan_upload[0] if floorplan_upload else None
        if show_floorplan_slot and floorplan_upload is not None:
            st.image(
                floorplan_upload,
                caption="Floorplan patch (G9 only)",
                use_container_width=True,
            )

        chat_text = st.text_area(
            "Field message",
            height=90,
            key=chat_key,
            placeholder="Describe the element you need to locate…",
            label_visibility="collapsed",
        )

        # Compact action row. Floorplan slot only appears when G9 is selected.
        if show_floorplan_slot:
            attach_col, plan_col, send_col = st.columns([0.7, 0.7, 3.6], gap="small")
        else:
            attach_col, send_col = st.columns([0.7, 4.3], gap="small")
            plan_col = None
        with attach_col.popover("＋", use_container_width=True, help="Attach site photo(s)"):
            st.file_uploader(
                "Attach site photos",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=images_key,
                label_visibility="collapsed",
            )
        if plan_col is not None:
            with plan_col.popover("＋ plan", use_container_width=True, help="Attach a floorplan patch (used by G9 OpenCV + ResNet)"):
                st.file_uploader(
                    "Attach floorplan patch",
                    type=["png", "jpg", "jpeg"],
                    accept_multiple_files=False,
                    key=floorplan_key,
                    label_visibility="collapsed",
                )
        run_btn = send_col.button(
            "Send", type="primary", use_container_width=True, key=send_key
        )

        # Files persist in session_state via the file_uploader key.
        uploaded_files = st.session_state.get(images_key) or []
        # Refresh after the popover may have set the value.
        floorplan_upload = st.session_state.get(floorplan_key)
        if isinstance(floorplan_upload, list):
            floorplan_upload = floorplan_upload[0] if floorplan_upload else None
        if not show_floorplan_slot:
            # Don't forward a stale floorplan to a non-G9 predictor.
            floorplan_upload = None

        with st.expander("Settings", expanded=False):
            st.selectbox(
                "Pipeline",
                list(_MODEL_VARIANTS.keys()),
                key="inf_model_variant",
            )
            st.selectbox(
                "IFC Model",
                ["AP (AdvancedProject)", "BH (BasicHouse)", "DXA (Duplex)"],
                key="inf_ifc_model",
            )
            _storey_options = [
                "1 - First Floor",
                "2 - Second Floor",
                "3 - Third Floor",
                "4 - Fourth Floor",
                "5 - Fifth Floor",
                "6 - Sixth Floor",
                "Level 1",
                "Level 2",
                "-1 - Garage",
                "(auto-detect: scan all storeys)",
            ]
            st.selectbox(
                "Storey (G9 perception)",
                _storey_options,
                index=0,
                key="inf_storey",
                help=(
                    "Storey used to localise the floorplan patch + ResNet crop. "
                    "'auto-detect' scans every calibrated storey and picks the "
                    "highest OpenCV match score — costs ~0.5s extra."
                ),
            )
            cols_modes = st.columns(2)
            cols_modes[0].selectbox(
                "size_band mode",
                ["off", "soft", "hard"],
                index=2,
                key="inf_size_band_mode",
                help="Cypher filter strength for the ResNet size_band signal.",
            )
            cols_modes[1].selectbox(
                "size_cluster mode",
                ["off", "soft", "hard"],
                index=1,
                key="inf_size_cluster_mode",
                help="Cypher filter strength for the VLM size_cluster signal.",
            )
            st.radio(
                "Ground truth (dev)",
                ["None", "Sidebar case", "Direct GUID"],
                horizontal=True,
                key="inf_gt_source_mode",
            )
            if st.session_state.get("inf_gt_source_mode") == "Direct GUID":
                st.text_input(
                    "Target GUID",
                    value=st.session_state.get("inf_gt_guid_manual", ""),
                    placeholder="e.g. 2BLn4xX2vF_gM9G5wfbU5X",
                    key="inf_gt_guid_manual",
                )

    if not run_btn:
        return

    if not (chat_text or "").strip():
        st.warning("Please enter a message.")
        return

    mv = st.session_state.get("inf_model_variant", "")
    variant_cfg = _MODEL_VARIANTS.get(mv) or next(iter(_MODEL_VARIANTS.values()))
    predictor_name = variant_cfg["predictor"]
    skip_p0 = bool(variant_cfg["skip_p0"])
    enable_rerank = bool(variant_cfg["rerank"])
    model_code = (st.session_state.get("inf_ifc_model") or "AP").split(" ", 1)[0].strip()

    storey_choice = st.session_state.get("inf_storey", "1 - First Floor")
    # Empty string → G9 predictor scans every calibrated storey and picks
    # the highest OpenCV match (auto-detect mode).
    storey_for_perception = "" if storey_choice.startswith("(auto") else storey_choice
    size_band_mode = st.session_state.get("inf_size_band_mode", "hard")
    size_cluster_mode = st.session_state.get("inf_size_cluster_mode", "soft")

    gt_mode = st.session_state.get("inf_gt_source_mode", "None")
    if gt_mode == "Sidebar case" and case_id:
        gt_guid = _extract_gt_guid(trace, case_id=case_id, allow_trace_fallback=True)
    elif gt_mode == "Direct GUID":
        gt_guid = (st.session_state.get("inf_gt_guid_manual") or "").strip()
    else:
        gt_guid = ""

    metadata_text = f"[IFC Model] {model_code}"
    if storey_for_perception:
        metadata_text += f"\n[Location] {storey_for_perception}"
    all_image_bytes = [f.getvalue() for f in (uploaded_files or [])]
    floorplan_bytes = floorplan_upload.getvalue() if floorplan_upload else None

    pipeline_label = mv.split(" — ", 1)[0] if " — " in mv else mv
    t0 = time.time()
    with st.spinner(f"Stage 1/3 — VLM inference ({pipeline_label}) on Modal A100..."):
        vlm_result = _call_modal_inference(
            all_image_bytes,
            chat_text,
            metadata_text,
            predictor_name=predictor_name,
            floorplan_patch_bytes=floorplan_bytes,
            storey_name=storey_for_perception or None,
        )
    if vlm_result is None:
        return

    vlm_result["_latency_ms"] = int((time.time() - t0) * 1000)
    vlm_result["_model_variant"] = mv
    vlm_result["_predictor"] = predictor_name

    floorplan_path_persist = ""
    if floorplan_upload is not None:
        floorplan_path_persist = "<live floorplan patch>"

    st.session_state["last_live_inputs"] = _persist_live_inputs(
        uploaded_files=uploaded_files or [],
        chat_text=chat_text,
        model_code=model_code,
        storey=storey_for_perception,
        phase="",
        task_status="",
        case_id=case_id,
        gt_guid=gt_guid,
        floorplan_upload=floorplan_upload,
    )

    retrieval_result = None
    if vlm_result.get("valid_json") and vlm_result.get("parsed"):
        t1 = time.time()
        with st.spinner("Stage 2-3 — Query planning + Neo4j retrieval..."):
            retrieval_result = _run_retrieval(
                vlm_result["parsed"],
                model_code=model_code,
                gt_guid=gt_guid,
                skip_p0=skip_p0,
                size_band_mode=size_band_mode,
                size_cluster_mode=size_cluster_mode,
                enable_rerank=enable_rerank,
                query_text=chat_text,
                site_image_bytes=all_image_bytes,
                floorplan_bytes=floorplan_bytes,
            )
        if retrieval_result:
            retrieval_result["_latency_ms"] = int((time.time() - t1) * 1000)
            default_guid = _top_candidate_guid(retrieval_result)
            if default_guid:
                st.session_state["approved_guid"] = default_guid
                st.session_state["approval_candidate_guid"] = default_guid
            else:
                st.session_state.pop("approved_guid", None)
                st.session_state.pop("approval_candidate_guid", None)
            st.session_state["rejected_candidate_guids"] = []
            st.session_state[
                _live_flow_state_key("live", case_id, "candidate_visible_count")
            ] = 5

    st.session_state["last_inference"] = vlm_result
    st.session_state["last_retrieval"] = retrieval_result

    # Bump the composer nonce so text_area / file_uploader remount fresh on
    # the next render. The submitted text and images are preserved in the
    # user chat bubble above via session_state["last_live_inputs"].
    st.session_state["composer_nonce"] = st.session_state.get("composer_nonce", 0) + 1

    st.rerun()


def render(
    *,
    static_base_url: str = "",
    trace: dict | None = None,
    case_id: str = "",
) -> None:
    """Live inference: chat thread (left) · stepper + 3D canvas + drawer (right)."""
    # Reset stale results when sidebar case changes.
    if st.session_state.get("live_case_id") != case_id:
        old_case_id = st.session_state.get("live_case_id", "")
        for k in (
            "last_inference", "last_retrieval", "last_live_inputs",
            "approved_guid", "approval_candidate_guid",
            "rejected_candidate_guids", "live_candidate_visible_count",
        ):
            st.session_state.pop(k, None)
        for flow_scope in ("live", "trace"):
            st.session_state.pop(
                _live_flow_state_key(flow_scope, old_case_id, "candidate_visible_count"),
                None,
            )
        # Bump composer nonce so widgets remount fresh for the new case.
        st.session_state["composer_nonce"] = st.session_state.get("composer_nonce", 0) + 1
        st.session_state["live_case_id"] = case_id

    left_col, right_col = st.columns([1, 1.6], gap="medium")

    with left_col:
        _render_chat_history(case_id=case_id)
        _render_composer(case_id=case_id, trace=trace)

    vlm_result = st.session_state.get("last_inference")
    retrieval_result = st.session_state.get("last_retrieval")

    with right_col:
        _render_stepper(vlm_result, retrieval_result)
        _render_perception_pill(vlm_result)
        view_tabs = st.tabs(["Symbolic Graph", "3D Canvas", "Rerank"])
        with view_tabs[0]:
            _render_symbolic_graph_panel(vlm_result, retrieval_result)
        with view_tabs[1]:
            _render_canvas(vlm_result, retrieval_result, static_base_url=static_base_url)
        with view_tabs[2]:
            _render_rerank_panel(retrieval_result)
        _render_drawer(vlm_result, retrieval_result)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline visualization
# ══════════════════════════════════════════════════════════════════════════════

def _top_candidate_guid(retrieval_result: dict | None) -> str:
    """Return the first candidate GUID from the winning result."""
    candidates = ((retrieval_result or {}).get("winning") or {}).get("candidates") or []
    if candidates:
        return candidates[0].get("guid", "")
    return ""


def _persist_live_inputs(
    *,
    uploaded_files,
    chat_text: str,
    model_code: str,
    storey: str,
    phase: str,
    task_status: str,
    case_id: str,
    gt_guid: str,
    floorplan_upload=None,
) -> dict:
    """Persist live demo inputs so the shared review tabs can render them."""
    base_dir = Path("/tmp/mscd_demo_live_inputs")
    base_dir.mkdir(parents=True, exist_ok=True)
    session_tag = str(int(time.time() * 1000))
    run_dir = base_dir / session_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    photo_paths: list[str] = []
    for idx, upload in enumerate(uploaded_files or []):
        name = Path(getattr(upload, "name", f"img_{idx}.png")).name
        dest = run_dir / f"img_{idx}_{name}"
        dest.write_bytes(upload.getvalue())
        photo_paths.append(str(dest))

    floorplan_path = ""
    if floorplan_upload is not None:
        name = Path(getattr(floorplan_upload, "name", "floorplan.png")).name
        dest = run_dir / f"floorplan_{name}"
        dest.write_bytes(floorplan_upload.getvalue())
        floorplan_path = str(dest)

    return {
        "photo_paths": photo_paths,
        "floorplan_path": floorplan_path,
        "chat_text": chat_text,
        "model_code": model_code,
        "storey": storey,
        "phase": phase,
        "task_status": task_status,
        "case_id": case_id,
        "gt_guid": gt_guid,
    }


def _trace_to_live_inputs(trace: dict, *, case_id: str = "") -> dict:
    """Adapt a stored review trace to the live-flow input shape."""
    scenario = trace.get("scenario") or {}
    ctx = scenario.get("context_meta") or {}
    image_parse = (trace.get("internals") or {}).get("image_parse_result") or {}
    photo_paths = [
        item.get("image_path", "")
        for item in (image_parse.get("site_photos") or [])
        if item.get("image_path")
    ]
    if not photo_paths:
        photo_paths = list(scenario.get("image_paths") or [])
    floorplan_entry = image_parse.get("floorplan") or {}
    floorplan_path = floorplan_entry.get("image_path", "")
    gt = scenario.get("ground_truth") or {}
    scenario_id = scenario.get("id") or case_id
    model_code = "AP"
    if "_BH_" in scenario_id:
        model_code = "BH"
    elif "_DXA_" in scenario_id:
        model_code = "DXA"

    return {
        "photo_paths": photo_paths,
        "floorplan_path": floorplan_path,
        "chat_text": scenario.get("query_text", ""),
        "model_code": model_code,
        "storey": gt.get("target_storey", ""),
        "phase": ctx.get("project_phase", ""),
        "task_status": ctx.get("task_status", ""),
        "case_id": scenario_id,
        "gt_guid": gt.get("target_guid", ""),
    }


def _trace_to_live_vlm_result(trace: dict) -> dict | None:
    """Adapt stored trace constraints to the live-flow VLM result shape."""
    constraints = (trace.get("internals") or {}).get("constraints") or {}
    if not constraints:
        return None
    parsed = {
        "storey_name": constraints.get("storey_name"),
        "ifc_class": constraints.get("ifc_class"),
        "space_name": constraints.get("space_name"),
        "target_name_keyword": constraints.get("target_name_keyword"),
        "spatial_relations": constraints.get("spatial_relations") or constraints.get("relations") or [],
    }
    return {
        "valid_json": True,
        "parsed": parsed,
        "_latency_ms": int((trace.get("internals") or {}).get("constraints_extraction_ms", 0)),
        "_model_variant": "Trace Replay",
        "raw_trace_constraints": constraints,
    }


def _trace_to_live_retrieval_result(trace: dict, *, case_id: str = "") -> dict | None:
    """Adapt stored trace retrieval internals to the live-flow retrieval shape."""
    internals = trace.get("internals") or {}
    plans = list(internals.get("query_plans") or [])
    results = list(internals.get("retrieval_results") or [])
    if not plans and not results:
        return None

    winning_result = None
    winning_index = None
    for idx, result in enumerate(results):
        if (result or {}).get("pool_size", 0) > 0:
            winning_result = result
            winning_index = idx
            break

    pool_guids = []
    if winning_result:
        pool_guids = [c.get("guid", "") for c in (winning_result.get("candidates") or []) if c.get("guid")]

    scenario = trace.get("scenario") or {}
    gt_guid = ((scenario.get("ground_truth") or {}).get("target_guid") or "")
    scenario_id = scenario.get("id") or case_id or trace.get("scenario_id", "")
    model_code = "AP"
    if "_BH_" in scenario_id:
        model_code = "BH"
    elif "_DXA_" in scenario_id:
        model_code = "DXA"

    return {
        "plans": plans,
        "results": results,
        "winning": winning_result,
        "winning_index": winning_index,
        "pool_guids": pool_guids,
        "ifc_model_code": model_code,
        "gt_guid": gt_guid,
        "guid_match": bool(trace.get("guid_match", False)),
    }


def render_trace_flow(
    trace: dict,
    *,
    static_base_url: str = "",
    case_id: str = "",
) -> None:
    """Render the same live-demo flow panel using stored review traces."""
    vlm_result = _trace_to_live_vlm_result(trace)
    retrieval_result = _trace_to_live_retrieval_result(trace, case_id=case_id)
    live_inputs = _trace_to_live_inputs(trace, case_id=case_id)

    if not vlm_result:
        st.info("This trace does not contain v2 constraints for flow replay.")
        return

    _render_live_flow(
        vlm_result,
        retrieval_result,
        static_base_url=static_base_url,
        case_id=case_id,
        live_inputs=live_inputs,
        flow_scope="trace",
    )


def _candidate_rows(retrieval_result: dict | None, limit: int = 8) -> list[dict]:
    """Flatten winning candidates into simple table rows."""
    candidates = ((retrieval_result or {}).get("winning") or {}).get("candidates") or []
    gt_guid = (retrieval_result or {}).get("gt_guid", "")
    if limit is None or limit <= 0:
        visible_candidates = candidates
    else:
        visible_candidates = candidates[:limit]
    rows = []
    for rank, cand in enumerate(visible_candidates, 1):
        guid = cand.get("guid", "")
        rows.append({
            "rank": rank,
            "guid": guid,
            "name": cand.get("name", ""),
            "type": cand.get("type", ""),
            "storey": cand.get("storey", ""),
            "gt": bool(gt_guid and guid == gt_guid),
        })
    return rows


def _live_flow_state_key(flow_scope: str, case_id: str, name: str) -> str:
    """Build a stable per-flow, per-case Streamlit state/widget key."""
    case_token = case_id or "no_case"
    return f"{flow_scope}_{case_token}_{name}"


def _render_live_flow(
    vlm_result: dict,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
    case_id: str = "",
    live_inputs: dict | None = None,
    flow_scope: str = "live",
) -> None:
    """Render the direct, presentation-friendly live demo flow."""
    parsed = vlm_result.get("parsed") or {}
    approved_guid = st.session_state.get("approved_guid", "")
    top_guid = _top_candidate_guid(retrieval_result)
    visible_count_key = _live_flow_state_key(flow_scope, case_id, "candidate_visible_count")
    approve_key_prefix = _live_flow_state_key(flow_scope, case_id, "approve")
    show_next_key = _live_flow_state_key(flow_scope, case_id, "show_next_candidates")
    if not approved_guid and top_guid:
        approved_guid = top_guid
        st.session_state["approved_guid"] = approved_guid
        st.session_state["approval_candidate_guid"] = approved_guid

    flow_col, backend_col = st.columns([1.45, 0.95], gap="medium")

    with flow_col:
        # ── Step 1 · Input — what was sent ──────────────────────────────────
        _step_header(1, "Input", "what was sent")
        with st.container(border=True):
            _render_live_modalities(live_inputs=live_inputs)
            chat_text = ((live_inputs or {}).get("chat_text") or "").strip()
            if chat_text:
                st.markdown(f"> {chat_text}")

        # ── Step 2 · Interpreter — VLM-extracted constraints ────────────────
        _step_header(2, "Interpreter", "VLM-extracted constraints")
        with st.container(border=True):
            st.markdown(
                _kv_pill("storey", parsed.get("storey_name") or "—", "#22c55e")
                + _kv_pill("ifc_class", parsed.get("ifc_class") or "—", "#3b82f6")
                + _kv_pill("space", parsed.get("space_name") or "—", "#8b5cf6")
                + _kv_pill("name_kw", parsed.get("target_name_keyword") or "—", "#f59e0b"),
                unsafe_allow_html=True,
            )
            rels = parsed.get("spatial_relations") or []
            if rels:
                first_rel = rels[0]
                st.markdown(
                    f"**Primary cue:** `{parsed.get('ifc_class') or 'Element'}` "
                    f"`{first_rel.get('predicate', '?')}` "
                    f"`{first_rel.get('object_type', '?')}`"
                )
                st.graphviz_chart(_build_vlm_relation_graph_dot(parsed), use_container_width=True)
            else:
                st.info("Attribute-only query. Retrieval will rely on storey, type, and optional name cues.")

        # ── Step 3 · Retrieval — symbolic execution waterfall ───────────────
        plans = (retrieval_result or {}).get("plans") or []
        results = (retrieval_result or {}).get("results") or []
        retrieval_ms = int((retrieval_result or {}).get("_latency_ms", 0))
        backend_label = ((retrieval_result or {}).get("winning") or {}).get("backend", "—")
        retrieval_hint = (
            f"{backend_label} · {retrieval_ms} ms" if retrieval_ms else backend_label
        )
        _step_header(3, "Retrieval", retrieval_hint)
        with st.container(border=True):
            if not plans:
                st.info("No symbolic retrieval results yet.")
            else:
                winning_idx = (retrieval_result or {}).get("winning_index")

                def _plan_line(idx, plan, result):
                    priority = plan.get("priority", "?")
                    strategy = plan.get("strategy", "?")
                    if result is None:
                        return f"⏭ `P{priority} · {strategy}` — no execution record"
                    pool_size = (result or {}).get("pool_size", 0)
                    raw_pool_size = result.get("raw_pool_size")
                    actual_strategy = result.get("strategy_actually_used", strategy)
                    fallback_note = " · fallback" if result.get("fallback_triggered") else ""
                    if strategy in ("spatial_triplet", "continuous_span") and raw_pool_size is not None:
                        pool_text = f"{raw_pool_size} → **{pool_size}** candidates"
                    else:
                        pool_text = f"**{pool_size}** candidates"
                    if winning_idx is not None and idx == winning_idx:
                        return f"✅ `P{priority} · {strategy}` — {pool_text} via `{actual_strategy}`{fallback_note}"
                    if winning_idx is not None and idx > winning_idx:
                        return f"⏭ `P{priority} · {strategy}` — skipped after winner · {pool_text}"
                    if pool_size == 0:
                        return f"❌ `P{priority} · {strategy}` — empty"
                    return f"• `P{priority} · {strategy}` — {pool_text}"

                # Winner first (prominent), others collapsed
                if winning_idx is not None and winning_idx < len(plans):
                    win_result = results[winning_idx] if winning_idx < len(results) else None
                    st.markdown(_plan_line(winning_idx, plans[winning_idx], win_result))
                others = [
                    (i, p, results[i] if i < len(results) else None)
                    for i, p in enumerate(plans)
                    if i != winning_idx
                ]
                if others:
                    with st.expander(f"Show full waterfall ({len(plans)} plans)", expanded=False):
                        for i, plan, result in others:
                            st.markdown(_plan_line(i, plan, result))

            # Embedded 3D viewer — uses current top/approved guid; reloads on confirm
            if approved_guid:
                pool_guids = list((retrieval_result or {}).get("pool_guids") or [])
                viewer_pool = [g for g in pool_guids if g and g != approved_guid]
                gt_guid_inline = (retrieval_result or {}).get("gt_guid", "")
                viewer_url_inline = _build_viewer_url(
                    static_base_url,
                    viewer_pool,
                    target_guid=approved_guid,
                    gt_guid=gt_guid_inline,
                    guid_match=bool(gt_guid_inline and approved_guid == gt_guid_inline) if gt_guid_inline else True,
                    ifc_model_code=(retrieval_result or {}).get("ifc_model_code", "AP"),
                )
                with st.expander("Open 3D Viewer", expanded=False):
                    components.iframe(viewer_url_inline, height=520, scrolling=False)

        # ── Step 4 · Confirm match — human-in-the-loop ──────────────────────
        rows = _candidate_rows(retrieval_result, limit=None)
        _step_header(4, "Confirm match", f"{len(rows)} candidates" if rows else "")
        with st.container(border=True):
            if not rows:
                st.warning("No candidates returned. Check retrieval mode or constraints.")
            else:
                approved_guid = st.session_state.get("approved_guid", approved_guid or rows[0]["guid"])
                visible_count = int(st.session_state.get(visible_count_key, 15) or 15)
                visible_count = max(15, min(visible_count, len(rows)))
                st.session_state[visible_count_key] = visible_count
                visible_rows = rows[:visible_count]

                header_cols = st.columns([0.6, 0.8, 1.8, 1.2, 1.3, 2.2, 1.2], gap="small")
                header_cols[0].markdown("**Rank**")
                header_cols[1].markdown("**GT**")
                header_cols[2].markdown("**Name**")
                header_cols[3].markdown("**Type**")
                header_cols[4].markdown("**Storey**")
                header_cols[5].markdown("**GUID**")
                header_cols[6].markdown("**Action**")

                for row in visible_rows:
                    guid = row["guid"]
                    is_approved = approved_guid == guid
                    row_cols = st.columns([0.6, 0.8, 1.8, 1.2, 1.3, 2.2, 1.2], gap="small")
                    row_cols[0].markdown(str(row["rank"]))
                    row_cols[1].markdown("✓" if row["gt"] else "")
                    row_cols[2].markdown(row["name"] or "Unnamed")
                    row_cols[3].markdown(row["type"] or "—")
                    row_cols[4].markdown(row["storey"] or "—")
                    row_cols[5].markdown(f"`{guid}`")

                    if is_approved:
                        row_cols[6].markdown(
                            "<div style='text-align:center;font-size:0.95em;font-weight:600;color:#15803d;'>✓ Approved</div>",
                            unsafe_allow_html=True,
                        )
                    elif row_cols[6].button(
                        "Confirm",
                        key=f"{approve_key_prefix}_{guid}",
                        help="Confirm this candidate",
                        use_container_width=True,
                    ):
                        st.session_state["approved_guid"] = guid
                        st.session_state["approval_candidate_guid"] = guid
                        st.rerun()

                if visible_count < len(rows):
                    more_col1, _more_col2 = st.columns([1, 3], gap="small")
                    if more_col1.button(
                        f"Show next 15 ({visible_count}/{len(rows)})",
                        key=show_next_key,
                        use_container_width=True,
                    ):
                        st.session_state[visible_count_key] = min(visible_count + 15, len(rows))
                        st.rerun()

                approved_guid = st.session_state.get("approved_guid", "")
                if approved_guid:
                    st.markdown(
                        f"<div style='font-size:0.9em;color:#15803d;font-weight:600;margin-top:6px;'>"
                        f"✓ Approved <code>{approved_guid}</code></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Confirm one candidate to lock the match.")

    with backend_col:
        _render_backend_panel(vlm_result, retrieval_result)


# ══════════════════════════════════════════════════════════════════════════════
# Backend calls
# ══════════════════════════════════════════════════════════════════════════════

def _call_modal_inference(
    image_bytes_list: list[bytes],
    chat_text: str,
    metadata_text: str,
    *,
    predictor_name: str = "G8Predictor",
    floorplan_patch_bytes: bytes | None = None,
    storey_name: str | None = None,
) -> dict | None:
    """Call the Modal VLM inference endpoint.

    Routes to G9Predictor (G9 + OpenCV + ResNet), G8ModelPredictor
    (G8 adapter), or G8Predictor (G4-Ultimate adapter). G9 additionally takes
    a floorplan patch and storey name and runs the perception layer
    server-side; G8/G4-Ultimate ignore those args. Strategy (full-topo vs
    P1+rerank) is controlled by skip_p0 in retrieval.
    """
    try:
        import modal
        predictor_cls = modal.Cls.from_name("mscd-vlm-lora3-inference", predictor_name)
        predictor = predictor_cls()
        kwargs: dict = dict(
            image_bytes_list=image_bytes_list,
            chat_text=chat_text,
            metadata_text=metadata_text,
        )
        if predictor_name == "G9Predictor":
            kwargs["floorplan_patch_bytes"] = floorplan_patch_bytes
            kwargs["storey_name"] = storey_name
        result = predictor.predict.remote(**kwargs)
        return result
    except Exception as e:
        err_msg = str(e)
        if "NotFound" in err_msg or "not found" in err_msg.lower():
            st.error(
                f"Modal app `mscd-vlm-lora3-inference` / `{predictor_name}` not found. "
                "Deploy first:\n\n"
                "```\nmodal deploy training/inference.py\n```"
            )
        else:
            st.error(f"Inference failed: {e}")
        return None


@st.cache_data(show_spinner=False)
def _load_demo_gt_index() -> dict[str, str]:
    """Load case_id -> target_guid from local demo JSONL datasets.

    Scans both demo test-data files and the synth_v0.5_* skeleton files so that
    eval-format IDs like 'AP_SK_282' resolve correctly.
    """
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data_curation" / "datasets"

    # Fixed test-data files
    candidates = [
        repo_root / "data" / "test_data" / "demo10_h1h3_top1" / "h1_h3_top1_success10.jsonl",
        repo_root / "data" / "test_data" / "demo10" / "h2_test_cases_demo10.jsonl",
        repo_root / "data" / "test_data" / "h2_test_cases.jsonl",
    ]
    # Skeleton files from all synth_v0.5_* sub-directories
    if data_root.exists():
        for skel_file in data_root.glob("synth_v0.5_*/skeletons/skeletons.jsonl"):
            candidates.append(skel_file)

    index: dict[str, str] = {}
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    gt = row.get("target_guid") or ((row.get("ground_truth") or {}).get("target_guid"))
                    if not gt:
                        continue
                    id_candidates = [
                        row.get("case_id"),
                        row.get("id"),
                        row.get("h2_id"),
                        row.get("skeleton_id"),
                    ]
                    for cid in id_candidates:
                        if not cid:
                            continue
                        cid_s = str(cid).strip()
                        index[cid_s] = gt
                        # Also index normalized form to bridge IDs with/without model segment.
                        cid_norm = _normalize_case_id(cid_s)
                        if cid_norm and cid_norm not in index:
                            index[cid_norm] = gt
        except Exception:
            continue
    return index


def _normalize_case_id(case_id: str) -> str:
    """Normalize IDs across formats, e.g. SYNTH_V3_002_AP_SK_002 -> SYNTH_V3_002_SK_002."""
    cid = (case_id or "").strip()
    if not cid:
        return ""
    return re.sub(r"_(AP|BH|DXA)(?=_SK_)", "", cid, flags=re.IGNORECASE)


def _extract_gt_guid(
    trace: dict | None,
    case_id: str = "",
    *,
    allow_trace_fallback: bool = True,
) -> str:
    """Resolve GT by explicit case_id first, with optional trace fallback."""
    gt_index = _load_demo_gt_index()
    lookup_candidates = []
    if case_id:
        lookup_candidates.append(case_id)
        lookup_candidates.append(_normalize_case_id(case_id))

    if allow_trace_fallback and trace:
        trace_case = (trace.get("scenario") or {}).get("id", "")
        if trace_case:
            lookup_candidates.append(trace_case)
            lookup_candidates.append(_normalize_case_id(trace_case))

    for cid in lookup_candidates:
        cid_s = (cid or "").strip()
        if cid_s and cid_s in gt_index:
            return gt_index[cid_s]

    if not allow_trace_fallback or not trace:
        return ""

    # Fallback to trace payload if case lookup misses.
    scenario = trace.get("scenario") or {}
    gt = scenario.get("ground_truth") or {}
    if gt.get("target_guid"):
        return gt["target_guid"]
    ev = trace.get("evaluation") or {}
    return ev.get("target_guid", "")


def _run_retrieval(
    parsed: dict,
    model_code: str = "AP",
    gt_guid: str = "",
    skip_p0: bool = False,
    *,
    size_band_mode: str = "hard",
    size_cluster_mode: str = "soft",
    enable_rerank: bool = False,
    query_text: str = "",
    site_image_bytes: list[bytes] | None = None,
    floorplan_bytes: bytes | None = None,
) -> dict | None:
    """Run the symbolic retrieval pipeline: Constraints → QueryPlanner → Neo4j."""
    try:
        from src.neurosym.types import Constraints, SpatialTriplet
        from src.neurosym.constraints_to_query import QueryPlanner
        from src.neurosym.retrieval_backend import RetrievalBackend
        import yaml
        from pathlib import Path

        # Build Constraints from parsed VLM output
        # Normalise: model may output 'relations' (old) or 'spatial_relations'
        sr_raw = parsed.get("spatial_relations") or []
        if not sr_raw:
            rel_raw = parsed.get("relations")
            if isinstance(rel_raw, list):
                sr_raw = [r for r in rel_raw if isinstance(r, dict) and "predicate" in r]
        spatial_rels = []
        for rel in sr_raw:
            direction = rel.get("direction")
            if isinstance(direction, str):
                direction = direction.lower().strip()
            if direction not in {"left", "right"}:
                direction = None
            spatial_rels.append(SpatialTriplet(
                subject_type=parsed.get("ifc_class", ""),
                predicate=rel.get("predicate", "ADJACENT_TO").upper(),
                object_type=rel.get("object_type", ""),
                object_subtype=rel.get("object_subtype"),
                direction=direction,
                object_material=rel.get("object_material"),
                confidence=rel.get("confidence", 0.0),
            ))

        conf = max((r.confidence for r in spatial_rels), default=0.85)
        constraints = Constraints(
            storey_name=parsed.get("storey_name"),
            ifc_class=parsed.get("ifc_class"),
            space_name=parsed.get("space_name"),
            target_name_keyword=parsed.get("target_name_keyword"),
            position_context=parsed.get("position_context"),
            position_context_confidence=parsed.get("position_context_confidence"),
            position_context_source=parsed.get("position_context_source"),
            size_cluster=parsed.get("size_cluster"),
            size_band=parsed.get("size_band"),
            size_band_confidence=parsed.get("size_band_confidence"),
            size_band_source=parsed.get("size_band_source"),
            spatial_relations=spatial_rels,
            confidence=conf,
            source="g9_live" if parsed.get("size_band") else "lora_live",
        )

        # Query planning
        planner = QueryPlanner()
        plans = planner.plan(constraints)

        # P1+Rerank mode: skip P0 spatial_triplet / continuous_span plans entirely
        if skip_p0:
            plans = [p for p in plans if p.priority >= 1]

        # Load config for Neo4j
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "config.yaml"
        config = {}
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}

        # Init engine + backend
        engine = _get_engine(config, model_code=model_code)
        retrieval_mode = "neo4j" if getattr(engine, "neo4j_conn", None) else "memory"
        backend = RetrievalBackend(
            engine=engine,
            retrieval_mode=retrieval_mode,
            size_cluster_mode=size_cluster_mode,
            size_band_mode=size_band_mode,
        )

        def _normalize_query(query: str) -> str:
            return re.sub(r"\n\s+\n", "\n\n", (query or "").strip())

        def _execute_plan_with_query_capture(plan):
            neo4j_conn = getattr(engine, "neo4j_conn", None)
            if not neo4j_conn or not hasattr(neo4j_conn, "run"):
                return asyncio.run(backend.execute_plan(plan)), []

            captured_queries: list[str] = []
            original_conn = neo4j_conn

            class _QueryCaptureProxy:
                def __init__(self, wrapped_conn):
                    self._wrapped_conn = wrapped_conn

                def run(self, query, *args, **kwargs):
                    captured_queries.append(_normalize_query(str(query)))
                    return self._wrapped_conn.run(query, *args, **kwargs)

                def __getattr__(self, name):
                    return getattr(self._wrapped_conn, name)

            engine.neo4j_conn = _QueryCaptureProxy(original_conn)
            try:
                result = asyncio.run(backend.execute_plan(plan))
            finally:
                engine.neo4j_conn = original_conn
            return result, captured_queries

        # Execute all plans so the live UI can show the full waterfall, while
        # still honoring the first non-empty result as the actual winner.
        all_results = []
        winning_result = None
        winning_index = None
        for plan in plans:
            result, executed_queries = _execute_plan_with_query_capture(plan)
            result_dict = {
                "priority": plan.priority,
                "strategy": plan.strategy,
                "pool_size": result.pool_size,
                "raw_pool_size": result.raw_pool_size,
                "candidates": result.candidates,
                "backend": result.backend,
                "fallback_triggered": result.fallback_triggered,
                "strategy_actually_used": result.strategy_actually_used,
                "executed_queries": executed_queries,
            }
            all_results.append(result_dict)
            if result.pool_size > 0 and winning_result is None:
                winning_result = result_dict
                winning_index = len(all_results) - 1

        pool_guids = []
        if winning_result:
            pool_guids = [c.get("guid", "") for c in winning_result["candidates"] if c.get("guid")]

        # ── Optional Graph-RAG rerank on the live top-K shortlist ───────────────
        rerank_payload: dict | None = None
        if (
            enable_rerank
            and winning_result
            and len(pool_guids) > 1
            and retrieval_mode == "neo4j"
            and getattr(engine, "neo4j_conn", None) is not None
        ):
            site_paths: list[str] = []
            floorplan_path_for_rerank: str | None = None
            if site_image_bytes or floorplan_bytes:
                tmp_dir = Path("/tmp/mscd_demo_rerank") / str(int(time.time() * 1000))
                tmp_dir.mkdir(parents=True, exist_ok=True)
                for idx, blob in enumerate(site_image_bytes or []):
                    p = tmp_dir / f"site_{idx}.png"
                    p.write_bytes(blob)
                    site_paths.append(str(p))
                if floorplan_bytes:
                    p = tmp_dir / "floorplan.png"
                    p.write_bytes(floorplan_bytes)
                    floorplan_path_for_rerank = str(p)
            try:
                from src.neurosym.graph_rag_rerank import rerank_topk
                top_k = min(10, len(pool_guids))
                rr = rerank_topk(
                    graph=engine.neo4j_conn,
                    candidate_guids=pool_guids[:top_k],
                    candidate_fallbacks=winning_result["candidates"][:top_k],
                    constraints=constraints.model_dump(mode="json"),
                    query_text=query_text,
                    site_image_paths=site_paths,
                    floorplan_path=floorplan_path_for_rerank,
                    top_k=top_k,
                )
                rerank_payload = rr.to_dict()
                if not rr.failed and rr.ordered_guids:
                    # Reorder the winning candidate list according to Gemini's ranking,
                    # preserving the tail (rank > top_k) order.
                    by_guid = {c.get("guid"): c for c in winning_result["candidates"]}
                    head = [by_guid[g] for g in rr.ordered_guids if g in by_guid]
                    seen = {g for g in rr.ordered_guids}
                    tail = [c for c in winning_result["candidates"] if c.get("guid") not in seen]
                    winning_result["candidates"] = head + tail
                    pool_guids = [c.get("guid", "") for c in winning_result["candidates"] if c.get("guid")]
                    winning_result["rerank_applied"] = True
            except Exception as exc:
                rerank_payload = {"failed": True, "reason": f"exception: {exc}"}

        # GT is explicit from UI case selection/manual case ID.
        gt_guid_final = (gt_guid or "").strip()

        subgraph = None
        graph_snapshot = None
        if winning_result and pool_guids and retrieval_mode == "neo4j":
            subgraph = _fetch_neo4j_subgraph(engine, pool_guids[0])
            graph_snapshot = _fetch_neo4j_graph_snapshot(
                engine,
                anchor_guid=pool_guids[0],
                gt_guid=gt_guid_final,
            )

        guid_match = bool(gt_guid_final and pool_guids and pool_guids[0] == gt_guid_final)

        return {
            "plans": [{"priority": p.priority, "strategy": p.strategy,
                        "params": p.params, "expected_pool_size": p.expected_pool_size}
                       for p in plans],
            "results": all_results,
            "winning": winning_result,
            "winning_index": winning_index,
            "pool_guids": pool_guids,
            "ifc_model_code": model_code,
            "subgraph": subgraph,
            "graph_snapshot": graph_snapshot,
            "gt_guid": gt_guid_final,
            "guid_match": guid_match,
            "rerank": rerank_payload,
            "constraints_used": constraints.model_dump(mode="json"),
        }

    except Exception as e:
        st.warning(f"Symbolic retrieval failed: {e}")
        import traceback
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc(), language="text")
        return None


@st.cache_resource(show_spinner="Loading IFC engine...")
def _get_engine(_config: dict, model_code: str = "AP"):
    """Cache the IFCEngine instance."""
    from src.ifc_engine import IFCEngine
    from src.common.config import init_registry_llm
    from pathlib import Path

    neo4j_cfg = _config.get("neo4j", {})
    ifc_cfg = _config.get("ifc", {})
    repo_root = Path(__file__).parent.parent.parent

    models = ifc_cfg.get("models") or {}
    selected_rel = models.get(model_code) or ifc_cfg.get("model_path") or _IFC_VIEWER_REL_PATH["AP"]
    selected_path = Path(selected_rel)
    if selected_path.is_absolute():
        ifc_path = selected_path
    else:
        ifc_path = (repo_root / selected_path).resolve()

    # Keep robust fallbacks for local demo setups.
    fallback_paths = {
        "AP": [
            repo_root / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc",
            Path("/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc"),
        ],
        "BH": [
            repo_root / "data" / "ifc" / "BasicHouse.ifc",
            Path("/root/cmu/master_thesis/data_curation/ifc_models/BasicHouse.ifc"),
        ],
        "DXA": [
            repo_root / "data" / "ifc" / "Duplex_A_20110505.ifc",
            Path("/root/cmu/master_thesis/data_curation/ifc_models/Duplex_A_20110505.ifc"),
        ],
    }
    if not ifc_path.exists():
        for candidate in fallback_paths.get(model_code, []):
            if candidate.exists():
                ifc_path = candidate
                break
    if not ifc_path.exists():
        raise FileNotFoundError(
            f"IFC file not found for model {model_code}: {selected_rel}. "
            "Check config.yaml `ifc.models` paths."
        )

    neo4j_conn = None
    if neo4j_cfg.get("enabled", False):
        try:
            from py2neo import Graph
            neo4j_conn = Graph(
                neo4j_cfg.get("uri", "bolt://localhost:7687"),
                auth=(neo4j_cfg.get("user", "neo4j"), neo4j_cfg.get("password", "password")),
            )
            neo4j_conn.run("RETURN 1")
        except Exception as e:
            st.warning(f"Neo4j connection failed: {e} — using memory mode")
            neo4j_conn = None

    # Pass registry LLM so storey/space parsing can use API key from .env.
    registry_llm = init_registry_llm(_config)
    engine = IFCEngine(str(ifc_path), neo4j_conn=neo4j_conn, llm_client=registry_llm)
    return engine


def _fetch_neo4j_subgraph(engine, anchor_guid: str, limit: int = 80) -> dict | None:
    """Fetch 1-hop neighborhood around a candidate GUID for graph visualization."""
    neo4j_conn = getattr(engine, "neo4j_conn", None)
    if not neo4j_conn or not anchor_guid:
        return None

    query = """
    MATCH (a {guid: $guid})
    OPTIONAL MATCH (a)-[r]-(n)
    RETURN
      a.guid AS anchor_guid,
      a.name AS anchor_name,
      labels(a) AS anchor_labels,
      n.guid AS n_guid,
      n.name AS n_name,
      n.ifc_type AS n_type,
      labels(n) AS n_labels,
      type(r) AS rel_type,
      startNode(r).guid AS source_guid,
      endNode(r).guid AS target_guid
    LIMIT $limit
    """
    try:
        records = [dict(r) for r in neo4j_conn.run(query, guid=anchor_guid, limit=limit)]
    except Exception:
        return None

    if not records:
        return None

    nodes_by_guid: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for rec in records:
        a_guid = rec.get("anchor_guid")
        if a_guid and a_guid not in nodes_by_guid:
            anchor_labels = rec.get("anchor_labels") or []
            nodes_by_guid[a_guid] = {
                "guid": a_guid,
                "name": rec.get("anchor_name") or "",
                "type": (anchor_labels[0] if anchor_labels else "IFCElement"),
            }

        n_guid = rec.get("n_guid")
        if n_guid and n_guid not in nodes_by_guid:
            n_labels = rec.get("n_labels") or []
            nodes_by_guid[n_guid] = {
                "guid": n_guid,
                "name": rec.get("n_name") or "",
                "type": rec.get("n_type") or (n_labels[0] if n_labels else "IFCElement"),
            }

        rel_type = rec.get("rel_type")
        src = rec.get("source_guid")
        dst = rec.get("target_guid")
        if rel_type and src and dst:
            edge_key = (src, rel_type, dst)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    {
                        "source_guid": src,
                        "target_guid": dst,
                        "rel_type": rel_type,
                    }
                )

    return {
        "anchor_guid": anchor_guid,
        "nodes": list(nodes_by_guid.values())[:limit],
        "edges": edges[:limit],
    }


def _fetch_neo4j_graph_snapshot(
    engine,
    anchor_guid: str = "",
    gt_guid: str = "",
    limit_edges: int = 260,
    limit_nodes: int = 170,
) -> dict | None:
    """Fetch a bounded whole-IFC graph snapshot for demo visualization."""
    neo4j_conn = getattr(engine, "neo4j_conn", None)
    if not neo4j_conn:
        return None

    # Anchor-centric sample keeps the graph visually coherent around the predicted element.
    core_query = """
    MATCH (a {guid: $anchor_guid})
    MATCH p=(a)-[*1..2]-(n)
    UNWIND relationships(p) AS rel
    WITH DISTINCT rel
    WITH startNode(rel) AS s, endNode(rel) AS t, type(rel) AS rel_type
    WHERE s.guid IS NOT NULL AND t.guid IS NOT NULL
    RETURN
      s.guid AS source_guid,
      coalesce(s.name, s.ifc_type, head(labels(s)), 'node') AS source_name,
      coalesce(s.ifc_type, head(labels(s)), 'IFCElement') AS source_type,
      t.guid AS target_guid,
      coalesce(t.name, t.ifc_type, head(labels(t)), 'node') AS target_name,
      coalesce(t.ifc_type, head(labels(t)), 'IFCElement') AS target_type,
      rel_type AS rel_type
    LIMIT $core_limit
    """
    context_query = """
    MATCH (s)-[r]->(t)
    WHERE s.guid IS NOT NULL AND t.guid IS NOT NULL
    RETURN
      s.guid AS source_guid,
      coalesce(s.name, s.ifc_type, head(labels(s)), 'node') AS source_name,
      coalesce(s.ifc_type, head(labels(s)), 'IFCElement') AS source_type,
      t.guid AS target_guid,
      coalesce(t.name, t.ifc_type, head(labels(t)), 'node') AS target_name,
      coalesce(t.ifc_type, head(labels(t)), 'IFCElement') AS target_type,
      type(r) AS rel_type
    LIMIT $context_limit
    """
    records: list[dict] = []
    try:
        if anchor_guid:
            core_limit = max(60, min(limit_edges // 2, 140))
            core_records = [dict(r) for r in neo4j_conn.run(
                core_query,
                anchor_guid=anchor_guid,
                core_limit=core_limit,
            )]
            records.extend(core_records)
    except Exception:
        # Fall back to global context query.
        pass

    if gt_guid and gt_guid != anchor_guid:
        try:
            gt_core_limit = max(30, min(limit_edges // 3, 90))
            gt_records = [dict(r) for r in neo4j_conn.run(
                core_query,
                anchor_guid=gt_guid,
                core_limit=gt_core_limit,
            )]
            records.extend(gt_records)
        except Exception:
            pass

    remaining = max(limit_edges - len(records), 0)
    if remaining > 0:
        try:
            context_records = [dict(r) for r in neo4j_conn.run(
                context_query,
                context_limit=remaining,
            )]
            records.extend(context_records)
        except Exception:
            if not records:
                return None

    if not records:
        return None

    nodes_by_guid: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for rec in records:
        src = (rec.get("source_guid") or "").strip()
        dst = (rec.get("target_guid") or "").strip()
        rel = (rec.get("rel_type") or "REL").strip() or "REL"
        if not src or not dst:
            continue

        if src not in nodes_by_guid:
            nodes_by_guid[src] = {
                "guid": src,
                "name": rec.get("source_name") or "",
                "type": rec.get("source_type") or "IFCElement",
            }
        if dst not in nodes_by_guid:
            nodes_by_guid[dst] = {
                "guid": dst,
                "name": rec.get("target_name") or "",
                "type": rec.get("target_type") or "IFCElement",
            }

        edge_key = (src, rel, dst)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edges.append(
            {
                "source_guid": src,
                "target_guid": dst,
                "rel_type": rel,
            }
        )

    kept_guids = list(nodes_by_guid.keys())[:limit_nodes]
    keep_set = set(kept_guids)
    nodes = [nodes_by_guid[g] for g in kept_guids]
    edges = [
        e for e in edges
        if e["source_guid"] in keep_set and e["target_guid"] in keep_set
    ][:limit_edges]

    # Ensure GT node appears in graph even if its edges are outside sampled slice.
    if gt_guid and gt_guid not in keep_set:
        try:
            gt_node_query = """
            MATCH (g {guid: $guid})
            RETURN
              g.guid AS guid,
              coalesce(g.name, g.ifc_type, head(labels(g)), 'node') AS name,
              coalesce(g.ifc_type, head(labels(g)), 'IFCElement') AS type
            LIMIT 1
            """
            gt_rows = [dict(r) for r in neo4j_conn.run(gt_node_query, guid=gt_guid)]
            if gt_rows:
                gt_row = gt_rows[0]
                nodes = [n for n in nodes if n.get("guid") != gt_guid]
                nodes.insert(0, {
                    "guid": gt_row.get("guid") or gt_guid,
                    "name": gt_row.get("name") or "",
                    "type": gt_row.get("type") or "IFCElement",
                })
                nodes = nodes[:limit_nodes]
        except Exception:
            pass

    return {
        "nodes": nodes,
        "edges": edges,
    }


def _build_viewer_url(
    static_base_url: str,
    pool_guids: list[str],
    target_guid: str = "",
    gt_guid: str = "",
    guid_match: bool = True,
    ifc_model_code: str = "AP",
) -> str:
    """Build URL for the 3D viewer with candidate GUIDs highlighted."""
    ifc_rel = _IFC_VIEWER_REL_PATH.get(ifc_model_code, _IFC_VIEWER_REL_PATH["AP"])
    ifc_url = static_base_url + "/" + ifc_rel
    viewer_base = static_base_url + "/demo/static/test_viewer.html"
    # Always pass GT GUID so viewer receives the true per-case target, even when GT == prediction.
    gt_param = gt_guid or ""
    # Highlighting cost scales with pool size on web-ifc; cap to top-10 so
    # the iframe paints fast. The full ranked shortlist still lives in the
    # **Confirm match** table — this is just the visual pool.
    # `_v` busts the browser cache when test_viewer.html or viewer.bundle.js
    # is rebuilt — without it the iframe silently loads the stale cached
    # copy and edits appear to do nothing.
    import time as _time
    params = {
        "ifc": ifc_url,
        "target": target_guid,
        "gt": gt_param,
        "match": "1" if guid_match else "0",
        "pool": ",".join(pool_guids[:10]),
        "base": static_base_url + "/demo/static",
        "_v": str(int(_time.time() // 60)),
    }
    return viewer_base + "?" + urlencode(params)


# ══════════════════════════════════════════════════════════════════════════════
# Symbolic graph panel — Cytoscape.js HTML template
# ══════════════════════════════════════════════════════════════════════════════
_SYMBOLIC_GRAPH_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0;background:#f8fafc;color:#0f172a;}
  .root{padding:10px;height:540px;display:flex;flex-direction:column;}
  .bar{display:flex;gap:4px;margin-bottom:10px;align-items:center;}
  .sb{flex:1;padding:7px 10px;font-size:11px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;border-radius:6px;color:#64748b;font-weight:600;font-family:inherit;}
  .sb.active{background:#3b82f6;color:#fff;border-color:#3b82f6;}
  .sb:hover:not(.active){border-color:#93c5fd;color:#0f172a;}
  .play{padding:7px 14px;font-size:11px;border-radius:6px;background:#15803d;color:#fff;border:none;cursor:pointer;font-weight:600;font-family:inherit;}
  .play:hover{background:#166534;}
  .content{flex:1;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px;overflow:hidden;display:flex;flex-direction:column;}
  pre.code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;line-height:1.55;margin:0;white-space:pre-wrap;word-break:break-word;}
  .hl-ifc{background:#dbeafe;color:#1e40af;padding:1px 5px;border-radius:3px;font-weight:600;}
  .hl-storey{background:#dcfce7;color:#166534;padding:1px 5px;border-radius:3px;font-weight:600;}
  .hl-kw{background:#fef3c7;color:#92400e;padding:1px 5px;border-radius:3px;font-weight:600;}
  .hl-pred{background:#fce7f3;color:#9d174d;padding:1px 5px;border-radius:3px;font-weight:600;}
  .hl-obj{background:#f3e8ff;color:#6b21a8;padding:1px 5px;border-radius:3px;font-weight:600;}
  .split{display:grid;grid-template-columns:1fr 1fr;gap:14px;flex:1;overflow:hidden;}
  .split>div{overflow:auto;}
  .col-label{font-size:10px;text-transform:uppercase;letter-spacing:0.5px;color:#64748b;margin-bottom:6px;font-weight:600;}
  .stage-hint{font-size:11px;color:#64748b;margin-bottom:8px;}
  .legend{display:flex;flex-wrap:wrap;gap:10px;font-size:10px;color:#64748b;padding-top:6px;}
  .legend span{display:inline-flex;align-items:center;gap:4px;}
  .legend i{width:9px;height:9px;border-radius:50%;display:inline-block;}
  .step-bar{display:flex;gap:6px;margin-bottom:6px;align-items:center;font-size:10px;}
  .step{padding:3px 9px;font-size:10px;border:1px solid #e2e8f0;background:#fff;cursor:pointer;border-radius:5px;color:#475569;font-weight:600;font-family:inherit;}
  .step:hover{border-color:#3b82f6;color:#3b82f6;}
  .step.done{background:#f0fdf4;border-color:#86efac;color:#15803d;}
  #cy{width:100%;height:100%;background:#fafafa;border-radius:6px;}
  .empty-note{font-size:10px;color:#94a3b8;font-style:italic;margin-left:auto;}
  .synth-banner{background:#fef3c7;border:1px solid #fbbf24;color:#92400e;font-size:11px;padding:7px 11px;border-radius:6px;margin-bottom:10px;font-weight:600;display:flex;align-items:center;gap:8px;line-height:1.4;}
  .synth-banner .icon{font-size:14px;}
  .synth-banner .body{flex:1;}
  .synth-banner b{color:#7c2d12;letter-spacing:0.4px;}
  .synth-banner em{font-style:normal;color:#a16207;font-weight:400;}
  .cy-wrap{position:relative;height:100%;}
  .cy-wrap.synth::after{content:"SYNTHETIC DEMO";position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) rotate(-22deg);font-size:54px;font-weight:900;color:rgba(251,191,36,0.22);pointer-events:none;z-index:1;letter-spacing:8px;font-family:inherit;white-space:nowrap;}
</style></head><body><div class="root">
  <div class="bar">
    <button class="sb" data-stage="1">1 · JSON</button>
    <button class="sb" data-stage="2">2 · JSON → Cypher</button>
    <button class="sb" data-stage="3">3 · Cypher on Graph</button>
    <button class="sb" data-stage="4">4 · Attribute filter</button>
    <button class="sb" data-stage="5">5 · Spatial topology</button>
    <button class="sb" data-stage="6">6 · Answer</button>
    <button class="play" id="play">▶ Play</button>
  </div>
  <div class="content" id="content"></div>
</div>
<script src="https://unpkg.com/cytoscape@3.30.0/dist/cytoscape.min.js"></script>
<script>
const PAYLOAD = __PAYLOAD__;
const SYNTH = (PAYLOAD.snapshot.nodes || []).length === 0;
let stage = 1, cy = null, playTimer = null;

function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function escRe(s){return String(s).replace(/[.*+?^${}()|[\]\\]/g,'\\$&');}

function jsonHtml(){
  const p = PAYLOAD.parsed || {};
  const L = ['{'];
  if (p.ifc_class) L.push('  "ifc_class": <span class="hl-ifc">"'+esc(p.ifc_class)+'"</span>,');
  if (p.storey_name) L.push('  "storey_name": <span class="hl-storey">"'+esc(p.storey_name)+'"</span>,');
  if (p.target_name_keyword) L.push('  "target_name_keyword": <span class="hl-kw">"'+esc(p.target_name_keyword)+'"</span>,');
  if (p.space_name) L.push('  "space_name": "'+esc(p.space_name)+'",');
  const rels = p.spatial_relations || [];
  if (rels.length){
    L.push('  "spatial_relations": [');
    rels.forEach((r,i)=>{
      L.push('    {');
      L.push('      "predicate": <span class="hl-pred">"'+esc(r.predicate||'')+'"</span>,');
      L.push('      "object_type": <span class="hl-obj">"'+esc(r.object_type||'')+'"</span>'+(r.object_material?',':''));
      if (r.object_material) L.push('      "object_material": "'+esc(r.object_material)+'"');
      L.push('    }'+(i<rels.length-1?',':''));
    });
    L.push('  ]');
  } else {
    L.push('  "spatial_relations": []');
  }
  L.push('}');
  return L.join('\n');
}

function cypherHtml(){
  let txt = esc(PAYLOAD.cypher || '// No Cypher captured for this run.');
  const p = PAYLOAD.parsed || {};
  const terms = [
    [p.ifc_class, 'hl-ifc'],
    [p.storey_name, 'hl-storey'],
    [p.target_name_keyword, 'hl-kw'],
  ];
  (p.spatial_relations || []).forEach(r=>{
    terms.push([r.predicate, 'hl-pred']);
    terms.push([r.object_type, 'hl-obj']);
  });
  terms.forEach(([term, cls])=>{
    if (!term) return;
    const re = new RegExp('('+escRe(String(term))+')', 'g');
    txt = txt.replace(re, '<span class="'+cls+'">$1</span>');
  });
  return txt;
}

function renderS1(){
  return ''
    + '<div class="stage-hint">Stage 1 — extracted constraint fields highlighted by role.</div>'
    + '<div style="flex:1;overflow:auto;"><pre class="code">'+jsonHtml()+'</pre></div>'
    + '<div class="legend">'
    + '<span><i style="background:#dbeafe;"></i>ifc_class</span>'
    + '<span><i style="background:#dcfce7;"></i>storey</span>'
    + '<span><i style="background:#fef3c7;"></i>name keyword</span>'
    + '<span><i style="background:#fce7f3;"></i>predicate</span>'
    + '<span><i style="background:#f3e8ff;"></i>object_type</span>'
    + '</div>';
}

function renderS2(){
  const wp = PAYLOAD.winningPlan || {};
  const planTag = wp.priority!=='' ? ('P'+esc(wp.priority)+' · '+esc(wp.strategy||'')) : 'no plan';
  const noCypher = !PAYLOAD.cypher;
  const banner = noCypher
    ? '<div class="synth-banner"><span class="icon">⚠</span><div class="body"><b>NO CYPHER CAPTURED</b> &nbsp;<em>Neo4j was unavailable for this run — the Cypher pane below shows placeholder text only, not a real query.</em></div></div>'
    : '';
  return ''
    + banner
    + '<div class="stage-hint">Stage 2 — each highlighted JSON field maps to its <code>WHERE</code> / <code>MATCH</code> clause.</div>'
    + '<div class="split">'
    +   '<div><div class="col-label">JSON (constraints)</div><pre class="code">'+jsonHtml()+'</pre></div>'
    +   '<div><div class="col-label">Cypher · '+planTag+'</div><pre class="code">'+cypherHtml()+'</pre></div>'
    + '</div>';
}

function renderS3(){
  const banner = SYNTH
    ? '<div class="synth-banner"><span class="icon">⚠</span><div class="body"><b>SYNTHETIC GRAPH</b> &nbsp;<em>graph_snapshot is empty (Neo4j unavailable). The 13-node graph below is illustrative only — it is NOT loaded from your IFC.</em></div></div>'
    : '';
  const wrapClass = SYNTH ? 'cy-wrap synth' : 'cy-wrap';
  return ''
    + banner
    + '<div class="step-bar">'
    +   '<span style="color:#64748b;">Stage 3 — apply filters in order:</span>'
    +   '<button class="step" data-step="storey">① storey filter</button>'
    +   '<button class="step" data-step="type">② type filter</button>'
    +   '<button class="step" data-step="edges">③ predicate edges</button>'
    +   '<button class="step" data-step="reset">reset</button>'
    + '</div>'
    + '<div style="flex:1;"><div class="'+wrapClass+'"><div id="cy"></div></div></div>'
    + '<div class="legend">'
    + '<span><i style="background:#cbd5e1;"></i>dim</span>'
    + '<span><i style="background:#22c55e;"></i>storey match</span>'
    + '<span><i style="background:#3b82f6;"></i>type match</span>'
    + '<span><i style="background:#ef4444;"></i>candidate</span>'
    + '<span><i style="background:#f59e0b;"></i>★ target (anchor)</span>'
    + '<span><i style="background:#f59e0b;height:2px;border-radius:0;"></i>predicate edge</span>'
    + '</div>';
}

function renderS4Attr(){
  const n = PAYLOAD.attribute_pool_size || 0;
  const ifc = PAYLOAD.ifc_class || '?';
  const storey = PAYLOAD.storey_name || '?';
  const banner = SYNTH
    ? '<div class="synth-banner"><span class="icon">⚠</span><div class="body"><b>SYNTHETIC GRAPH</b> &nbsp;<em>graph_snapshot is empty — the highlight below is illustrative only.</em></div></div>'
    : '';
  const wrapClass = SYNTH ? 'cy-wrap synth' : 'cy-wrap';
  return ''
    + banner
    + '<div class="stage-hint">Stage 4 — attribute filter on graph: <span class="hl-storey">storey = '+esc(storey)+'</span> AND <span class="hl-ifc">ifc_class = '+esc(ifc)+'</span> &nbsp;→&nbsp; <strong style="color:#15803d;">'+n+' candidates</strong></div>'
    + '<div style="flex:1;"><div class="'+wrapClass+'"><div id="cy"></div></div></div>'
    + '<div class="legend">'
    + '<span><i style="background:#cbd5e1;"></i>dim</span>'
    + '<span><i style="background:#22c55e;"></i>storey + type match</span>'
    + '</div>';
}

function renderS5Topo(){
  const attrN = PAYLOAD.attribute_pool_size || 0;
  const topoN = PAYLOAD.topology_pool_size || (PAYLOAD.pool_guids ? PAYLOAD.pool_guids.length : 0);
  const rels = (PAYLOAD.parsed && PAYLOAD.parsed.spatial_relations) || [];
  const r0 = rels[0] || {};
  const relText = r0.predicate ? ('<span class="hl-pred">'+esc(r0.predicate)+'</span> → <span class="hl-obj">'+esc(r0.object_type||'?')+'</span>') : 'spatial relations';
  const banner = SYNTH
    ? '<div class="synth-banner"><span class="icon">⚠</span><div class="body"><b>SYNTHETIC GRAPH</b> &nbsp;<em>topology overlay is illustrative only.</em></div></div>'
    : '';
  const wrapClass = SYNTH ? 'cy-wrap synth' : 'cy-wrap';
  return ''
    + banner
    + '<div class="stage-hint">Stage 5 — spatial topology compresses the attribute pool: '+attrN+' &nbsp;→&nbsp; <strong style="color:#dc2626;">'+topoN+'</strong> via '+relText+'</div>'
    + '<div style="flex:1;"><div class="'+wrapClass+'"><div id="cy"></div></div></div>'
    + '<div class="legend">'
    + '<span><i style="background:#f59e0b;"></i>★ anchor (object_type)</span>'
    + '<span><i style="background:#ef4444;"></i>topology candidate (strict P0)</span>'
    + '<span><i style="background:#f59e0b;height:2px;border-radius:0;"></i>predicate edge</span>'
    + '</div>';
}

function renderS6(){
  const sub = PAYLOAD.subgraph || {};
  const empty = (sub.nodes || []).length === 0;
  const banner = empty
    ? '<div class="synth-banner"><span class="icon">⚠</span><div class="body"><b>SYNTHETIC SUBGRAPH</b> &nbsp;<em>1-hop subgraph is empty (Neo4j unavailable). The anchor and neighbors below are illustrative only — they are NOT loaded from your IFC.</em></div></div>'
    : '';
  const wrapClass = empty ? 'cy-wrap synth' : 'cy-wrap';
  return ''
    + banner
    + '<div class="stage-hint">Stage 6 — answer: anchor + 1-hop neighbors. Satisfying edges (FILLS / ADJACENT_TO / CONTINUOUS) drawn solid green.</div>'
    + '<div style="flex:1;"><div class="'+wrapClass+'"><div id="cy"></div></div></div>';
}

function buildSnapshotElements(){
  const nodes = (PAYLOAD.snapshot.nodes || []).map(n=>({
    data:{id:n.guid, label:(n.name||n.type||n.guid.slice(0,6)).slice(0,18), type:n.type||'IFCElement', guid:n.guid}
  }));
  const edges = (PAYLOAD.snapshot.edges || []).map(e=>({
    data:{id:e.source_guid+'>'+e.rel_type+'>'+e.target_guid, source:e.source_guid, target:e.target_guid, rel:e.rel_type}
  }));
  if (nodes.length > 0) return {nodes, edges};
  // synth demo — anchor + 12 distractors
  const anchor = PAYLOAD.anchor_guid || 'synth_anchor';
  const ifc = PAYLOAD.ifc_class || 'IfcElement';
  const sn = [{data:{id:anchor, label:'target', type:ifc, guid:anchor}}];
  for (let i = 0; i < 12; i++){
    const id = 'synth_'+i;
    const t = i % 3 === 0 ? ifc : (i % 3 === 1 ? 'IfcWall' : 'IfcSpace');
    sn.push({data:{id, label:t.replace('Ifc','')+(i), type:t, guid:id}});
  }
  const se = [
    {data:{id:'se1', source:'synth_0', target:anchor, rel:'ADJACENT_TO'}},
    {data:{id:'se2', source:anchor, target:'synth_3', rel:'FILLS'}},
    {data:{id:'se3', source:'synth_6', target:'synth_3', rel:'CONTAINS'}},
    {data:{id:'se4', source:'synth_9', target:anchor, rel:'ADJACENT_TO'}},
    {data:{id:'se5', source:'synth_1', target:'synth_4', rel:'CONTAINS'}},
  ];
  return {nodes:sn, edges:se};
}

function buildSubgraphElements(){
  const sub = PAYLOAD.subgraph || {};
  const nodes = (sub.nodes || []).map(n=>({
    data:{id:n.guid, label:(n.name||n.type||n.guid.slice(0,6)).slice(0,20), type:n.type||'IFCElement', guid:n.guid}
  }));
  const edges = (sub.edges || []).map(e=>({
    data:{id:e.source_guid+'>'+e.rel_type+'>'+e.target_guid, source:e.source_guid, target:e.target_guid, rel:e.rel_type}
  }));
  if (nodes.length > 0) return {nodes, edges};
  const anchor = PAYLOAD.anchor_guid || 'synth_anchor';
  const ifc = PAYLOAD.ifc_class || 'IfcElement';
  return {
    nodes:[
      {data:{id:anchor, label:ifc.replace('Ifc',''), type:ifc, guid:anchor}},
      {data:{id:'demo_wall', label:'Wall', type:'IfcWallStandardCase', guid:'demo_wall'}},
      {data:{id:'demo_space', label:'Space', type:'IfcSpace', guid:'demo_space'}},
    ],
    edges:[
      {data:{id:'de1', source:anchor, target:'demo_wall', rel:'FILLS'}},
      {data:{id:'de2', source:anchor, target:'demo_space', rel:'ADJACENT_TO'}},
    ],
  };
}

function initS3(){
  const {nodes, edges} = buildSnapshotElements();
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [...nodes, ...edges],
    style: [
      {selector:'node', style:{'background-color':'#cbd5e1','label':'data(label)','font-size':8,'color':'#94a3b8','width':16,'height':16,'text-valign':'bottom','text-margin-y':3,'text-max-width':80,'text-wrap':'ellipsis'}},
      {selector:'edge', style:{'line-color':'#e2e8f0','width':1,'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#e2e8f0','arrow-scale':0.5}},
      {selector:'node.storey', style:{'background-color':'#22c55e','color':'#15803d'}},
      {selector:'node.type', style:{'background-color':'#3b82f6','color':'#1e40af','width':20,'height':20}},
      {selector:'node.cand', style:{'background-color':'#ef4444','width':24,'height':24,'color':'#991b1b'}},
      {selector:'node.anchor', style:{'background-color':'#f59e0b','width':38,'height':38,'border-width':3,'border-color':'#92400e','color':'#92400e','font-size':10,'font-weight':'bold','z-index':99}},
      {selector:'edge.pred', style:{'line-color':'#f59e0b','width':2.4,'target-arrow-color':'#f59e0b','arrow-scale':0.9}},
    ],
    layout: {name:'cose', animate:false, idealEdgeLength:55, nodeRepulsion:3500, fit:true, padding:20},
  });

  // Pre-highlight the target so the audience can locate it before any filter runs.
  const a0 = PAYLOAD.anchor_guid ? cy.getElementById(PAYLOAD.anchor_guid) : null;
  if (a0 && a0.length) {
    a0.data('label', '★ TARGET');
    a0.addClass('anchor');
    // Gentle pulse to draw the eye.
    (function pulse(){
      a0.animate({style:{'border-width':6}}, {duration:600})
        .animate({style:{'border-width':3}}, {duration:600, complete:pulse});
    })();
    cy.center(a0);
  }

  const stepBtns = document.querySelectorAll('.step');
  function markDone(name){
    stepBtns.forEach(b => { if (b.dataset.step === name) b.classList.add('done'); });
  }

  document.querySelector('.step[data-step="storey"]').onclick = () => {
    const target = (PAYLOAD.storey_name || '').toLowerCase();
    cy.batch(() => {
      cy.nodes().forEach((n, i) => {
        // Real storey property if present; otherwise heuristic by index
        const ns = (n.data('storey') || '').toLowerCase();
        const pick = target ? (ns && ns.includes(target.split(' ').pop())) : (i % 3 === 0);
        if (pick || (!target && i % 3 === 0)) n.addClass('storey');
      });
    });
    markDone('storey');
  };
  document.querySelector('.step[data-step="type"]').onclick = () => {
    const ifc = (PAYLOAD.ifc_class || '').toLowerCase();
    cy.batch(() => {
      cy.nodes().forEach(n => {
        const t = (n.data('type') || '').toLowerCase();
        if (ifc && t && t.indexOf(ifc) === 0) n.addClass('type');
      });
    });
    markDone('type');
  };
  document.querySelector('.step[data-step="edges"]').onclick = () => {
    const pool = new Set(PAYLOAD.pool_guids || []);
    cy.batch(() => {
      cy.nodes().forEach(n => {
        if (pool.has(n.data('guid'))) n.addClass('cand');
      });
      const a = PAYLOAD.anchor_guid && cy.getElementById(PAYLOAD.anchor_guid);
      if (a && a.length) a.addClass('anchor');
      cy.edges().forEach(e => {
        const r = e.data('rel');
        if (r === 'FILLS' || r === 'ADJACENT_TO' || r === 'CONTINUOUS') e.addClass('pred');
      });
    });
    markDone('edges');
  };
  document.querySelector('.step[data-step="reset"]').onclick = () => {
    cy.batch(() => {
      cy.nodes().removeClass('storey type cand anchor');
      cy.edges().removeClass('pred');
    });
    stepBtns.forEach(b => b.classList.remove('done'));
  };
}

function initS4Attr(){
  // Auto-animated attribute filter: dim the world, then pulse-highlight every
  // node whose storey + type match the constraints. Runs without requiring
  // the audience to click — the animation IS the explanation.
  const {nodes, edges} = buildSnapshotElements();
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [...nodes, ...edges],
    style: [
      {selector:'node', style:{'background-color':'#cbd5e1','label':'data(label)','font-size':8,'color':'#94a3b8','width':16,'height':16,'text-valign':'bottom','text-margin-y':3,'text-max-width':80,'text-wrap':'ellipsis'}},
      {selector:'edge', style:{'line-color':'#e2e8f0','width':1,'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#e2e8f0','arrow-scale':0.5}},
      {selector:'node.attr', style:{'background-color':'#22c55e','color':'#15803d','width':22,'height':22,'border-width':2,'border-color':'#15803d'}},
    ],
    layout: {name:'cose', animate:false, idealEdgeLength:55, nodeRepulsion:3500, fit:true, padding:20},
  });
  const ifc = (PAYLOAD.ifc_class || '').toLowerCase();
  const storey = (PAYLOAD.storey_name || '').toLowerCase();
  const attrSet = new Set(PAYLOAD.attribute_pool_guids || []);
  // Step in after a short delay so the audience sees the dim baseline first.
  setTimeout(() => {
    cy.batch(() => {
      cy.nodes().forEach((n, i) => {
        const guid = n.data('guid');
        if (attrSet.size > 0) {
          if (attrSet.has(guid)) n.addClass('attr');
          return;
        }
        // Fallback heuristic when explicit attribute_pool_guids weren't sent.
        const t = (n.data('type') || '').toLowerCase();
        const ns = (n.data('storey') || '').toLowerCase();
        const typeMatch = ifc && t && t.indexOf(ifc) === 0;
        const storeyMatch = !storey || (ns && ns.includes(storey.split(' ').pop()));
        if (typeMatch && storeyMatch) n.addClass('attr');
        else if (!ifc && !storey && i % 3 === 0) n.addClass('attr');
      });
    });
  }, 350);
}

function initS5Topo(){
  // Stage 5 renders a *focused* subgraph: only the strict-P0 topology
  // candidates plus a synthetic anchor node representing the spatial-relation
  // object_type, with predicate edges from anchor → each candidate. The full
  // building graph is intentionally hidden — the audience already saw the
  // attribute pool in Stage 4.
  const topoGuids = (PAYLOAD.topology_pool_guids || PAYLOAD.pool_guids || []).slice(0, 30);
  const rels = (PAYLOAD.parsed && PAYLOAD.parsed.spatial_relations) || [];
  const r0 = rels[0] || {};
  const predicate = r0.predicate || 'TOPOLOGY';
  const anchorType = r0.object_type || 'Anchor';
  const anchorId = '__topology_anchor__';

  // Look up each candidate's metadata from the snapshot — falls back to a
  // generic label when the snapshot didn't include the node.
  const snapshotById = new Map();
  (PAYLOAD.snapshot.nodes || []).forEach(n => { snapshotById.set(n.guid, n); });

  const subjectType = PAYLOAD.ifc_class || 'IfcElement';

  const nodes = [
    {data:{id:anchorId, label:'★ ' + anchorType, type:anchorType, guid:anchorId, role:'anchor'}},
  ];
  const edges = [];
  if (topoGuids.length === 0) {
    // No strict-topology hits — show a placeholder so the panel isn't empty.
    nodes.push({data:{id:'__none__', label:'No exact topology matches', type:'note', guid:'__none__', role:'empty'}});
    edges.push({data:{id:'e_none', source:anchorId, target:'__none__', rel:predicate}});
  } else {
    topoGuids.forEach((guid, idx) => {
      const meta = snapshotById.get(guid) || {};
      const label = (meta.name || meta.type || subjectType).slice(0, 20);
      const type = meta.type || subjectType;
      nodes.push({data:{id:guid, label:label, type:type, guid:guid, role:'cand'}});
      edges.push({data:{id:'e_'+idx, source:anchorId, target:guid, rel:predicate}});
    });
  }

  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [...nodes, ...edges],
    style: [
      {selector:'node', style:{'background-color':'#ef4444','label':'data(label)','font-size':10,'color':'#7f1d1d','width':30,'height':30,'text-valign':'bottom','text-margin-y':4,'text-max-width':140,'text-wrap':'ellipsis','border-width':2,'border-color':'#991b1b'}},
      {selector:'node[role = "anchor"]', style:{'background-color':'#f59e0b','color':'#92400e','width':54,'height':54,'border-width':3,'border-color':'#92400e','font-size':12,'font-weight':'bold','z-index':99}},
      {selector:'node[role = "empty"]', style:{'background-color':'#f1f5f9','color':'#64748b','width':140,'height':30,'shape':'roundrectangle','border-color':'#cbd5e1','font-size':10}},
      {selector:'edge', style:{'line-color':'#f59e0b','width':2.4,'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#f59e0b','arrow-scale':0.9,'label':'data(rel)','font-size':9,'color':'#92400e','text-background-color':'#fff','text-background-opacity':0.9,'text-background-padding':2}},
    ],
    layout: {
      name: 'concentric',
      animate: false,
      concentric: function(node) { return node.data('role') === 'anchor' ? 10 : 1; },
      levelWidth: function() { return 1; },
      minNodeSpacing: 35,
      fit: true,
      padding: 30,
    },
  });
}

function initS6(){
  const {nodes, edges} = buildSubgraphElements();
  const anchorId = PAYLOAD.anchor_guid || (nodes[0] && nodes[0].data.id) || '';
  cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [...nodes, ...edges],
    style: [
      {selector:'node', style:{'background-color':'#3b82f6','label':'data(label)','font-size':10,'color':'#0f172a','width':28,'height':28,'text-valign':'bottom','text-margin-y':4,'text-max-width':120,'text-wrap':'ellipsis'}},
      {selector:'node[guid = "'+anchorId+'"]', style:{'background-color':'#f59e0b','width':44,'height':44,'border-width':3,'border-color':'#92400e','font-size':11,'font-weight':'bold'}},
      {selector:'edge', style:{'line-color':'#94a3b8','width':1.5,'curve-style':'bezier','target-arrow-shape':'triangle','target-arrow-color':'#94a3b8','arrow-scale':0.7,'label':'data(rel)','font-size':9,'color':'#475569','text-background-color':'#fff','text-background-opacity':0.85,'text-background-padding':2}},
      {selector:'edge[rel = "FILLS"], edge[rel = "ADJACENT_TO"], edge[rel = "CONTINUOUS"]', style:{'line-color':'#15803d','target-arrow-color':'#15803d','width':3,'color':'#15803d'}},
    ],
    layout: {name:'cose', animate:false, idealEdgeLength:90, nodeRepulsion:9000, fit:true, padding:30},
  });
}

function setStage(s){
  stage = s;
  document.querySelectorAll('.sb[data-stage]').forEach(b => {
    b.classList.toggle('active', Number(b.dataset.stage) === s);
  });
  const c = document.getElementById('content');
  if (s === 1) c.innerHTML = renderS1();
  else if (s === 2) c.innerHTML = renderS2();
  else if (s === 3) { c.innerHTML = renderS3(); setTimeout(initS3, 30); }
  else if (s === 4) { c.innerHTML = renderS4Attr(); setTimeout(initS4Attr, 30); }
  else if (s === 5) { c.innerHTML = renderS5Topo(); setTimeout(initS5Topo, 30); }
  else if (s === 6) { c.innerHTML = renderS6(); setTimeout(initS6, 30); }
}

document.querySelectorAll('.sb[data-stage]').forEach(b => {
  b.onclick = () => setStage(Number(b.dataset.stage));
});

document.getElementById('play').onclick = () => {
  const btn = document.getElementById('play');
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    btn.textContent = '▶ Play';
    return;
  }
  btn.textContent = '⏸ Stop';
  let s = stage;
  playTimer = setInterval(() => {
    s = s >= 6 ? 1 : s + 1;
    setStage(s);
  }, 2400);
};

setStage(1);
</script></body></html>
"""
