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
# Main render
# ══════════════════════════════════════════════════════════════════════════════

def render(
    *,
    static_base_url: str = "",
    trace: dict | None = None,
    case_id: str = "",
) -> None:
    # Reset stale results when sidebar case changes.
    if st.session_state.get("live_case_id") != case_id:
        old_case_id = st.session_state.get("live_case_id", "")
        st.session_state.pop("last_inference", None)
        st.session_state.pop("last_retrieval", None)
        st.session_state.pop("last_live_inputs", None)
        st.session_state.pop("approved_guid", None)
        st.session_state.pop("approval_candidate_guid", None)
        st.session_state.pop("rejected_candidate_guids", None)
        st.session_state.pop("live_candidate_visible_count", None)
        for flow_scope in ("live", "trace"):
            st.session_state.pop(
                _live_flow_state_key(flow_scope, old_case_id, "candidate_visible_count"),
                None,
            )
        st.session_state["live_case_id"] = case_id

    # ── Two-column layout ─────────────────────────────────────────────────
    #   LEFT  (1): compact multimodal evidence input
    #   RIGHT (2): live demo flow
    left_col, right_col = st.columns([1, 2], gap="medium")

    # ═════════════════════════════════════════════════════════════════════
    # LEFT — Evidence input
    # ═════════════════════════════════════════════════════════════════════
    with left_col:
        with st.container(border=True):
            chat_text = st.text_area(
                "Field message",
                value="",
                height=120,
                key="inference_chat",
                placeholder="Describe what the field team is seeing and what element they need help locating.",
            )

            uploaded_files = st.file_uploader(
                "Images",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="inference_images",
                help="Site photos, floorplans, or any visual evidence.",
            )
            if uploaded_files:
                thumb_cols = st.columns(min(len(uploaded_files), 4))
                for i, f in enumerate(uploaded_files):
                    thumb_cols[i % 4].image(f, caption=f.name, width=90)

            run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

            with st.expander("Settings", expanded=False):
                model_variant = st.selectbox(
                    "VLM Model",
                    [
                        "G8 — Full Topology (P0 spatial + P1…)",
                        "G8 — P1 + Rerank (skip spatial, storey+type first)",
                        "G4-Ultimate — Full Topology (P0 spatial + P1…)",
                        "G4-Ultimate — P1 + Rerank (skip spatial, storey+type first)",
                    ],
                    key="inf_model_variant",
                )
                ifc_model = st.selectbox(
                    "IFC Model",
                    ["AP (AdvancedProject)", "BH (BasicHouse)", "DXA (Duplex)"],
                    key="inf_ifc_model",
                )
            model_code = (st.session_state.get("inf_ifc_model") or "AP").split(" ", 1)[0].strip()

            with st.expander("4D Metadata", expanded=False):
                meta_c1, meta_c2 = st.columns(2)
                meta_c1.text_input("Storey", value="", key="inf_storey")
                meta_c2.text_input("Phase", value="", key="inf_phase")
                meta_c1.selectbox(
                    "Task Status",
                    ["", "IN_PROGRESS", "PENDING_INSPECTION", "REVIEW_REQUIRED", "ON_HOLD"],
                    key="inf_status",
                )

            storey      = st.session_state.get("inf_storey",  "")
            phase       = st.session_state.get("inf_phase",   "")
            task_status = st.session_state.get("inf_status",  "")

            with st.expander("Ground Truth (optional)", expanded=False):
                gt_mode_widget = st.radio(
                    "GT source",
                    ["None", "Sidebar case", "Case ID", "Direct GUID"],
                    horizontal=True,
                    key="inf_gt_source_mode",
                )
                if gt_mode_widget == "Sidebar case" and case_id:
                    st.caption(f"Using: `{case_id}`")
                elif gt_mode_widget == "Case ID":
                    st.text_input(
                        "Case ID",
                        value=st.session_state.get("inf_gt_case_manual", ""),
                        placeholder="e.g. AP_SK_282  or  SYNTH_V3_002_SK_002",
                        key="inf_gt_case_manual",
                    )
                elif gt_mode_widget == "Direct GUID":
                    st.text_input(
                        "Target GUID",
                        value=st.session_state.get("inf_gt_guid_manual", ""),
                        placeholder="e.g. 2BLn4xX2vF_gM9G5wfbU5X",
                        key="inf_gt_guid_manual",
                    )

            gt_mode = st.session_state.get("inf_gt_source_mode", "None")
            if gt_mode == "Sidebar case":
                gt_case_id = case_id
            elif gt_mode == "Case ID":
                gt_case_id = st.session_state.get("inf_gt_case_manual", "").strip()
            else:
                gt_case_id = ""

            if gt_mode == "Direct GUID":
                gt_guid = st.session_state.get("inf_gt_guid_manual", "").strip()
            else:
                allow_trace_fallback = (gt_mode == "Sidebar case")
                gt_guid = _extract_gt_guid(
                    trace, case_id=gt_case_id, allow_trace_fallback=allow_trace_fallback,
                )

            metadata_text = (
                f"[4D Task Status] TASK_0001: Inspection — {task_status}\n"
                f"[Project Phase] {phase}\n"
                f"[Location] {storey}\n"
                f"[IFC Model] {model_code}"
            )

        vlm_result      = st.session_state.get("last_inference")
        retrieval_result = st.session_state.get("last_retrieval")
        if vlm_result is not None and not vlm_result.get("valid_json", False):
            st.error("VLM returned invalid JSON")

    # ── Handle run (outside columns to avoid nested widget issues) ────────
    if run_btn:
        if not chat_text.strip():
            st.warning("Please enter a chat message.")
        else:
            all_image_bytes = [f.getvalue() for f in (uploaded_files or [])]

            # Resolve model variant from session state (widget is inside left_col scope)
            _mv = st.session_state.get("inf_model_variant", "")
            _skip_p0 = "P1 + Rerank" in _mv  # True for p1+rerank mode
            _use_g8 = _mv.startswith("G8")    # True → route to G8ModelPredictor

            _model_label = "G8" if _use_g8 else "G4-Ultimate"
            t0 = time.time()
            with st.spinner(f"Stage 1/3 — VLM inference ({_model_label}) on Modal A100..."):
                vlm_result = _call_modal_inference(all_image_bytes, chat_text, metadata_text, use_g8=_use_g8)
            vlm_ms = int((time.time() - t0) * 1000)

            if vlm_result is not None:
                vlm_result["_latency_ms"] = vlm_ms
                vlm_result["_model_variant"] = _mv
                st.session_state["last_live_inputs"] = _persist_live_inputs(
                    uploaded_files=uploaded_files or [],
                    chat_text=chat_text,
                    model_code=model_code,
                    storey=storey,
                    phase=phase,
                    task_status=task_status,
                    case_id=case_id,
                    gt_guid=gt_guid,
                )
                retrieval_result = None
                if vlm_result.get("valid_json") and vlm_result.get("parsed"):
                    t1 = time.time()
                    with st.spinner("Stage 2-3 — Query planning + Neo4j retrieval..."):
                        retrieval_result = _run_retrieval(
                            vlm_result["parsed"],
                            model_code=model_code,
                            gt_guid=gt_guid,
                            skip_p0=_skip_p0,
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
                        ] = 15

                st.session_state["last_inference"]  = vlm_result
                st.session_state["last_retrieval"]  = retrieval_result
                st.rerun()

    # Re-read after potential rerun
    vlm_result       = st.session_state.get("last_inference")
    retrieval_result = st.session_state.get("last_retrieval")

    # ═════════════════════════════════════════════════════════════════════
    # RIGHT — Live demo flow
    # ═════════════════════════════════════════════════════════════════════
    with right_col:
        if vlm_result is None:
            st.info(
                "Run the pipeline to inspect the inference flow and grounded result."
            )
        else:
            _render_live_flow(
                vlm_result,
                retrieval_result,
                static_base_url=static_base_url,
                case_id=case_id,
                flow_scope="live",
            )


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

    return {
        "photo_paths": photo_paths,
        "floorplan_path": "",
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
    use_g8: bool = False,
) -> dict | None:
    """Call the Modal VLM inference endpoint.

    Routes to G8ModelPredictor (G8 adapter) or G8Predictor (G4-Ultimate adapter)
    depending on use_g8. Both use lora_system_g7 prompt and base+PEFT two-step loading.
    Strategy (full-topo vs P1+rerank) is controlled by skip_p0 in retrieval.
    """
    try:
        import modal
        cls_name = "G8ModelPredictor" if use_g8 else "G8Predictor"
        predictor_cls = modal.Cls.from_name("mscd-vlm-lora3-inference", cls_name)
        predictor = predictor_cls()
        result = predictor.predict.remote(
            image_bytes_list=image_bytes_list,
            chat_text=chat_text,
            metadata_text=metadata_text,
        )
        return result
    except Exception as e:
        err_msg = str(e)
        if "NotFound" in err_msg or "not found" in err_msg.lower():
            cls_name = "G8ModelPredictor" if use_g8 else "G8Predictor"
            st.error(
                f"Modal app `mscd-vlm-lora3-inference` / `{cls_name}` not found. "
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
            spatial_rels.append(SpatialTriplet(
                subject_type=parsed.get("ifc_class", ""),
                predicate=rel.get("predicate", "ADJACENT_TO").upper(),
                object_type=rel.get("object_type", ""),
                object_material=rel.get("object_material"),
                confidence=rel.get("confidence", 0.0),
            ))

        conf = max((r.confidence for r in spatial_rels), default=0.85)
        constraints = Constraints(
            storey_name=parsed.get("storey_name"),
            ifc_class=parsed.get("ifc_class"),
            space_name=parsed.get("space_name"),
            target_name_keyword=parsed.get("target_name_keyword"),
            spatial_relations=spatial_rels,
            confidence=conf,
            source="lora3_live",
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
    params = {
        "ifc": ifc_url,
        "target": target_guid,
        "gt": gt_param,
        "match": "1" if guid_match else "0",
        "pool": ",".join(pool_guids[:60]),
        "base": static_base_url + "/demo/static",
    }
    return viewer_base + "?" + urlencode(params)
