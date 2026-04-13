"""
Tab 1 — Live Inference: upload images + enter chat → G4-Ultimate VLM extracts constraints
→ QueryPlanner builds plans → RetrievalBackend executes → 3D viewer shows pool.

Full neuro-symbolic pipeline: VLM (Modal GPU) → Constraints → Cypher → GUIDs → 3D.
Two modes: Full Topology (P0 spatial + P1 fallback) and P1+Rerank (storey+type first).
"""
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st


_PREDICATE_COLORS = {
    "FILLS": "#3b82f6",
    "ADJACENT_TO": "#f59e0b",
    "CONTINUOUS": "#8b5cf6",
}

# ── Stage styling ─────────────────────────────────────────────────────────────

_STAGE_STYLE = (
    "padding:16px 20px;background:#0f172a;border-radius:10px;"
    "border:1px solid #1e293b;margin-bottom:2px;"
)

_ARROW_HTML = (
    '<div style="text-align:center;color:#475569;font-size:1.4em;'
    'line-height:1;margin:2px 0;">▼</div>'
)

_CONF_THRESHOLD = 0.7

# IFC model code -> static path used by demo static server (for 3D viewer URL).
_IFC_VIEWER_REL_PATH = {
    "AP": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc",
    "BH": "data/ifc/BasicHouse.ifc",
    "DXA": "data/ifc/Duplex_A_20110505.ifc",
}


def _stage_header(number: int, title: str, subtitle: str, latency_ms: int = 0) -> str:
    """Render a numbered stage header with optional latency badge."""
    lat = ""
    if latency_ms > 0:
        lat = (
            f'<span style="float:right;background:#1e293b;color:#94a3b8;'
            f'padding:2px 8px;border-radius:4px;font-size:0.75em;">'
            f'{latency_ms}ms</span>'
        )
    return (
        f'<div style="margin-bottom:10px;">'
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:26px;height:26px;border-radius:50%;background:#3b82f6;color:white;'
        f'font-weight:700;font-size:0.85em;margin-right:8px;">{number}</span>'
        f'<span style="color:#f1f5f9;font-weight:600;font-size:1.05em;">{title}</span>'
        f'<span style="color:#64748b;font-size:0.85em;margin-left:8px;">{subtitle}</span>'
        f'{lat}</div>'
    )


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
        st.session_state.pop("last_inference", None)
        st.session_state.pop("last_retrieval", None)
        st.session_state.pop("last_explain", None)
        st.session_state["live_case_id"] = case_id

    # ── Two-column layout ─────────────────────────────────────────────────
    #   LEFT  (1): compact multimodal evidence input
    #   RIGHT (2): closeable analysis / pipeline trace panel
    left_col, right_col = st.columns([1, 2], gap="medium")

    # ═════════════════════════════════════════════════════════════════════
    # LEFT — Evidence input
    # ═════════════════════════════════════════════════════════════════════
    with left_col:
        with st.container(border=True):
            st.markdown(
                '<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
                'letter-spacing:0.5px;color:#64748b;margin-bottom:4px;">Evidence Input</p>',
                unsafe_allow_html=True,
            )

            # VLM model variant selector
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
            model_code = ifc_model.split(" ", 1)[0].strip()

            st.caption("Site photos")
            uploaded_files = st.file_uploader(
                "site_photos",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key="inference_images",
                label_visibility="collapsed",
            )
            if uploaded_files:
                thumb_cols = st.columns(min(len(uploaded_files), 3))
                for i, f in enumerate(uploaded_files):
                    thumb_cols[i % 3].image(f, caption=f.name, width=110)

            st.caption("Floorplan")
            floorplan_file = st.file_uploader(
                "floorplan",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=False,
                key="inference_floorplan",
                label_visibility="collapsed",
            )
            if floorplan_file:
                st.image(floorplan_file, caption=floorplan_file.name, width=110)

            st.caption("Chat message")
            chat_text = st.text_area(
                "chat_msg",
                value="There's a crack on the window next to the railing, third floor",
                height=72,
                key="inference_chat",
                label_visibility="collapsed",
            )

            with st.expander("4D Metadata", expanded=False):
                meta_c1, meta_c2 = st.columns(2)
                meta_c1.text_input("Storey", value="3 - Third Floor", key="inf_storey")
                meta_c2.text_input("Phase", value="Interior Fit-out", key="inf_phase")
                meta_c1.selectbox(
                    "Task Status",
                    ["IN_PROGRESS", "PENDING_INSPECTION", "REVIEW_REQUIRED", "ON_HOLD"],
                    key="inf_status",
                )

            storey      = st.session_state.get("inf_storey",  "3 - Third Floor")
            phase       = st.session_state.get("inf_phase",   "Interior Fit-out")
            task_status = st.session_state.get("inf_status",  "IN_PROGRESS")

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

            # Direct GUID bypasses case_id lookup entirely
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

            run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

        # Post-run status badge
        vlm_result      = st.session_state.get("last_inference")
        retrieval_result = st.session_state.get("last_retrieval")
        if vlm_result is not None:
            pool_guids  = (retrieval_result or {}).get("pool_guids") or []
            guid_match  = (retrieval_result or {}).get("guid_match", False)
            total_ms    = (vlm_result.get("_latency_ms", 0)
                           + (retrieval_result or {}).get("_latency_ms", 0))
            if not vlm_result.get("valid_json", False):
                st.error("VLM returned invalid JSON")
            else:
                match_color = "#22c55e" if guid_match else ("#ef4444" if pool_guids else "#f59e0b")
                match_icon  = "✓ match" if guid_match else ("✗ miss" if pool_guids else "no pool")
                _used_mv = vlm_result.get("_model_variant", "")
                _mode_tag = "P1+rerank" if "P1 + Rerank" in _used_mv else "full topo"
                _model_tag = "G8" if _used_mv.startswith("G8") else "G4"
                st.markdown(
                    f'<div style="padding:6px 12px;border-radius:6px;margin-top:4px;'
                    f'background:rgba(15,23,42,0.6);border:1px solid #1e293b;'
                    f'font-family:monospace;font-size:0.82em;">'
                    f'<span style="color:{match_color};font-weight:700;">{match_icon}</span>'
                    f'<span style="color:#64748b;margin-left:8px;">'
                    f'pool={len(pool_guids)}  ·  {total_ms}ms  ·  {_model_tag}/{_mode_tag}</span></div>',
                    unsafe_allow_html=True,
                )

    # ── Handle run (outside columns to avoid nested widget issues) ────────
    if run_btn:
        if not chat_text.strip():
            st.warning("Please enter a chat message.")
        else:
            all_image_bytes = [f.getvalue() for f in (uploaded_files or [])]
            if floorplan_file:
                all_image_bytes.append(floorplan_file.getvalue())

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

                st.session_state["last_inference"]  = vlm_result
                st.session_state["last_retrieval"]  = retrieval_result
                st.rerun()

    # Re-read after potential rerun
    vlm_result       = st.session_state.get("last_inference")
    retrieval_result = st.session_state.get("last_retrieval")

    # ═════════════════════════════════════════════════════════════════════
    # RIGHT — Closeable analysis panel
    # ═════════════════════════════════════════════════════════════════════
    with right_col:
        hdr_c, tog_c = st.columns([3, 1])
        hdr_c.markdown(
            '<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.5px;color:#64748b;margin-bottom:0;">Analysis / Trace Inspector</p>',
            unsafe_allow_html=True,
        )
        show_analysis = tog_c.toggle("Show", value=True, key="show_analysis_panel")

        if show_analysis:
            if vlm_result is None:
                st.info(
                    "Run the pipeline to see analysis — "
                    "**VLM → Constraints → Query Plan → Cypher → Pool → 3D**"
                )
            else:
                _render_pipeline(vlm_result, retrieval_result, static_base_url=static_base_url)

                # ── Occlusion saliency (only when images + spatial relations present) ──
                has_images   = bool(uploaded_files or floorplan_file)
                has_spatial  = bool((vlm_result.get("parsed") or {}).get("spatial_relations"))
                if has_images and has_spatial:
                    st.markdown("---")
                    st.markdown("#### VLM Spatial Grounding")
                    st.caption(
                        "Occlusion saliency: masks each image patch, re-runs VLM, "
                        "measures which regions drive the spatial relation prediction."
                    )
                    grid_col, btn_col = st.columns([1, 2])
                    grid_size = grid_col.select_slider(
                        "Grid resolution", options=[3, 4, 5, 6], value=4, key="explain_grid",
                    )
                    explain_btn = btn_col.button(
                        f"Run Explain ({grid_size}×{grid_size} = {grid_size**2} passes)",
                        key="explain_btn",
                    )
                    if explain_btn:
                        all_bytes = [f.getvalue() for f in (uploaded_files or [])]
                        if floorplan_file:
                            all_bytes.append(floorplan_file.getvalue())
                        with st.spinner(f"Running occlusion saliency ({grid_size**2} passes)..."):
                            explain_result = _call_modal_explain(
                                all_bytes, chat_text, metadata_text, grid_size
                            )
                        if explain_result:
                            st.session_state["last_explain"] = explain_result
                    explain_result = st.session_state.get("last_explain")
                    if explain_result and "heatmaps" in explain_result:
                        all_files = list(uploaded_files or []) + (
                            [floorplan_file] if floorplan_file else []
                        )
                        _render_saliency(explain_result, all_files)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline visualization
# ══════════════════════════════════════════════════════════════════════════════

def _render_pipeline(
    vlm_result: dict,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
) -> None:
    """Render key outputs first, then optional step-by-step details."""
    valid = vlm_result.get("valid_json", False)
    parsed = vlm_result.get("parsed") or {}
    raw = vlm_result.get("raw_output", "")
    vlm_ms = vlm_result.get("_latency_ms", 0)
    _mv = vlm_result.get("_model_variant", "")
    _model_label = "G8" if _mv.startswith("G8") else "G4-Ultimate"

    if not valid:
        st.error("VLM returned invalid JSON — pipeline halted")
        st.code(raw, language="text")
        return

    _render_key_outputs(parsed, retrieval_result, static_base_url=static_base_url)

    st.markdown("#### Step-by-Step Details")
    with st.expander("Show pipeline execution steps", expanded=False):
        _render_pipeline_details(
            parsed=parsed,
            vlm_ms=vlm_ms,
            retrieval_result=retrieval_result,
            model_label=_model_label,
        )

    _render_raw_outputs_panel(vlm_result, retrieval_result)


def _render_key_outputs(
    parsed: dict,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
) -> None:
    """Show the two demo-critical outputs immediately: graph retrieval + 3D pool."""
    st.markdown("#### Key Outputs")

    if retrieval_result is None:
        st.warning("Symbolic retrieval unavailable, so graph and 3D outputs are not shown.")
        return

    _render_graph_retrieval_panel(parsed, retrieval_result)

    pool_guids = retrieval_result.get("pool_guids") or []
    if pool_guids:
        _render_3d_pool(retrieval_result, static_base_url)
        return

    if static_base_url:
        viewer_url = _build_viewer_url(
            static_base_url,
            [],
            target_guid="",
            gt_guid=(retrieval_result or {}).get("gt_guid", ""),
            guid_match=bool((retrieval_result or {}).get("guid_match", False)),
            ifc_model_code=retrieval_result.get("ifc_model_code", "AP"),
        )
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
            f'<button style="margin-top:8px;padding:8px 18px;background:#1e293b;color:#e2e8f0;'
            f'border:1px solid #334155;border-radius:6px;font-family:monospace;'
            f'font-size:13px;cursor:pointer;">Open 3D Viewer (empty) &#8599;</button></a>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No 3D candidates returned for this query.")


def _render_pipeline_details(
    *,
    parsed: dict,
    vlm_ms: int,
    retrieval_result: dict | None,
    model_label: str = "G4-Ultimate",
) -> None:
    """Render detailed stage trace below the key outputs."""
    # ━━ Stage 1: Neuro Layer (VLM) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(1, "Neuro Layer", f"{model_label} VLM → Structured JSON", vlm_ms)
        + '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("VLM raw output", expanded=False):
        st.code(json.dumps(parsed, indent=2, ensure_ascii=False), language="json")

    st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    # ━━ Stage 2: Constraint Parsing + Confidence Gate ━━━━━━━━━━━━━━━━━━━
    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(2, "Constraint Parsing", "JSON → Constraints + confidence gate")
        + _render_constraints_inline(parsed)
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    # ━━ Stage 3: Query Planning ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if retrieval_result is None:
        st.markdown(
            f'<div style="{_STAGE_STYLE};border-color:#b91c1c;">'
            + _stage_header(3, "Query Planning", "BLOCKED — Neo4j not available")
            + '<div style="color:#fca5a5;font-size:0.9em;">'
            'Symbolic retrieval failed. Start Neo4j with: '
            '<code>./script/neo4j_init.sh</code></div>'
            + '</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Raw VLM JSON", expanded=False):
            st.json(parsed)
        return

    ret_ms = retrieval_result.get("_latency_ms", 0)
    plans = retrieval_result.get("plans") or []
    results = retrieval_result.get("results") or []
    winning = retrieval_result.get("winning")

    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(3, "Query Planning", f"{len(plans)} plans generated (priority cascade)", ret_ms)
        + _render_plan_cascade_inline(plans)
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    # ━━ Stage 4: Cypher Execution ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(4, "Cypher Execution", "Priority cascade → Neo4j")
        + '</div>',
        unsafe_allow_html=True,
    )

    # Execution results per plan
    for i, plan in enumerate(plans):
        result = results[i] if i < len(results) else None
        _render_execution_step(i, plan, result, winning)

    st.markdown(_ARROW_HTML, unsafe_allow_html=True)

    # ━━ Stage 5: Candidate Pool ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    pool_guids = retrieval_result.get("pool_guids") or []

    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(5, "Candidate Pool", f"{len(pool_guids)} elements retrieved")
        + '</div>',
        unsafe_allow_html=True,
    )

    if winning and winning.get("candidates"):
        _render_candidates_table(winning)
    else:
        st.warning("No candidates returned — all plans returned empty pools.")

    # ── Pipeline summary ──────────────────────────────────────────────────
    total_ms = vlm_ms + ret_ms
    st.markdown("---")
    summary_parts = [f"Total: **{total_ms}ms**"]
    if winning:
        summary_parts.append(
            f"Winner: **P{winning['priority']} {winning['strategy']}** "
            f"→ **{winning['pool_size']}** candidates"
        )
        if winning.get("fallback_triggered"):
            summary_parts.append("(fallback triggered)")
    st.caption(" · ".join(summary_parts))


# ══════════════════════════════════════════════════════════════════════════════
# Inline renderers (return HTML strings for embedding inside stage divs)
# ══════════════════════════════════════════════════════════════════════════════

def _render_constraints_inline(parsed: dict) -> str:
    """Render constraints as compact pills inside the stage div."""
    html = '<div style="margin-top:8px;">'

    # Attribute constraints
    html += '<div style="margin-bottom:6px;">'
    html += _kv_pill("storey", parsed.get("storey_name") or "null", "#22c55e")
    html += _kv_pill("ifc_class", parsed.get("ifc_class") or "null", "#3b82f6")
    html += _kv_pill("space", parsed.get("space_name") or "null", "#a78bfa")
    html += _kv_pill("name_kw", parsed.get("target_name_keyword") or "null", "#f59e0b")
    html += '</div>'

    # Spatial relations
    rels = parsed.get("spatial_relations") or []
    if rels:
        for rel in rels:
            pred = rel.get("predicate", "?")
            obj_type = rel.get("object_type", "?")
            obj_mat = rel.get("object_material")
            conf = rel.get("confidence", 0)
            color = _PREDICATE_COLORS.get(pred, "#6b7280")
            mat_tag = f" ({obj_mat})" if obj_mat else ""

            html += (
                f'<div style="display:flex;align-items:center;gap:8px;'
                f'padding:6px 12px;background:#1e293b;border-radius:6px;'
                f'border-left:3px solid {color};margin:4px 0;">'
                f'<span style="color:#e2e8f0;font-family:monospace;font-size:0.9em;">'
                f'{parsed.get("ifc_class", "?")} '
                f'<span style="background:{color};color:white;padding:1px 8px;'
                f'border-radius:3px;font-weight:700;font-size:0.85em;">'
                f'{pred}</span>'
                f' → {obj_type}{mat_tag}</span>'
                f'<span style="margin-left:auto;color:#94a3b8;font-size:0.8em;">'
                f'conf={conf:.2f}</span></div>'
            )

        # Confidence gate (must match backend logic: max confidence across relations)
        max_conf = max((rel.get("confidence", 0) for rel in rels), default=0)
        gate_pass = max_conf >= _CONF_THRESHOLD
        gate_color = "#22c55e" if gate_pass else "#f59e0b"
        gate_icon = "PASS" if gate_pass else "SKIP"
        gate_action = "Priority 0 Cypher executes" if gate_pass else "Falls back to P1-P8"
        html += (
            f'<div style="margin-top:6px;padding:4px 10px;border-radius:4px;'
            f'background:rgba(30,41,59,0.5);font-size:0.82em;font-family:monospace;">'
            f'<span style="color:{gate_color};font-weight:700;">GATE {gate_icon}</span>'
            f'<span style="color:#94a3b8;"> — max_conf {max_conf:.2f} vs '
            f'threshold {_CONF_THRESHOLD} → {gate_action}</span></div>'
        )
    else:
        html += (
            '<div style="color:#94a3b8;font-size:0.85em;padding:4px 0;">'
            'No spatial relations — attribute-only case → Priority 1-8 cascade</div>'
        )

    html += '</div>'
    return html


def _render_plan_cascade_inline(plans: list) -> str:
    """Render query plans as a compact cascade inside the stage div."""
    if not plans:
        return '<div style="color:#94a3b8;">No plans generated</div>'

    html = '<div style="margin-top:8px;font-family:monospace;font-size:0.82em;">'
    for p in plans:
        pri = p["priority"]
        strat = p["strategy"]
        est = p.get("expected_pool_size", "?")
        params = p.get("params", {})
        param_str = ", ".join(f'{k}={v}' for k, v in params.items() if v)
        if len(param_str) > 80:
            param_str = param_str[:77] + "..."

        html += (
            f'<div style="padding:3px 8px;margin:2px 0;border-left:2px solid #334155;">'
            f'<span style="color:#3b82f6;font-weight:600;">P{pri}</span>'
            f'<span style="color:#e2e8f0;margin-left:6px;">{strat}</span>'
            f'<span style="color:#64748b;margin-left:8px;">~{est} est.</span>'
            f'<span style="color:#475569;margin-left:8px;">{param_str}</span>'
            f'</div>'
        )
    html += '</div>'
    return html


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit-based renderers (use st.* calls)
# ══════════════════════════════════════════════════════════════════════════════

def _render_execution_step(
    idx: int,
    plan: dict,
    result: dict | None,
    winning: dict | None,
) -> None:
    """Render one execution step as an expander."""
    pool_size = result["pool_size"] if result else 0
    is_winner = (
        result is not None
        and pool_size > 0
        and winning is not None
        and winning.get("strategy") == result.get("strategy")
    )

    if result is None:
        icon, badge_text, badge_color = "⏭", "skipped", "#475569"
    elif pool_size > 0:
        icon = "✅" if is_winner else "🔹"
        badge_text = f"{pool_size} candidates"
        badge_color = "#22c55e" if is_winner else "#3b82f6"
    else:
        icon, badge_text, badge_color = "❌", "0 results", "#ef4444"

    label = f"{icon}  P{plan['priority']}: {plan['strategy']}  —  {badge_text}"

    with st.expander(label, expanded=is_winner):
        # Params
        params = plan.get("params", {})
        if params:
            pills = ""
            for k, v in params.items():
                if v:
                    pills += _kv_pill(k, str(v))
            if pills:
                st.markdown(pills, unsafe_allow_html=True)

        if result:
            meta_parts = [f"Backend: `{result.get('backend', '?')}`"]
            if result.get("fallback_triggered"):
                meta_parts.append(
                    f"Fallback → `{result.get('strategy_actually_used', '?')}`"
                )
            st.caption(" · ".join(meta_parts))

            # Show first few candidates inline for the winner
            if is_winner and result.get("candidates"):
                cands = result["candidates"][:5]
                rows = ""
                for c in cands:
                    rows += (
                        f'<tr style="border-bottom:1px solid #1e293b;">'
                        f'<td style="padding:2px 8px;color:#94a3b8;">{c.get("guid", "")[:12]}...</td>'
                        f'<td style="padding:2px 8px;color:#e2e8f0;">{c.get("name", "")}</td>'
                        f'<td style="padding:2px 8px;color:#64748b;">{c.get("type", "")}</td>'
                        f'<td style="padding:2px 8px;color:#64748b;">{c.get("storey", "")}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;font-size:0.8em;font-family:monospace;'
                    f'border-collapse:collapse;">'
                    f'<tr style="color:#64748b;border-bottom:1px solid #334155;">'
                    f'<th style="text-align:left;padding:2px 8px;">GUID</th>'
                    f'<th style="text-align:left;padding:2px 8px;">Name</th>'
                    f'<th style="text-align:left;padding:2px 8px;">Type</th>'
                    f'<th style="text-align:left;padding:2px 8px;">Storey</th></tr>'
                    f'{rows}</table>',
                    unsafe_allow_html=True,
                )
                if len(result["candidates"]) > 5:
                    st.caption(f"+{len(result['candidates']) - 5} more...")


def _render_candidates_table(winning: dict) -> None:
    """Render the winning strategy's candidate table."""
    candidates = winning["candidates"]

    st.markdown(
        f'<div style="padding:8px 14px;background:#0f172a;border-radius:8px;'
        f'border-left:3px solid #22c55e;margin:8px 0;font-family:monospace;font-size:0.9em;">'
        f'<span style="color:#22c55e;font-weight:700;">WINNER</span>'
        f'<span style="color:#e2e8f0;margin-left:8px;">'
        f'P{winning["priority"]} {winning["strategy"]}</span>'
        f'<span style="color:#94a3b8;margin-left:8px;">'
        f'→ {winning["pool_size"]} candidates</span></div>',
        unsafe_allow_html=True,
    )

    rows = []
    for rank, cand in enumerate(candidates[:15], 1):
        rows.append({
            "Rank": rank,
            "Name": cand.get("name", ""),
            "Type": cand.get("type", ""),
            "Storey": cand.get("storey", ""),
            "GUID": cand.get("guid", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Graph explainability
# ══════════════════════════════════════════════════════════════════════════════

def _gv_escape(value: str) -> str:
    return str(value).replace('"', "'")


def _gv_node_id(guid: str) -> str:
    """Build a DOT-safe node id from an IFC GUID."""
    raw = str(guid or "").strip()
    if not raw:
        return "n_empty"
    safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"n_{safe[:18]}_{digest}"


def _build_plan_graph_dot(parsed: dict, plans: list, results: list, winning: dict | None) -> str:
    """Build Graphviz DOT string for VLM -> constraints -> plan cascade."""
    subject = parsed.get("ifc_class") or "UnknownIfcClass"
    storey = parsed.get("storey_name") or "UnknownStorey"

    lines = [
        "digraph G {",
        'rankdir=LR;',
        'graph [bgcolor="transparent"];',
        'node [shape=box, style="rounded,filled", fillcolor="#0f172a", color="#334155", fontcolor="#e2e8f0"];',
        'edge [color="#64748b", fontcolor="#94a3b8"];',
        f'vlm [label="VLM Output\\n{_gv_escape(subject)} @ {_gv_escape(storey)}", fillcolor="#1e293b", color="#3b82f6"];',
        'constraints [label="Constraints\\nstructured JSON", fillcolor="#1e293b", color="#22c55e"];',
        'vlm -> constraints [label="parse"];',
    ]

    winner_key = (winning or {}).get("strategy")
    for i, plan in enumerate(plans):
        result = results[i] if i < len(results) else None
        pool_size = (result or {}).get("pool_size", 0)
        strategy = plan.get("strategy", "?")
        priority = plan.get("priority", "?")
        node_id = f"p{i}"
        is_winner = winner_key == strategy and pool_size > 0
        color = "#22c55e" if is_winner else "#334155"
        fill = "#14532d" if is_winner else "#0f172a"
        lines.append(
            f'{node_id} [label="P{priority}: {_gv_escape(strategy)}\\npool={pool_size}", color="{color}", fillcolor="{fill}"];'
        )
        lines.append(f"constraints -> {node_id};")

    lines.append("}")
    return "\n".join(lines)


def _build_subgraph_dot(subgraph: dict, gt_guid: str = "") -> str:
    """Build Graphviz DOT for a 1-hop Neo4j subgraph around top-1 candidate."""
    anchor_guid = subgraph.get("anchor_guid")
    nodes = subgraph.get("nodes") or []
    edges = subgraph.get("edges") or []

    lines = [
        "digraph SG {",
        'rankdir=LR;',
        'graph [bgcolor="transparent"];',
        'node [shape=ellipse, style="filled", fillcolor="#334155", color="#64748b", fontcolor="#e2e8f0"];',
        'edge [color="#6b7280", fontcolor="#cbd5e1"];',
    ]

    node_ids: dict[str, str] = {}
    for n in nodes[:40]:
        guid = n.get("guid", "")
        if not guid:
            continue
        node_ids[guid] = _gv_node_id(guid)
        name = n.get("name") or n.get("type") or "node"
        if len(name) > 26:
            name = name[:23] + "..."
        label = f"{name}\\n{guid[:10]}..."
        is_anchor = guid == anchor_guid
        is_gt = bool(gt_guid and guid == gt_guid)
        if is_anchor:
            # Keep top-1 green; if GT==top-1 add blue outline for dual meaning.
            color = "#3b82f6" if is_gt else "#22c55e"
            fill = "#14532d"
            pen = "2.8" if is_gt else "2.2"
        elif is_gt:
            color = "#3b82f6"
            fill = "#1e3a8a"
            pen = "2.2"
        else:
            color = "#64748b"
            fill = "#334155"
            pen = "1.0"
        lines.append(
            f'{node_ids[guid]} [label="{_gv_escape(label)}", color="{color}", fillcolor="{fill}", penwidth={pen}];'
        )

    for e in edges[:80]:
        src_guid = e.get("source_guid", "")
        dst_guid = e.get("target_guid", "")
        src = node_ids.get(src_guid)
        dst = node_ids.get(dst_guid)
        rel = _gv_escape(e.get("rel_type", "REL"))
        if not src or not dst:
            continue
        lines.append(f"{src} -> {dst} [label=\"{rel}\"];")

    lines.append("}")
    return "\n".join(lines)


def _build_ifc_context_dot(
    graph_snapshot: dict,
    subgraph: dict | None,
    gt_guid: str = "",
) -> str:
    """Build a Neo4j-style bubble DOT view with highlighted 1-hop around top-1."""
    snap_nodes = graph_snapshot.get("nodes") or []
    snap_edges = graph_snapshot.get("edges") or []
    hop_nodes_raw = (subgraph or {}).get("nodes") or []
    hop_edges_raw = (subgraph or {}).get("edges") or []
    anchor_guid = (subgraph or {}).get("anchor_guid", "")

    nodes_by_guid: dict[str, dict] = {}
    for n in snap_nodes + hop_nodes_raw:
        guid = (n.get("guid") or "").strip()
        if not guid:
            continue
        existing = nodes_by_guid.get(guid, {})
        nodes_by_guid[guid] = {
            "guid": guid,
            "name": n.get("name") or existing.get("name") or "",
            "type": n.get("type") or existing.get("type") or "IFCElement",
        }

    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def _add_edge(edge: dict) -> None:
        src = (edge.get("source_guid") or "").strip()
        dst = (edge.get("target_guid") or "").strip()
        rel = (edge.get("rel_type") or "REL").strip() or "REL"
        if not src or not dst:
            return
        key = (src, rel, dst)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"source_guid": src, "target_guid": dst, "rel_type": rel})

    for e in snap_edges:
        _add_edge(e)
    for e in hop_edges_raw:
        _add_edge(e)

    hop_nodes = {
        (n.get("guid") or "").strip()
        for n in hop_nodes_raw
        if n.get("guid")
    }
    hop_edges = {
        (
            (e.get("source_guid") or "").strip(),
            (e.get("rel_type") or "REL").strip() or "REL",
            (e.get("target_guid") or "").strip(),
        )
        for e in hop_edges_raw
        if e.get("source_guid") and e.get("target_guid")
    }

    ordered_guids: list[str] = []
    if anchor_guid and anchor_guid in nodes_by_guid:
        ordered_guids.append(anchor_guid)
    if gt_guid and gt_guid in nodes_by_guid and gt_guid not in ordered_guids:
        ordered_guids.append(gt_guid)
    for guid in hop_nodes:
        if guid not in ordered_guids and guid in nodes_by_guid:
            ordered_guids.append(guid)
    for guid in nodes_by_guid:
        if guid not in ordered_guids:
            ordered_guids.append(guid)

    keep_guids = set(ordered_guids[:170])
    node_ids = {guid: _gv_node_id(guid) for guid in keep_guids}

    lines = [
        "graph IFC {",
        'graph [bgcolor="transparent", layout="sfdp", overlap=false, splines=true, K=1.15, repulsiveforce=1.25];',
        'node [shape=ellipse, style="filled", fillcolor="#334155", color="#64748b", fontcolor="#cbd5e1"];',
        'edge [color="#6b7280", fontcolor="#cbd5e1", penwidth=0.9];',
        '__hub [label="", shape=point, width=0.01, height=0.01, style=invis];',
    ]

    for guid in ordered_guids[:170]:
        node = nodes_by_guid.get(guid, {})
        if not node:
            continue
        node_id = node_ids.get(guid)
        if not node_id:
            continue

        name = node.get("name") or node.get("type") or "node"
        type_name = node.get("type") or "IFCElement"
        if len(name) > 22:
            name = name[:19] + "..."
        label = f"{name}\\n{guid[:10]}..."

        is_anchor = guid == anchor_guid
        is_gt = bool(gt_guid and guid == gt_guid)
        in_hop = guid in hop_nodes

        if is_anchor:
            # Top-1 always green; when GT==top-1, keep green fill and add blue outline.
            color = "#22c55e"
            fill = "#14532d"
            font = "#ecfdf5"
            pen = "2.6"
            if is_gt:
                color = "#3b82f6"
                pen = "3.0"
        elif is_gt:
            color = "#3b82f6"
            fill = "#1e3a8a"
            font = "#dbeafe"
            pen = "2.4"
        elif in_hop:
            color = "#f59e0b"
            fill = "#78350f"
            font = "#fef3c7"
            pen = "2.0"
        else:
            color = "#64748b"
            fill = "#334155"
            font = "#cbd5e1"
            pen = "1.0"
            label = f"{type_name}\\n{guid[:8]}..."

        lines.append(
            f'{node_id} [label="{_gv_escape(label)}", color="{color}", fillcolor="{fill}", '
            f'fontcolor="{font}", penwidth={pen}];'
        )
        # Invisible links keep sampled nodes in one visual collection (single bubble cloud).
        lines.append(f"__hub -- {node_id} [style=invis, weight=0.05];")

    kept_edges = 0
    for e in edges:
        src_guid = e.get("source_guid", "")
        dst_guid = e.get("target_guid", "")
        rel = e.get("rel_type", "REL")
        if src_guid not in keep_guids or dst_guid not in keep_guids:
            continue
        src = node_ids.get(src_guid)
        dst = node_ids.get(dst_guid)
        if not src or not dst:
            continue

        is_hop_edge = (
            (src_guid, rel, dst_guid) in hop_edges
            or (dst_guid, rel, src_guid) in hop_edges
        )
        if is_hop_edge:
            lines.append(
                f'{src} -- {dst} [color="#f59e0b", penwidth=2.1, '
                f'label="{_gv_escape(rel)}", fontcolor="#fbbf24"];'
            )
        else:
            lines.append(
                f'{src} -- {dst} [color="#6b7280", penwidth=0.9];'
            )

        kept_edges += 1
        if kept_edges >= 260:
            break

    lines.append("}")
    return "\n".join(lines)


def _build_spatial_constraint_dot(parsed: dict) -> str:
    """Build DOT graph of spatial relation triplets extracted by the VLM.

    Shows the target element type as the central node with edges to each
    object_type labelled by predicate (FILLS / ADJACENT_TO / CONTINUOUS).
    Used as a fallback when Neo4j subgraph data is not available.
    """
    ifc_class = parsed.get("ifc_class") or "Target"
    rels      = parsed.get("spatial_relations") or []

    _pred_fill = {
        "FILLS":       ("#1e3a8a", "#3b82f6"),   # dark-blue fill, blue border
        "ADJACENT_TO": ("#78350f", "#f59e0b"),   # dark-amber fill, amber border
        "CONTINUOUS":  ("#3b0764", "#8b5cf6"),   # dark-purple fill, purple border
    }

    lines = [
        "digraph spatial {",
        "rankdir=LR;",
        'graph [bgcolor="transparent"];',
        'node [shape=box, style="rounded,filled", fontcolor="#e2e8f0", '
        'fontname="Helvetica", fontsize=11];',
        'edge [fontname="Helvetica", fontsize=10];',
        f'target [label="{_gv_escape(ifc_class)}", fillcolor="#14532d", '
        f'color="#22c55e", penwidth=2.2, fontcolor="#ecfdf5"];',
    ]

    for i, rel in enumerate(rels):
        pred     = rel.get("predicate", "REL")
        obj_type = rel.get("object_type") or "Element"
        obj_mat  = rel.get("object_material")
        conf     = rel.get("confidence", 0)
        fill_c, border_c = _pred_fill.get(pred, ("#334155", "#64748b"))

        mat_label = f"\\n({_gv_escape(obj_mat)})" if obj_mat else ""
        node_id   = f"obj_{i}"
        conf_tag  = f"\\nconf={conf:.2f}" if conf else ""

        lines.append(
            f'{node_id} [label="{_gv_escape(obj_type)}{mat_label}", '
            f'fillcolor="{fill_c}", color="{border_c}", penwidth=1.8];'
        )
        lines.append(
            f'target -> {node_id} [label="{_gv_escape(pred)}{conf_tag}", '
            f'color="{border_c}", fontcolor="{border_c}", penwidth=2.0];'
        )

    if not rels:
        lines.append(
            'no_rels [label="No spatial relations\\n(attribute-only case)", '
            'fillcolor="#1e293b", color="#475569", fontcolor="#94a3b8"];'
        )

    lines.append("}")
    return "\n".join(lines)


def _render_graph_retrieval_panel(parsed: dict, retrieval_result: dict | None) -> None:
    """Render graph-centric explainability: spatial relationship graph nodes."""
    if not retrieval_result:
        return

    plans    = retrieval_result.get("plans") or []
    results  = retrieval_result.get("results") or []
    winning  = retrieval_result.get("winning") or {}
    subgraph = retrieval_result.get("subgraph")
    gt_guid  = (retrieval_result.get("gt_guid") or "").strip()
    graph_snapshot = retrieval_result.get("graph_snapshot")

    if not plans:
        return

    st.markdown("#### Graph Retrieval Explainability")

    # ── Build plan summary rows ───────────────────────────────────────────
    rows = []
    graph_size = attr_size = None
    for i, plan in enumerate(plans):
        result = results[i] if i < len(results) else None
        pool   = (result or {}).get("pool_size", 0)
        strat  = plan.get("strategy", "?")
        rows.append({
            "Priority": f"P{plan.get('priority', '?')}",
            "Strategy": strat,
            "Pool":     pool,
            "Winner":   "✓" if winning and strat == winning.get("strategy") else "",
            "Fallback": "↩" if (result or {}).get("fallback_triggered") else "",
        })
        if strat in {"spatial_triplet", "continuous_span"} and graph_size is None:
            graph_size = pool
        if strat in {"storey+type", "type_only"} and attr_size is None:
            attr_size = pool

    c1, c2 = st.columns([1, 1], gap="large")

    # ── LEFT: plan table ─────────────────────────────────────────────────
    with c1:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        if graph_size is not None and attr_size is not None:
            delta = attr_size - graph_size
            st.caption(
                f"Graph gain: attribute pool {attr_size} → graph pool {graph_size} (Δ −{delta})"
            )

    # ── RIGHT: spatial relationship graph ────────────────────────────────
    with c2:
        if graph_snapshot and graph_snapshot.get("edges"):
            # Full IFC context bubble cloud with 1-hop spatial neighbourhood highlighted
            snap_nodes = len(graph_snapshot.get("nodes") or [])
            snap_edges = len(graph_snapshot.get("edges") or [])
            hop_nodes  = len((subgraph or {}).get("nodes") or [])
            hop_edges  = len((subgraph or {}).get("edges") or [])
            st.caption(
                f"IFC graph sample: {snap_nodes} nodes / {snap_edges} edges — "
                f"spatial 1-hop: {hop_nodes} nodes / {hop_edges} edges"
            )
            st.graphviz_chart(
                _build_ifc_context_dot(graph_snapshot, subgraph, gt_guid=gt_guid),
                use_container_width=True,
            )
            st.caption(
                "🟢 top-1 · 🟡 1-hop spatial neighbours · 🔵 ground truth · ⚫ context"
            )
            if subgraph and subgraph.get("edges"):
                with st.expander("Isolated 1-hop neighbourhood", expanded=False):
                    st.graphviz_chart(
                        _build_subgraph_dot(subgraph, gt_guid=gt_guid),
                        use_container_width=True,
                    )

        elif subgraph and subgraph.get("edges"):
            # Neo4j returned a 1-hop subgraph without the full snapshot
            st.caption("Spatial 1-hop neighbourhood around top-1 candidate (Neo4j)")
            st.graphviz_chart(
                _build_subgraph_dot(subgraph, gt_guid=gt_guid),
                use_container_width=True,
            )

        else:
            # Fallback: draw VLM-extracted spatial constraint triplets
            rels = (parsed.get("spatial_relations") or [])
            if rels:
                st.caption("Spatial relation triplets extracted by VLM (Neo4j graph not available)")
            else:
                st.caption("Attribute-only case — no spatial relations extracted")
            st.graphviz_chart(
                _build_spatial_constraint_dot(parsed),
                use_container_width=True,
            )


def _render_raw_outputs_panel(vlm_result: dict, retrieval_result: dict | None) -> None:
    """Single raw-output panel for demo transparency and Q&A."""
    st.markdown("---")
    with st.expander("Raw key outputs (VLM → Constraints → Plans → Retrieval)", expanded=False):
        st.markdown("**VLM raw output**")
        st.code(vlm_result.get("raw_output", ""), language="text")

        st.markdown("**Parsed constraints JSON**")
        st.json(vlm_result.get("parsed") or {})

        if retrieval_result:
            plans = retrieval_result.get("plans") or []
            results = retrieval_result.get("results") or []
            winning = retrieval_result.get("winning") or {}

            compact_results = []
            for r in results:
                cands = r.get("candidates") or []
                compact = {k: v for k, v in r.items() if k != "candidates"}
                compact["candidates_total"] = len(cands)
                compact["candidates_preview"] = cands[:10]
                compact_results.append(compact)

            st.markdown("**Query plans**")
            st.json(plans)
            st.markdown("**Execution results (compact)**")
            st.json(compact_results)
            st.markdown("**Winning result**")
            st.json({
                **{k: v for k, v in winning.items() if k != "candidates"},
                "candidates_total": len(winning.get("candidates") or []),
                "candidates_preview": (winning.get("candidates") or [])[:10],
            })
            if retrieval_result.get("subgraph"):
                st.markdown("**Neo4j subgraph payload**")
                st.json(retrieval_result.get("subgraph"))
            if retrieval_result.get("graph_snapshot"):
                st.markdown("**Neo4j IFC graph snapshot payload**")
                st.json(retrieval_result.get("graph_snapshot"))


# ══════════════════════════════════════════════════════════════════════════════
# 3D Viewer
# ══════════════════════════════════════════════════════════════════════════════

def _render_3d_pool(
    retrieval_result: dict | None,
    static_base_url: str,
) -> None:
    """Render 3D viewer section with GUID chips + viewer link."""
    if not static_base_url:
        return

    winning = (retrieval_result or {}).get("winning")
    pool_guids = (retrieval_result or {}).get("pool_guids") or []
    ifc_model_code = (retrieval_result or {}).get("ifc_model_code", "AP")
    gt_guid = (retrieval_result or {}).get("gt_guid", "")
    candidates = (winning or {}).get("candidates") or [] if winning else []

    if not pool_guids:
        return

    with st.container(border=True):
        st.markdown("#### 3D Pool Viewer")

        top1_guid = pool_guids[0]
        # Keep top-1 out of pool highlights so it stays visually distinct.
        pool_for_viewer = [g for g in pool_guids if g and g != top1_guid]
        has_gt = bool(gt_guid)
        guid_match = (top1_guid == gt_guid) if has_gt else True
        strategy = winning.get("strategy", "?") if winning else "?"
        pool_size = winning.get("pool_size", len(pool_guids)) if winning else len(pool_guids)

        status_suffix = ", GT in blue" if has_gt else ""
        st.markdown(
            f"**{pool_size} candidates** from `{strategy}` "
            f"— top-1 highlighted green/red, pool in amber{status_suffix}"
        )

        # Explicit GUID chips for demo clarity.
        pred_color = "#22c55e" if guid_match else "#ef4444"
        st.markdown(
            f'<div style="margin:6px 0 4px 0;font-family:monospace;font-size:0.84em;">'
            f'<span style="color:{pred_color};font-weight:700;">Predicted</span> '
            f'<code style="color:#e2e8f0;">{top1_guid}</code></div>',
            unsafe_allow_html=True,
        )
        if gt_guid:
            st.markdown(
                f'<div style="margin:0 0 8px 0;font-family:monospace;font-size:0.84em;">'
                f'<span style="color:#3b82f6;font-weight:700;">Ground Truth</span> '
                f'<code style="color:#e2e8f0;">{gt_guid}</code></div>',
                unsafe_allow_html=True,
            )

        # GUID chips
        chip_html = ""
        for i, guid in enumerate(pool_guids[:10]):
            if i == 0:
                color, bg = "#22c55e", "rgba(34,197,94,0.15)"
                label = "TOP-1"
            else:
                color, bg = "#f59e0b", "rgba(245,158,11,0.10)"
                label = f"#{i+1}"

            name = ""
            if i < len(candidates):
                name = candidates[i].get("name", "")
                if len(name) > 40:
                    name = name[:37] + "..."

            chip_html += (
                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                f'padding:4px 10px;margin:2px 4px 2px 0;background:{bg};'
                f'border:1px solid {color};border-radius:6px;'
                f'font-size:0.78em;font-family:monospace;">'
                f'<span style="color:{color};font-weight:700;">{label}</span>'
                f'<code style="color:#e2e8f0;">{guid[:12]}...</code>'
            )
            if name:
                chip_html += f'<span style="color:#94a3b8;font-size:0.9em;">{name}</span>'
            chip_html += "</div>"

        if len(pool_guids) > 10:
            chip_html += (
                f'<span style="color:#64748b;font-size:0.82em;margin-left:6px;">'
                f'+{len(pool_guids) - 10} more</span>'
            )
        st.markdown(chip_html, unsafe_allow_html=True)

        # Viewer button
        viewer_url = _build_viewer_url(
            static_base_url,
            pool_for_viewer,
            target_guid=top1_guid,
            gt_guid=gt_guid,
            guid_match=guid_match,
            ifc_model_code=ifc_model_code,
        )
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
            f'<button style="'
            f'margin-top:10px;padding:10px 24px;background:#1e40af;color:#e2e8f0;'
            f'border:none;border-radius:8px;font-family:monospace;'
            f'font-size:14px;cursor:pointer;font-weight:600;">'
            f'Open 3D Viewer — {len(pool_guids)} candidates &#8599;</button></a>',
            unsafe_allow_html=True,
        )


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


def _call_modal_explain(
    image_bytes_list: list[bytes],
    chat_text: str,
    metadata_text: str,
    grid_size: int = 4,
) -> dict | None:
    """Call the Modal explain endpoint for occlusion saliency."""
    try:
        import modal
        predictor_cls = modal.Cls.from_name("mscd-vlm-lora3-inference", "G8Predictor")
        predictor = predictor_cls()
        result = predictor.explain.remote(
            image_bytes_list=image_bytes_list,
            chat_text=chat_text,
            metadata_text=metadata_text,
            grid_size=grid_size,
        )
        return result
    except Exception as e:
        st.error(f"Explain failed: {e}")
        return None


def _render_saliency(explain_result: dict, uploaded_files) -> None:
    """Render occlusion saliency heatmaps overlaid on input images."""
    import io
    import numpy as np
    from PIL import Image, ImageFilter

    heatmaps = explain_result.get("heatmaps") or []
    image_sizes = explain_result.get("image_sizes") or []
    grid_size = explain_result.get("grid_size", 4)
    focus_tokens = explain_result.get("spatial_focus_tokens") or []

    if focus_tokens:
        st.caption(f"Tracked spatial predicate: **{', '.join(focus_tokens)}**")

    cols = st.columns(max(len(heatmaps), 1))

    for idx, heatmap_2d in enumerate(heatmaps):
        if idx >= len(uploaded_files):
            break

        hm = np.array(heatmap_2d, dtype=np.float32)

        # Load original image
        img_bytes = uploaded_files[idx].getvalue()
        orig = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h = orig.size

        # Upscale heatmap to image size with smooth interpolation
        hm_img = Image.fromarray((hm * 255).astype(np.uint8), mode="L")
        hm_img = hm_img.resize((w, h), Image.BILINEAR)
        hm_img = hm_img.filter(ImageFilter.GaussianBlur(radius=max(w, h) // 20))

        # Create colormap overlay (blue-to-red)
        hm_array = np.array(hm_img, dtype=np.float32) / 255.0
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        # Low importance = blue, high importance = red
        overlay[..., 0] = (hm_array * 255).astype(np.uint8)       # Red channel
        overlay[..., 2] = ((1 - hm_array) * 180).astype(np.uint8)  # Blue channel

        overlay_img = Image.fromarray(overlay, mode="RGB")

        # Blend original + overlay
        blended = Image.blend(orig, overlay_img, alpha=0.45)

        with cols[idx]:
            st.image(blended, caption=f"Image {idx+1}: saliency ({grid_size}x{grid_size})", use_container_width=True)
            # Show raw heatmap values
            with st.expander("Raw heatmap", expanded=False):
                for row in heatmap_2d:
                    st.text("  ".join(f"{v:.2f}" for v in row))


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
        from src.v2.types import Constraints, SpatialTriplet
        from src.v2.constraints_to_query import QueryPlanner
        from src.v2.retrieval_backend import RetrievalBackend
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

        neo4j_cfg = config.get("neo4j", {})
        retrieval_mode = "neo4j" if neo4j_cfg.get("enabled", False) else "memory"

        # Init engine + backend
        engine = _get_engine(config, model_code=model_code)
        backend = RetrievalBackend(
            engine=engine,
            retrieval_mode=retrieval_mode,
        )

        # Execute plans in cascade (first non-empty wins)
        all_results = []
        winning_result = None
        for plan in plans:
            result = asyncio.run(backend.execute_plan(plan))
            result_dict = {
                "priority": plan.priority,
                "strategy": plan.strategy,
                "pool_size": result.pool_size,
                "candidates": result.candidates,
                "backend": result.backend,
                "fallback_triggered": result.fallback_triggered,
                "strategy_actually_used": result.strategy_actually_used,
            }
            all_results.append(result_dict)
            if result.pool_size > 0 and winning_result is None:
                winning_result = result_dict

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
