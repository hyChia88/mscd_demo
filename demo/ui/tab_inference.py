"""
Tab 4 — Live Inference: upload images + enter chat → LoRA_3 extracts constraints
→ QueryPlanner builds plans → RetrievalBackend executes → 3D viewer shows pool.

Full neuro-symbolic pipeline: VLM (Modal GPU) → Constraints → Cypher → GUIDs → 3D.
"""
import asyncio
import json
import time
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

def render(*, static_base_url: str = "") -> None:
    st.markdown("#### Neuro-Symbolic Inference Pipeline")
    st.caption(
        "Upload site photos / floorplans and enter chat text. "
        "Watch each pipeline stage execute in sequence: "
        "**VLM → Constraints → Query Plan → Cypher → Pool → 3D**"
    )

    # ── Input form ────────────────────────────────────────────────────────
    col_img, col_text = st.columns([1, 1], gap="medium")

    with col_img:
        st.markdown("**Images**")
        uploaded_files = st.file_uploader(
            "Site photo + floorplan",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            key="inference_images",
        )
        if uploaded_files:
            img_cols = st.columns(min(len(uploaded_files), 3))
            for i, f in enumerate(uploaded_files):
                img_cols[i % 3].image(f, caption=f.name, width=180)

    with col_text:
        st.markdown("**Chat Message**")
        chat_text = st.text_area(
            "Site worker message",
            value="There's a crack on the window next to the railing, third floor",
            height=80,
            key="inference_chat",
        )

        st.markdown("**4D Metadata** (optional)")
        meta_col1, meta_col2 = st.columns(2)
        storey = meta_col1.text_input("Location / Storey", value="3 - Third Floor", key="inf_storey")
        phase = meta_col2.text_input("Project Phase", value="Interior Fit-out", key="inf_phase")
        task_status = meta_col1.selectbox(
            "Task Status",
            ["IN_PROGRESS", "PENDING_INSPECTION", "REVIEW_REQUIRED", "ON_HOLD"],
            key="inf_status",
        )
        ifc_model = meta_col2.selectbox(
            "IFC Model",
            ["AP (AdvancedProject)", "BH (BasicHouse)", "DXA (Duplex)"],
            key="inf_ifc_model",
        )

    metadata_text = (
        f"[4D Task Status] TASK_0001: Inspection — {task_status}\n"
        f"[Project Phase] {phase}\n"
        f"[Location] {storey}"
    )

    # ── Run inference ─────────────────────────────────────────────────────
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

    if run_btn:
        if not chat_text.strip():
            st.warning("Please enter a chat message.")
            return

        image_bytes_list = [f.getvalue() for f in (uploaded_files or [])]

        # Stage 1: VLM
        t0 = time.time()
        with st.spinner("Stage 1/3 — VLM inference on Modal A100..."):
            vlm_result = _call_modal_inference(image_bytes_list, chat_text, metadata_text)
        vlm_ms = int((time.time() - t0) * 1000)

        if vlm_result is None:
            return

        vlm_result["_latency_ms"] = vlm_ms

        # Stage 2+3: Symbolic retrieval
        retrieval_result = None
        if vlm_result.get("valid_json") and vlm_result.get("parsed"):
            t1 = time.time()
            with st.spinner("Stage 2-3 — Query planning + Neo4j retrieval..."):
                retrieval_result = _run_retrieval(vlm_result["parsed"])
            if retrieval_result:
                retrieval_result["_latency_ms"] = int((time.time() - t1) * 1000)

        st.session_state["last_inference"] = vlm_result
        st.session_state["last_retrieval"] = retrieval_result

    # ── Display pipeline ──────────────────────────────────────────────────
    vlm_result = st.session_state.get("last_inference")
    if vlm_result is None:
        st.info("Click **Run Pipeline** to execute the full neuro-symbolic pipeline.")
        return

    retrieval_result = st.session_state.get("last_retrieval")
    _render_pipeline(vlm_result, retrieval_result, static_base_url=static_base_url)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline visualization
# ══════════════════════════════════════════════════════════════════════════════

def _render_pipeline(
    vlm_result: dict,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
) -> None:
    """Render the full pipeline as numbered stages with data flowing between them."""
    valid = vlm_result.get("valid_json", False)
    parsed = vlm_result.get("parsed") or {}
    raw = vlm_result.get("raw_output", "")
    vlm_ms = vlm_result.get("_latency_ms", 0)

    # ━━ Stage 1: Neuro Layer (VLM) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown(
        f'<div style="{_STAGE_STYLE}">'
        + _stage_header(1, "Neuro Layer", "LoRA_3 VLM → Structured JSON", vlm_ms)
        + '</div>',
        unsafe_allow_html=True,
    )

    if not valid:
        st.error("VLM returned invalid JSON — pipeline halted")
        st.code(raw, language="text")
        return

    # Show extracted JSON compactly
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

    # ━━ Stage 5: Candidate Pool + 3D ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

    # 3D viewer
    if pool_guids:
        _render_3d_pool(retrieval_result, static_base_url)
    elif static_base_url:
        viewer_url = _build_viewer_url(static_base_url, [], target_guid="")
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
            f'<button style="margin-top:8px;padding:8px 18px;background:#1e293b;color:#e2e8f0;'
            f'border:1px solid #334155;border-radius:6px;font-family:monospace;'
            f'font-size:13px;cursor:pointer;">Open 3D Viewer (empty) &#8599;</button></a>',
            unsafe_allow_html=True,
        )

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

        # Confidence gate
        first_conf = rels[0].get("confidence", 0)
        gate_pass = first_conf >= _CONF_THRESHOLD
        gate_color = "#22c55e" if gate_pass else "#f59e0b"
        gate_icon = "PASS" if gate_pass else "SKIP"
        gate_action = "Priority 0 Cypher executes" if gate_pass else "Falls back to P1-P8"
        html += (
            f'<div style="margin-top:6px;padding:4px 10px;border-radius:4px;'
            f'background:rgba(30,41,59,0.5);font-size:0.82em;font-family:monospace;">'
            f'<span style="color:{gate_color};font-weight:700;">GATE {gate_icon}</span>'
            f'<span style="color:#94a3b8;"> — conf {first_conf:.2f} vs '
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
    candidates = (winning or {}).get("candidates") or [] if winning else []

    if not pool_guids:
        return

    with st.container(border=True):
        st.markdown("#### 3D Pool Viewer")

        top1_guid = pool_guids[0]
        strategy = winning.get("strategy", "?") if winning else "?"
        pool_size = winning.get("pool_size", len(pool_guids)) if winning else len(pool_guids)

        st.markdown(
            f"**{pool_size} candidates** from `{strategy}` "
            f"— top-1 highlighted green, pool in amber"
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
            static_base_url, pool_guids, target_guid=top1_guid
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
) -> dict | None:
    """Call the Modal serverless inference endpoint."""
    try:
        import modal
        predictor_cls = modal.Cls.from_name("mscd-vlm-lora3-inference", "LoRA3Predictor")
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
            st.error(
                "Modal app `mscd-vlm-lora3-inference` not found. "
                "Deploy it first:\n\n"
                "```\nmodal deploy training/inference.py\n```"
            )
        else:
            st.error(f"Inference failed: {e}")
        return None


def _run_retrieval(parsed: dict) -> dict | None:
    """Run the symbolic retrieval pipeline: Constraints → QueryPlanner → Neo4j."""
    try:
        from src.v2.types import Constraints, SpatialTriplet
        from src.v2.constraints_to_query import QueryPlanner
        from src.v2.retrieval_backend import RetrievalBackend
        import yaml
        from pathlib import Path

        # Build Constraints from parsed VLM output
        spatial_rels = []
        for rel in (parsed.get("spatial_relations") or []):
            spatial_rels.append(SpatialTriplet(
                subject_type=parsed.get("ifc_class", ""),
                predicate=rel.get("predicate", "ADJACENT_TO"),
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

        # Load config for Neo4j
        repo_root = Path(__file__).parent.parent.parent
        config_path = repo_root / "config.yaml"
        config = {}
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text()) or {}

        neo4j_cfg = config.get("neo4j", {})
        retrieval_mode = "neo4j" if neo4j_cfg.get("enabled", False) else "memory"

        # Init engine + backend
        engine = _get_engine(config)
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
                "candidates": result.candidates[:20],
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

        return {
            "plans": [{"priority": p.priority, "strategy": p.strategy,
                        "params": p.params, "expected_pool_size": p.expected_pool_size}
                       for p in plans],
            "results": all_results,
            "winning": winning_result,
            "pool_guids": pool_guids,
        }

    except Exception as e:
        st.warning(f"Symbolic retrieval failed: {e}")
        import traceback
        with st.expander("Error details", expanded=False):
            st.code(traceback.format_exc(), language="text")
        return None


@st.cache_resource(show_spinner="Loading IFC engine...")
def _get_engine(_config: dict):
    """Cache the IFCEngine instance."""
    from src.ifc_engine import IFCEngine
    from src.common.config import init_registry_llm
    from pathlib import Path

    neo4j_cfg = _config.get("neo4j", {})
    repo_root = Path(__file__).parent.parent.parent

    ifc_path = repo_root / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc"
    if not ifc_path.exists():
        ifc_path = Path("/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc")

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


def _build_viewer_url(
    static_base_url: str,
    pool_guids: list[str],
    target_guid: str = "",
) -> str:
    """Build URL for the 3D viewer with candidate GUIDs highlighted."""
    ifc_url = static_base_url + "/data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
    viewer_base = static_base_url + "/demo/static/test_viewer.html"
    params = {
        "ifc": ifc_url,
        "target": target_guid,
        "gt": "",
        "match": "1",
        "pool": ",".join(pool_guids[:60]),
        "base": static_base_url + "/demo/static",
    }
    return viewer_base + "?" + urlencode(params)
