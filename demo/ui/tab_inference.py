"""
Tab 4 — Live Inference: upload images + enter chat → LoRA_3 extracts constraints
→ QueryPlanner builds plans → RetrievalBackend executes → 3D viewer shows pool.

Full neuro-symbolic pipeline: VLM (Modal GPU) → Constraints → Cypher → GUIDs → 3D.
"""
import asyncio
import json
from urllib.parse import urlencode

import streamlit as st


_PREDICATE_COLORS = {
    "FILLS": "#3b82f6",
    "ADJACENT_TO": "#f59e0b",
    "CONTINUOUS": "#8b5cf6",
}


def render(*, static_base_url: str = "") -> None:
    st.markdown("#### Live LoRA_3 Inference")
    st.caption(
        "Upload site photos / floorplans and enter chat text. "
        "The VLM extracts constraints, then the symbolic layer retrieves candidate GUIDs."
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
    run_btn = st.button("Run LoRA_3 Inference", type="primary", use_container_width=True)

    if run_btn:
        if not chat_text.strip():
            st.warning("Please enter a chat message.")
            return

        image_bytes_list = []
        for f in (uploaded_files or []):
            image_bytes_list.append(f.getvalue())

        # Step 1: VLM inference (Modal GPU)
        with st.spinner("Step 1/2: Running VLM inference on Modal A100..."):
            vlm_result = _call_modal_inference(image_bytes_list, chat_text, metadata_text)

        if vlm_result is None:
            return

        # Step 2: Symbolic retrieval (local Neo4j)
        retrieval_result = None
        if vlm_result.get("valid_json") and vlm_result.get("parsed"):
            with st.spinner("Step 2/2: Running symbolic retrieval (Neo4j)..."):
                retrieval_result = _run_retrieval(vlm_result["parsed"])

        st.session_state["last_inference"] = vlm_result
        st.session_state["last_retrieval"] = retrieval_result

    # ── Display result ────────────────────────────────────────────────────
    vlm_result = st.session_state.get("last_inference")
    if vlm_result is None:
        st.info("Click **Run LoRA_3 Inference** to extract constraints from your inputs.")
        return

    retrieval_result = st.session_state.get("last_retrieval")
    _render_result(vlm_result, retrieval_result, static_base_url=static_base_url)


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
    """Run the symbolic retrieval pipeline: Constraints → QueryPlanner → Neo4j.

    Returns dict with keys: plans, results, candidates, pool_guids, strategy_used.
    """
    try:
        from src.v2.types import Constraints, SpatialTriplet
        from src.v2.constraints_to_query import QueryPlanner
        from src.v2.retrieval_backend import RetrievalBackend
        from src.ifc_engine import IFCEngine
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

        constraints = Constraints(
            storey_name=parsed.get("storey_name"),
            ifc_class=parsed.get("ifc_class"),
            space_name=parsed.get("space_name"),
            target_name_keyword=parsed.get("target_name_keyword"),
            spatial_relations=spatial_rels,
            confidence=0.9,
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

        # Execute plans in cascade
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
        st.warning(f"Symbolic retrieval failed (Neo4j may not be running): {e}")
        return None


@st.cache_resource(show_spinner="Loading IFC engine...")
def _get_engine(_config: dict):
    """Cache the IFCEngine instance."""
    from src.ifc_engine import IFCEngine
    from pathlib import Path

    neo4j_cfg = _config.get("neo4j", {})
    repo_root = Path(__file__).parent.parent.parent

    # Find IFC file
    ifc_path = repo_root / "data" / "ifc" / "AdvancedProject" / "IFC" / "AdvancedProject.ifc"
    if not ifc_path.exists():
        ifc_path = Path("/root/cmu/master_thesis/data_curation/ifc_models/AdvancedProject.ifc")

    # Connect to Neo4j if enabled
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
            import streamlit as _st
            _st.warning(f"Neo4j connection failed: {e} — using memory mode")
            neo4j_conn = None

    engine = IFCEngine(str(ifc_path), neo4j_conn=neo4j_conn)
    return engine


def _render_result(
    vlm_result: dict,
    retrieval_result: dict | None,
    *,
    static_base_url: str = "",
) -> None:
    """Display VLM output + retrieval results + 3D viewer."""
    valid = vlm_result.get("valid_json", False)
    parsed = vlm_result.get("parsed")
    raw = vlm_result.get("raw_output", "")

    # Status badge
    if valid:
        st.success("Valid JSON output")
    else:
        st.error("Invalid JSON — raw output shown below")
        st.code(raw, language="json")
        return

    # ── Extracted Constraints ─────────────────────────────────────────
    with st.container(border=True):
        st.markdown("#### Extracted Constraints")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spatial**")
            st.markdown(f"- Storey: `{parsed.get('storey_name') or '—'}`")
            st.markdown(f"- Space: `{parsed.get('space_name') or '—'}`")
        with col2:
            st.markdown("**Semantic**")
            st.markdown(f"- IFC class: `{parsed.get('ifc_class') or '—'}`")
            st.markdown(f"- Name keyword: `{parsed.get('target_name_keyword') or '—'}`")

    # ── Spatial Relations ─────────────────────────────────────────────
    rels = parsed.get("spatial_relations") or []

    with st.container(border=True):
        if rels:
            st.markdown("#### Spatial Relations (Priority 0)")
            for rel in rels:
                pred = rel.get("predicate", "?")
                obj_type = rel.get("object_type", "?")
                obj_mat = rel.get("object_material")
                conf = rel.get("confidence", 0)
                color = _PREDICATE_COLORS.get(pred, "#6b7280")
                mat_tag = f" ({obj_mat})" if obj_mat else ""

                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'padding:8px 14px;background:#1e293b;border-radius:8px;'
                    f'border-left:4px solid {color};margin-bottom:8px;">'
                    f'<span style="color:#e2e8f0;font-family:monospace;font-size:1em;">'
                    f'{parsed.get("ifc_class", "?")} '
                    f'<span style="background:{color};color:white;padding:2px 10px;'
                    f'border-radius:4px;font-weight:700;font-size:0.9em;">'
                    f'{pred}</span>'
                    f' → {obj_type}{mat_tag}'
                    f'</span>'
                    f'<span style="margin-left:auto;color:#94a3b8;font-size:0.85em;">'
                    f'confidence: {conf:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            threshold = 0.7
            first_conf = rels[0].get("confidence", 0) if rels else 0
            gate_icon = "✅" if first_conf >= threshold else "⚠️"
            gate_msg = (
                f"→ Priority 0 Cypher query executed"
                if first_conf >= threshold
                else f"→ would fall back to Priority 1–8 cascade"
            )
            st.caption(f"{gate_icon} Confidence {first_conf:.2f} vs threshold {threshold} {gate_msg}")
        else:
            st.markdown("#### Spatial Relations")
            st.info(
                "No spatial relations extracted — attribute-only case. "
                "System uses Priority 1–8 cascade (storey+type, name keyword, etc.)"
            )

    # ── Retrieval Results + 3D Viewer ─────────────────────────────────
    if retrieval_result is None:
        st.info("Symbolic retrieval not available (Neo4j may not be running).")
        with st.expander("Raw JSON output", expanded=False):
            st.json(parsed)
        return

    with st.container(border=True):
        st.markdown("#### Symbolic Retrieval")

        # Query plans cascade
        plans = retrieval_result.get("plans") or []
        results = retrieval_result.get("results") or []

        for i, plan in enumerate(plans):
            result = results[i] if i < len(results) else None
            pool_size = result["pool_size"] if result else None
            tried = result is not None

            if not tried:
                icon, badge = "⏭", f"~{plan.get('expected_pool_size', '?')} est."
            elif pool_size and pool_size > 0:
                icon, badge = "✅", f"**{pool_size} items**"
            else:
                icon, badge = "❌", "0 items"

            label = f"{icon}  P{plan['priority']}: `{plan['strategy']}`  —  {badge}"
            is_winner = (result is not None and pool_size and pool_size > 0
                         and retrieval_result.get("winning", {}).get("strategy") == result.get("strategy"))
            with st.expander(label, expanded=is_winner):
                st.json(plan.get("params", {}), expanded=False)
                if result:
                    st.caption(
                        f"Backend: `{result.get('backend', '?')}` · "
                        f"Fallback: {result.get('fallback_triggered', False)} · "
                        f"Strategy used: `{result.get('strategy_actually_used', '?')}`"
                    )

        # Candidates table
        winning = retrieval_result.get("winning")
        pool_guids = retrieval_result.get("pool_guids") or []

        if winning and winning.get("candidates"):
            candidates = winning["candidates"]
            st.markdown(
                f"**Winning strategy:** P{winning['priority']} `{winning['strategy']}` "
                f"— **{winning['pool_size']} candidates**"
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

        else:
            st.warning("No candidates found. Neo4j may not have the required edges.")

    # ── View Pool in 3D ──────────────────────────────────────────────
    _render_3d_pool(retrieval_result, static_base_url)

    # ── Raw JSON ──────────────────────────────────────────────────────
    with st.expander("Raw JSON output", expanded=False):
        st.json(parsed)


def _render_3d_pool(
    retrieval_result: dict | None,
    static_base_url: str,
) -> None:
    """Render a dedicated 'View Pool in 3D' section with GUID chips + viewer link."""
    if not static_base_url:
        return

    winning = (retrieval_result or {}).get("winning")
    pool_guids = (retrieval_result or {}).get("pool_guids") or []
    candidates = (winning or {}).get("candidates") or [] if winning else []

    with st.container(border=True):
        st.markdown("#### View Pool in 3D")

        if not pool_guids:
            st.info(
                "No candidate pool to visualise. "
                "Run inference with a query that produces retrieval results."
            )
            # Still allow opening the bare model
            viewer_url = _build_viewer_url(static_base_url, [], target_guid="")
            st.markdown(
                f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
                f'<button style="'
                f"margin-top:4px;padding:8px 18px;background:#1e293b;color:#e2e8f0;"
                f"border:1px solid #334155;border-radius:6px;font-family:monospace;"
                f'font-size:13px;cursor:pointer;">'
                f"Open 3D Viewer (empty model) &#8599;</button></a>",
                unsafe_allow_html=True,
            )
            return

        top1_guid = pool_guids[0] if pool_guids else ""
        strategy = winning.get("strategy", "?") if winning else "?"
        pool_size = winning.get("pool_size", len(pool_guids)) if winning else len(pool_guids)

        # Pool summary
        st.markdown(
            f"**{pool_size} candidates** from `{strategy}` strategy "
            f"&mdash; top-1 highlighted green, pool in amber"
        )

        # GUID chips for top candidates
        chip_html = ""
        for i, guid in enumerate(pool_guids[:10]):
            if i == 0:
                # Top-1: green
                color, bg = "#22c55e", "rgba(34,197,94,0.15)"
                label = "TOP-1"
            else:
                # Pool: amber
                color, bg = "#f59e0b", "rgba(245,158,11,0.10)"
                label = f"#{i+1}"

            name = ""
            if i < len(candidates):
                name = candidates[i].get("name", "")
                if len(name) > 40:
                    name = name[:37] + "..."

            chip_html += (
                f'<div style="display:inline-flex;align-items:center;gap:6px;'
                f"padding:4px 10px;margin:2px 4px 2px 0;background:{bg};"
                f'border:1px solid {color};border-radius:6px;font-size:0.78em;font-family:monospace;">'
                f'<span style="color:{color};font-weight:700;">{label}</span>'
                f'<code style="color:#e2e8f0;">{guid[:12]}...</code>'
            )
            if name:
                chip_html += f'<span style="color:#94a3b8;font-size:0.9em;">{name}</span>'
            chip_html += "</div>"

        if len(pool_guids) > 10:
            chip_html += (
                f'<span style="color:#64748b;font-size:0.82em;margin-left:6px;">'
                f"+{len(pool_guids) - 10} more</span>"
            )

        st.markdown(chip_html, unsafe_allow_html=True)

        # Viewer button
        viewer_url = _build_viewer_url(
            static_base_url, pool_guids, target_guid=top1_guid
        )
        st.markdown(
            f'<a href="{viewer_url}" target="_blank" style="text-decoration:none;">'
            f'<button style="'
            f"margin-top:10px;padding:10px 24px;background:#1e40af;color:#e2e8f0;"
            f"border:none;border-radius:8px;font-family:monospace;"
            f'font-size:14px;cursor:pointer;font-weight:600;">'
            f"Open 3D Viewer &mdash; {len(pool_guids)} candidates &#8599;</button></a>",
            unsafe_allow_html=True,
        )


def _build_viewer_url(
    static_base_url: str,
    pool_guids: list[str],
    target_guid: str = "",
) -> str:
    """Build URL for the 3D viewer with candidate GUIDs highlighted."""
    ifc_url = static_base_url + "/" + "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"
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
