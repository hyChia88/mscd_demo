"""Tab 2 — Pipeline Trace: constraints → query plans → retrieval results."""
import streamlit as st
from urllib.parse import urlencode


def render(trace: dict, *, static_base_url: str = "", case_id: str = "") -> None:
    pipeline_type = trace.get("pipeline_type", "unknown")
    internals     = trace.get("internals") or {}

    if pipeline_type == "v2" and internals:
        _render_v2(internals, trace, static_base_url=static_base_url, case_id=case_id)
    elif pipeline_type == "v1":
        _render_v1(trace)
    else:
        st.info(f"Pipeline type: `{pipeline_type}` — no internals available.")
        st.json(trace.get("interpreter_output") or {})


def _build_viewer_url(
    static_base_url: str,
    case_id: str,
    pool_guids: list[str],
    gt_guid: str,
    target_guid: str = "",
    guid_match: bool = False,
) -> str:
    """Build URL for the 3D pool viewer with candidate GUIDs highlighted."""
    from demo import loader as _loader
    ifc_url     = _loader.get_ifc_url(case_id, static_base_url)
    viewer_base = static_base_url + "/demo/static/test_viewer.html"
    params = {
        "ifc":    ifc_url,
        "target": target_guid or (pool_guids[0] if pool_guids else ""),
        "gt":     gt_guid,
        "match":  "1" if guid_match else "0",
        "pool":   ",".join(pool_guids[:60]),   # cap at 60 GUIDs in URL
        "base":   static_base_url + "/demo/static",
    }
    return viewer_base + "?" + urlencode(params)


def _render_v2(
    internals: dict,
    trace: dict,
    *,
    static_base_url: str = "",
    case_id: str = "",
) -> None:
    gt_guid    = (trace.get("scenario") or {}).get("ground_truth", {}).get("target_guid", "")
    guid_match = trace.get("guid_match", False)

    # ── Constraints Extraction ──────────────────────────────────────────────
    c = internals.get("constraints") or {}
    with st.container(border=True):
        st.markdown("#### Constraints Extraction")
        if c:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Spatial**")
                st.markdown(f"- Storey: `{c.get('storey_name') or '—'}`")
                st.markdown(f"- Space: `{c.get('space_name') or '—'}`")
            with col2:
                st.markdown("**Semantic**")
                st.markdown(f"- IFC class: `{c.get('ifc_class') or '—'}`")
                st.markdown(f"- Name keyword: `{c.get('target_name_keyword') or '—'}`")
                st.markdown(f"- Neighbor type: `{c.get('neighbor_type') or '—'}`")

            conf = c.get("confidence", 0)
            src  = c.get("source", "?")
            st.progress(conf, text=f"Confidence {conf:.2f}  ·  source: `{src}`")
        else:
            st.info("No constraints extracted.")

    # ── Merge plans with actual retrieval results ───────────────────────────
    # Plans are tried in order until pool > 0 — retrieval_results[i] maps to plans[i].
    plans   = internals.get("query_plans") or []
    results = internals.get("retrieval_results") or []

    steps = []
    for i, plan in enumerate(plans):
        result      = results[i] if i < len(results) else None
        actual_pool = result["pool_size"] if result is not None else None
        tried       = result is not None
        success     = tried and actual_pool is not None and actual_pool > 0
        steps.append({
            "plan":    plan,
            "result":  result,
            "actual":  actual_pool,
            "tried":   tried,
            "success": success,
        })

    # ── Query Plans ─────────────────────────────────────────────────────────
    st.markdown("#### Query Plans")
    if steps:
        for s in steps:
            plan  = s["plan"]
            pri   = plan.get("priority", "?")
            strat = plan.get("strategy", "?")
            est   = plan.get("expected_pool_size")

            if not s["tried"]:
                icon  = "⏭"
                badge = f"~{est} est." if est else "~?"
            elif s["success"]:
                icon  = "✅"
                badge = f"**{s['actual']} items**"
            else:
                icon  = "❌"
                badge = "0 items"

            label = f"{icon}  P{pri}: `{strat}`  —  {badge}"
            with st.expander(label, expanded=s["success"]):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.json(plan.get("params", {}), expanded=False)
                with col_b:
                    status_str = "not tried" if not s["tried"] else ("✓ used" if s["success"] else "✗ empty")
                    st.markdown(f"**Status:** {status_str}")
                    actual_str = str(s["actual"]) if s["tried"] else "—"
                    st.markdown(f"**Actual pool:** {actual_str}")
                    st.markdown(f"**Expected:** ~{est if est else '?'}")
    else:
        st.info("No query plans available.")

    # ── Pool Visualization (step slider + 3D viewer) ────────────────────────
    tried_steps = [s for s in steps if s["tried"]]
    st.markdown("#### Pool Visualization")

    if not tried_steps:
        st.info("No retrieval results to visualize.")
        _render_timing(internals)
        return

    # Step selector — only visible if more than one plan was tried
    if len(tried_steps) == 1:
        sel_idx = 0
    else:
        options = list(range(len(tried_steps)))
        sel_idx = st.select_slider(
            "Query Plan Step",
            options=options,
            format_func=lambda i: (
                f"{'✅' if tried_steps[i]['success'] else '❌'}  "
                f"P{tried_steps[i]['plan'].get('priority', '?')}: "
                f"{tried_steps[i]['plan'].get('strategy', '?')}  "
                f"({tried_steps[i]['actual']} items)"
            ),
            help="Slide to inspect each query plan's candidate pool.",
        )

    sel        = tried_steps[sel_idx]
    plan       = sel["plan"]
    result     = sel.get("result") or {}
    candidates = result.get("candidates") or []
    pool_size  = sel.get("actual", 0)
    backend    = result.get("backend", "?")
    reranked   = result.get("rerank_applied", False)

    with st.container(border=True):
        # Caption bar
        pool_guids = [c.get("guid", "") for c in candidates if c.get("guid")]
        in_pool    = gt_guid in pool_guids if gt_guid else False
        recall_tag = "  ·  GT ✓ in pool" if in_pool else ("  ·  GT ✗ not in pool" if gt_guid else "")

        st.caption(
            f"**P{plan.get('priority','?')} · {plan.get('strategy','?')}**"
            f"  ·  Backend: `{backend}`"
            f"  ·  Pool: **{pool_size}**"
            f"  ·  Reranked: {reranked}"
            f"{recall_tag}"
        )

        if candidates:
            rows = []
            for rank, cand in enumerate(candidates[:10], 1):
                guid  = cand.get("guid", "")
                score = cand.get("clip_score")
                rows.append({
                    "Rank":   rank,
                    "GT":     "✓" if guid == gt_guid else "",
                    "Name":   cand.get("name", ""),
                    "Type":   cand.get("type", ""),
                    "Storey": cand.get("storey", ""),
                    "CLIP":   f"{score:.3f}" if score is not None else "—",
                    "GUID":   guid,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # 3D viewer button
            if static_base_url and case_id and pool_guids:
                top_guid = pool_guids[0]
                is_match = (top_guid == gt_guid)
                viewer_url = _build_viewer_url(
                    static_base_url, case_id, pool_guids, gt_guid,
                    target_guid=top_guid, guid_match=is_match,
                )
                st.link_button(
                    f"🔭  View Pool in 3D  ({pool_size} elements · amber = candidate)",
                    viewer_url,
                    use_container_width=False,
                )
        else:
            st.info("No candidates in this plan step.")

    _render_timing(internals)


def _render_timing(internals: dict) -> None:
    t_ext  = internals.get("constraints_extraction_ms", 0)
    t_plan = internals.get("query_planning_ms", 0)
    t_ret  = internals.get("retrieval_ms", 0)
    cols = st.columns(3)
    cols[0].metric("Constraints", f"{t_ext:.0f} ms")
    cols[1].metric("Query planning", f"{t_plan:.1f} ms")
    cols[2].metric("Retrieval", f"{t_ret:.0f} ms")


def _render_v1(trace: dict) -> None:
    tool_steps = trace.get("tool_steps") or []
    st.markdown(f"#### Agent Tool Calls ({len(tool_steps)})")
    if not tool_steps:
        agent = trace.get("agent") or {}
        calls = agent.get("tool_calls") or []
        for i, call in enumerate(calls, 1):
            with st.expander(f"#{i}  `{call.get('name','?')}`"):
                st.json(call.get("args", {}))
        if not calls:
            st.info("No tool calls recorded.")
        return

    for step in tool_steps:
        name = step.get("tool_name", "?")
        with st.expander(f"`{name}`  →  {step.get('candidate_count', 0)} candidates"):
            st.json(step.get("tool_args", {}))
            if step.get("tool_result"):
                st.text(step["tool_result"][:500])
