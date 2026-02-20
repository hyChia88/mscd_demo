"""Tab 2 — Pipeline Trace: constraints → query plans → retrieval results."""
import streamlit as st


def render(trace: dict) -> None:
    pipeline_type = trace.get("pipeline_type", "unknown")
    internals     = trace.get("internals") or {}

    if pipeline_type == "v2" and internals:
        _render_v2(internals, trace)
    elif pipeline_type == "v1":
        _render_v1(trace)
    else:
        st.info(f"Pipeline type: `{pipeline_type}` — no internals available.")
        st.json(trace.get("interpreter_output") or {})


def _render_v2(internals: dict, trace: dict) -> None:
    # ── Constraints ────────────────────────────────────────────────────────
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

    # ── Query Plans ────────────────────────────────────────────────────────
    st.markdown("#### Query Plans")
    plans = internals.get("query_plans") or []
    if plans:
        for plan in plans:
            pri      = plan.get("priority", "?")
            strategy = plan.get("strategy", "?")
            pool     = plan.get("expected_pool_size")
            label    = f"P{pri}: `{strategy}`"
            if pool:
                label += f"  →  ~{pool} candidates"
            with st.expander(label, expanded=(pri == 1)):
                st.json(plan.get("params", {}))
    else:
        st.info("No query plans available.")

    # ── Retrieval Results ──────────────────────────────────────────────────
    results = internals.get("retrieval_results") or []
    with st.container(border=True):
        st.markdown("#### Retrieval Results")
        if results:
            for r in results:
                backend  = r.get("backend", "?")
                pool     = r.get("pool_size", 0)
                reranked = r.get("rerank_applied", False)
                st.caption(f"Backend: `{backend}`  ·  Pool: {pool}  ·  Reranked: {reranked}")
                candidates = r.get("candidates") or []
                if candidates:
                    gt_guid = (
                        (trace.get("scenario") or {})
                        .get("ground_truth", {})
                        .get("target_guid", "")
                    )
                    rows = []
                    for i, cand in enumerate(candidates[:10], 1):
                        guid  = cand.get("guid", "")
                        score = cand.get("clip_score")
                        rows.append({
                            "Rank":   i,
                            "GT":     "✓" if guid == gt_guid else "",
                            "Name":   cand.get("name", ""),
                            "Type":   cand.get("type", ""),
                            "Storey": cand.get("storey", ""),
                            "CLIP":   f"{score:.3f}" if score else "—",
                            "GUID":   guid,
                        })
                    st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No retrieval results.")

    # ── Timing ────────────────────────────────────────────────────────────
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
