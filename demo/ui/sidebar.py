"""Query review selector panel."""

import streamlit as st

from demo.loader import list_cases, list_runs


def get_selection() -> tuple[str, str] | tuple[None, None]:
    """Return the current run/case selection from session state."""
    runs = list_runs()
    if not runs:
        return None, None

    run_id = st.session_state.get("review_run_id", runs[-1])
    if run_id not in runs:
        run_id = runs[-1]
    st.session_state["review_run_id"] = run_id

    cases = list_cases(run_id)
    if not cases:
        return run_id, None

    case_id = st.session_state.get("review_case_id", cases[0])
    if case_id not in cases:
        case_id = cases[0]
    st.session_state["review_case_id"] = case_id
    return run_id, case_id


def render(trace: dict) -> None:
    """Render the query review selector as a live-demo-style evidence panel."""
    with st.container(border=True):
        st.markdown(
            '<p style="font-size:11px;font-weight:600;text-transform:uppercase;'
            'letter-spacing:0.5px;color:#64748b;margin-bottom:4px;">Evidence Input</p>',
            unsafe_allow_html=True,
        )

        runs = list_runs()
        if not runs:
            st.warning("No trace files found in outputs/traces/")
            return

        current_run = st.session_state.get("review_run_id", runs[-1])
        if current_run not in runs:
            current_run = runs[-1]
            st.session_state["review_run_id"] = current_run

        top_c1, top_c2 = st.columns([1, 1])
        with top_c1:
            run_id = st.selectbox("Run", runs, key="review_run_id")

        cases = list_cases(run_id)
        if not cases:
            st.warning(f"No cases in {run_id}")
            return

        if st.session_state.get("review_case_id") not in cases:
            st.session_state["review_case_id"] = cases[0]

        with top_c2:
            st.selectbox("Case", cases, key="review_case_id")

        st.caption("Evaluation")
        ev = trace.get("evaluation") or {}
        guid_match = trace.get("guid_match") or ev.get("guid_match", False)
        name_match = trace.get("name_match") or ev.get("name_match", False)
        storey_match = trace.get("storey_match") or ev.get("storey_match", False)

        m1, m2, m3 = st.columns(3)
        m1.metric("GUID", "✓" if guid_match else "✗")
        m2.metric("Name", "✓" if name_match else "✗")
        m3.metric("Storey", "✓" if storey_match else "✗")

        gt = (trace.get("scenario") or {}).get("ground_truth") or {}
        if gt:
            with st.expander("Ground Truth", expanded=True):
                st.code(gt.get("target_guid", "—"), language=None)
                if gt.get("target_name"):
                    st.caption(gt.get("target_name", ""))
                st.caption(f"Storey: {gt.get('target_storey', '—')}")
                st.caption(f"IFC Class: {gt.get('target_ifc_class', '—')}")

        timing = trace.get("total_latency_ms")
        if timing:
            st.caption(f"Total latency: {timing/1000:.1f}s")
