"""Sidebar — run / case selector + evaluation metrics summary."""
import streamlit as st
from demo.loader import list_runs, list_cases


def render() -> tuple[str, str] | tuple[None, None]:
    """Render sidebar and return (run_id, case_id), or (None, None) if nothing selected."""
    st.sidebar.title("MSCD Demo")
    st.sidebar.caption("BIM Issue Localisation")
    st.sidebar.divider()

    runs = list_runs()
    if not runs:
        st.sidebar.warning("No trace files found in outputs/traces/")
        return None, None

    run_id = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)

    cases = list_cases(run_id)
    if not cases:
        st.sidebar.warning(f"No cases in {run_id}")
        return None, None

    case_id = st.sidebar.selectbox("Case", cases)

    return run_id, case_id


def render_metrics(trace: dict) -> None:
    """Show eval metrics below the selector."""
    st.sidebar.divider()
    st.sidebar.markdown("**Evaluation**")

    ev = trace.get("evaluation") or {}
    guid_match   = trace.get("guid_match")   or ev.get("guid_match",  False)
    name_match   = trace.get("name_match")   or ev.get("name_match",  False)
    storey_match = trace.get("storey_match") or ev.get("storey_match", False)

    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("GUID", "✓" if guid_match   else "✗")
    col2.metric("Name", "✓" if name_match   else "✗")
    col3.metric("Storey","✓" if storey_match else "✗")

    gt = (trace.get("scenario") or {}).get("ground_truth") or {}
    if gt:
        st.sidebar.divider()
        st.sidebar.markdown("**Ground Truth**")
        st.sidebar.code(gt.get("target_guid", "—"), language=None)
        st.sidebar.caption(gt.get("target_name", ""))
        st.sidebar.caption(f"Storey: {gt.get('target_storey', '—')}")

    timing = trace.get("total_latency_ms")
    if timing:
        st.sidebar.caption(f"⏱ {timing/1000:.1f}s total latency")
