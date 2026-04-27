"""
MSCD Demo UI — Streamlit entry point.

Run:
    cd mscd_demo
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

# Ensure repo root is on path so `from src...` and `from demo...` work
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))  # for `from common.config` etc.

from dotenv import load_dotenv
import streamlit as st
import streamlit.components.v1 as components

from demo import server, loader
from demo.demo_dashboard import build_dashboard_url
from demo.ui import sidebar, tab_inference

# Load API keys from project-root .env (e.g., GOOGLE_API_KEY for registry LLM).
load_dotenv(REPO_ROOT / ".env")

# ── page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MSCD Demo",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── start static file server (once per session) ───────────────────────────
@st.cache_resource
def get_static_base() -> str:
    """Start background HTTP server serving the repo root on port 8502."""
    return server.start(root=str(REPO_ROOT), port=8502)

static_base = get_static_base()

# ── query review selection state ──────────────────────────────────────────
run_id, case_id = sidebar.get_selection()

if run_id is None or case_id is None:
    st.info("No review traces are available yet.")
    st.stop()

# ── load trace ────────────────────────────────────────────────────────────
trace = loader.load_trace(run_id, case_id)

if trace is None:
    st.error(f"Trace not found: {run_id} / {case_id}")
    st.stop()

# ── header ────────────────────────────────────────────────────────────────
st.markdown("### AEC Interpreter Demo")

# ── main tabs ─────────────────────────────────────────────────────────────
t1, t2, t3 = st.tabs([
    "Live Walkthrough", "Trace Replay", "Impact Dashboard"
])

with t1:
    tab_inference.render(static_base_url=static_base, trace=trace, case_id=case_id)

with t2:
    left_col, right_col = st.columns([1, 2], gap="medium")

    with left_col:
        sidebar.render(trace)

    with right_col:
        tab_inference.render_trace_flow(
            trace,
            static_base_url=static_base,
            case_id=case_id,
        )

with t3:
    dashboard_url = build_dashboard_url(
        static_base,
        case_id=case_id,
        guid=st.session_state.get("approved_guid", ""),
        mode="grounding",
    )
    components.iframe(dashboard_url, height=920, scrolling=True)
