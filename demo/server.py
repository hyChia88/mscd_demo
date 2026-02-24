"""
Static file server — serves IFC files + JS/WASM assets to the browser.

Runs in a background daemon thread so Streamlit doesn't block.
"""
import http.server
import threading
from pathlib import Path

_server: http.server.HTTPServer | None = None
_lock = threading.Lock()


def start(root: str, port: int = 8502) -> str:
    """
    Start the static server (idempotent — safe to call multiple times).

    Returns the base URL, e.g. 'http://localhost:8502'.
    """
    global _server
    with _lock:
        if _server is not None:
            return f"http://localhost:{port}"

        root_dir = root

        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=root_dir, **kwargs)

            def log_message(self, *_):
                pass

        class _QuietServer(http.server.HTTPServer):
            def handle_error(self, request, client_address):
                pass  # suppress BrokenPipeError from browser mid-download cancels

        _server = _QuietServer(("localhost", port), _QuietHandler)
        t = threading.Thread(target=_server.serve_forever, daemon=True)
        t.start()

    return f"http://localhost:{port}"
