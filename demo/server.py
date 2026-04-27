"""
Static file server — serves IFC files + JS/WASM assets to the browser.

Runs in a background daemon thread so Streamlit doesn't block.
"""
import http.server
import posixpath
import threading
from pathlib import Path
from urllib.parse import unquote, urlsplit

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

        root_dir = Path(root).resolve()
        repo_root = root_dir.parent
        mounted_roots = {
            "data_curation": repo_root / "data_curation",
        }

        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(root_dir), **kwargs)

            def translate_path(self, path: str) -> str:
                request_path = posixpath.normpath(unquote(urlsplit(path).path))
                parts = [part for part in request_path.split("/") if part]

                if parts and parts[0] in mounted_roots:
                    base = mounted_roots[parts[0]].resolve()
                    candidate = base.joinpath(*parts[1:]).resolve()
                    try:
                        candidate.relative_to(base)
                    except ValueError:
                        return str(base)
                    return str(candidate)

                return super().translate_path(path)

            def log_message(self, *_):
                pass

        class _QuietServer(http.server.HTTPServer):
            def handle_error(self, request, client_address):
                pass  # suppress BrokenPipeError from browser mid-download cancels

        _server = _QuietServer(("localhost", port), _QuietHandler)
        t = threading.Thread(target=_server.serve_forever, daemon=True)
        t.start()

    return f"http://localhost:{port}"
