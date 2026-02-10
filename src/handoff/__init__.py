"""
Handoff Package - BCF Issue Generation and Trace Management

This package provides:
- Trace building and persistence (trace.py)
- BCF-lite JSON issue output (bcf_lite.py)
- BCFzip generation for BIM tool interoperability (bcf_zip.py)
"""

from common.guid import extract_first_ifc_guid as extract_ifc_guid  # backward compat
from .trace import build_trace, write_trace_json
from .bcf_lite import write_issue_json
from .bcf_zip import write_bcfzip

__all__ = [
    "build_trace",
    "extract_ifc_guid",
    "write_trace_json",
    "write_issue_json",
    "write_bcfzip",
]
