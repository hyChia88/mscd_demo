"""
Unified IFC GUID extraction.

IFC GlobalIds are 22-character strings from the Base64 alphabet [0-9A-Za-z_$].
This module provides the single canonical implementation used everywhere.

Consolidates:
- main_mcp.py:extract_guids_from_response()
- eval/runner.py:extract_guids_from_text()
- handoff/trace.py:extract_ifc_guid()
"""

import re
from typing import List, Optional

# Compiled pattern for IFC GlobalId: exactly 22 characters at word boundaries
_IFC_GUID_RE = re.compile(r"\b[0-9A-Za-z_$]{22}\b")


def extract_guids_from_text(text: str) -> List[str]:
    """
    Extract all IFC GUIDs from text, deduplicated, preserving first-occurrence order.

    Includes false-positive filtering:
    - Rejects all-digit matches
    - Rejects strings with 3 or fewer distinct characters
    """
    if not text:
        return []
    matches = _IFC_GUID_RE.findall(text)
    seen = set()
    result = []
    for m in matches:
        if m in seen:
            continue
        if m.isdigit():
            continue
        if len(set(m)) <= 3:
            continue
        seen.add(m)
        result.append(m)
    return result


def extract_first_ifc_guid(text: str) -> Optional[str]:
    """
    Extract the first IFC GUID from text, or None.

    Replaces handoff/trace.py:extract_ifc_guid() with consistent
    word-boundary matching and false-positive filtering.
    """
    if not text:
        return None
    guids = extract_guids_from_text(text)
    return guids[0] if guids else None
