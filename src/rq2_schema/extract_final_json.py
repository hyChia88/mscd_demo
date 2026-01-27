"""
Extract FINAL_JSON block from agent output.

The agent is expected to output a machine-readable JSON block
starting with the exact tag: FINAL_JSON=
"""

import json
import re
from typing import Any, Dict, Tuple

FINAL_TAG = "FINAL_JSON="


def extract_final_json(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract FINAL_JSON object from agent output.

    Args:
        text: Raw agent output text

    Returns:
        Tuple of (parsed_dict, error_message)
        If successful: (dict, "")
        If failed: ({}, "error description")
    """
    if not text:
        return {}, "empty response"

    idx = text.rfind(FINAL_TAG)
    if idx == -1:
        return {}, "FINAL_JSON tag not found"

    payload = text[idx + len(FINAL_TAG) :].strip()

    # Try direct JSON parse
    try:
        return json.loads(payload), ""
    except Exception:
        pass

    # Fallback: find first {...} after tag
    m = re.search(r"\{.*\}", payload, flags=re.DOTALL)
    if not m:
        return {}, "JSON object not found after FINAL_JSON tag"

    try:
        return json.loads(m.group(0)), ""
    except Exception as e:
        return {}, f"JSON parse failed: {e}"
