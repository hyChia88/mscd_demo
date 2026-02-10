"""
Unified response parsing for LangGraph agent output.

Extracts structured data from the agent's response messages:
- Final response text
- Tool calls (name + args)
- Tool results (ToolMessage content)
- GUID fallback chain

Consolidates:
- main_mcp.py lines 561-620 (tool extraction + 3-level GUID fallback)
- chat_cli.py lines 249-276 (tool extraction + 2-level fallback)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ParsedResponse:
    """Structured extraction from LangGraph agent response."""
    final_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[str] = field(default_factory=list)
    messages: List[Any] = field(default_factory=list)


def extract_response_content(response: Dict[str, Any]) -> ParsedResponse:
    """
    Extract structured content from a LangGraph agent response.

    Handles:
    - response["messages"] format (standard LangGraph)
    - Fallback to str(response) if no messages
    """
    result = ParsedResponse()

    if "messages" not in response:
        result.final_text = str(response)
        return result

    messages = response["messages"]
    result.messages = messages

    # Final text is always the last message's content
    if messages:
        result.final_text = getattr(messages[-1], "content", str(messages[-1]))

    # Extract tool calls and tool results
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                result.tool_calls.append({
                    "name": tool_call.get("name", "unknown"),
                    "args": tool_call.get("args", {}),
                })
        # Capture tool results (ToolMessage content)
        if hasattr(msg, "type") and msg.type == "tool":
            result.tool_results.append(str(msg.content))

    return result


def apply_guid_fallback(
    parsed: ParsedResponse,
    extract_guids_fn: Callable[[str], List[str]],
) -> str:
    """
    Apply the 3-level GUID fallback chain to ensure the response contains GUIDs.

    Fallback levels:
    1. GUIDs already in final_text -> no change
    2. Extract from get_element_details tool call args
    3. Extract from tool results

    Returns the (possibly augmented) final text.
    """
    output = parsed.final_text

    # Check if response already has GUIDs
    if extract_guids_fn(output):
        return output

    # Priority 1: Extract from get_element_details calls (explicit selections)
    explicit_guids = []
    for tc in parsed.tool_calls:
        if tc.get("name") == "get_element_details":
            guid = tc.get("args", {}).get("guid")
            if guid and guid not in explicit_guids:
                explicit_guids.append(guid)

    if explicit_guids:
        guid_list = ", ".join(explicit_guids[:5])
        return f"{output}\n\n[AUTO-EXTRACTED] Selected element GUID: {guid_list}"

    # Priority 2: Extract from tool results
    if parsed.tool_results:
        all_tool_content = "\n".join(parsed.tool_results)
        tool_guids = extract_guids_fn(all_tool_content)
        if tool_guids:
            guid_list = ", ".join(tool_guids[:5])
            return f"{output}\n\n[AUTO-EXTRACTED] Top candidate GUIDs from tool results: {guid_list}"

    return output


def handle_empty_response(
    parsed: ParsedResponse,
    extract_guids_fn: Callable[[str], List[str]],
) -> str:
    """
    Handle empty agent responses by extracting from tool results.

    Returns fallback response text if final_text is empty, otherwise
    returns final_text unchanged.
    """
    if parsed.final_text and parsed.final_text.strip():
        return parsed.final_text

    # Try to extract GUIDs from tool results
    if parsed.tool_results:
        all_tool_content = "\n".join(parsed.tool_results)
        tool_guids = extract_guids_fn(all_tool_content)
        if tool_guids:
            guid_list = ", ".join(tool_guids[:5])
            return f"[FALLBACK] Top candidates from tool results: {guid_list}"

    tc_desc = parsed.tool_calls if parsed.tool_calls else "None"
    return f"[FALLBACK] Agent returned empty response. Tool calls made: {tc_desc}. Please check tool results."
