"""
Evaluation Runner for Pipeline v2

Provides run_one_scenario() that:
1. Takes a ScenarioInput and agent context
2. Executes the agent with tracing
3. Captures tool calls with timestamps
4. Extracts candidates from response
5. Returns complete EvalTrace
"""

import ast
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .contracts import (
    CandidateElement,
    EvalTrace,
    InterpreterOutput,
    RQ2Result,
    RQCategory,
    ScenarioInput,
    ToolStepRecord,
)

# RQ2 imports (optional - only used if rq2_schema is available)
try:
    from rq2_schema.extract_final_json import extract_final_json
    from rq2_schema.pipeline import run_rq2_postprocess

    RQ2_AVAILABLE = True
except ImportError:
    RQ2_AVAILABLE = False


def format_scenario_input(scenario: ScenarioInput) -> str:
    """
    Format scenario into agent input string.
    Matches existing format_test_input() pattern from main_mcp.py.
    """
    meta = scenario.context_meta

    parts = [
        "=" * 50,
        "[CONTEXT]",
        f"  Timestamp: {meta.timestamp}",
        f"  Sender Role: {meta.sender_role}",
        f"  Project Phase: {meta.project_phase}",
        f"  4D Task Status: {meta.task_status or 'N/A'}",
        "",
        "[CHAT HISTORY]",
    ]

    for msg in scenario.chat_history:
        parts.append(f"  {msg.role}: {msg.text}")

    parts.extend(["", "[USER QUERY]", f"  {scenario.query_text}", "=" * 50])

    return "\n".join(parts)


def extract_guids_from_text(text: str) -> List[str]:
    """
    Extract IFC GUIDs from text using regex pattern.
    IFC GUIDs are 22-character base64-like strings.
    """
    # IFC GUID pattern: 22 characters, alphanumeric + $ and _
    pattern = r"\b[0-9A-Za-z_$]{22}\b"
    matches = re.findall(pattern, text)

    # Filter out common false positives
    guids = []
    for m in matches:
        # Skip if all digits or all same character
        if not m.isdigit() and len(set(m)) > 3:
            guids.append(m)

    return list(dict.fromkeys(guids))  # Deduplicate while preserving order


def extract_element_names_from_text(text: str) -> List[str]:
    """
    Extract BIM element names from text.
    Looks for patterns like "ElementType:SubType:ID" or quoted names.
    """
    names = []

    # Pattern: Type:Subtype:Number format (e.g., "Basic Wall:Generic - 200mm:308895")
    pattern1 = r"([A-Z][a-zA-Z\s]+:[^:]+:\d+)"
    names.extend(re.findall(pattern1, text))

    # Pattern: BALANS window names
    pattern2 = r"(BALANS[^,\n\)]+)"
    names.extend(re.findall(pattern2, text))

    # Pattern: Common element prefixes followed by identifiers
    pattern3 = r"((?:Wall|Door|Window|Slab|Column|Beam)[_\s][A-Za-z0-9_\-:]+)"
    names.extend(re.findall(pattern3, text))

    return list(dict.fromkeys(names))


def parse_candidates_from_tool_result(
    tool_name: str, tool_result: str, storey_context: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Extract candidate elements from a tool result.

    Args:
        tool_name: Name of the tool that produced the result
        tool_result: String output from the tool
        storey_context: Optional storey filter that was applied

    Returns:
        List of candidate element dicts with guid, name, type, location
    """
    candidates = []

    try:
        # Try to parse as Python literal (list of dicts)
        if tool_result.strip().startswith("["):
            data = ast.literal_eval(tool_result)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "guid" in item:
                        candidates.append(
                            {
                                "guid": item.get("guid", ""),
                                "name": item.get("name", ""),
                                "type": item.get("type", ""),
                                "location": item.get("location", storey_context),
                            }
                        )
    except (ValueError, SyntaxError):
        # Not a parseable list, extract GUIDs from text
        guids = extract_guids_from_text(tool_result)
        for guid in guids:
            candidates.append({"guid": guid, "name": None, "type": None})

    return candidates


def infer_element_type(target_name: str) -> str:
    """
    Infer IFC element type from ground truth target name.
    """
    name_lower = target_name.lower()
    if "wall" in name_lower:
        return "IfcWall"
    elif "window" in name_lower:
        return "IfcWindow"
    elif "door" in name_lower:
        return "IfcDoor"
    elif "slab" in name_lower:
        return "IfcSlab"
    elif "column" in name_lower:
        return "IfcColumn"
    elif "beam" in name_lower:
        return "IfcBeam"
    return "IfcBuildingElement"


def compute_initial_pool_size(scenario: ScenarioInput, engine: Any = None) -> int:
    """
    Compute the initial pool size for search-space reduction.

    Strategy:
    1. Determine target element type from ground truth name
    2. Query total count of that type in model

    Args:
        scenario: Input scenario with ground truth
        engine: IFC engine for querying (optional)

    Returns:
        Initial pool size (total elements of target type)
    """
    target_name = scenario.ground_truth.target_name
    element_type = infer_element_type(target_name)

    # If we have direct engine access
    if engine is not None and hasattr(engine, "file"):
        try:
            elements = engine.file.by_type(element_type)
            return len(elements)
        except Exception:
            pass

    # Default fallback based on typical model sizes
    type_defaults = {
        "IfcWall": 200,
        "IfcWindow": 150,
        "IfcDoor": 50,
        "IfcSlab": 30,
        "IfcColumn": 100,
        "IfcBeam": 80,
        "IfcBuildingElement": 500,
    }
    return type_defaults.get(element_type, 100)


def parse_interpreter_output(
    response_messages: List[Any], scenario: ScenarioInput
) -> InterpreterOutput:
    """
    Parse the agent's response messages into structured InterpreterOutput.

    Args:
        response_messages: List of LangChain message objects
        scenario: Original scenario for context

    Returns:
        InterpreterOutput with extracted data
    """
    # Get final response text
    final_response = ""
    if response_messages:
        last_msg = response_messages[-1]
        final_response = getattr(last_msg, "content", str(last_msg))

    output = InterpreterOutput(raw_response=final_response)

    # Extract GUIDs and names
    output.mentioned_guids = extract_guids_from_text(final_response)
    output.mentioned_names = extract_element_names_from_text(final_response)

    # Build candidate list from all messages with GUIDs
    seen_guids = set()
    all_candidates = []
    for msg in response_messages:
        if hasattr(msg, "content") and msg.content:
            guids = extract_guids_from_text(str(msg.content))
            for guid in guids:
                if guid not in seen_guids:
                    seen_guids.add(guid)
                    all_candidates.append(CandidateElement(guid=guid))

    output.candidates = all_candidates

    # Detect clarification/escalation
    clarification_phrases = [
        "could you please provide",
        "could you tell me",
        "which room",
        "which floor",
        "what is the guid",
        "i cannot determine",
        "i need more information",
        "please clarify",
        "unable to identify",
    ]
    lower_response = final_response.lower()
    output.is_clarification_request = any(
        p in lower_response for p in clarification_phrases
    )
    output.is_escalation = output.is_clarification_request or not output.mentioned_guids

    if output.is_escalation:
        if output.is_clarification_request:
            output.escalation_reason = "clarification_needed"
        else:
            output.escalation_reason = "no_candidates_found"

    # Field population tracking
    target_guid = scenario.ground_truth.target_guid
    target_name = scenario.ground_truth.target_name
    target_name_prefix = target_name.split(":")[0] if ":" in target_name else target_name

    output.fields_populated = {
        "guid": target_guid in final_response,
        "name": target_name_prefix.lower() in final_response.lower(),
        "storey": scenario.ground_truth.target_storey.lower() in final_response.lower(),
        "properties": "Pset_" in final_response or "FireRating" in final_response,
        "compliance_check": any(
            w in final_response.lower()
            for w in ["compliant", "compliance", "code", "regulation"]
        ),
        "bcf_issue": "BCF" in final_response or "issue" in final_response.lower(),
    }

    return output


async def run_one_scenario(
    scenario: ScenarioInput,
    agent_executor: Any,
    engine: Any = None,
    run_id: Optional[str] = None,
    rq2_enabled: bool = False,
    rq2_schema: Optional[Dict[str, Any]] = None,
    rq2_schema_id: Optional[str] = None,
    tool_by_name: Optional[Dict[str, Any]] = None,
) -> EvalTrace:
    """
    Execute a single evaluation scenario and return complete trace.

    This is the core function that:
    1. Formats the scenario input
    2. Invokes the agent
    3. Captures tool calls with timestamps
    4. Parses the response
    5. Computes matches against ground truth
    6. (For RQ2) Runs schema validation post-processing

    Args:
        scenario: ScenarioInput with ground truth
        agent_executor: LangGraph agent executor
        engine: Optional IFC engine for pool size computation
        run_id: Optional run identifier (generated if not provided)
        rq2_enabled: Whether to run RQ2 post-processing
        rq2_schema: JSON Schema dict for RQ2 validation
        rq2_schema_id: Schema identifier string
        tool_by_name: Dict mapping tool name -> LangChain tool (for RQ2)

    Returns:
        Complete EvalTrace with all execution data
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    trace = EvalTrace(
        scenario_id=scenario.id,
        run_id=run_id,
        scenario=scenario,
        timestamp=datetime.now(),
    )

    # Compute initial pool size
    trace.initial_pool_size = compute_initial_pool_size(scenario, engine)

    # Format input
    formatted_input = format_scenario_input(scenario)
    scenario.formatted_input = formatted_input

    start_time = time.time()

    try:
        # Invoke agent
        response = await agent_executor.ainvoke(
            {"messages": [("user", formatted_input)]}
        )

        end_time = time.time()
        trace.total_latency_ms = (end_time - start_time) * 1000

        # Process response messages
        if "messages" in response:
            messages = response["messages"]

            # Extract tool steps
            step_index = 0
            for i, msg in enumerate(messages):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_step = ToolStepRecord(
                            step_index=step_index,
                            tool_name=tool_call.get("name", "unknown"),
                            tool_args=tool_call.get("args", {}),
                            start_time=datetime.now(),  # Approximate
                            success=True,
                        )

                        # Find corresponding tool result in next message
                        if i + 1 < len(messages):
                            next_msg = messages[i + 1]
                            if hasattr(next_msg, "content"):
                                tool_step.tool_result = str(next_msg.content)
                                tool_step.candidates_extracted = (
                                    parse_candidates_from_tool_result(
                                        tool_step.tool_name, tool_step.tool_result
                                    )
                                )
                                tool_step.candidate_count = len(
                                    tool_step.candidates_extracted
                                )

                        trace.tool_steps.append(tool_step)
                        step_index += 1

            # Parse interpreter output
            trace.interpreter_output = parse_interpreter_output(messages, scenario)

            # Compute final pool size from last tool step with candidates
            for step in reversed(trace.tool_steps):
                if step.candidate_count > 0:
                    trace.final_pool_size = step.candidate_count
                    break

            if trace.final_pool_size is None:
                trace.final_pool_size = (
                    len(trace.interpreter_output.candidates)
                    if trace.interpreter_output
                    else 0
                )

        # Compute ground truth matches
        gt = scenario.ground_truth
        final_response = (
            trace.interpreter_output.raw_response if trace.interpreter_output else ""
        )

        # GUID match: exact match in response
        trace.guid_match = gt.target_guid in final_response if gt.target_guid else False

        # Name match: partial match (first part of name)
        name_parts = (
            gt.target_name.split(":")[0] if ":" in gt.target_name else gt.target_name
        )
        trace.name_match = (
            name_parts.lower() in final_response.lower() if name_parts else False
        )

        # TODO
        # Storey match 
        trace.storey_match = (
            gt.target_storey.lower() in final_response.lower()
            if gt.target_storey
            else False
        )

        # RQ2: Schema validation post-processing (only for RQ2 cases)
        if (
            rq2_enabled
            and RQ2_AVAILABLE
            and scenario.ground_truth.rq_category == RQCategory.RQ2
            and rq2_schema is not None
            and tool_by_name is not None
        ):
            try:
                # Parse FINAL_JSON from agent output
                agent_final, parse_err = extract_final_json(final_response)

                # Build evidence list
                evidence = [
                    {
                        "type": "chat",
                        "ref": scenario.id,
                        "note": "ground truth chat context + query",
                    }
                ]
                for img_path in scenario.image_paths:
                    evidence.append(
                        {"type": "image", "ref": img_path, "note": "ground truth image"}
                    )

                rq2_context = {
                    "storey_name": (
                        agent_final.get("selected_storey_name", "")
                        if agent_final
                        else ""
                    ),
                    "evidence": evidence,
                }

                rq2_raw_result = await run_rq2_postprocess(
                    schema_id=rq2_schema_id,
                    schema=rq2_schema,
                    agent_final=agent_final,
                    parse_error=parse_err,
                    rq2_context=rq2_context,
                    tool_by_name=tool_by_name,
                )

                # Convert to typed RQ2Result
                trace.rq2_result = RQ2Result.from_pipeline_result(rq2_raw_result)

            except Exception as rq2_err:
                # Log but don't fail the trace
                trace.error = f"RQ2 post-processing error: {rq2_err}"

        trace.success = True

    except Exception as e:
        trace.error = str(e)
        trace.success = False
        trace.total_latency_ms = (time.time() - start_time) * 1000

    return trace
