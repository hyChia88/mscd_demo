"""
MCP-Based AI Agent Orchestrator for BIM Inspection

This is the production-ready version using Model Context Protocol (MCP) for tool integration.
Instead of hardcoded LangChain tools, this version dynamically connects to MCP servers
that expose BIM analysis capabilities as standardized services.

Architecture:
    LangGraph ReAct Agent → MCP Clients → MCP Servers (IFC, Visual, Rendering)

Benefits over legacy approach:
    1. Tools are decoupled and reusable across different AI applications
    2. MCP servers can be developed/deployed/scaled independently
    3. Standard protocol enables ecosystem integration (Claude Desktop, VS Code, etc.)
    4. Better separation of concerns: orchestration vs. domain logic

Usage:
    python src/main_mcp.py
"""

import os
import sys
import time
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# LangChain and LangGraph imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

# MCP imports
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: MCP dependencies not installed. Please run: pip install -r requirements.txt")
    MCP_AVAILABLE = False
    sys.exit(1)

# Try to import langchain-mcp-adapters, fall back to custom adapter
try:
    from langchain_mcp_adapters import convert_mcp_to_langchain_tools
    print("[MCP] Using official langchain-mcp-adapters", file=sys.stderr)
except ImportError:
    from mcp_langchain_adapter import convert_mcp_to_langchain_tools
    print("[MCP] Using custom MCP-LangChain adapter", file=sys.stderr)

from chat_logger import ConversationLogger

# RQ2: Schema-aware mapping and validation
from rq2_schema.extract_final_json import extract_final_json
from rq2_schema.schema_registry import SchemaRegistry
from rq2_schema.pipeline import run_rq2_postprocess

# P2: BCF handoff - trace, issue, and BCFzip generation
from handoff.trace import build_trace, write_trace_json
from handoff.bcf_lite import write_issue_json
from handoff.bcf_zip import write_bcfzip


def _get_experiment_description(experiment_mode, query_mode, visual_enabled):
    """Get human-readable description for experiment mode."""
    if not experiment_mode:
        return "Using config.yaml setting"
    base = "Neo4j graph queries" if query_mode == "neo4j" else "In-memory spatial index"
    if visual_enabled:
        return f"{base} + CLIP visual matching"
    return base


def load_config(config_file="config.yaml"):
    """Load centralized configuration from YAML file"""
    base_dir = Path(__file__).parent.parent
    config_path = base_dir / config_file

    if not config_path.exists():
        print(f"⚠️  Warning: Config file '{config_path}' not found. Using defaults.")
        return {
            "ifc": {"model_path": "data/ifc/AdvancedProject/IFC/AdvancedProject.ifc"},
            "ground_truth": {
                "file": "data/ground_truth/gt_1/gt_1.json",
                "image_dir": "data/ground_truth/gt_1/imgs"
            },
            "llm": {"model": "gemini-2.5-flash", "temperature": 0, "max_retries": 2},
            "agent": {"delay_between_tests": 7, "system_prompt_file": "prompts/system_prompt.yaml"},
            "output": {"evaluations_dir": "logs/evaluations", "logs_dir": "logs"}
        }

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_system_prompt(prompt_file="prompts/system_prompt.yaml"):
    """Load system prompt from YAML file"""
    base_dir = Path(__file__).parent.parent
    prompt_path = base_dir / prompt_file

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("system_prompt", "")


def load_scenarios(file_path="prompts/tests/test_2.yaml"):
    """Load test scenarios from YAML file"""
    if not os.path.exists(file_path):
        print(f"❌ Error: Configuration file '{file_path}' not found.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
            return []


def load_ground_truth(file_path="data/ground_truth/gt_1/gt_1.json"):
    """
    Load ground truth test cases from JSON or JSONL.

    Supports:
    1. Single JSON array file (gt_1.json)
    2. JSONL file with one case per line (cases_v2.jsonl)

    Args:
        file_path: Path to JSON/JSONL file (relative to project root)

    Returns:
        list: List of test case dictionaries
    """
    base_dir = Path(__file__).parent.parent
    gt_path = base_dir / file_path

    if not gt_path.exists():
        print(f"❌ Error: Ground truth path '{gt_path}' not found.")
        return []

    # JSONL format (cases_v2.jsonl)
    if gt_path.suffix == ".jsonl":
        cases = []
        with open(gt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
        print(f"📂 Loaded {len(cases)} cases from JSONL")
        return cases

    # Single JSON array file (gt_1.json)
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_test_input(case, image_dir):
    """
    Format a ground truth case into agent input string.

    Supports both formats:
    - Legacy (gt_1.json): context_payload.meta, image_file
    - Standardized (cases_v2.jsonl): inputs.project_context, inputs.images

    Args:
        case: Ground truth test case dict
        image_dir: Path to directory containing test images

    Returns:
        tuple: (formatted_input_string, list_of_image_paths)
    """
    # Handle both legacy and standardized field names
    if "context_payload" in case:
        # Legacy format (gt_1.json)
        meta = case["context_payload"]["meta"]
        chat_history = case["context_payload"]["chat_history"]
        image_files = case.get("image_file", [])
    else:
        # Standardized format (cases_v2.jsonl)
        inputs = case.get("inputs", {})
        meta = inputs.get("project_context", {})
        chat_history = inputs.get("chat_history", [])
        image_files = [Path(p).name for p in inputs.get("images", [])]

    # Build context string
    input_parts = [
        "=" * 50,
        "[CONTEXT]",
        f"  Timestamp: {meta.get('timestamp', 'N/A')}",
        f"  Sender Role: {meta.get('sender_role', 'N/A')}",
        f"  Project Phase: {meta.get('project_phase', 'N/A')}",
        f"  4D Task Status: {meta.get('4d_task_status', 'N/A')}",
        "",
        "[CHAT HISTORY]"
    ]

    for msg in chat_history:
        input_parts.append(f"  {msg['role']}: {msg['text']}")

    input_parts.extend([
        "",
        "[USER QUERY]",
        f"  {case.get('query_text', '')}",
    ])

    # Build image paths
    image_paths = []
    for img_file in image_files:
        img_path = Path(image_dir) / img_file
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")

    # Include image paths in the message so agent can analyze them
    if image_paths:
        input_parts.append("")
        input_parts.append("[ATTACHED IMAGES]")
        for img_path in image_paths:
            input_parts.append(f"  Image path: {img_path}")
        input_parts.append("  Note: Use analyze_site_image(image_path) to analyze these images")

    input_parts.append("=" * 50)
    formatted_input = "\n".join(input_parts)

    return formatted_input, image_paths


def extract_guids_from_response(response_text: str) -> list:
    """
    Extract all IFC GUIDs mentioned in the response.

    IFC GUIDs are 22-character base64 strings (GlobalId format).
    Pattern: alphanumeric + $ + _ characters, exactly 22 chars.

    Returns:
        list: Ordered list of GUIDs as they appear in the response (preserves order for top-k)
    """
    import re
    # IFC GlobalId pattern: 22 characters of [0-9A-Za-z_$]
    guid_pattern = r'\b([0-9A-Za-z_$]{22})\b'
    matches = re.findall(guid_pattern, response_text)
    # Remove duplicates while preserving order
    seen = set()
    unique_guids = []
    for guid in matches:
        if guid not in seen:
            seen.add(guid)
            unique_guids.append(guid)
    return unique_guids


def evaluate_response(response_text, ground_truth):
    """
    Evaluate agent response against ground truth with top-k metrics.

    Args:
        response_text: The agent's response string
        ground_truth: Ground truth dict with target_guid, expected_reasoning, etc.

    Returns:
        dict: Evaluation results with top-k metrics, precision, recall
    """
    target_guid = ground_truth.get("target_guid", "")
    target_name = ground_truth.get("target_name", "")
    rq_category = ground_truth.get("rq_category", "")

    # Extract all GUIDs mentioned in response (ordered by appearance)
    mentioned_guids = extract_guids_from_response(response_text)

    results = {
        "guid_match": False,
        "name_match": False,
        "target_guid": target_guid,
        "target_name": target_name,
        "rq_category": rq_category,
        "details": [],
        # New top-k metrics
        "mentioned_guids": mentioned_guids,
        "num_candidates": len(mentioned_guids),
        "top1_hit": False,
        "top3_hit": False,
        "top5_hit": False,
        "target_rank": None,  # Position of target in mentioned GUIDs (1-indexed), None if not found
        # Retrieval metrics
        "precision_at_1": 0.0,
        "precision_at_3": 0.0,
        "recall": 0.0,
    }

    # Skip special target GUIDs
    if target_guid and target_guid not in ["MULTIPLE", "CLARIFICATION_NEEDED", "INSUFFICIENT_DATA", "INVALID_LOCATION"]:
        # Check if target GUID is found in response (backward compatible)
        if target_guid in response_text:
            results["guid_match"] = True
            results["details"].append(f"✅ Target GUID found: {target_guid}")
        else:
            results["details"].append(f"❌ Target GUID not found: {target_guid}")

        # Top-k evaluation
        if mentioned_guids:
            if target_guid in mentioned_guids:
                rank = mentioned_guids.index(target_guid) + 1  # 1-indexed
                results["target_rank"] = rank

                # Top-k hits
                results["top1_hit"] = (rank == 1)
                results["top3_hit"] = (rank <= 3)
                results["top5_hit"] = (rank <= 5)

                results["details"].append(f"📊 Target GUID rank: {rank}/{len(mentioned_guids)}")

                # Precision@k (for single target, precision = 1/k if hit in top-k, else 0)
                results["precision_at_1"] = 1.0 if rank == 1 else 0.0
                results["precision_at_3"] = 1.0 / min(3, rank) if rank <= 3 else 0.0

                # Recall (single target: 1 if found, 0 if not)
                results["recall"] = 1.0
            else:
                results["details"].append(f"📊 Target GUID not in {len(mentioned_guids)} mentioned candidates")
                results["recall"] = 0.0
        else:
            results["details"].append("📊 No GUIDs extracted from response")
            results["recall"] = 0.0 if target_guid else 1.0  # If no target expected, recall=1

    # Check if target name is mentioned
    if target_name:
        name_parts = target_name.split(":")[0] if ":" in target_name else target_name
        if name_parts.lower() in response_text.lower():
            results["name_match"] = True
            results["details"].append(f"✅ Target name found: {name_parts}")
        else:
            results["details"].append(f"❌ Target name not found: {name_parts}")

    return results


async def main_async(args=None):
    """Main asynchronous orchestrator"""
    load_dotenv()

    # Determine experiment mode
    experiment_mode = args.experiment if args and args.experiment else None

    # Load centralized configuration
    config = load_config()
    base_dir = Path(__file__).parent.parent

    # RQ2: Load schema configuration
    rq2_cfg = config.get("rq2", {})
    rq2_enabled = rq2_cfg.get("enabled", True)
    rq2_schema_path = str(base_dir / rq2_cfg.get("schema_path", "schemas/corenetx_min/v0.schema.json"))
    schema_registry = None
    rq2_schema = None
    rq2_schema_id = None
    if rq2_enabled:
        try:
            schema_registry = SchemaRegistry(rq2_schema_path)
            rq2_schema = schema_registry.schema
            rq2_schema_id = schema_registry.schema_id
            print(f"[RQ2] Schema loaded: {rq2_schema_id}")
        except FileNotFoundError as e:
            print(f"⚠️  Warning: RQ2 schema not found: {e}")
            rq2_enabled = False

    # Initialize conversation logger
    logger = ConversationLogger()

    # Setup LLM from config
    llm_config = config.get("llm", {})
    llm = ChatGoogleGenerativeAI(
        model=llm_config.get("model", "gemini-2.5-flash"),
        temperature=llm_config.get("temperature", 0),
        max_retries=llm_config.get("max_retries", 2),
    )

    # Load system prompt from config
    agent_config = config.get("agent", {})
    try:
        system_prompt = load_system_prompt(agent_config.get("system_prompt_file", "prompts/system_prompt.yaml"))
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        system_prompt = "You are a helpful BIM inspection assistant. Use the available tools to help users query the IFC model."

    # Define MCP server configuration
    python_exe = sys.executable
    ifc_server_path = str(base_dir / "mcp_servers" / "ifc_server.py")

    # Ground truth / dataset configuration
    # Priority: --cases > --dataset > config.yaml
    gt_config = config.get("ground_truth", {})

    # Dataset presets
    DATASET_PRESETS = {
        "gt1": {
            "file": "data/ground_truth/gt_1/gt_1.json",
            "image_dir": "data/ground_truth/gt_1/imgs"
        },
        "synth": {
            "file": "../data_curation/datasets/synth_v0.2/cases_v2.jsonl",
            "image_dir": "../data_curation/datasets/synth_v0.2/cases/imgs"
        }
    }

    # Resolve dataset path
    if args and args.cases:
        # Custom path specified
        gt_file = args.cases
        gt_image_dir = base_dir / gt_config.get("image_dir", "data/ground_truth/gt_1/imgs")
        dataset_name = "custom"
    elif args and args.dataset:
        # Preset selected
        preset = DATASET_PRESETS.get(args.dataset, DATASET_PRESETS["gt1"])
        gt_file = preset["file"]
        gt_image_dir = base_dir / preset["image_dir"]
        dataset_name = args.dataset
    else:
        # Use config.yaml
        gt_file = gt_config.get("file", "data/ground_truth/gt_1/gt_1.json")
        gt_image_dir = base_dir / gt_config.get("image_dir", "data/ground_truth/gt_1/imgs")
        dataset_name = "config"

    # Print active dataset to terminal
    print(f"\n{'='*60}")
    print(f"  DATASET: {dataset_name.upper()}  ({gt_file})")
    print(f"  IMAGES:  {gt_image_dir}")
    print(f"{'='*60}\n")

    # Output configuration from config
    output_config = config.get("output", {})
    evaluations_dir = base_dir / output_config.get("evaluations_dir", "logs/evaluations")

    # Delay between tests from config
    delay_between_tests = agent_config.get("delay_between_tests", 7)

    # Build environment for MCP server (inherit current env + add QUERY_MODE if specified)
    server_env = dict(os.environ)  # Inherit current environment

    # Parse experiment mode: "memory", "neo4j", "memory+clip", "neo4j+clip"
    query_mode = None
    visual_enabled = False
    if experiment_mode:
        if "+clip" in experiment_mode:
            query_mode = experiment_mode.replace("+clip", "")
            visual_enabled = True
        else:
            query_mode = experiment_mode
            visual_enabled = False
        server_env["QUERY_MODE"] = query_mode
        server_env["VISUAL_ENABLED"] = "true" if visual_enabled else "false"

    # Determine display mode
    display_mode = experiment_mode or "config"  # "config" means using config.yaml setting

    print("\n" + "="*70)
    print("🚀 Initializing MCP-based Agent")
    print(f"   IFC Model: {config.get('ifc', {}).get('model_path', 'N/A')}")
    print(f"   Dataset: {dataset_name.upper()} → {gt_file}")
    print(f"   Image Dir: {gt_image_dir}")
    if experiment_mode:
        print(f"   Experiment: {display_mode.upper()}")
        print(f"   Query Mode: {query_mode.upper()}")
        print(f"   Visual (CLIP): {'ENABLED' if visual_enabled else 'DISABLED'}")
    else:
        print(f"   Query Mode: CONFIG (from config.yaml)")
    print("="*70)

    server_params = StdioServerParameters(
        command=python_exe,
        args=[ifc_server_path],
        env=server_env
    )

    # Connect to MCP server and run agent INSIDE the session context
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✅ Connected to IFC Query Service")

            # List available tools
            tools_result = await session.list_tools()

            if not tools_result or not tools_result.tools:
                print("❌ No tools found. Cannot proceed.")
                return

            print(f"   Found {len(tools_result.tools)} tools:")
            for tool in tools_result.tools:
                desc = tool.description[:60] if tool.description else "No description"
                print(f"      - {tool.name}: {desc}...")

            # Convert MCP tools to LangChain tools
            langchain_tools = convert_mcp_to_langchain_tools(tools_result.tools, session)

            # RQ2: Build tool lookup map for post-processing
            tool_by_name = {t.name: t for t in langchain_tools}

            print(f"\n✅ {len(langchain_tools)} tools loaded from MCP server")
            print("="*70 + "\n")

            # Create LangGraph ReAct agent with MCP tools
            agent_executor = create_react_agent(
                llm.bind(system=system_prompt),
                langchain_tools
            )

            # Load ground truth test cases
            test_cases = load_ground_truth(gt_file)
            print(f"✅ Agent initialized. Loaded {len(test_cases)} ground truth test cases.\n")
            logger.log_agent_message(f"MCP-based agent initialized with {len(langchain_tools)} tools")

            # P2: Create run_id for this evaluation session (shared across all cases)
            # Include dataset and experiment mode in run_id for easy identification
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id_parts = [timestamp, dataset_name]
            if experiment_mode:
                run_id_parts.append(experiment_mode)
            run_id = "_".join(run_id_parts)

            # Track evaluation results
            evaluation_results = []

            for i, case in enumerate(test_cases, 1):
                case_id = case.get("case_id", case.get("id", f"case_{i}"))
                rq_category = case.get("ground_truth", {}).get("rq_category", "Unknown")

                print(f"\n{'='*70}")
                print(f"📋 Test Case {i}/{len(test_cases)}: {case_id}")
                print(f"   RQ Category: {rq_category}")
                print(f"{'='*70}")

                # Format input from ground truth case
                user_input, image_paths = format_test_input(case, gt_image_dir)

                print(f"📥 Input:\n{user_input}\n")
                if image_paths:
                    print(f"📷 Images: {', '.join([Path(p).name for p in image_paths])}\n")

                logger.log_user_message(user_input)

                try:
                    start_time = time.time()

                    # TODO: Add image support when multimodal is enabled
                    response = await agent_executor.ainvoke({"messages": [("user", user_input)]})

                    elapsed = time.time() - start_time

                    print(f"\n📤 Final Response ({elapsed:.2f}s):")
                    print("-" * 70)

                    # Extract response
                    if "messages" in response:
                        output = response["messages"][-1].content

                        # Extract tool calls and tool results
                        tool_calls = []
                        tool_results = []
                        for msg in response["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_calls.append({
                                        "name": tool_call.get("name", "unknown"),
                                        "args": tool_call.get("args", {})
                                    })
                            # Capture tool results (ToolMessage content)
                            if hasattr(msg, 'type') and msg.type == 'tool':
                                tool_results.append(str(msg.content))
                    else:
                        output = str(response)
                        tool_calls = []
                        tool_results = []

                    # FALLBACK: If response has no GUIDs, extract from tool calls/results
                    response_guids = extract_guids_from_response(output)
                    if not response_guids:
                        # Priority 1: Extract GUIDs from get_element_details calls (explicit selections)
                        explicit_guids = []
                        for tc in tool_calls:
                            if tc.get("name") == "get_element_details":
                                guid = tc.get("args", {}).get("guid")
                                if guid and guid not in explicit_guids:
                                    explicit_guids.append(guid)

                        # Priority 2: Extract from tool results if no explicit selections
                        if explicit_guids:
                            guid_list = ", ".join(explicit_guids[:5])
                            output = f"{output}\n\n[AUTO-EXTRACTED] Selected element GUID: {guid_list}"
                            print(f"⚠️  No GUIDs in response, extracted {len(explicit_guids)} from get_element_details calls")
                        elif tool_results:
                            # Fallback to search results
                            all_tool_content = "\n".join(tool_results)
                            tool_guids = extract_guids_from_response(all_tool_content)
                            if tool_guids:
                                guid_list = ", ".join(tool_guids[:5])
                                output = f"{output}\n\n[AUTO-EXTRACTED] Top candidate GUIDs from tool results: {guid_list}"
                                print(f"⚠️  No GUIDs in response, extracted {len(tool_guids)} from tool results")

                    # Type C Fix: Handle empty responses with fallback
                    if not output or not output.strip():
                        # Try to extract GUIDs from tool results for empty response
                        if tool_results:
                            all_tool_content = "\n".join(tool_results)
                            tool_guids = extract_guids_from_response(all_tool_content)
                            if tool_guids:
                                guid_list = ", ".join(tool_guids[:5])
                                output = f"[FALLBACK] Top candidates from tool results: {guid_list}"
                            else:
                                output = f"[FALLBACK] Agent returned empty response. Tool calls made: {tool_calls if tool_calls else 'None'}. Please check tool results."
                        else:
                            output = f"[FALLBACK] Agent returned empty response. Tool calls made: {tool_calls if tool_calls else 'None'}. Please check tool results."
                        print(f"⚠️  Empty response detected, using fallback message")

                    print(output)
                    print("-" * 70)

                    logger.log_agent_message(output, tool_calls=tool_calls if tool_calls else None)

                    # RQ2: Parse FINAL_JSON and run post-processing
                    agent_final, parse_err = extract_final_json(output)

                    # Build evidence list
                    evidence = [{"type": "chat", "ref": case_id, "note": "ground truth chat context + query"}]
                    for p in image_paths:
                        evidence.append({"type": "image", "ref": p, "note": "ground truth image"})

                    rq2_result = None
                    if rq2_enabled and rq_category == "RQ2" and rq2_schema is not None:
                        rq2_context = {
                            "storey_name": agent_final.get("selected_storey_name", "") if agent_final else "",
                            "evidence": evidence
                        }
                        rq2_result = await run_rq2_postprocess(
                            schema_id=rq2_schema_id,
                            schema=rq2_schema,
                            agent_final=agent_final,
                            parse_error=parse_err,
                            rq2_context=rq2_context,
                            tool_by_name=tool_by_name
                        )

                    # Evaluate response against ground truth
                    eval_result = evaluate_response(output, case.get("ground_truth", {}))
                    eval_result["case_id"] = case_id
                    eval_result["elapsed_time"] = elapsed
                    eval_result["tool_calls"] = tool_calls
                    eval_result["rq2"] = rq2_result  # Attach RQ2 results (may be None for non-RQ2 cases)
                    # Capture ambiguity tier for synthetic dataset analysis
                    tier = case.get("ambiguity_tier") or case.get("difficulty_tags", {}).get("tier")
                    if tier:
                        eval_result["ambiguity_tier"] = tier

                    # P2: Build trace and generate handoff artifacts
                    # Prepare rq2_result for trace (convert to dict format if needed)
                    eval_result_for_trace = dict(eval_result)
                    if rq2_result:
                        eval_result_for_trace["rq2_result"] = rq2_result

                    trace = build_trace(
                        run_id=run_id,
                        case_id=case_id,
                        test_case=case,
                        agent_response=output,
                        tool_calls=tool_calls,
                        eval_result=eval_result_for_trace,
                        config=config
                    )

                    # Write trace, issue.json, and bcfzip
                    trace_path = write_trace_json(trace, out_dir=str(base_dir / "outputs/traces"))
                    issue_path = write_issue_json(out_dir=str(base_dir / "outputs/issues"), trace=trace)
                    bcf_path = write_bcfzip(out_dir=str(base_dir / "outputs/bcf"), trace=trace)

                    # Add handoff paths to eval_result
                    eval_result["handoff"] = {
                        "trace": trace_path,
                        "issue_json": issue_path,
                        "bcfzip": bcf_path
                    }

                    evaluation_results.append(eval_result)

                    # Print evaluation results
                    print(f"\n📊 Evaluation:")
                    for detail in eval_result["details"]:
                        print(f"   {detail}")

                    # P2: Print handoff artifact paths
                    print(f"\n📁 Handoff Artifacts:")
                    print(f"   Trace:  {trace_path}")
                    print(f"   Issue:  {issue_path}")
                    print(f"   BCFzip: {bcf_path}")

                except Exception as e:
                    error_msg = f"Error during execution: {e}"
                    print(f"\n❌ {error_msg}")
                    logger.log_agent_message(f"ERROR: {error_msg}")
                    evaluation_results.append({
                        "case_id": case_id,
                        "guid_match": False,
                        "name_match": False,
                        "error": str(e)
                    })

                # Pause between test cases
                if i < len(test_cases):
                    print(f"\n⏳ Proceeding to next test case in {delay_between_tests} seconds...")
                    await asyncio.sleep(delay_between_tests)

            # Print summary report
            print(f"\n{'='*70}")
            print("📊 GROUND TRUTH EVALUATION SUMMARY")
            print(f"{'='*70}")

            total = len(evaluation_results)
            guid_matches = sum(1 for r in evaluation_results if r.get("guid_match", False))
            name_matches = sum(1 for r in evaluation_results if r.get("name_match", False))

            # Top-k retrieval metrics
            top1_hits = sum(1 for r in evaluation_results if r.get("top1_hit", False))
            top3_hits = sum(1 for r in evaluation_results if r.get("top3_hit", False))
            top5_hits = sum(1 for r in evaluation_results if r.get("top5_hit", False))
            avg_precision_at_1 = sum(r.get("precision_at_1", 0) for r in evaluation_results) / total if total > 0 else 0
            avg_recall = sum(r.get("recall", 0) for r in evaluation_results) / total if total > 0 else 0
            # F1 score
            f1_score = 2 * (avg_precision_at_1 * avg_recall) / (avg_precision_at_1 + avg_recall) if (avg_precision_at_1 + avg_recall) > 0 else 0

            print(f"   Total Test Cases: {total}")
            print(f"\n   📈 Retrieval Metrics:")
            print(f"      Top-1 Accuracy: {top1_hits}/{total} ({100*top1_hits/total:.1f}%)")
            print(f"      Top-3 Accuracy: {top3_hits}/{total} ({100*top3_hits/total:.1f}%)")
            print(f"      Top-5 Accuracy: {top5_hits}/{total} ({100*top5_hits/total:.1f}%)")
            print(f"      Precision@1:    {avg_precision_at_1:.3f}")
            print(f"      Recall:         {avg_recall:.3f}")
            print(f"      F1 Score:       {f1_score:.3f}")

            print(f"\n   📋 Legacy Metrics (backward compatible):")
            print(f"      GUID Matches: {guid_matches}/{total} ({100*guid_matches/total:.1f}%)")
            print(f"      Name Matches: {name_matches}/{total} ({100*name_matches/total:.1f}%)")

            # Group by RQ category with top-k metrics
            rq_stats = {}
            for r in evaluation_results:
                rq = r.get("rq_category", "Unknown")
                if rq not in rq_stats:
                    rq_stats[rq] = {"total": 0, "guid_match": 0, "top1_hit": 0, "top3_hit": 0}
                rq_stats[rq]["total"] += 1
                if r.get("guid_match", False):
                    rq_stats[rq]["guid_match"] += 1
                if r.get("top1_hit", False):
                    rq_stats[rq]["top1_hit"] += 1
                if r.get("top3_hit", False):
                    rq_stats[rq]["top3_hit"] += 1

            print(f"\n   📊 Results by RQ Category:")
            for rq, stats in sorted(rq_stats.items()):
                top1_pct = 100 * stats["top1_hit"] / stats["total"] if stats["total"] > 0 else 0
                top3_pct = 100 * stats["top3_hit"] / stats["total"] if stats["total"] > 0 else 0
                print(f"      {rq}: Top-1={stats['top1_hit']}/{stats['total']} ({top1_pct:.1f}%) | Top-3={stats['top3_hit']}/{stats['total']} ({top3_pct:.1f}%)")

            # Ambiguity Tier statistics (for synthetic dataset)
            tier_stats = {}
            for r in evaluation_results:
                tier = r.get("ambiguity_tier", None)
                if tier:
                    if tier not in tier_stats:
                        tier_stats[tier] = {"total": 0, "top1_hit": 0, "top3_hit": 0}
                    tier_stats[tier]["total"] += 1
                    if r.get("top1_hit", False):
                        tier_stats[tier]["top1_hit"] += 1
                    if r.get("top3_hit", False):
                        tier_stats[tier]["top3_hit"] += 1

            if tier_stats:
                print(f"\n   🎯 Results by Ambiguity Tier (Enough Thinking):")
                for tier, stats in sorted(tier_stats.items()):
                    top1_pct = 100 * stats["top1_hit"] / stats["total"] if stats["total"] > 0 else 0
                    top3_pct = 100 * stats["top3_hit"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"      {tier}: Top-1={stats['top1_hit']}/{stats['total']} ({top1_pct:.1f}%) | Top-3={stats['top3_hit']}/{stats['total']} ({top3_pct:.1f}%)")

            # RQ2 Schema Validation summary
            rq2_rows = [r for r in evaluation_results if r.get("rq_category") == "RQ2" and r.get("rq2")]
            if rq2_rows:
                passed = sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"])
                avg_fill = sum(r["rq2"]["submission"]["validation_metadata"]["required_fill_rate"] for r in rq2_rows) / len(rq2_rows)
                print(f"\n   📝 RQ2 Schema Validation:")
                print(f"      Validation Pass Rate: {passed}/{len(rq2_rows)} ({100*passed/len(rq2_rows):.1f}%)")
                print(f"      Avg Field Fill Rate:  {avg_fill:.3f}")

            # Save evaluation results to JSON with enhanced metrics
            evaluations_dir.mkdir(parents=True, exist_ok=True)
            eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Include dataset and experiment mode in filename for easy identification
            filename_parts = ["eval", eval_timestamp, dataset_name]
            if experiment_mode:
                filename_parts.append(experiment_mode)
            results_file = evaluations_dir / f"{'_'.join(filename_parts)}.json"

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": eval_timestamp,
                    "run_id": run_id,
                    "dataset": {
                        "name": dataset_name,
                        "path": gt_file,
                        "image_dir": str(gt_image_dir),
                        "total_cases": total
                    },
                    "experiment": {
                        "mode": experiment_mode or "config",
                        "query_mode": query_mode if experiment_mode else "config",
                        "visual_enabled": visual_enabled if experiment_mode else False,
                        "description": _get_experiment_description(experiment_mode, query_mode, visual_enabled)
                    },
                    "ground_truth_file": gt_file,  # Kept for backward compatibility
                    "summary": {
                        "total": total,
                        # Retrieval metrics (new)
                        "retrieval": {
                            "top1_accuracy": top1_hits / total if total > 0 else 0,
                            "top3_accuracy": top3_hits / total if total > 0 else 0,
                            "top5_accuracy": top5_hits / total if total > 0 else 0,
                            "precision_at_1": avg_precision_at_1,
                            "recall": avg_recall,
                            "f1_score": f1_score
                        },
                        # Legacy metrics (backward compatible)
                        "guid_matches": guid_matches,
                        "name_matches": name_matches,
                        # RQ2 Schema metrics
                        "rq2_schema": {
                            "total": len(rq2_rows),
                            "passed": sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"]) if rq2_rows else 0,
                            "pass_rate": (sum(1 for r in rq2_rows if r["rq2"]["submission"]["validation_metadata"]["passed"]) / len(rq2_rows)) if rq2_rows else 0,
                            "avg_fill_rate": (sum(r["rq2"]["submission"]["validation_metadata"]["required_fill_rate"] for r in rq2_rows) / len(rq2_rows)) if rq2_rows else 0
                        },
                        "by_rq_category": rq_stats,
                        # Ambiguity tier metrics (for synthetic dataset)
                        "by_ambiguity_tier": tier_stats if tier_stats else None
                    },
                    "results": evaluation_results
                }, f, indent=2)

            print(f"\n   Results saved to: {results_file}")

            # P2: Print handoff summary
            print(f"\n   Handoff Artifacts (run_id: {run_id}):")
            print(f"      Traces:  outputs/traces/{run_id}/")
            print(f"      Issues:  outputs/issues/{run_id}/")
            print(f"      BCFzips: outputs/bcf/{run_id}/")

            # Save conversation summary
            logger.save_summary(f"Completed {len(test_cases)} ground truth test cases using MCP architecture")
            print(f"\n📊 Session complete. Check logs/ directory for conversation history.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MCP-Based AI Agent Orchestrator for BIM Inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main_mcp.py                         # Default: use config.yaml setting
  python src/main_mcp.py --experiment memory     # In-memory spatial index (baseline)
  python src/main_mcp.py --experiment neo4j      # Neo4j graph queries
  python src/main_mcp.py --experiment memory+clip  # In-memory + CLIP visual matching
  python src/main_mcp.py --experiment neo4j+clip   # Neo4j + CLIP visual matching
  python src/main_mcp.py -e neo4j+clip           # Short form

Dataset selection:
  python src/main_mcp.py --dataset synth         # Use synthetic dataset (synth_v0.2)
  python src/main_mcp.py --dataset gt1           # Use original ground truth (gt_1)
  python src/main_mcp.py -d synth -e neo4j       # Combine dataset + experiment mode

Run all experiments in sequence:
  for mode in memory neo4j memory+clip neo4j+clip; do
    python src/main_mcp.py -e $mode
    sleep 10
  done
        """
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        choices=["memory", "neo4j", "memory+clip", "neo4j+clip"],
        default=None,
        help="Experiment mode: 'memory', 'neo4j', 'memory+clip', 'neo4j+clip'. The '+clip' variants enable CLIP visual matching tools."
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=["gt1", "synth"],
        default=None,
        help="Dataset to use: 'gt1' (original ground truth), 'synth' (synthetic dataset synth_v0.2). Default: use config.yaml setting."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Custom path to cases file or directory (overrides --dataset and config.yaml)"
    )
    return parser.parse_args()


def main():
    """Synchronous entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not available. Install with: pip install -r requirements.txt")
        return

    args = parse_args()

    # Run the async main function with args
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
