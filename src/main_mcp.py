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
    """Load ground truth test cases from JSON file"""
    base_dir = Path(__file__).parent.parent
    gt_path = base_dir / file_path

    if not gt_path.exists():
        print(f"❌ Error: Ground truth file '{gt_path}' not found.")
        return []

    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_test_input(case, image_dir):
    """
    Format a ground truth case into agent input string.

    Args:
        case: Ground truth test case dict
        image_dir: Path to directory containing test images

    Returns:
        tuple: (formatted_input_string, list_of_image_paths)
    """
    context = case["context_payload"]
    meta = context["meta"]

    # Build context string
    input_parts = [
        "=" * 50,
        "[CONTEXT]",
        f"  Timestamp: {meta['timestamp']}",
        f"  Sender Role: {meta['sender_role']}",
        f"  Project Phase: {meta['project_phase']}",
        f"  4D Task Status: {meta.get('4d_task_status', 'N/A')}",
        "",
        "[CHAT HISTORY]"
    ]

    for msg in context["chat_history"]:
        input_parts.append(f"  {msg['role']}: {msg['text']}")

    input_parts.extend([
        "",
        "[USER QUERY]",
        f"  {case['query_text']}",
        "=" * 50
    ])

    formatted_input = "\n".join(input_parts)

    # Build image paths
    image_paths = []
    for img_file in case.get("image_file", []):
        img_path = Path(image_dir) / img_file
        if img_path.exists():
            image_paths.append(str(img_path))
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")

    return formatted_input, image_paths


def evaluate_response(response_text, ground_truth):
    """
    Evaluate agent response against ground truth.

    Args:
        response_text: The agent's response string
        ground_truth: Ground truth dict with target_guid, expected_reasoning, etc.

    Returns:
        dict: Evaluation results with scores and details
    """
    target_guid = ground_truth.get("target_guid", "")
    target_name = ground_truth.get("target_name", "")
    expected_reasoning = ground_truth.get("expected_reasoning", "")
    rq_category = ground_truth.get("rq_category", "")

    results = {
        "guid_match": False,
        "name_match": False,
        "target_guid": target_guid,
        "target_name": target_name,
        "rq_category": rq_category,
        "details": []
    }

    # Check if target GUID is found in response
    if target_guid and target_guid not in ["MULTIPLE", "CLARIFICATION_NEEDED", "INSUFFICIENT_DATA", "INVALID_LOCATION"]:
        if target_guid in response_text:
            results["guid_match"] = True
            results["details"].append(f"✅ Target GUID found: {target_guid}")
        else:
            results["details"].append(f"❌ Target GUID not found: {target_guid}")

    # Check if target name is mentioned
    if target_name:
        # Check partial match (element names can be truncated)
        name_parts = target_name.split(":")[0] if ":" in target_name else target_name
        if name_parts.lower() in response_text.lower():
            results["name_match"] = True
            results["details"].append(f"✅ Target name found: {name_parts}")
        else:
            results["details"].append(f"❌ Target name not found: {name_parts}")

    return results


async def main_async():
    """Main asynchronous orchestrator"""
    load_dotenv()

    # Load centralized configuration
    config = load_config()
    base_dir = Path(__file__).parent.parent

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

    # Ground truth configuration from config
    gt_config = config.get("ground_truth", {})
    gt_file = gt_config.get("file", "data/ground_truth/gt_1/gt_1.json")
    gt_image_dir = base_dir / gt_config.get("image_dir", "data/ground_truth/gt_1/imgs")

    # Output configuration from config
    output_config = config.get("output", {})
    evaluations_dir = base_dir / output_config.get("evaluations_dir", "logs/evaluations")

    # Delay between tests from config
    delay_between_tests = agent_config.get("delay_between_tests", 7)

    print("\n" + "="*70)
    print("🚀 Initializing MCP-based Agent")
    print(f"   IFC Model: {config.get('ifc', {}).get('model_path', 'N/A')}")
    print(f"   Ground Truth: {gt_file}")
    print("="*70)

    server_params = StdioServerParameters(
        command=python_exe,
        args=[ifc_server_path],
        env=None
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

            # Track evaluation results
            evaluation_results = []

            for i, case in enumerate(test_cases, 1):
                case_id = case.get("id", f"case_{i}")
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

                        # Extract tool calls
                        tool_calls = []
                        for msg in response["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_calls.append({
                                        "name": tool_call.get("name", "unknown"),
                                        "args": tool_call.get("args", {})
                                    })
                    else:
                        output = str(response)
                        tool_calls = []

                    print(output)
                    print("-" * 70)

                    logger.log_agent_message(output, tool_calls=tool_calls if tool_calls else None)

                    # Evaluate response against ground truth
                    eval_result = evaluate_response(output, case.get("ground_truth", {}))
                    eval_result["case_id"] = case_id
                    eval_result["elapsed_time"] = elapsed
                    eval_result["tool_calls"] = tool_calls
                    evaluation_results.append(eval_result)

                    # Print evaluation results
                    print(f"\n📊 Evaluation:")
                    for detail in eval_result["details"]:
                        print(f"   {detail}")

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

            print(f"   Total Test Cases: {total}")
            print(f"   GUID Matches: {guid_matches}/{total} ({100*guid_matches/total:.1f}%)")
            print(f"   Name Matches: {name_matches}/{total} ({100*name_matches/total:.1f}%)")

            # Group by RQ category
            rq_stats = {}
            for r in evaluation_results:
                rq = r.get("rq_category", "Unknown")
                if rq not in rq_stats:
                    rq_stats[rq] = {"total": 0, "guid_match": 0}
                rq_stats[rq]["total"] += 1
                if r.get("guid_match", False):
                    rq_stats[rq]["guid_match"] += 1

            print(f"\n   Results by RQ Category:")
            for rq, stats in sorted(rq_stats.items()):
                pct = 100 * stats["guid_match"] / stats["total"] if stats["total"] > 0 else 0
                print(f"      {rq}: {stats['guid_match']}/{stats['total']} ({pct:.1f}%)")

            # Save evaluation results to JSON
            evaluations_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = evaluations_dir / f"eval_{timestamp}.json"

            with open(results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": timestamp,
                    "ground_truth_file": gt_file,
                    "summary": {
                        "total": total,
                        "guid_matches": guid_matches,
                        "name_matches": name_matches,
                        "by_rq_category": rq_stats
                    },
                    "results": evaluation_results
                }, f, indent=2)

            print(f"\n   Results saved to: {results_file}")

            # Save conversation summary
            logger.save_summary(f"Completed {len(test_cases)} ground truth test cases using MCP architecture")
            print(f"\n📊 Session complete. Check logs/ directory for conversation history.")


def main():
    """Synchronous entry point"""
    if not MCP_AVAILABLE:
        print("❌ MCP dependencies not available. Install with: pip install -r requirements.txt")
        return

    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
