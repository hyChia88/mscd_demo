"""
Interactive CLI Chat for BIM Inspection Agent (MCP-based)

Uses the same MCP architecture as production for realistic testing.

Usage:
    python src/chat_cli.py                    # Normal interactive mode
    python src/chat_cli.py --scenario GT_007  # Pre-load a test scenario

Commands:
    - Type your question and press Enter
    - 'quit' or 'q' - Exit
    - 'clear' - Clear conversation history
    - 'tools' - List available MCP tools
    - 'scenarios' - List available test scenarios
    - 'load <id>' - Load scenario context (e.g., 'load GT_007')
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

# Try official adapter first, fall back to custom
try:
    from langchain_mcp_adapters import convert_mcp_to_langchain_tools
    print("[MCP] Using official langchain-mcp-adapters", file=sys.stderr)
except ImportError:
    from mcp_langchain_adapter import convert_mcp_to_langchain_tools
    print("[MCP] Using custom MCP-LangChain adapter", file=sys.stderr)


def load_config():
    """Load configuration from config.yaml"""
    config_path = BASE_DIR / "config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {"llm": {"model": "gemini-2.5-flash", "temperature": 0}}


def load_system_prompt(prompt_file="prompts/system_prompt.yaml"):
    """Load system prompt from YAML file"""
    prompt_path = BASE_DIR / prompt_file
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("system_prompt", "")


def load_ground_truth():
    """Load ground truth scenarios for context simulation"""
    config = load_config()
    gt_file = config.get("ground_truth", {}).get("file", "data/ground_truth/gt_1/gt_1.json")
    gt_path = BASE_DIR / gt_file
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def format_scenario_context(scenario: dict) -> str:
    """Format a ground truth scenario into context string (simulates real 4D task)"""
    context = scenario.get("context_payload", {})
    meta = context.get("meta", {})

    parts = [
        "=" * 50,
        "[CONTEXT]",
        f"  Timestamp: {meta.get('timestamp', 'N/A')}",
        f"  Sender Role: {meta.get('sender_role', 'N/A')}",
        f"  Project Phase: {meta.get('project_phase', 'N/A')}",
        f"  4D Task Status: {meta.get('4d_task_status', 'N/A')}",
        "",
        "[CHAT HISTORY]"
    ]
    for msg in context.get("chat_history", []):
        parts.append(f"  {msg['role']}: {msg['text']}")
    parts.extend(["", "[USER QUERY]", f"  {scenario.get('query_text', '')}", "=" * 50])
    return "\n".join(parts)


async def main_async(args):
    """Main async interactive session with MCP tools"""

    config = load_config()
    llm_config = config.get("llm", {})

    print("\n" + "=" * 70)
    print("  BIM Inspection Agent - Interactive MCP Chat")
    print("=" * 70)
    print("\nInitializing MCP connection...")

    # MCP server configuration
    python_exe = sys.executable
    ifc_server_path = str(BASE_DIR / "mcp_servers" / "ifc_server.py")

    server_params = StdioServerParameters(
        command=python_exe,
        args=[ifc_server_path],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Get MCP tools
            tools_result = await session.list_tools()
            if not tools_result or not tools_result.tools:
                print("No MCP tools found. Exiting.")
                return

            langchain_tools = convert_mcp_to_langchain_tools(tools_result.tools, session)

            print(f"Connected to MCP server with {len(langchain_tools)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}")

            # Setup LLM
            llm = ChatGoogleGenerativeAI(
                model=llm_config.get("model", "gemini-2.5-flash"),
                temperature=llm_config.get("temperature", 0),
                max_retries=llm_config.get("max_retries", 2),
            )

            # Load system prompt
            try:
                system_prompt = load_system_prompt()
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                system_prompt = "You are a helpful BIM inspection assistant."

            # Create agent
            agent_executor = create_react_agent(
                llm.bind(system=system_prompt),
                langchain_tools
            )

            # Load ground truth for scenario simulation
            ground_truth = load_ground_truth()
            gt_by_id = {gt.get("id", ""): gt for gt in ground_truth}

            print("\nAgent ready!")
            print("\nCommands:")
            print("  'quit'/'q'    - Exit")
            print("  'clear'       - Clear conversation")
            print("  'tools'       - List available tools")
            print("  'scenarios'   - List test scenarios")
            print("  'load <id>'   - Load scenario (e.g., 'load GT_007')")
            print("\n" + "-" * 70)

            messages = []
            current_context = None

            # Load initial scenario if specified via CLI arg
            if args.scenario and args.scenario in gt_by_id:
                scenario = gt_by_id[args.scenario]
                current_context = format_scenario_context(scenario)
                print(f"\nLoaded scenario: {args.scenario}")
                print(current_context)
                print("-" * 70)

            while True:
                try:
                    user_input = input("\nYou: ").strip()

                    if not user_input:
                        continue

                    # Commands
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("\nGoodbye!")
                        break

                    if user_input.lower() == 'clear':
                        messages = []
                        current_context = None
                        print("\nConversation cleared.")
                        continue

                    if user_input.lower() == 'tools':
                        print("\nAvailable MCP tools:")
                        for tool in tools_result.tools:
                            desc = (tool.description or "")[:60]
                            print(f"  - {tool.name}: {desc}...")
                        continue

                    if user_input.lower() == 'scenarios':
                        print("\nAvailable test scenarios:")
                        for gt in ground_truth[:10]:
                            gt_id = gt.get("id", "Unknown")
                            query = gt.get("query_text", "")[:50]
                            rq = gt.get("ground_truth", {}).get("rq_category", "")
                            print(f"  - {gt_id} [{rq}]: {query}...")
                        if len(ground_truth) > 10:
                            print(f"  ... and {len(ground_truth) - 10} more")
                        continue

                    if user_input.lower().startswith('load '):
                        scenario_id = user_input[5:].strip()
                        if scenario_id in gt_by_id:
                            scenario = gt_by_id[scenario_id]
                            current_context = format_scenario_context(scenario)
                            messages = []
                            print(f"\nLoaded scenario: {scenario_id}")
                            print(current_context)
                            print(f"\nTarget GUID: {scenario.get('ground_truth', {}).get('target_guid', 'N/A')}")
                            print("Type your query or 'send' to use original query.")
                        else:
                            print(f"\nScenario '{scenario_id}' not found. Use 'scenarios' to list.")
                        continue

                    if user_input.lower() == 'send' and current_context:
                        for gt in ground_truth:
                            if format_scenario_context(gt) == current_context:
                                user_input = gt.get("query_text", "")
                                print(f"Sending: {user_input}")
                                break

                    # Build full input with context
                    if current_context and not messages:
                        full_input = f"{current_context}\n\n[USER QUERY]\n{user_input}"
                    else:
                        full_input = user_input

                    messages.append(("user", full_input))

                    print("\nAgent: ", end="", flush=True)

                    response = await agent_executor.ainvoke({"messages": messages})

                    if "messages" in response:
                        # Show tool calls
                        for msg in response["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"\n  [Tool: {tc.get('name')}]")

                        agent_message = response["messages"][-1].content
                        messages = response["messages"]
                    else:
                        agent_message = str(response)

                    print(agent_message)
                    print("\n" + "-" * 70)

                except KeyboardInterrupt:
                    print("\n\nInterrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    import traceback
                    traceback.print_exc()
                    print("Please try again.\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive BIM Agent CLI (MCP-based)")
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        help="Pre-load a ground truth scenario (e.g., GT_007)"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
