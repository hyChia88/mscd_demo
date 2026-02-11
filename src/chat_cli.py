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

import asyncio
import argparse
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent

from common.config import load_config, load_system_prompt, load_ground_truth, get_base_dir, init_llm
from common.evaluation import format_context_string
from common.mcp import mcp_session

load_dotenv()


def format_scenario_context(scenario: dict) -> str:
    """Format a ground truth scenario into context string (simulates real 4D task)."""
    context = scenario.get("context_payload", {})
    return format_context_string(
        meta=context.get("meta", {}),
        chat_history=context.get("chat_history", []),
        query_text=scenario.get("query_text", ""),
    )


async def main_async(args):
    """Main async interactive session with MCP tools"""

    config = load_config()

    print("\n" + "=" * 70)
    print("  BIM Inspection Agent - Interactive MCP Chat")
    print("=" * 70)
    print("\nInitializing MCP connection...")

    async with mcp_session(get_base_dir()) as ctx:
            langchain_tools = ctx.tools
            tools_result = ctx.tools_result

            print(f"Connected to MCP server with {len(langchain_tools)} tools:")
            for tool in tools_result.tools:
                print(f"  - {tool.name}")

            # Setup LLM
            llm = init_llm(config)

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
            gt_file = config.get("ground_truth", {}).get("file", "data/ground_truth/gt_1/gt_1.json")
            ground_truth = load_ground_truth(gt_file)
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
                        tool_names = []
                        for msg in response["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_names.append(tc.get('name'))
                                    print(f"\n  [Tool: {tc.get('name')}]")

                        agent_message = response["messages"][-1].content
                        messages = response["messages"]
                    else:
                        agent_message = str(response)
                        tool_names = []

                    # Handle empty responses - try to extract tool results
                    if not agent_message or not agent_message.strip():
                        # Look for ToolMessage results in the response
                        for msg in reversed(response["messages"]):
                            msg_type = getattr(msg, 'type', None) or msg.__class__.__name__
                            if msg_type in ('tool', 'ToolMessage') and getattr(msg, 'content', None):
                                agent_message = msg.content
                                break

                    # Final fallback if still empty
                    if not agent_message or not agent_message.strip():
                        agent_message = f"[Empty response] Tools called: {tool_names if tool_names else 'None'}. Try rephrasing your query."
                        print("\n⚠️  Agent returned empty response")

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
