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
import yaml
import asyncio
from pathlib import Path
from dotenv import load_dotenv

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


async def main_async():
    """Main asynchronous orchestrator"""
    load_dotenv()

    # Initialize conversation logger
    logger = ConversationLogger()

    # Setup LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_retries=2,
    )

    # Load system prompt
    try:
        system_prompt = load_system_prompt()
    except FileNotFoundError as e:
        print(f"⚠️  Warning: {e}")
        system_prompt = "You are a helpful BIM inspection assistant. Use the available tools to help users query the IFC model."

    # Define MCP server configuration (single server for simplicity)
    base_dir = Path(__file__).parent.parent
    python_exe = sys.executable
    ifc_server_path = str(base_dir / "mcp_servers" / "ifc_server.py")

    print("\n" + "="*70)
    print("🚀 Initializing MCP-based Agent")
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

            # Load and execute test scenarios
            scenarios = load_scenarios()
            print(f"✅ Agent initialized. Loaded {len(scenarios)} scenarios.\n")
            logger.log_agent_message(f"MCP-based agent initialized with {len(langchain_tools)} tools")

            for i, scenario in enumerate(scenarios, 1):
                print(f"\n{'='*70}")
                print(f"📋 Scenario {i}: {scenario['name']}")
                print(f"{'='*70}")
                print(f"Description: {scenario['description']}\n")

                user_input = scenario['input']
                print(f"📥 Input:\n{user_input}\n")

                logger.log_user_message(user_input)

                try:
                    start_time = time.time()

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

                except Exception as e:
                    error_msg = f"Error during execution: {e}"
                    print(f"\n❌ {error_msg}")
                    logger.log_agent_message(f"ERROR: {error_msg}")

                # Pause between scenarios
                if i < len(scenarios):
                    print(f"\n⏳ Proceeding to next scenario in 3 seconds...")
                    await asyncio.sleep(3)

            # Save conversation summary
            logger.save_summary(f"Completed {len(scenarios)} test scenarios using MCP architecture")
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
