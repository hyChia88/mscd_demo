"""
Shared MCP connection helper.

Consolidates the repeated MCP client-side connection boilerplate from:
- main_mcp.py
- chat_cli.py

Usage::

    from common.mcp import mcp_session

    async with mcp_session(base_dir, env=server_env) as ctx:
        agent = create_react_agent(llm, ctx.tools)
        ...
"""

import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Adapter import: official package → custom fallback
try:
    from langchain_mcp_adapters import convert_mcp_to_langchain_tools
except ImportError:
    from mcp_langchain_adapter import convert_mcp_to_langchain_tools


@dataclass
class MCPContext:
    """Container returned by mcp_session()."""
    session: Any
    tools: List[Any]
    tool_by_name: Dict[str, Any]
    tools_result: Any


@asynccontextmanager
async def mcp_session(
    base_dir: Path,
    env: Optional[Dict[str, str]] = None,
    server_script: str = "mcp_servers/ifc_server.py",
):
    """
    Async context manager that connects to the IFC MCP server,
    converts tools to LangChain format, and yields an MCPContext.

    Args:
        base_dir: Project root directory.
        env: Optional environment dict passed to the subprocess.
        server_script: Relative path (from base_dir) to the MCP server script.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(base_dir / server_script)],
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            if not tools_result or not tools_result.tools:
                raise RuntimeError("No MCP tools found on server")

            langchain_tools = convert_mcp_to_langchain_tools(
                tools_result.tools, session
            )
            tool_by_name = {t.name: t for t in langchain_tools}

            yield MCPContext(
                session=session,
                tools=langchain_tools,
                tool_by_name=tool_by_name,
                tools_result=tools_result,
            )
