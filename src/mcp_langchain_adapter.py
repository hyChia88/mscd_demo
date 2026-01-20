"""
MCP to LangChain Tool Adapter

Converts MCP tools to LangChain-compatible tools when langchain-mcp-adapters
is not available.

This is a simple implementation that bridges MCP tool definitions to LangChain's
tool calling interface.
"""

from typing import Any, Callable, Dict, List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, create_model


def create_langchain_tool_from_mcp(mcp_tool, mcp_session):
    """
    Convert an MCP tool to a LangChain StructuredTool.

    Args:
        mcp_tool: MCP tool object with name, description, and inputSchema
        mcp_session: Active MCP client session for calling the tool

    Returns:
        StructuredTool: LangChain-compatible tool
    """

    # Create async function that calls the MCP tool
    async def mcp_tool_func(**kwargs) -> str:
        """Execute MCP tool via the session"""
        try:
            result = await mcp_session.call_tool(mcp_tool.name, arguments=kwargs)

            # Extract result content
            if hasattr(result, 'content') and result.content:
                # Handle list of content items
                if isinstance(result.content, list) and len(result.content) > 0:
                    return str(result.content[0].text)
                return str(result.content)

            return str(result)

        except Exception as e:
            return f"Error calling MCP tool {mcp_tool.name}: {str(e)}"

    # Create Pydantic model from inputSchema if available
    if hasattr(mcp_tool, 'inputSchema') and mcp_tool.inputSchema:
        schema = mcp_tool.inputSchema

        # Extract properties for Pydantic model
        properties = schema.get('properties', {})
        required = schema.get('required', [])

        # Build field definitions
        fields = {}
        for prop_name, prop_schema in properties.items():
            prop_type = str  # Default to string
            prop_default = ... if prop_name in required else None

            # Map JSON schema types to Python types
            json_type = prop_schema.get('type', 'string')
            if json_type == 'integer':
                prop_type = int
            elif json_type == 'number':
                prop_type = float
            elif json_type == 'boolean':
                prop_type = bool
            elif json_type == 'array':
                prop_type = list
            elif json_type == 'object':
                prop_type = dict

            description = prop_schema.get('description', '')
            fields[prop_name] = (prop_type, prop_default)

        # Create Pydantic model dynamically
        ArgsModel = create_model(
            f"{mcp_tool.name}_args",
            **fields
        )
    else:
        # No schema, create empty model
        ArgsModel = create_model(f"{mcp_tool.name}_args")

    # Create LangChain StructuredTool
    return StructuredTool(
        name=mcp_tool.name,
        description=mcp_tool.description or f"MCP tool: {mcp_tool.name}",
        coroutine=mcp_tool_func,
        args_schema=ArgsModel
    )


def convert_mcp_to_langchain_tools(mcp_tools: List[Any], mcp_session: Any) -> List[StructuredTool]:
    """
    Convert a list of MCP tools to LangChain tools.

    Args:
        mcp_tools: List of MCP tool objects
        mcp_session: Active MCP client session

    Returns:
        List of LangChain StructuredTool objects
    """
    langchain_tools = []

    for mcp_tool in mcp_tools:
        try:
            lc_tool = create_langchain_tool_from_mcp(mcp_tool, mcp_session)
            langchain_tools.append(lc_tool)
        except Exception as e:
            print(f"⚠️  Warning: Failed to convert MCP tool {mcp_tool.name}: {e}")
            continue

    return langchain_tools
