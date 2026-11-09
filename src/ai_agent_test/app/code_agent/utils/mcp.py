"""MCP stdio 客户端封装。

所有 tools/*.py 都只是薄薄一层转发：真正的工具实现在 app/code_agent/mcp/
或 rag/ 下的 server 脚本里，通过 stdio 协议暴露，由这里统一拉起。
"""

from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_agent_test.config import mcp_python, mcp_script


async def create_mcp_stdio_client(name: str, params: dict):
    config = {name: {"transport": "stdio", **params}}
    client = MultiServerMCPClient(config)
    tools = await client.get_tools()
    return client, tools


async def get_stdio_tools(server_name: str, script: str):
    """按脚本路径加载一个 MCP server 暴露的全部工具。

    Args:
        server_name: 客户端内的逻辑名，仅用于日志区分。
        script: 相对 app/code_agent/ 的脚本路径，如 "mcp/mysql_tools.py"。
    """
    params = {"command": mcp_python(), "args": [str(mcp_script(script))]}
    _, tools = await create_mcp_stdio_client(server_name, params)
    return tools
