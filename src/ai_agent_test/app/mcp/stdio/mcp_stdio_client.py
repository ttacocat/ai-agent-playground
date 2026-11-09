import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 用当前解释器 + 本文件同级目录定位 server，不写死任何绝对路径
server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(Path(__file__).with_name("mcp_stdio_server.py"))],
)


async def main():
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # 把 MCP 工具"翻译"成 LangChain 工具对象列表
        tools = await load_mcp_tools(session)

        # 每个 tool 都是标准的 LangChain StructuredTool，
        # 有 name / description / args_schema
        add_tool = next(t for t in tools if t.name == "add")
        result = await add_tool.ainvoke({"a": 3, "b": 5})
        print(result)


asyncio.run(main())
