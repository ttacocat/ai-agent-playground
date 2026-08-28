# import asyncio
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
#
# params = StdioServerParameters(command="python", args=["/Users/wxy/ai-agent-test/src/ai_agent_test/app/mcp/stdio/mcp_stdio_server.py"])
#
# async def main():
#     async with stdio_client(params) as (r, w):
#         async with ClientSession(r, w) as session:
#             await session.initialize()
#             result = await session.call_tool("add", {"a": 1, "b": 2})
#             print(result.content[0].text)
#
# asyncio.run(main())
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

server_params = StdioServerParameters(command="python", args=["/Users/wxy/ai-agent-test/src/ai_agent_test/app/mcp/stdio/mcp_stdio_server.py"])


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 把 MCP 工具"翻译"成 LangChain 工具对象列表
            tools = await load_mcp_tools(session)

            # 每个 tool 都是标准的 LangChain StructuredTool，
            # 有 name / description / args_schema
            add_tool = next(t for t in tools if t.name == "add")
            result = await add_tool.ainvoke({"a": 3, "b": 5})
            print(result)


asyncio.run(main())