from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import StdioServerParameters, stdio_client, ClientSession
import asyncio
from langchain.agents import create_agent

from ai_agent_test.app.qwen.common import llm


async def mcp_playwright_client():
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@executeautomation/playwright-mcp-server"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 获取 MCP Tools
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_agent(model=llm, tools=tools, debug=True)
            response = await agent.ainvoke(
                input={"messages": [("user", "在bing中查询北京今天的天气")]}
            )
            messages = response["messages"]

            # 4. 遍历每条消息，根据类型打印不同内容
            for message in messages:
                # 4.1 处理用户发送的原始消息（HumanMessage）
                if isinstance(message, HumanMessage):
                    print(f"用户: {message.content}")

                # 4.2 处理 AI 的回复消息（AIMessage）
                elif isinstance(message, AIMessage):
                    # 4.2.1 如果 AI 直接回复了文本内容，打印出来
                    if message.content:
                        print("助理:", message.content)
                    else:
                        # 4.2.2 如果 AI 没有文本内容，则可能调用了工具
                        # 注意：AIMessage 可能有 tool_calls 属性（LangChain 的新格式）
                        if hasattr(message, "tool_calls") and message.tool_calls:
                            for tool_call in message.tool_calls:
                                # tool_call 是一个 dict，包含 name, args, id 等
                                print(f"助理[调用工具]: {tool_call['name']} 参数: {tool_call['args']}")
                        # 如果既无内容又无工具调用，可能是空消息，跳过或打印提示
                        else:
                            print("助理: (无操作)")

                # 4.3 处理工具执行后的返回消息（ToolMessage）
                elif isinstance(message, ToolMessage):
                    # 可以打印工具执行结果，通常包含 tool_call_id 和 content
                    print(f"工具结果: {message.content[:100]}...")  # 截断过长内容

                # 4.4 其他类型的消息（比如 SystemMessage）可根据需要处理
                else:
                    print(f"其他消息类型: {type(message).__name__}")

asyncio.run(mcp_playwright_client())