import asyncio
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from ai_agent_test.app.qwen.common import llm, file_tools


async def create_amap_mcp_client():
    mcp_config = {
        "amap-maps": {
            "url": "https://mcp.amap.com/sse?key=",
            "transport": "sse"
        }
    }

    client = MultiServerMCPClient(mcp_config)
    tools = await client.get_tools()
    return client, tools


async def create_and_run_agent():
    client, tools = await create_amap_mcp_client()

    system_prompt = "你是一个智能助手，可以调用高德 MCP 工具帮助用户规划出行路线、查询地点坐标等。"

    agent = create_agent(
        model=llm,
        tools=tools + file_tools,
        system_prompt=system_prompt,
    )

    user_question = """
    目标：
    - 明天上午10点我要从北京四惠地铁站到颐和园
    - 线路选择： 公交地铁或打车
    - 考虑出行时间和路线，以及天气状况和穿衣建议
    - 输出一个 HTML 页面到：/Users/wxy/ai-agent-test/.temp 目录下
    要求：
    - 制作网页来展示出线路和位置
    - 网页使用简约美观的页面风格，以及卡片展示
    - 行程规划的结果要能够在高德app中展示，并集成到h5页面中
    """

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": user_question}]}
    )

    # 打印完整消息链，方便调试每一步工具调用
    for msg in result["messages"]:
        role = getattr(msg, "type", msg.__class__.__name__)
        content = getattr(msg, "content", "")
        print(f"[{role}] {content}")

    # 打印最终回答（最后一条 AI 消息）
    final_answer = result["messages"][-1].content
    print("\n=== 最终回答 ===")
    print(final_answer)

    return result

# asyncio.run(create_amap_mcp_client())
asyncio.run(create_and_run_agent())