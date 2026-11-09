import asyncio
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.mysql_tools import get_stdio_mysql_tools
from ai_agent_test.config import settings
from ai_agent_test.guardrails import apply_guard
from ai_agent_test.observability import build_context_middleware, build_tracer


async def run_agent():
    async with AsyncRedisSaver.from_conn_string(settings.checkpoint.redis_url) as memory:
        await memory.asetup()

        # 只加载实际用到的工具：每个 MCP server 都是一个子进程，
        # 加载了不用等于白白拖慢启动
        mysql_tools = await get_stdio_mysql_tools()
        # 危险操作在执行前过一道人工审批（destructive 才拦），并落审计日志
        tools = apply_guard(mysql_tools)
        #方案二：提供一个rag工具，让智能体通过工具查询知识
        name = "bot"
        system_prompt = f"""
        # 角色
        你是一名优秀的工程师，你的名字叫做{name}
        """

        agent = create_agent(
            model=qwen_llm,
            tools=tools,
            checkpointer=memory,
            debug=False,
            system_prompt=system_prompt,
            middleware=build_context_middleware(),
        )

        config = RunnableConfig(configurable={"thread_id": "11"})
        tracer = build_tracer("chat-single")
        if tracer:
            config["callbacks"] = [tracer]

        step = 0  # 一轮对话
        print("输入 exit / quit 退出对话\n")

        while True:
            user_input = input("user: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("已退出。")
                break

            step += 1
            print(f"\n{'='*60}")
            print(f"📊 第 {step} 轮对话:")
            print("-" * 60)

            # final_reply = None
            start_time = time.perf_counter()  # 本轮计时起点

            # 注：这里曾有一个拼好的 prompt 要求"先查 query_rag 再执行任务"，
            # 但当前模式只加载了 mysql 工具、没有 rag 工具，那个提示词会诱导
            # 模型去调用不存在的工具。要启用知识库前置，请改用 --mode supervisor。
            try:
                async for chunk in agent.astream(
                    {"messages": [("user", user_input)]},
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in chunk.items():
                        if "messages" not in node_output:
                            continue

                        for msg in node_output["messages"]:
                            if node_name == "tools":
                                print("🔄 【工具调用】工具执行结果")
                                print("-" * 60)
                                print(f"{msg.name}: {msg.content}")
                                print("-" * 60)

                            elif node_name == "model" and isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    print("🔧 【工具调用】")
                                    print("-" * 60)
                                    for tc in msg.tool_calls:
                                        print(f"{tc['name']}: {tc['args']}")
                                    print("-" * 60)
                                elif msg.content:
                                    print("💭 【AI思考】")
                                    print("-" * 60)
                                    print(msg.content)
                                    print("-" * 60)
                                    # final_reply = msg.content

                elapsed = time.perf_counter() - start_time
                print("✅ 状态: 执行完成，可以开始下一个任务")
                print(f"⏱️ 执行时间: {elapsed:.2f}秒")
                print("-" * 60)

                # if final_reply:
                #     print(f"\nAgent: {final_reply}\n")

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                print("❌ 状态: 执行出错")
                print(f"⏱️ 执行时间: {elapsed:.2f}秒")
                print(f"调用出错: {e}\n")

        if tracer:
            path = tracer.save()
            if path:
                print(f"trace 已写入：{path}\n查看方式：uv run agent trace")


if __name__ == "__main__":
    asyncio.run(run_agent())
