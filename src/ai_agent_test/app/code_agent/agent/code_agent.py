import asyncio
import time

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.file_tools import file_tools
from ai_agent_test.app.code_agent.tools.rag_self_tools import get_stdio_rag_self_tools
from ai_agent_test.app.code_agent.tools.terminal_tools import get_stdio_terminal_tools

async def run_agent():
    async with AsyncRedisSaver.from_conn_string("redis://localhost:6379") as memory:
        await memory.asetup()

        # shell_tools = await get_stdio_shell_tools()
        terminal_tools = await get_stdio_terminal_tools()
        # rag_tools = await get_stdio_rag_tools()
        rag_self_tools = await get_stdio_rag_self_tools()
        tools = file_tools + terminal_tools + rag_self_tools
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
        )

        config = RunnableConfig(configurable={"thread_id": "6"})

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

            #方案一：从rag阿里云百炼知识库中读取知识，并拼接到提示词中
            # rag = query_rag_from_bailian(user_input)
            prompt = f"""
            # 要求
            执行任务之前先试用 query_rag 工具查询知识库，根据知识库中的知识执行任务
            用户问题：{user_input}
            """
            try:
                async for chunk in agent.astream(
                    {"messages": [("user", prompt)]},
                    config=config,
                    stream_mode="updates",
                ):
                    for node_name, node_output in chunk.items():
                        if "messages" not in node_output:
                            continue

                        for msg in node_output["messages"]:
                            if node_name == "tools":
                                print(f"🔄 【工具调用】工具执行结果")
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
                print(f"❌ 状态: 执行出错")
                print(f"⏱️ 执行时间: {elapsed:.2f}秒")
                print(f"调用出错: {e}\n")


if __name__ == "__main__":
    asyncio.run(run_agent())