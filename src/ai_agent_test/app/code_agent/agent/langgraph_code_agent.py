"""多智能体 chat 入口：supervisor（主管 LLM 分派）或 team（固定的 方案→实现→审查 流程）。

两种架构的构建在 team.py，评测用的是同一套构建函数。
"""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from ai_agent_test.app.code_agent.agent.lead import build_lead
from ai_agent_test.app.code_agent.agent.team import build_supervisor, build_team
from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.browser_tools import get_stdio_browser_tools
from ai_agent_test.app.code_agent.tools.file_tools import file_tools
from ai_agent_test.app.code_agent.tools.terminal_tools import get_stdio_terminal_tools
from ai_agent_test.app.code_agent.tools.vm import get_stdio_vm_tools
from ai_agent_test.config import settings
from ai_agent_test.guardrails import apply_guard, drive, pending_approvals, terminal_ask
from ai_agent_test.observability import build_context_middleware, build_tracer

TASK_TIMEOUT = 6000  # 单次任务最长等待秒数
RECURSION_LIMIT = 60  # 防止 agent 无限循环调用工具


async def read_input(prompt: str = "\nuser：") -> str:
    lines = []
    current_prompt = prompt
    while True:
        line = await asyncio.to_thread(input, current_prompt)
        if line.endswith("\\"):
            lines.append(line[:-1])
            current_prompt = "... "
            continue
        lines.append(line)
        return "\n".join(lines).strip()


def print_update(namespace: tuple, update: dict) -> None:
    for node, data in update.items():
        if not isinstance(data, dict):
            continue
        messages = data.get("messages") or []
        if not messages:
            continue
        where = "/".join(namespace) if namespace else "main"
        print(f"\n--- [{where}] {node} ---")
        messages[-1].pretty_print()


def explore_model():
    """EXPLORE_MODEL_NAME 配了就用它，否则返回 None（与主模型相同）。"""
    if not settings.model.explore_name:
        return None
    from ai_agent_test.app.code_agent.model.model import build_chat_model

    return build_chat_model(settings.model.explore_name)


async def load_tools():
    """chat 与 Web 服务共用的工具集：文件 + 终端 + 浏览器 + 虚拟机，全部套上护栏。"""
    terminal_tools = await get_stdio_terminal_tools()
    browser_tools = await get_stdio_browser_tools()
    vm_tools = await get_stdio_vm_tools()
    from ai_agent_test.app.code_agent.tools.test_runner import build_run_tests_tool

    run_tests = build_run_tests_tool(settings.paths.work_dir)
    return apply_guard([*file_tools, run_tests, *terminal_tools, *browser_tools, *vm_tools])


async def run_agent(thread_id: str = "default-thread", mode: str = "supervisor"):
    async with AsyncRedisSaver.from_conn_string(settings.checkpoint.redis_url) as memory:
        await memory.asetup()

        print("正在加载工具...")
        tools = await load_tools()
        print(f"已加载 {len(tools)} 个工具（护栏：{settings.guard.approver}）")

        if mode == "team":
            app = build_team(qwen_llm, tools, middleware=build_context_middleware(qwen_llm), checkpointer=memory)
        elif mode == "lead":
            app = build_lead(
                qwen_llm,
                tools,
                explore_model=explore_model(),
                middleware=build_context_middleware(qwen_llm),
                checkpointer=memory,
            )
        else:
            app = build_supervisor(qwen_llm, tools, middleware=build_context_middleware(qwen_llm), checkpointer=memory)
        print(f"架构：{mode}")

        # 两种架构的图状态结构不同，同一个会话 ID 不能混用一份 checkpoint
        checkpoint_thread = thread_id if mode == "supervisor" else f"{mode}:{thread_id}"
        config = {
            "configurable": {"thread_id": checkpoint_thread},
            "recursion_limit": RECURSION_LIMIT,
        }
        tracer = build_tracer(f"chat-{mode}")
        if tracer:
            config["callbacks"] = [tracer]

        stream_kwargs = {
            "stream_mode": "updates",
            "subgraphs": True,  # 同时输出子 agent 内部的每一步
        }

        # 审批挂起的状态在 checkpoint 里：上次退出时没批完的操作，这次接着批
        leftover = await pending_approvals(app, config)
        if leftover:
            print(f"\n会话 {thread_id} 有 {len(leftover)} 个操作在等待确认，先处理它们。")
            async for namespace, update in drive(app, None, config, terminal_ask, **stream_kwargs):
                print_update(namespace, update)

        print("就绪。输入 exit 退出；行尾加 \\ 可换行继续输入。")
        while True:
            try:
                user_input = await read_input()
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in {"exit", "quit", "q"}:
                break
            if not user_input:
                continue

            print("思考中...", flush=True)
            try:
                async with asyncio.timeout(TASK_TIMEOUT):
                    # drive 在遇到危险操作时挂起图、问人、再恢复，直到本轮结束
                    async for namespace, update in drive(
                        app,
                        {"messages": [HumanMessage(content=user_input)]},
                        config,
                        terminal_ask,
                        **stream_kwargs,
                    ):
                        print_update(namespace, update)
            except TimeoutError:
                print(f"\n[超时] 任务超过 {TASK_TIMEOUT}s，已中断")
            except KeyboardInterrupt:
                print("\n[已中断当前任务]")
            except Exception as e:
                print(f"\n[错误] {type(e).__name__}: {e}")

        if tracer:
            path = tracer.save()
            if path:
                print(f"\ntrace 已写入：{path}\n查看方式：uv run agent trace")


if __name__ == "__main__":
    asyncio.run(run_agent(thread_id="user4"))
