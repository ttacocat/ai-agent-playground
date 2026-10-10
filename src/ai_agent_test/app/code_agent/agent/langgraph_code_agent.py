import asyncio

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph_supervisor import create_supervisor

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.browser_tools import get_stdio_browser_tools
from ai_agent_test.app.code_agent.tools.file_tools import file_tools
from ai_agent_test.app.code_agent.tools.shell_tools import get_stdio_shell_tools
from ai_agent_test.app.code_agent.tools.terminal_tools import get_stdio_terminal_tools
from ai_agent_test.app.code_agent.tools.vm import get_stdio_vm_tools

REDIS_URL = "redis://localhost:6379"
TASK_TIMEOUT = 6000  # 单次任务最长等待秒数
RECURSION_LIMIT = 60  # 防止 agent 无限循环调用工具

RESEARCH_PROMPT = (
    "你是一个技术方案设计专家，专门负责设计技术方案。\n"
    "要求：先阅读相关文件，再给出方案；方案包含目标、思路、涉及文件、实施步骤、风险点；"
    "不要直接编写或修改代码，实现工作交给 code_expert。"
)

CODE_PROMPT = (
    "你是一个代码实现专家，负责根据技术方案编写、修改并验证代码。\n"
    "要求：严格按照 research_expert 的方案实现；修改后运行必要命令验证；"
    "完成后简要说明改动了哪些文件。"
)

SUPERVISOR_PROMPT = (
    "你是团队主管，管理两个专家：\n"
    "- research_expert：负责技术方案设计，不写代码；\n"
    "- code_expert：负责代码实现与验证。\n"
    "流程：先让 research_expert 出方案，再让 code_expert 按方案实现。\n"
    "每次只分配给一个专家，不要自己做具体工作。"
)


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


async def run_agent(thread_id: str = "default-thread"):
    async with AsyncRedisSaver.from_conn_string(REDIS_URL) as memory:
        await memory.asetup()

        print("正在加载工具...")
        # shell_tools = await get_stdio_shell_tools()
        terminal_tools = await get_stdio_terminal_tools()
        browser_tools = await get_stdio_browser_tools()
        vm_tools = await get_stdio_vm_tools()
        tools = [*file_tools, *terminal_tools, *browser_tools, *vm_tools]
        print(f"已加载 {len(tools)} 个工具")

        research_agent = create_agent(
            model=qwen_llm,
            tools=tools,
            name="research_expert",
            system_prompt=RESEARCH_PROMPT,
        )
        code_agent = create_agent(
            model=qwen_llm,
            tools=tools,
            name="code_expert",
            system_prompt=CODE_PROMPT,
        )

        workflow = create_supervisor(
            agents=[research_agent, code_agent],
            model=qwen_llm,
            prompt=SUPERVISOR_PROMPT,
            output_mode="last_message",
        )
        app = workflow.compile(checkpointer=memory)

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        }

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
                    async for namespace, update in app.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="updates",
                        subgraphs=True,  # 同时输出子 agent 内部的每一步
                    ):
                        print_update(namespace, update)
            except TimeoutError:
                print(f"\n[超时] 任务超过 {TASK_TIMEOUT}s，已中断")
            except KeyboardInterrupt:
                print("\n[已中断当前任务]")
            except Exception as e:
                print(f"\n[错误] {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(run_agent(thread_id="user4"))