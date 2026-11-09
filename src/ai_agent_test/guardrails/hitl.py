"""人工审批（human-in-the-loop）的驱动循环。

护栏在工具里调用 ``interrupt()`` 后，图会挂起并把状态写进 checkpoint。
这里负责另一半：读出待审批项 → 交给"回答者"→ 用 ``Command(resume=...)`` 恢复，
直到图不再挂起。

"回答者"是一个异步函数 ``ask(request) -> bool``，与界面无关：
终端里是 y/N 提示（``terminal_ask``），Web 前端可以是一次 HTTP 往返。

待审批项从 ``app.aget_state(config).interrupts`` 读取，而不是从流式输出里捞：
它是 checkpoint 里的事实，进程重启后依然在，子图（Supervisor 下的专家 agent）
里发起的 interrupt 也会汇总到顶层。同一步里并行的多个危险调用各有一个 id，
用 ``{id: 答案}`` 一次性恢复。
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from langgraph.types import Command, Interrupt

from ai_agent_test.guardrails.approval import APPROVAL_KIND

AskFn = Callable[[dict[str, Any]], Awaitable[Any]]


def is_approval(item: Interrupt) -> bool:
    return isinstance(item.value, dict) and item.value.get("kind") == APPROVAL_KIND


async def pending_approvals(app: Any, config: dict) -> list[Interrupt]:
    """当前会话里等待人工审批的请求（含子图）。"""
    state = await app.aget_state(config)
    return [item for item in state.interrupts if is_approval(item)]


async def resolve(app: Any, config: dict, ask: AskFn) -> Command | None:
    """逐条询问待审批项，返回用于恢复的 Command；没有待审批项时返回 None。"""
    pending = await pending_approvals(app, config)
    if not pending:
        return None
    answers = {}
    for item in pending:
        answers[item.id] = await ask(item.value)
    return Command(resume=answers)


async def drive(
    app: Any,
    inputs: Any,
    config: dict,
    ask: AskFn,
    **stream_kwargs: Any,
) -> AsyncIterator[Any]:
    """运行图直到结束，中途遇到审批就问 ``ask``，然后自动恢复。

    ``inputs`` 为 None 表示不开新一轮，只处理上次遗留的待审批项（进程重启后的恢复）。
    产出的内容与 ``app.astream(...)`` 相同，调用方照常打印即可。

    开新一轮之前如果还有遗留的待审批项（上一轮审批时超时或按了 Ctrl-C），
    先把它们处理完：否则会话里会留下一条没有结果的 tool_call，
    OpenAI 兼容接口会直接返回 400。
    """
    if inputs is not None and await pending_approvals(app, config):
        async for item in drive(app, None, config, ask, **stream_kwargs):
            yield item
    payload = inputs if inputs is not None else await resolve(app, config, ask)
    while payload is not None:
        async for item in app.astream(payload, config, **stream_kwargs):
            yield item
        payload = await resolve(app, config, ask)


async def terminal_ask(request: dict[str, Any]) -> bool:
    """终端里的回答者：打印审批说明，读 y/N。"""
    banner = "\n" + "─" * 60 + f"\n⚠️  需要确认（{request.get('level')}）\n" + str(request.get("prompt", "")) + "\n"
    # 打印到 stderr，避免污染标准输出里模型的消息流
    print(banner, file=sys.stderr, flush=True)
    answer = await asyncio.to_thread(input, "   执行吗？[y/N] ")
    return answer.strip().lower() in {"y", "yes"}
