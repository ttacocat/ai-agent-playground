"""上下文预算：把消息裁进预算，而不是等它撑爆。

两个坑必须避开：

1. **不能拆散 tool_call 配对**。`AIMessage(tool_calls=[...])` 后面必须跟着对应的
   `ToolMessage`，否则 OpenAI 兼容接口直接 400。所以保留区间要向前补齐发起调用的
   AIMessage，并向前级联它的全部结果。
2. **裁的是模型看到的，不是会话状态**。所以用 `wrap_model_call` 改写请求，
   checkpoint 里的完整历史一条不少——回放和审计都还在。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

OMIT_NOTICE = "（为控制上下文长度，已省略 {count} 条较早的消息；当前任务目标见下方最后一条用户消息）"
TRUNCATE_HEAD = 800
TRUNCATE_TAIL = 300


def _char_len(message: BaseMessage) -> int:
    return len(str(message.content or ""))


def total_chars(messages: Sequence[BaseMessage]) -> int:
    return sum(_char_len(m) for m in messages)


def truncate_tool_output(message: BaseMessage, limit: int = TRUNCATE_HEAD + TRUNCATE_TAIL) -> BaseMessage:
    """把超长工具结果压成 head + tail，中间留省略标记。

    只压工具结果：它是最容易炸的那一类（网页正文、SQL 结果集、命令输出）。
    """
    if not isinstance(message, ToolMessage):
        return message
    text = str(message.content or "")
    if len(text) <= limit:
        return message
    head, tail = text[:TRUNCATE_HEAD], text[-TRUNCATE_TAIL:]
    omitted = len(text) - TRUNCATE_HEAD - TRUNCATE_TAIL
    return message.model_copy(
        update={"content": f"{head}\n…（省略 {omitted} 字符）…\n{tail}"}
    )


def trim_messages(
    messages: Sequence[BaseMessage],
    max_chars: int = 24_000,
    keep_last: int = 12,
    tool_output_limit: int = TRUNCATE_HEAD + TRUNCATE_TAIL,
) -> list[BaseMessage]:
    """超预算时保留：全部 system + 第一条 human + 最近 keep_last 条。"""
    messages = [truncate_tool_output(m, tool_output_limit) for m in messages]
    if total_chars(messages) <= max_chars or len(messages) <= keep_last:
        return list(messages)

    keep: set[int] = set()
    for index, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            keep.add(index)
    first_human = next((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), None)
    if first_human is not None:
        keep.add(first_human)
    keep.update(range(len(messages) - keep_last, len(messages)))
    keep = {i for i in keep if i >= 0}

    # 向前补齐：被保留的 ToolMessage 必须有发起调用的 AIMessage
    changed = True
    while changed:
        changed = False
        for index in sorted(keep):
            if isinstance(messages[index], ToolMessage) and index - 1 >= 0 and index - 1 not in keep:
                keep.add(index - 1)
                changed = True

    kept = [messages[i] for i in sorted(keep)]
    omitted = len(messages) - len(kept)
    if omitted <= 0:
        return kept

    # 提示插在 system 之后、正文之前，避免打断 system 与首条消息的关联
    insert_at = sum(1 for m in kept if isinstance(m, SystemMessage))
    notice = SystemMessage(content=OMIT_NOTICE.format(count=omitted))
    return kept[:insert_at] + [notice] + kept[insert_at:]


class ContextBudgetMiddleware(AgentMiddleware):
    """在模型调用前裁剪，会话状态本身不动。"""

    def __init__(self, max_chars: int = 24_000, keep_last: int = 12) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.keep_last = keep_last

    def _trimmed_request(self, request: Any) -> Any:
        messages = request.messages
        trimmed = trim_messages(messages, self.max_chars, self.keep_last)
        if len(trimmed) == len(messages) and all(a is b for a, b in zip(trimmed, messages, strict=False)):
            return request
        return request.override(messages=trimmed)

    def wrap_model_call(self, request: ModelRequest, handler):  # type: ignore[override]
        return handler(self._trimmed_request(request))

    async def awrap_model_call(self, request: ModelRequest, handler):  # type: ignore[override]
        return await handler(self._trimmed_request(request))


def build_context_middleware() -> list[Any]:
    """按配置装配；CONTEXT_MAX_CHARS<=0 表示关闭。"""
    from ai_agent_test.config import settings

    cfg = settings.context
    if cfg.max_chars <= 0:
        return []
    return [ContextBudgetMiddleware(max_chars=cfg.max_chars, keep_last=cfg.keep_last)]


__all__ = [
    "ContextBudgetMiddleware",
    "build_context_middleware",
    "total_chars",
    "trim_messages",
    "truncate_tool_output",
]
