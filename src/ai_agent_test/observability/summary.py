"""上下文摘要压缩：超预算时把早期对话压成一份结构化摘要，而不是直接丢掉。

``ContextBudgetMiddleware`` 的做法是保留 system、第一条用户消息和最近 N 条，中间的直接扔。
长任务里这会丢掉关键结论：查到的表结构、已经改过的文件、**用户拒绝过的操作**。
最后一条尤其要命——模型忘了用户拒绝过什么，就会换个方式再试一次，
把护栏的拒绝记忆（``guardrails/memory.py``）也绕过去。

为什么不用 LangChain 自带的 ``SummarizationMiddleware``：它用 ``RemoveMessage(REMOVE_ALL)``
把会话状态里的旧消息替换成一条摘要，checkpoint 里的原始历史就没了。这里坚持
"裁剪只改发给模型的请求，不动会话状态"，所以：

1. **摘要存在独立的状态字段里**（``context_summary``），消息历史一条不删，回放和审计照常；
2. **增量滚动**：摘要记录自己覆盖到哪条消息，下次只把新超出预算的部分和旧摘要合并；
3. **回差**：一次压到预算的一半（``target_ratio``），不会每轮都触发一次总结；
4. **不拆 tool_call 配对**：切点不会落在 ToolMessage 上，否则接口直接 400；
5. **结构化**：固定小节，"被拒绝或被拦截的操作"单列，要求原样保留；
6. **失败降级**：总结调用失败就不更新摘要，由裁剪兜底，主流程不受影响。

发给模型的请求 = system + 第一条用户消息（原文）+ 摘要 + 摘要之后的全部消息，
最后再过一遍 ``trim_messages`` 作为硬上限。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent_test.observability.context import total_chars, trim_messages

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """你在为一个工程助手压缩早期对话记录，压缩结果会替代这些记录交给它继续工作。
把【已有摘要】和【新增对话】合并成一份新的摘要，按下面的小节输出，没有内容的小节写"无"：

## 用户目标与要求
## 已确认的事实
（查到的表结构、路径、配置、数值等，保留具体的名字和值）
## 已完成的操作
（做了什么、结果如何）
## 被拒绝或被拦截的操作
（必须完整保留：用户拒绝过什么、安全护栏拦下过什么。后续不得换写法、换工具或分步重试这些操作）
## 未完成事项

要求：只写对话里出现过的信息，不要推测；不要写客套话；总长度不超过 {limit} 字。

【已有摘要】
{previous}

【新增对话】
{transcript}"""

SUMMARY_HEADER = (
    "【早期对话摘要】以下是此前对话的压缩版（已压缩 {times} 次，覆盖 {count} 条消息），原始记录仍保存在会话历史中。"
)

MESSAGE_CHAR_LIMIT = 1500


class ContextSummaryState(AgentState):
    context_summary: NotRequired[dict[str, Any]]


@dataclass(frozen=True)
class CompactionPlan:
    """一次压缩要做的事：把 messages[start:cut] 并进摘要。"""

    start: int
    cut: int

    def pending(self, messages: Sequence[BaseMessage]) -> list[BaseMessage]:
        return list(messages[self.start : self.cut])


# --------------------------------------------------------------------------- #
# 纯函数：切点与视图
# --------------------------------------------------------------------------- #
def _first_human(messages: Sequence[BaseMessage]) -> int | None:
    return next((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), None)


def covered_index(messages: Sequence[BaseMessage], summary: dict | None) -> int | None:
    """摘要覆盖到的最后一条消息的下标；找不到（历史被改过）时返回 None。"""
    upto = (summary or {}).get("upto")
    if not upto:
        return None
    return next((i for i, m in enumerate(messages) if m.id == upto), None)


def _safe_cut(messages: Sequence[BaseMessage], cut: int) -> int:
    """切点落在 ToolMessage 上就往前挪，让发起调用的 AIMessage 和它的结果都留在后半段。"""
    while 0 < cut < len(messages) and isinstance(messages[cut], ToolMessage):
        cut -= 1
    return cut


def plan_compaction(
    messages: Sequence[BaseMessage],
    summary: dict | None,
    max_chars: int,
    keep_last: int,
    target_ratio: float = 0.5,
) -> CompactionPlan | None:
    """当前视图超预算时，决定把哪一段并进摘要；不需要压缩时返回 None。"""
    first = _first_human(messages)
    if first is None:
        return None
    covered = covered_index(messages, summary)
    start = (covered + 1) if covered is not None else first + 1
    summary_chars = len((summary or {}).get("text", ""))
    view_chars = total_chars([messages[first]]) + summary_chars + total_chars(messages[start:])
    if view_chars <= max_chars:
        return None

    # 从后往前累加，找到"剩下的部分不超过预算一半"的位置；但至少保留最近 keep_last 条
    budget = max_chars * target_ratio
    cut, tail = len(messages), 0
    while cut > start and tail + total_chars([messages[cut - 1]]) <= budget:
        cut -= 1
        tail += total_chars([messages[cut]])
    cut = min(cut, len(messages) - keep_last)
    cut = _safe_cut(messages, cut)
    if cut <= start:
        return None
    return CompactionPlan(start=start, cut=cut)


def build_view(messages: Sequence[BaseMessage], summary: dict | None) -> list[BaseMessage]:
    """发给模型的消息：system + 第一条用户消息 + 摘要 + 摘要之后的消息。没有摘要时原样返回。"""
    covered = covered_index(messages, summary)
    first = _first_human(messages)
    if covered is None or first is None or covered < first:
        return list(messages)
    systems = [m for m in messages[:first] if isinstance(m, SystemMessage)]
    header = SUMMARY_HEADER.format(times=summary.get("times", 1), count=summary.get("count", 0))
    note = SystemMessage(content=f"{header}\n\n{summary['text']}")
    return [*systems, messages[first], note, *messages[covered + 1 :]]


def render_transcript(messages: Sequence[BaseMessage]) -> str:
    """把要压缩的消息转成给总结模型看的纯文本。每条截断，避免一次网页正文撑爆总结请求。"""

    def clip(text: str) -> str:
        text = text.strip()
        return text if len(text) <= MESSAGE_CHAR_LIMIT else text[:MESSAGE_CHAR_LIMIT] + "…（截断）"

    lines = []
    for message in messages:
        content = clip(str(message.content or ""))
        if isinstance(message, HumanMessage):
            lines.append(f"[用户] {content}")
        elif isinstance(message, ToolMessage):
            lines.append(f"[工具 {message.name or ''} 返回] {content}")
        elif isinstance(message, AIMessage):
            if content:
                lines.append(f"[助手] {content}")
            for call in message.tool_calls or []:
                lines.append(f"[助手调用工具] {call['name']}({clip(str(call.get('args', {})))})")
        elif isinstance(message, SystemMessage):
            continue
        else:
            lines.append(f"[{message.type}] {content}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 中间件
# --------------------------------------------------------------------------- #
class ContextSummaryMiddleware(AgentMiddleware[ContextSummaryState]):
    state_schema = ContextSummaryState

    def __init__(
        self,
        model: BaseChatModel | None = None,
        max_chars: int = 24_000,
        keep_last: int = 12,
        summary_max_chars: int = 2_000,
        target_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self._model = model
        self.max_chars = max_chars
        self.keep_last = keep_last
        self.summary_max_chars = summary_max_chars
        self.target_ratio = target_ratio

    @property
    def model(self) -> BaseChatModel:
        if self._model is None:  # 惰性取默认模型：import 本模块不需要凭证
            from ai_agent_test.app.code_agent.model.model import qwen_llm

            self._model = qwen_llm
        return self._model

    # ---- 压缩 ---------------------------------------------------------- #
    def _plan(self, state: dict) -> tuple[CompactionPlan, list[BaseMessage], dict | None] | None:
        messages = state.get("messages") or []
        if any(m.id is None for m in messages):
            return None  # 没有 id 就没法记录覆盖位置，交给裁剪
        summary = state.get("context_summary")
        plan = plan_compaction(messages, summary, self.max_chars, self.keep_last, self.target_ratio)
        if plan is None:
            return None
        return plan, messages, summary

    def _prompt(self, summary: dict | None, pending: list[BaseMessage]) -> list[BaseMessage]:
        text = SUMMARY_PROMPT.format(
            limit=self.summary_max_chars,
            previous=(summary or {}).get("text") or "无",
            transcript=render_transcript(pending),
        )
        return [HumanMessage(content=text)]

    def _update(self, summary: dict | None, plan: CompactionPlan, messages, text: str) -> dict:
        text = text.strip()
        if len(text) > self.summary_max_chars:
            text = text[: self.summary_max_chars] + "…"
        return {
            "context_summary": {
                "text": text,
                "upto": messages[plan.cut - 1].id,
                "count": plan.cut - (_first_human(messages) or 0) - 1,
                "times": (summary or {}).get("times", 0) + 1,
            }
        }

    # 总结调用在 trace 里单独标出来，成本统计能看到压缩花了多少 token
    _CONFIG: ClassVar[dict[str, Any]] = {"run_name": "context_summary", "tags": ["context_summary"]}

    def before_model(self, state, runtime) -> dict | None:  # type: ignore[override]
        found = self._plan(state)
        if not found:
            return None
        plan, messages, summary = found
        try:
            reply = self.model.invoke(self._prompt(summary, plan.pending(messages)), config=self._CONFIG)
        except Exception as exc:  # 总结失败不能拖垮主流程：不更新摘要，裁剪兜底
            logger.warning("上下文摘要失败，退回裁剪：%s", exc)
            return None
        return self._update(summary, plan, messages, str(reply.content))

    async def abefore_model(self, state, runtime) -> dict | None:  # type: ignore[override]
        found = self._plan(state)
        if not found:
            return None
        plan, messages, summary = found
        try:
            reply = await self.model.ainvoke(self._prompt(summary, plan.pending(messages)), config=self._CONFIG)
        except Exception as exc:
            logger.warning("上下文摘要失败，退回裁剪：%s", exc)
            return None
        return self._update(summary, plan, messages, str(reply.content))

    # ---- 组装请求 ------------------------------------------------------ #
    def _request(self, request: ModelRequest) -> ModelRequest:
        summary = (request.state or {}).get("context_summary")
        view = build_view(request.messages, summary)
        view = trim_messages(view, self.max_chars, self.keep_last)  # 硬上限兜底
        if len(view) == len(request.messages) and all(a is b for a, b in zip(view, request.messages, strict=False)):
            return request
        return request.override(messages=view)

    def wrap_model_call(self, request: ModelRequest, handler):  # type: ignore[override]
        return handler(self._request(request))

    async def awrap_model_call(self, request: ModelRequest, handler):  # type: ignore[override]
        return await handler(self._request(request))
