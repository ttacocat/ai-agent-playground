"""lead 架构：一个主 agent 负责到底，子 agent 作为它可以调用的工具。

为什么放弃"按阶段拆分"（supervisor / team）
-------------------------------------------
实测（编码用例，qwen3.8-max、deepseek-v4.1-flash）：单 agent 自己就会先读 README、
看已有代码、改完跑测试，9 条编码用例全过；按"调研 → 实现 → 审查"拆给不同 agent 后，
准确率没有提高，token 却是 2.5 到 4.8 倍。读文件的次数正好翻倍：后一个角色看不到前一个
角色读过的东西，只能重读。在方案里附原文（team.py 的 context）只能缓解，堵不住。

强模型在一个 agent 里本来就会"先调研再动手"。拆开之后多出来的是交接时丢失的上下文，
以及分散的决策：定方案的人看不到实现中遇到的问题，实现的人又改不了方案。

这里的结构
----------
::

    lead（主模型，持有完整上下文，自己读写文件、跑测试、做所有决定）
      ├─ explore(question)  → 探索助手：独立上下文、只读工具，可以用更便宜的模型；
      │                       只把压缩后的结论和文件位置还给 lead。多个可以并行。
      └─ review(summary)    → 审查助手：独立核实改动，只在 lead 觉得需要时调用

- 不再有交接：做决定、写代码的始终是同一个 agent；
- 子 agent 只做"读得多、结论短"的事。它读 20 个文件，返回给 lead 的可能只有十几行，
  lead 的上下文反而更干净；
- 贵的模型只做需要判断的部分，探索可以交给便宜模型（``explore_model``）；
- 子 agent 不写 checkpoint：它们是一次性的只读调查，不需要中途恢复；并行的两个
  explore 也不会在同一个命名空间里互相覆盖状态。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

import openai
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import Field

from ai_agent_test.app.code_agent.agent.team import Review
from ai_agent_test.guardrails import read_only

EXPLORE_RESULT_LIMIT = 4_000
# 子 agent 最多调用几轮工具（一轮可以并行多个）。实测没有上限时，探索助手为了"找全"
# 会连跑三四十次 grep，一个问题花掉 4 万 token，比主 agent 自己做还贵
EXPLORE_MAX_ROUNDS = 10
REVIEW_MAX_ROUNDS = 8

LEAD_PROMPT = (
    "你是工程助手，负责从头到尾完成任务：自己读写文件、跑测试、做所有决定。\n"
    "- 需要大范围搜索或梳理时（找出所有调用某个函数的地方、项目对某件事有什么约定、某个行为是哪段代码实现的），"
    "用 explore 交给探索助手。它在独立的上下文里读文件，只把结论还给你。"
    "互不相关的问题可以在同一步里并行发起多个 explore。\n"
    "- 已知路径、范围明确的读取直接自己做，不要用 explore。\n"
    "- 改动跨多个文件、涉及删除，或你拿不准时，完成后调用 review 请审查助手独立核实；"
    "简单改动不需要。\n"
    "- 危险操作会被安全护栏拦下等待用户确认；被拒绝时不要换个写法、换个工具或拆成多步绕过，停止这部分并说明原因。\n"
    "完成后用几句话说明做了什么、结果如何。"
)

EXPLORER_PROMPT = (
    "你是探索助手，只有只读工具。主 agent 交给你一个具体问题，你负责查清楚并回答。\n"
    "- 先搜索（grep、file_search、list_directory），再读相关文件，不要猜。\n"
    "- 结论要短：直接回答问题，列出相关文件路径，必要时附上关键几行原文；不要贴整份文件。\n"
    "- 没找到就说没找到，并说明搜过哪些地方。\n"
    f"- 最终回答不超过 {EXPLORE_RESULT_LIMIT // 2} 字。"
)

REVIEWER_PROMPT = (
    "你是审查助手，只有只读工具。主 agent 说明了它做了什么改动，你要独立核实：\n"
    "不要只相信它的说明；读改动过的文件、跑测试（run_tests）、检查有没有遗漏的调用方或违反项目约定的地方。\n"
    "核实完成后必须调用 Review 工具提交结论；未达成的点写进 issues，写清楚现象和期望。"
)


class StepBudget(AgentMiddleware):
    """工具调用轮数到上限后，收走工具，让子 agent 立刻根据已有信息作答。

    和 ToolCallLimitMiddleware 直接结束不同：这里保留已经查到的东西，最后一次调用
    只是不能再调工具。结构化输出用的工具（如 Review）不在 tools 里，不受影响。
    """

    def __init__(self, max_rounds: int) -> None:
        super().__init__()
        self.max_rounds = max_rounds

    def _apply(self, request):
        rounds = sum(1 for m in request.messages if isinstance(m, AIMessage) and m.tool_calls)
        if rounds < self.max_rounds:
            return request
        note = HumanMessage(
            content=f"已经查了 {rounds} 轮，不能再调用工具了。立刻根据已经查到的内容给出结论；没查到的写明没查到。"
        )
        return request.override(tools=[], messages=[*request.messages, note])

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n…（结论过长，已截断）"


def _last_text(result: dict) -> str:
    messages = result.get("messages") or []
    return str(messages[-1].content) if messages else ""


def build_subagent_tools(
    tools: Sequence[BaseTool],
    explore_model: BaseChatModel,
    review_model: BaseChatModel,
    middleware: Sequence[Any] = (),
) -> list[BaseTool]:
    """explore / review 两个子 agent 工具。子 agent 只拿只读工具（guardrails.read_only）。"""
    readonly = read_only(tools)
    explorer = create_agent(
        model=explore_model,
        tools=readonly,
        name="explorer",
        system_prompt=EXPLORER_PROMPT,
        middleware=[*middleware, StepBudget(EXPLORE_MAX_ROUNDS)],
        checkpointer=False,
    )
    reviewer = create_agent(
        model=review_model,
        tools=readonly,
        name="reviewer",
        system_prompt=REVIEWER_PROMPT,
        response_format=ToolStrategy(Review),
        middleware=[*middleware, StepBudget(REVIEW_MAX_ROUNDS)],
        checkpointer=False,
    )

    async def _explore(
        question: Annotated[str, Field(description="要查清楚的一个具体问题，写明范围和想要的结果")],
    ) -> str:
        try:
            result = await explorer.ainvoke({"messages": [HumanMessage(content=question)]})
        except openai.BadRequestError as exc:  # 子 agent 出错不该拖垮整个任务：告诉 lead，让它自己查
            return f"探索助手出错，没有结论，请自己查：{_clip(str(exc), 300)}"
        return _clip(_last_text(result), EXPLORE_RESULT_LIMIT)

    async def _review(
        summary: Annotated[str, Field(description="你做了哪些改动、为什么、改了哪些文件")],
    ) -> str:
        try:
            result = await reviewer.ainvoke({"messages": [HumanMessage(content=f"主 agent 的改动说明：\n{summary}")]})
        except openai.BadRequestError as exc:
            return f"审查助手出错，没有结论，请自己跑 run_tests 核实：{_clip(str(exc), 300)}"
        verdict = result.get("structured_response")
        if not isinstance(verdict, Review):
            return f"审查未给出结构化结论：{_clip(_last_text(result), 1000)}"
        if verdict.passed:
            return f"审查通过：{verdict.summary}"
        return "审查未通过：" + verdict.summary + "".join(f"\n- {i}" for i in verdict.issues)

    explore = StructuredTool.from_function(
        coroutine=_explore,
        name="explore",
        description=(
            "把一个只读的调研问题交给探索助手（独立上下文、只读工具）。适合大范围搜索、梳理项目约定、"
            "查找所有调用方。返回简短结论和相关文件位置。互不相关的问题可以并行调用多次。"
        ),
    )
    review = StructuredTool.from_function(
        coroutine=_review,
        name="review",
        description="改动完成后，请审查助手独立核实（读文件、跑测试、查遗漏）。返回通过或问题列表。",
    )
    return [explore, review]


def build_lead(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    explore_model: BaseChatModel | None = None,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
):
    """主 agent + explore / review。explore_model 为空时用主模型。"""
    subagents = build_subagent_tools(tools, explore_model or model, model, middleware)
    return create_agent(
        model=model,
        tools=[*tools, *subagents],
        name="lead",
        system_prompt=LEAD_PROMPT,
        middleware=list(middleware),
        checkpointer=checkpointer,
    )
