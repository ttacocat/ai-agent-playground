"""多智能体编排：两种架构，chat 与评测共用同一套构建函数。

``build_supervisor``
    langgraph-supervisor 的标准用法：主管 LLM 决定把任务交给谁。改进点只有一个——
    research_expert 只拿只读工具（``guardrails.read_only``），分工不再只靠 prompt。

``build_team``
    自己用 StateGraph 写的固定流程，把"主管 LLM 自由分派"换成确定的状态机::

        planner ──(需要改动)──▶ coder ──▶ reviewer ──(通过)──▶ finish
           │                     ▲          │
           └──(纯问答)──▶ finish  └─(未通过且未超轮次)─┘

    - planner / reviewer 只读，coder 拿全部工具（仍然经过护栏和人工审批）；
    - 方案和审查结论是结构化输出（``Plan`` / ``Review``），存进图状态，而不是埋在消息里；
    - 审查不通过时，coder 带着具体问题重做，最多 ``max_attempts`` 轮；
    - 每个角色每次都从干净的上下文开始，只看到任务、方案和上一轮审查意见，
      不继承别人的完整对话——上下文更短，也避免被前一轮的错误思路带偏。

    为什么要换：Supervisor 的分派本身是一次 LLM 决策，可能跳过方案直接写、
    写完不验证就结束，这些在评测里都很难归因。固定流程把"一定会审查""最多重试几次"
    变成代码保证，LLM 只负责每个角色内部的事。代价是灵活性：简单问题也要先过 planner，
    所以 planner 可以判定"不需要改动"直接作答。

两种架构可以用同一套评测用例对比（``agent eval --target supervisor team``）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field

from ai_agent_test.guardrails import read_only

# --------------------------------------------------------------------------- #
# Supervisor 架构
# --------------------------------------------------------------------------- #
RESEARCH_PROMPT = (
    "你是一个技术方案设计专家，专门负责设计技术方案。\n"
    "要求：先阅读相关文件，再给出方案；方案包含目标、思路、涉及文件、实施步骤、风险点；"
    "你只有只读工具，不要尝试修改任何东西，实现工作交给 code_expert。\n"
    "code_expert 看不到你读过的内容，只能看到你最后的方案：把要修改的文件的完整原文、"
    "要遵守的项目约定原文、要复用的函数原文附在方案末尾，每段注明文件路径。"
)

CODE_PROMPT = (
    "你是一个代码实现专家，负责根据技术方案编写、修改并验证代码。\n"
    "要求：严格按照 research_expert 的方案实现；方案末尾附了文件原文，已附的文件不要重复读取；修改后运行必要命令验证；"
    "完成后简要说明改动了哪些文件。\n"
    "危险操作会被安全护栏拦下等待用户确认；被拒绝时不要换个写法、换个工具或拆成多步绕过，"
    "停止这部分操作并说明原因。"
)

SUPERVISOR_PROMPT = (
    "你是团队主管，管理两个专家：\n"
    "- research_expert：负责技术方案设计，只读，不写代码；\n"
    "- code_expert：负责代码实现与验证。\n"
    "流程：先让 research_expert 出方案，再让 code_expert 按方案实现。\n"
    "每次只分配给一个专家，不要自己做具体工作。"
)


def build_supervisor(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
):
    research_agent = create_agent(
        model=model,
        tools=read_only(tools),
        name="research_expert",
        system_prompt=RESEARCH_PROMPT,
        middleware=list(middleware),
    )
    code_agent = create_agent(
        model=model,
        tools=list(tools),
        name="code_expert",
        system_prompt=CODE_PROMPT,
        middleware=list(middleware),
    )
    workflow = create_supervisor(
        agents=[research_agent, code_agent],
        model=model,
        prompt=SUPERVISOR_PROMPT,
        output_mode="last_message",
    )
    return workflow.compile(checkpointer=checkpointer)


# --------------------------------------------------------------------------- #
# Team 架构：结构化状态
# --------------------------------------------------------------------------- #
class Excerpt(BaseModel):
    path: str = Field(description="文件路径")
    content: str = Field(description="原文，不要改写或省略")
    note: str = Field(default="", description="为什么需要它：要修改的文件 / 要遵守的约定 / 要复用的函数")


# 交接成本的实测（qwen3.8-max，编码用例 C01）：单 agent 读 9 次文件；supervisor 读 18 到 27 次，
# 列目录正好翻倍——方案专家读一遍，实现专家拿到的只有一段方案文字，又从头读一遍。
# 所以方案里直接附上读到的原文，实现者不必重新调研。
CONTEXT_LIMIT = 16_000


class Plan(BaseModel):
    """技术方案。完成调研后调用它提交方案。"""

    needs_changes: bool = Field(
        description="完成任务是否需要修改代码、文件、数据库或执行有副作用的命令；纯查询或问答为 false"
    )
    goal: str = Field(description="一句话说明要达成的结果")
    steps: list[str] = Field(default_factory=list, description="实施步骤，按顺序")
    files: list[str] = Field(default_factory=list, description="涉及的文件、表或目录")
    risks: list[str] = Field(default_factory=list, description="风险点")
    answer: str = Field(default="", description="needs_changes 为 false 时，直接给用户的回答")
    context: list[Excerpt] = Field(
        default_factory=list,
        description=(
            "交给实现者的调研摘录。实现者看不到你读过的内容：要修改的文件给出完整原文，"
            "要遵守的约定、要复用的函数给出原文片段。摘录不全，实现者就得重新读一遍。"
        ),
    )


class Review(BaseModel):
    """审查结论。核实完成后调用它提交结论。"""

    passed: bool = Field(description="任务目标是否已经真正达成（以你核实到的事实为准）")
    issues: list[str] = Field(default_factory=list, description="未达成的点，每条写清楚现象和期望")
    summary: str = Field(default="", description="一句话结论")


class TeamState(MessagesState):
    task: str
    plan: dict | None
    attempts: int
    review: dict | None
    report: str


PLANNER_PROMPT = (
    "你是方案专家。你只有只读工具：先调研现状（读文件、查表结构、查知识库），再提交方案。\n"
    "调研完成后必须调用 Plan 工具提交结构化方案。\n"
    "实现者看不到你读过的内容：把要修改的文件的完整原文、要遵守的项目约定、要复用的函数原文放进 context。\n"
    "如果任务只是查询或问答、不需要任何修改，needs_changes 设为 false，并把答案写进 answer。"
)

CODER_PROMPT = (
    "你是实现专家，按给定方案完成任务，可以使用全部工具。\n"
    "方案附了调研摘录（原文）：摘录里已有的文件不要重新读取，直接以摘录为基础修改；"
    "只有摘录里没有、又确实需要的内容才去读。\n"
    "危险操作会被安全护栏拦下等待用户确认；被拒绝时不要换个写法绕过，改用安全的方式或说明原因。\n"
    "如果附带了审查意见，逐条修正。完成后用几句话说明做了什么、结果如何。"
)

REVIEWER_PROMPT = (
    "你是审查专家。你只有只读工具。\n"
    "不要只相信实现者的自述：用只读工具核实结果（查看目录、读文件、SELECT 查询等）。\n"
    "核实完成后必须调用 Review 工具提交结论；未达成的点写进 issues，写清楚现象和期望。"
)


def _format_context(plan: dict, limit: int = CONTEXT_LIMIT) -> str:
    """调研摘录 → 给实现者看的文本。总长度有上限，超出的部分截断并注明，实现者可以自己去读。"""
    blocks, used = [], 0
    for item in plan.get("context") or []:
        content = str(item.get("content", ""))
        room = limit - used
        if room <= 0:
            blocks.append(f"（还有摘录因长度限制没有附上：{item.get('path', '')}，需要时请自己读取）")
            continue
        if len(content) > room:
            content = content[:room] + "\n…（截断，完整内容请自己读取）"
        used += len(content)
        note = f"（{item['note']}）" if item.get("note") else ""
        blocks.append(f"### {item.get('path', '')}{note}\n```\n{content}\n```")
    return "调研摘录（原文）：\n" + "\n\n".join(blocks) if blocks else ""


def _format_plan(plan: dict, with_context: bool = False) -> str:
    """with_context=True 只给实现者：审查者要独立核实，不看方案专家的摘录。"""
    lines = [f"目标：{plan.get('goal', '')}"]
    for index, step in enumerate(plan.get("steps") or [], 1):
        lines.append(f"{index}. {step}")
    if plan.get("files"):
        lines.append("涉及：" + "、".join(plan["files"]))
    if plan.get("risks"):
        lines.append("风险：" + "；".join(plan["risks"]))
    if with_context and (excerpts := _format_context(plan)):
        lines += ["", excerpts]
    elif plan.get("context"):
        lines.append(f"附调研摘录 {len(plan['context'])} 段")
    return "\n".join(lines)


def _format_review(review: dict) -> str:
    verdict = "通过" if review.get("passed") else "未通过"
    lines = [f"审查{verdict}：{review.get('summary', '')}"]
    lines += [f"- {issue}" for issue in review.get("issues") or []]
    return "\n".join(lines)


def _last_text(result: dict) -> str:
    messages = result.get("messages") or []
    return str(messages[-1].content) if messages else ""


def _latest_user_task(state: TeamState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def build_team(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    max_attempts: int = 2,
    middleware: Sequence[Any] = (),
    checkpointer: Any = None,
):
    """planner → coder → reviewer（不通过则带着问题回到 coder，最多 max_attempts 轮）。"""
    if max_attempts < 1:
        raise ValueError("max_attempts 至少为 1")

    readonly_tools = read_only(tools)
    planner = create_agent(
        model=model,
        tools=readonly_tools,
        name="planner",
        system_prompt=PLANNER_PROMPT,
        response_format=ToolStrategy(Plan),
        middleware=list(middleware),
    )
    coder = create_agent(
        model=model,
        tools=list(tools),
        name="coder",
        system_prompt=CODER_PROMPT,
        middleware=list(middleware),
    )
    reviewer = create_agent(
        model=model,
        tools=readonly_tools,
        name="reviewer",
        system_prompt=REVIEWER_PROMPT,
        response_format=ToolStrategy(Review),
        middleware=list(middleware),
    )

    # 注意：节点在人工审批恢复时会整体重跑一遍，所以子 agent 调用之前不能有副作用，
    # 状态更新全部放在返回值里。
    async def plan_node(state: TeamState) -> dict:
        task = _latest_user_task(state)
        result = await planner.ainvoke({"messages": [HumanMessage(content=f"任务：{task}")]})
        structured = result.get("structured_response")
        if isinstance(structured, Plan):
            plan = structured.model_dump()
        else:  # 模型没按格式提交：把它的文字当成方案，仍然交给 coder
            plan = Plan(needs_changes=True, goal=task, steps=[_last_text(result)]).model_dump()
        return {
            "task": task,
            "plan": plan,
            "attempts": 0,
            "review": None,
            "report": "",
            "messages": [AIMessage(content=_format_plan(plan), name="planner")],
        }

    async def code_node(state: TeamState) -> dict:
        parts = [f"任务：{state['task']}", f"方案：\n{_format_plan(state['plan'] or {}, with_context=True)}"]
        review = state.get("review")
        if review and not review.get("passed"):
            parts.append("上一轮审查未通过，请逐条修正：\n" + "\n".join(f"- {i}" for i in review.get("issues") or []))
        result = await coder.ainvoke({"messages": [HumanMessage(content="\n\n".join(parts))]})
        report = _last_text(result)
        return {
            "report": report,
            "attempts": state.get("attempts", 0) + 1,
            "messages": [AIMessage(content=report, name="coder")],
        }

    async def review_node(state: TeamState) -> dict:
        prompt = (
            f"任务：{state['task']}\n\n方案：\n{_format_plan(state['plan'] or {})}\n\n"
            f"实现者的说明（需要你核实）：\n{state.get('report', '')}"
        )
        result = await reviewer.ainvoke({"messages": [HumanMessage(content=prompt)]})
        structured = result.get("structured_response")
        if isinstance(structured, Review):
            review = structured.model_dump()
        else:  # 没有结构化结论 = 无法确认通过，按未通过处理
            review = Review(passed=False, issues=["审查未给出结构化结论"], summary=_last_text(result)).model_dump()
        return {"review": review, "messages": [AIMessage(content=_format_review(review), name="reviewer")]}

    async def finish_node(state: TeamState) -> dict:
        plan = state.get("plan") or {}
        review = state.get("review")
        if not plan.get("needs_changes", True):
            text = plan.get("answer") or plan.get("goal", "")
        elif review and review.get("passed"):
            text = f"已完成：{review.get('summary') or plan.get('goal', '')}\n\n{state.get('report', '')}".strip()
        else:
            issues = "\n".join(f"- {i}" for i in (review or {}).get("issues") or [])
            text = (
                f"尝试 {state.get('attempts', 0)} 轮后审查仍未通过：\n{issues}\n\n"
                f"最后一轮实现说明：\n{state.get('report', '')}"
            )
        return {"messages": [AIMessage(content=text, name="team")]}

    def after_plan(state: TeamState) -> Literal["coder", "finish"]:
        return "coder" if (state.get("plan") or {}).get("needs_changes", True) else "finish"

    def after_review(state: TeamState) -> Literal["coder", "finish"]:
        review = state.get("review") or {}
        if review.get("passed") or state.get("attempts", 0) >= max_attempts:
            return "finish"
        return "coder"

    graph = StateGraph(TeamState)
    graph.add_node("planner", plan_node)
    graph.add_node("coder", code_node)
    graph.add_node("reviewer", review_node)
    graph.add_node("finish", finish_node)
    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", after_plan)
    graph.add_edge("coder", "reviewer")
    graph.add_conditional_edges("reviewer", after_review)
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
