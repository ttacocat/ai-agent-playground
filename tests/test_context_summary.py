"""上下文摘要压缩的离线测试。

切点、视图这些纯函数直接测；中间件接进真实的 create_agent，用脚本化模型驱动，
验证三件事：发给模型的请求变短了且带着摘要、会话状态里的历史一条没少、
总结失败时主流程照常。
"""

from __future__ import annotations

from fakes import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from ai_agent_test.observability.context import total_chars
from ai_agent_test.observability.summary import (
    SUMMARY_PROMPT,
    ContextSummaryMiddleware,
    build_view,
    plan_compaction,
    render_transcript,
)


def history(rounds: int, size: int = 400, prefix: str = "") -> list:
    """一段带工具调用的长历史：每轮 = AI 调工具 + 工具结果 + AI 回答 + 用户追问。

    prefix 用来生成不重复的消息 id：id 相同的消息进入状态时是"更新"而不是"追加"。
    """
    p = prefix
    messages = [HumanMessage("任务：整理 test 库", id=f"{p}h0")]
    for i in range(rounds):
        call = {"name": "mysql_execute_query", "args": {"command": f"SELECT {i}"}, "id": f"{p}c{i}"}
        messages += [
            AIMessage("", tool_calls=[call], id=f"{p}a{i}"),
            ToolMessage("x" * size, tool_call_id=f"{p}c{i}", name="mysql_execute_query", id=f"{p}t{i}"),
            AIMessage(f"第 {i} 轮结论", id=f"{p}r{i}"),
            HumanMessage(f"继续第 {i + 1} 轮", id=f"{p}u{i}"),
        ]
    return messages


# --------------------------------------------------------------------------- #
# 切点
# --------------------------------------------------------------------------- #
def test_no_compaction_under_budget():
    assert plan_compaction(history(2), None, max_chars=10_000, keep_last=4) is None


def test_compaction_never_splits_tool_call_pairs():
    messages = history(10)
    plan = plan_compaction(messages, None, max_chars=2_000, keep_last=4)
    assert plan is not None
    assert plan.start == 1  # 第一条用户消息永远原样保留
    assert not isinstance(messages[plan.cut], ToolMessage)
    tail = messages[plan.cut :]
    answered = {m.tool_call_id for m in tail if isinstance(m, ToolMessage)}
    called = {tc["id"] for m in tail if isinstance(m, AIMessage) for tc in m.tool_calls}
    assert answered <= called  # 后半段里每个工具结果都有对应的调用


def test_compaction_targets_half_budget_and_keeps_recent():
    messages = history(10)
    plan = plan_compaction(messages, None, max_chars=2_000, keep_last=4)
    assert total_chars(messages[plan.cut :]) <= 1_000 + 400  # 一半预算（允许为保配对多留一条）
    assert len(messages) - plan.cut >= 4


def test_compaction_is_incremental():
    messages = history(10)
    first = plan_compaction(messages, None, max_chars=2_000, keep_last=4)
    summary = {"text": "旧摘要", "upto": messages[first.cut - 1].id, "times": 1, "count": first.cut - 1}
    # 刚压缩完：不再触发
    assert plan_compaction(messages, summary, max_chars=2_000, keep_last=4) is None
    # 又聊了很多轮：只压缩上次之后的部分
    longer = messages + history(10, prefix="b")[1:]
    second = plan_compaction(longer, summary, max_chars=2_000, keep_last=4)
    assert second.start == first.cut


def test_unknown_summary_anchor_falls_back_to_full_history():
    messages = history(10)
    plan = plan_compaction(messages, {"text": "x", "upto": "不存在的 id"}, max_chars=2_000, keep_last=4)
    assert plan.start == 1


# --------------------------------------------------------------------------- #
# 视图
# --------------------------------------------------------------------------- #
def test_view_pins_task_inserts_summary_and_drops_covered():
    messages = [SystemMessage("sys", id="s"), *history(6)]
    summary = {"text": "## 用户目标与要求\n整理 test 库", "upto": "u2", "times": 2, "count": 12}
    view = build_view(messages, summary)
    assert [type(m).__name__ for m in view[:3]] == ["SystemMessage", "HumanMessage", "SystemMessage"]
    assert view[1].content == "任务：整理 test 库"
    assert "已压缩 2 次" in view[2].content and "整理 test 库" in view[2].content
    assert view[3].id == "a3"  # 紧接着是摘要之后的第一条
    assert len(messages) == 1 + 1 + 6 * 4  # 原列表没被改动


def test_view_without_summary_is_unchanged():
    messages = history(3)
    assert build_view(messages, None) == messages


def test_transcript_keeps_guard_denials():
    messages = [
        AIMessage("", tool_calls=[{"name": "mysql_execute_query", "args": {"command": "DROP TABLE user"}, "id": "c"}]),
        ToolMessage("⛔ 操作被安全护栏拦截：mysql_execute_query（DROP 语句不可回滚）" + "x" * 5000, tool_call_id="c"),
    ]
    text = render_transcript(messages)
    assert "DROP TABLE user" in text
    assert "⛔ 操作被安全护栏拦截" in text
    assert "（截断）" in text and len(text) < 2_000


def test_summary_prompt_asks_to_keep_denials():
    assert "被拒绝或被拦截的操作" in SUMMARY_PROMPT
    assert "不得换写法" in SUMMARY_PROMPT


# --------------------------------------------------------------------------- #
# 中间件接进真实的 agent
# --------------------------------------------------------------------------- #
class RecordingModel(ScriptedModel):
    """记下每次收到的消息。"""

    seen: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


class BrokenModel(ScriptedModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("总结模型挂了")


def build(summarizer, max_chars: int = 2_000):
    main = RecordingModel(replies=[AIMessage("好的")], seen=[])
    middleware = ContextSummaryMiddleware(model=summarizer, max_chars=max_chars, keep_last=4)
    agent = create_agent(model=main, tools=[], middleware=[middleware], checkpointer=InMemorySaver())
    return agent, main


CONFIG = {"configurable": {"thread_id": "t"}}


async def test_agent_receives_summary_and_history_stays_intact():
    summarizer = RecordingModel(
        replies=[AIMessage("## 用户目标与要求\n整理 test 库\n## 被拒绝或被拦截的操作\n无")], seen=[]
    )
    agent, main = build(summarizer)
    original = history(10)
    await agent.ainvoke({"messages": original}, CONFIG)

    state = (await agent.aget_state(CONFIG)).values
    assert len(state["messages"]) == len(original) + 1  # 历史一条没少，只多了这次的回答
    assert state["context_summary"]["times"] == 1

    sent = main.seen[-1]
    assert any(isinstance(m, SystemMessage) and "早期对话摘要" in m.content for m in sent)
    assert total_chars(sent) < total_chars(original)
    assert sent[0].content == "任务：整理 test 库"
    assert len(summarizer.seen) == 1  # 总结只调了一次


async def test_next_turn_reuses_summary_without_resummarizing():
    summarizer = RecordingModel(replies=[AIMessage("摘要 v1"), AIMessage("摘要 v2")], seen=[])
    agent, _ = build(summarizer)
    await agent.ainvoke({"messages": history(10)}, CONFIG)
    await agent.ainvoke({"messages": [HumanMessage("再看一眼")]}, CONFIG)  # 只多了一点点
    assert len(summarizer.seen) == 1


async def test_second_compaction_merges_previous_summary():
    summarizer = RecordingModel(replies=[AIMessage("摘要 v1"), AIMessage("摘要 v2")], seen=[])
    agent, _ = build(summarizer)
    await agent.ainvoke({"messages": history(10)}, CONFIG)
    await agent.ainvoke({"messages": history(10, prefix="b")[1:]}, CONFIG)  # 又来了一大段
    assert len(summarizer.seen) == 2
    assert "摘要 v1" in summarizer.seen[1][0].content  # 增量：旧摘要作为输入
    state = (await agent.aget_state(CONFIG)).values
    assert state["context_summary"]["text"] == "摘要 v2"
    assert state["context_summary"]["times"] == 2


async def test_summarizer_failure_falls_back_to_trim():
    agent, main = build(BrokenModel(replies=[]))
    await agent.ainvoke({"messages": history(10)}, CONFIG)
    state = (await agent.aget_state(CONFIG)).values
    assert "context_summary" not in state
    assert total_chars(main.seen[-1]) <= 2_000 + 500  # 裁剪兜底仍然生效


async def test_long_summary_is_capped():
    summarizer = RecordingModel(replies=[AIMessage("长" * 5000)], seen=[])
    main = RecordingModel(replies=[AIMessage("好的")], seen=[])
    middleware = ContextSummaryMiddleware(model=summarizer, max_chars=2_000, keep_last=4, summary_max_chars=300)
    agent = create_agent(model=main, tools=[], middleware=[middleware], checkpointer=InMemorySaver())
    await agent.ainvoke({"messages": history(10)}, CONFIG)
    assert len((await agent.aget_state(CONFIG)).values["context_summary"]["text"]) <= 301


def test_build_context_middleware_respects_strategy(monkeypatch):
    from ai_agent_test.config import Settings
    from ai_agent_test.observability import ContextBudgetMiddleware
    from ai_agent_test.observability import context as ctx

    for strategy, expected in (("summarize", ContextSummaryMiddleware), ("trim", ContextBudgetMiddleware)):
        monkeypatch.setenv("CONTEXT_STRATEGY", strategy)
        monkeypatch.setattr("ai_agent_test.config.settings", Settings())
        [mw] = ctx.build_context_middleware()
        assert isinstance(mw, expected)


async def test_works_inside_supervisor_subagents():
    """子 agent 的状态多了 context_summary 字段，放进 Supervisor 里不能和父图冲突。"""
    from fakes import tool_call

    from ai_agent_test.app.code_agent.agent.team import build_supervisor

    model = ScriptedModel(
        replies=[tool_call("transfer_to_research_expert", "t1"), AIMessage("方案：无需改动"), AIMessage("结束")]
    )
    summarizer = RecordingModel(replies=[AIMessage("摘要")], seen=[])
    middleware = [ContextSummaryMiddleware(model=summarizer, max_chars=2_000, keep_last=4)]
    app = build_supervisor(model, [], middleware=middleware, checkpointer=InMemorySaver())
    await app.ainvoke({"messages": history(10)}, CONFIG)
    assert len(summarizer.seen) == 1  # 专家收到的长历史触发了一次压缩
    assert (await app.aget_state(CONFIG)).values["messages"][-1].content == "结束"
