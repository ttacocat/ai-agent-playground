"""人工审批（interrupt）的端到端测试。

用脚本化的假模型搭真实的 LangGraph 图（create_agent / create_supervisor + checkpointer），
验证的是图层面的行为：挂起时工具没执行、恢复后才执行、换一个进程也能恢复。
全部离线，不需要 API 凭证。
"""

from __future__ import annotations

from fakes import ScriptedModel
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph_supervisor import create_supervisor

from ai_agent_test.guardrails import (
    InteractiveApprover,
    NullAuditLog,
    drive,
    guard_tool,
    parse_answer,
    pending_approvals,
)


def call(command: str, call_id: str) -> dict:
    return {"name": "run_shell", "args": {"command": command}, "id": call_id}


EXECUTED: list[str] = []


@tool
def run_shell(command: str) -> str:
    """run a shell command"""
    EXECUTED.append(command)
    return f"executed {command}"


def build_agent(*tool_calls: dict, saver=None, audit=None, name: str | None = None):
    model = ScriptedModel(replies=[AIMessage("", tool_calls=list(tool_calls)), AIMessage("完成")])
    guarded = guard_tool(run_shell, approver=InteractiveApprover(), audit=audit or NullAuditLog())
    return create_agent(model=model, tools=[guarded], checkpointer=saver, name=name)


def config(thread: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread}}


def tool_messages(state) -> list[ToolMessage]:
    return [m for m in state.values["messages"] if isinstance(m, ToolMessage)]


def setup_function():
    EXECUTED.clear()


# --------------------------------------------------------------------------- #
# 单 agent
# --------------------------------------------------------------------------- #
async def test_destructive_call_suspends_graph_before_execution():
    app = build_agent(call("rm -rf /srv/app", "c1"), saver=InMemorySaver())
    await app.ainvoke({"messages": [("user", "删掉")]}, config())

    pending = await pending_approvals(app, config())
    assert len(pending) == 1
    request = pending[0].value
    assert request["tool"] == "run_shell"
    assert request["level"] == "destructive"
    assert "rm" in request["reason"]
    assert EXECUTED == []  # 挂起时底层工具一次都没跑


async def test_approve_resumes_and_executes_once():
    audit = NullAuditLog()
    app = build_agent(call("rm -rf /srv/app", "c1"), saver=InMemorySaver(), audit=audit)
    await app.ainvoke({"messages": [("user", "删掉")]}, config())
    pending = await pending_approvals(app, config())

    await app.ainvoke(Command(resume={pending[0].id: True}), config())

    assert EXECUTED == ["rm -rf /srv/app"]
    assert await pending_approvals(app, config()) == []
    # 审计只在拿到决定后写一次，挂起那一下不算
    assert [r.outcome for r in audit.records] == ["executed"]


async def test_deny_resumes_with_explanation_and_no_execution():
    app = build_agent(call("rm -rf /srv/app", "c1"), saver=InMemorySaver())
    await app.ainvoke({"messages": [("user", "删掉")]}, config())
    pending = await pending_approvals(app, config())

    await app.ainvoke(Command(resume={pending[0].id: False}), config())

    state = await app.aget_state(config())
    assert EXECUTED == []
    assert "未执行" in tool_messages(state)[-1].content


async def test_read_only_call_does_not_suspend():
    app = build_agent(call("ls -la", "c1"), saver=InMemorySaver())
    await app.ainvoke({"messages": [("user", "看看")]}, config())

    assert await pending_approvals(app, config()) == []
    assert EXECUTED == ["ls -la"]


async def test_parallel_calls_are_answered_independently():
    app = build_agent(call("rm -rf a", "c1"), call("rm -rf b", "c2"), saver=InMemorySaver())
    await app.ainvoke({"messages": [("user", "删掉 a 和 b")]}, config())
    pending = await pending_approvals(app, config())
    assert len(pending) == 2

    answers = {item.id: item.value["preview"].endswith("a") for item in pending}  # 只批 a
    await app.ainvoke(Command(resume=answers), config())

    assert EXECUTED == ["rm -rf a"]


async def test_pending_approval_survives_process_restart():
    """模拟进程重启：图对象重建，只有 checkpointer 是同一个（生产中是 Redis）。"""
    saver = InMemorySaver()
    first = build_agent(call("rm -rf /srv/app", "c1"), saver=saver)
    await first.ainvoke({"messages": [("user", "删掉")]}, config())
    del first

    second = build_agent(call("rm -rf /srv/app", "c1"), saver=saver)
    pending = await pending_approvals(second, config())
    assert len(pending) == 1

    await second.ainvoke(Command(resume={pending[0].id: True}), config())
    assert EXECUTED == ["rm -rf /srv/app"]


# --------------------------------------------------------------------------- #
# drive 循环
# --------------------------------------------------------------------------- #
async def test_drive_asks_and_runs_to_completion():
    asked: list[dict] = []

    async def ask(request: dict) -> bool:
        asked.append(request)
        return True

    app = build_agent(call("rm -rf /srv/app", "c1"), saver=InMemorySaver())
    async for _ in drive(app, {"messages": [("user", "删掉")]}, config(), ask, stream_mode="updates"):
        pass

    assert [r["tool"] for r in asked] == ["run_shell"]
    assert EXECUTED == ["rm -rf /srv/app"]
    state = await app.aget_state(config())
    assert state.values["messages"][-1].content == "完成"


async def test_drive_with_no_input_only_resolves_leftovers():
    saver = InMemorySaver()
    app = build_agent(call("rm -rf /srv/app", "c1"), saver=saver)
    await app.ainvoke({"messages": [("user", "删掉")]}, config())

    async def approve(_request: dict) -> bool:
        return True

    async for _ in drive(app, None, config(), approve):
        pass
    assert EXECUTED == ["rm -rf /srv/app"]

    # 没有遗留时 inputs=None 什么都不做
    async for _ in drive(app, None, config(), approve):
        raise AssertionError("不应该产生任何输出")


async def test_drive_settles_leftovers_before_a_new_turn():
    """上一轮审批被中断（超时 / Ctrl-C）后用户直接发新消息，遗留的审批要先处理掉。"""
    app = build_agent(call("rm -rf /srv/app", "c1"), saver=InMemorySaver())
    await app.ainvoke({"messages": [("user", "删掉")]}, config())

    asked: list[str] = []

    async def deny(request: dict) -> bool:
        asked.append(request["preview"])
        return False

    async for _ in drive(app, {"messages": [("user", "算了，看看目录")]}, config(), deny):
        pass

    assert asked == ["rm -rf /srv/app"]
    assert await pending_approvals(app, config()) == []
    # 每条 tool_call 都有对应的 ToolMessage，会话历史是合法的
    state = await app.aget_state(config())
    call_ids = {tc["id"] for m in state.values["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls}
    answered = {m.tool_call_id for m in tool_messages(state)}
    assert call_ids <= answered


async def test_supervisor_subagent_interrupt_bubbles_up():
    """Supervisor 下专家 agent 里发起的审批，要在顶层图上能看到、能恢复。"""
    worker = build_agent(call("rm -rf /srv/app", "c1"), name="code_expert")
    supervisor_model = ScriptedModel(
        replies=[
            AIMessage("", tool_calls=[{"name": "transfer_to_code_expert", "args": {}, "id": "t1"}]),
            AIMessage("全部完成"),
        ]
    )
    app = create_supervisor(agents=[worker], model=supervisor_model, prompt="分派任务").compile(
        checkpointer=InMemorySaver()
    )

    asked: list[dict] = []

    async def ask(request: dict) -> bool:
        asked.append(request)
        return True

    async for _ in drive(app, {"messages": [("user", "删掉")]}, config(), ask, stream_mode="updates", subgraphs=True):
        pass

    assert len(asked) == 1
    assert EXECUTED == ["rm -rf /srv/app"]


# --------------------------------------------------------------------------- #
# 审批器本身
# --------------------------------------------------------------------------- #
async def test_interactive_approver_fails_closed_outside_a_graph():
    guarded = guard_tool(run_shell, approver=InteractiveApprover(), audit=NullAuditLog())
    result = await guarded.ainvoke({"command": "rm -rf /srv/app"})
    assert "未执行" in result
    assert EXECUTED == []


def test_parse_answer():
    assert parse_answer(True) is True
    assert parse_answer("y") is True
    assert parse_answer({"approved": True}) is True
    assert parse_answer(False) is False
    assert parse_answer("") is False
    assert parse_answer(None) is False
    assert parse_answer({"approved": "nope"}) is False
