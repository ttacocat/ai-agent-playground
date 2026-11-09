"""lead 架构的离线测试：主 agent 负责到底，explore / review 是子 agent 工具。"""

from __future__ import annotations

from fakes import RecordingScriptedModel, ScriptedModel, tool_call
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_test.app.code_agent.agent.lead import EXPLORE_RESULT_LIMIT, build_lead
from ai_agent_test.guardrails import InteractiveApprover, NullAuditLog, drive, guard_tool, pending_approvals

EXECUTED: list[str] = []
SECRET_CONTENT = "def apply_discount(price, discount):  # 这一行只应该出现在探索助手的上下文里"


@tool
def read_file(file_path: str) -> str:
    """Read file from disk"""
    EXECUTED.append(f"read {file_path}")
    return SECRET_CONTENT


@tool
def write_file(file_path: str, text: str, append: bool = False) -> str:
    """Write file to disk"""
    EXECUTED.append(f"write {file_path}")
    return f"File written successfully to {file_path}."


@tool
def run_shell(command: str) -> str:
    """run a shell command"""
    EXECUTED.append(command)
    return "ok"


def tools():
    approver, audit = InteractiveApprover(), NullAuditLog()
    return [guard_tool(t, approver=approver, audit=audit) for t in (read_file, write_file, run_shell)]


def config(thread: str = "t") -> dict:
    return {"configurable": {"thread_id": thread}}


def setup_function():
    EXECUTED.clear()


async def run(app, task: str = "修 apply_discount"):
    async def approve(_request: dict) -> bool:
        return True

    async for _ in drive(app, {"messages": [("user", task)]}, config(), approve, stream_mode="updates"):
        pass
    return await app.aget_state(config())


def explore_call(*questions: str) -> AIMessage:
    return AIMessage(
        "",
        tool_calls=[{"name": "explore", "args": {"question": q}, "id": f"x{i}"} for i, q in enumerate(questions)],
    )


async def test_main_agent_sees_only_the_explorer_conclusion():
    lead_model = RecordingScriptedModel(
        replies=[
            explore_call("apply_discount 在哪、怎么写的"),
            tool_call("write_file", "w1", file_path="shop/pricing.py", text="fixed"),
            AIMessage("改好了"),
        ]
    )
    explorer_model = RecordingScriptedModel(
        replies=[tool_call("read_file", "r1", file_path="shop/pricing.py"), AIMessage("在 shop/pricing.py 第 6 行")]
    )
    app = build_lead(lead_model, tools(), explore_model=explorer_model, checkpointer=InMemorySaver())
    state = await run(app)

    assert EXECUTED == ["read shop/pricing.py", "write shop/pricing.py"]
    # 探索助手读到了原文；主 agent 收到的只有结论
    assert any(SECRET_CONTENT in str(m.content) for msgs in explorer_model.seen for m in msgs)
    assert not any(SECRET_CONTENT in str(m.content) for msgs in lead_model.seen for m in msgs)
    explore_result = next(m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.name == "explore")
    assert explore_result.content == "在 shop/pricing.py 第 6 行"


async def test_parallel_explores_each_get_a_fresh_context():
    lead_model = ScriptedModel(replies=[explore_call("问题一", "问题二"), AIMessage("完成")])
    explorer_model = RecordingScriptedModel(replies=[AIMessage("结论")])
    app = build_lead(lead_model, tools(), explore_model=explorer_model, checkpointer=InMemorySaver())
    state = await run(app)

    questions = [next(str(m.content) for m in msgs if m.type == "human") for msgs in explorer_model.seen]
    assert sorted(questions) == ["问题一", "问题二"]
    assert all(len([m for m in msgs if m.type == "human"]) == 1 for msgs in explorer_model.seen)  # 互不可见
    results = [m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.name == "explore"]
    assert len(results) == 2


async def test_explorer_is_read_only():
    lead_model = ScriptedModel(replies=[explore_call("顺便把文件删了"), AIMessage("完成")])
    explorer_model = ScriptedModel(
        replies=[tool_call("run_shell", "s1", command="rm -rf shop"), AIMessage("我没有权限删除")]
    )
    app = build_lead(lead_model, tools(), explore_model=explorer_model, checkpointer=InMemorySaver())
    await run(app)
    assert EXECUTED == []
    assert await pending_approvals(app, config()) == []  # 角色边界直接拒绝，不打扰人


async def test_review_returns_structured_verdict():
    lead_model = ScriptedModel(replies=[tool_call("review", "v1", summary="改了 pricing.py"), AIMessage("完成")])
    reviewer_model = tool_call("Review", "rv", passed=False, issues=["没处理负数折扣"], summary="还有问题")
    app = build_lead(
        lead_model, tools(), explore_model=ScriptedModel(replies=[AIMessage("x")]), checkpointer=InMemorySaver()
    )
    # review 用主模型：把审查回复接在主模型脚本里
    lead_model.replies.insert(1, reviewer_model)
    state = await run(app)
    verdict = next(m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.name == "review")
    assert verdict.content.startswith("审查未通过") and "没处理负数折扣" in verdict.content


async def test_dangerous_call_by_lead_still_needs_approval():
    lead_model = ScriptedModel(replies=[tool_call("run_shell", "s1", command="rm -rf build"), AIMessage("清理完了")])
    app = build_lead(lead_model, tools(), checkpointer=InMemorySaver())
    await app.ainvoke({"messages": [("user", "清理 build")]}, config())
    [pending] = await pending_approvals(app, config())
    assert pending.value["tool"] == "run_shell"
    assert EXECUTED == []


async def test_long_explore_answer_is_capped():
    lead_model = ScriptedModel(replies=[explore_call("全部列出来"), AIMessage("完成")])
    explorer_model = ScriptedModel(replies=[AIMessage("长" * (EXPLORE_RESULT_LIMIT * 2))])
    app = build_lead(lead_model, tools(), explore_model=explorer_model, checkpointer=InMemorySaver())
    state = await run(app)
    result = next(m for m in state.values["messages"] if isinstance(m, ToolMessage) and m.name == "explore")
    assert len(result.content) < EXPLORE_RESULT_LIMIT + 50 and "截断" in result.content


async def test_explorer_is_forced_to_answer_after_step_budget():
    from fakes import ScriptedModel
    from langchain_core.messages import AIMessage

    from ai_agent_test.app.code_agent.agent.lead import EXPLORE_MAX_ROUNDS
    from ai_agent_test.evals.cases import CASES
    from ai_agent_test.evals.runner import run_cases

    class Probe(ScriptedModel):
        bound: int = -1  # 本次调用拿到了几个工具；没 bind = -1

        def bind_tools(self, tools, **kwargs):
            return self.model_copy(update={"bound": len(tools)})

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatGeneration, ChatResult

            if self.bound <= 0:
                reply = AIMessage("规范要求先用 pwd 确认目录")
            else:
                n = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
                reply = AIMessage("", tool_calls=[{"name": "query_rag", "args": {"query": f"q{n}"}, "id": f"e{n}"}])
            return ChatResult(generations=[ChatGeneration(message=reply)])

    def lead():
        return ScriptedModel(
            replies=[
                AIMessage("", tool_calls=[{"name": "explore", "args": {"question": "终端规范"}, "id": "x1"}]),
                AIMessage("规范要求先用 pwd 确认目录"),
            ]
        )

    case = next(c for c in CASES if c.id == "E06")
    session = await run_cases(
        [case], targets=["lead"], model_factory=lead, explore_model_factory=lambda: Probe(replies=[AIMessage("")])
    )
    attempt = session.targets[0].runs[0].attempts[0]
    assert attempt.result.actual_calls == ["query_rag"] * EXPLORE_MAX_ROUNDS
    assert attempt.delegations == {"explore": 1}
