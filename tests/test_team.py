"""角色权限与多智能体编排的离线测试。

用一个 ScriptedModel 驱动 planner / coder / reviewer（或 supervisor 下的专家）：
回复按图的执行顺序依次消耗，所以每个测试的脚本就是一次完整的协作过程。
"""

from __future__ import annotations

from fakes import ScriptedModel, tool_call
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_test.app.code_agent.agent.team import build_supervisor, build_team
from ai_agent_test.guardrails import InteractiveApprover, NullAuditLog, drive, guard_tool, pending_approvals, read_only

EXECUTED: list[str] = []


@tool
def run_shell(command: str) -> str:
    """run a shell command"""
    EXECUTED.append(command)
    return f"executed {command}"


@tool
def mysql_list_tables(database: str) -> str:
    """获取指定数据库中的所有表"""
    return "user, orders"


@tool
def mysql_delete_data(database: str, table: str, data: dict[str, str]) -> str:
    """根据条件删除指定数据库表中的数据"""
    EXECUTED.append(f"delete {table} {data}")
    return "1 row deleted"


@tool
def write_file(file_path: str, text: str) -> str:
    """Write file to disk"""
    EXECUTED.append(f"write {file_path}")
    return "ok"


def guarded_tools():
    approver, audit = InteractiveApprover(), NullAuditLog()
    return [guard_tool(t, approver=approver, audit=audit) for t in (run_shell, mysql_list_tables, write_file)]


def plan(needs_changes: bool = True, answer: str = "") -> AIMessage:
    return tool_call(
        "Plan", "plan", needs_changes=needs_changes, goal="生成 out 目录", steps=["mkdir out"], answer=answer
    )


def review(passed: bool, *issues: str) -> AIMessage:
    return tool_call("Review", f"review-{passed}-{len(issues)}", passed=passed, issues=list(issues), summary="核实完毕")


def config(thread: str = "t1") -> dict:
    return {"configurable": {"thread_id": thread}}


def setup_function():
    EXECUTED.clear()


async def run(app, task: str = "建一个 out 目录"):
    async def approve(_request: dict) -> bool:
        return True

    async for _ in drive(app, {"messages": [("user", task)]}, config(), approve, stream_mode="updates"):
        pass
    return await app.aget_state(config())


# --------------------------------------------------------------------------- #
# 角色权限
# --------------------------------------------------------------------------- #
def test_read_only_drops_write_tools_and_keeps_content_decided_ones():
    names = [t.name for t in read_only([mysql_list_tables, mysql_delete_data, write_file, run_shell])]
    assert names == ["mysql_list_tables", "run_shell"]


async def test_read_only_shell_allows_ls_but_refuses_rm():
    shell = next(t for t in read_only([run_shell]) if t.name == "run_shell")
    assert "executed ls" in await shell.ainvoke({"command": "ls -la"})

    refused = await shell.ainvoke({"command": "rm -rf /srv/app"})
    assert "只读权限" in refused
    assert EXECUTED == ["ls -la"]


async def test_read_only_view_keeps_tool_contract():
    shell = read_only([run_shell])[0]
    assert shell.description == run_shell.description
    assert shell.args_schema.model_json_schema() == run_shell.args_schema.model_json_schema()


# --------------------------------------------------------------------------- #
# Team：planner → coder → reviewer
# --------------------------------------------------------------------------- #
async def test_team_happy_path_plans_codes_and_verifies():
    model = ScriptedModel(
        replies=[
            plan(),  # planner 提交方案
            tool_call("run_shell", "c1", command="mkdir -p out"),  # coder 实现
            AIMessage("已创建 out 目录"),
            tool_call("run_shell", "r1", command="ls"),  # reviewer 自己核实
            review(True),
        ]
    )
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    state = await run(app)

    assert EXECUTED == ["mkdir -p out", "ls"]
    assert state.values["attempts"] == 1
    assert state.values["review"]["passed"] is True
    assert state.values["messages"][-1].content.startswith("已完成")
    assert [m.name for m in state.values["messages"][1:]] == ["planner", "coder", "reviewer", "team"]


async def test_team_retries_with_review_issues():
    model = ScriptedModel(
        replies=[
            plan(),
            AIMessage("做完了"),  # 第一轮：coder 什么都没做就说做完了
            review(False, "out 目录不存在"),
            tool_call("run_shell", "c2", command="mkdir -p out"),  # 第二轮：按审查意见修正
            AIMessage("已创建 out 目录"),
            review(True),
        ]
    )
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    state = await run(app)

    assert state.values["attempts"] == 2
    assert EXECUTED == ["mkdir -p out"]
    assert state.values["review"]["passed"] is True


async def test_team_stops_after_max_attempts():
    model = ScriptedModel(
        replies=[plan(), AIMessage("做完了"), review(False, "没做"), AIMessage("真的做完了"), review(False, "还是没做")]
    )
    app = build_team(model, guarded_tools(), max_attempts=2, checkpointer=InMemorySaver())
    state = await run(app)

    assert state.values["attempts"] == 2
    final = state.values["messages"][-1].content
    assert "2 轮后审查仍未通过" in final
    assert "还是没做" in final


async def test_team_answers_directly_when_no_changes_needed():
    model = ScriptedModel(replies=[plan(needs_changes=False, answer="3 加 5 乘以 12 等于 63")])
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    state = await run(app, "3 加 5 乘以 12 等于多少")

    assert state.values["messages"][-1].content == "3 加 5 乘以 12 等于 63"
    assert state.values["attempts"] == 0  # coder 没被叫起来
    assert EXECUTED == []


async def test_team_planner_cannot_write_even_if_it_tries():
    model = ScriptedModel(
        replies=[
            tool_call("run_shell", "p1", command="rm -rf /srv/app"),  # planner 越权
            plan(needs_changes=False, answer="不需要改动"),
        ]
    )
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    state = await run(app)

    assert EXECUTED == []
    assert await pending_approvals(app, config()) == []  # 角色边界直接拒绝，不打扰人
    assert state.values["messages"][-1].content == "不需要改动"


async def test_team_coder_destructive_call_goes_through_human_approval():
    model = ScriptedModel(
        replies=[
            plan(),
            tool_call("run_shell", "c1", command="rm -rf build"),
            AIMessage("已清理"),
            review(True),
        ]
    )
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    asked: list[str] = []

    async def ask(request: dict) -> bool:
        asked.append(request["preview"])
        return True

    async for _ in drive(
        app, {"messages": [("user", "清理 build")]}, config(), ask, stream_mode="updates", subgraphs=True
    ):
        pass

    assert asked == ["rm -rf build"]
    assert EXECUTED == ["rm -rf build"]  # 恢复后只执行一次


async def test_team_without_structured_review_counts_as_failed():
    model = ScriptedModel(replies=[plan(), AIMessage("做完了"), AIMessage("看起来没问题")])
    app = build_team(model, guarded_tools(), max_attempts=1, checkpointer=InMemorySaver())
    state = await run(app)

    assert state.values["review"]["passed"] is False
    assert "审查未给出结构化结论" in state.values["review"]["issues"]


# --------------------------------------------------------------------------- #
# Supervisor：research_expert 只读
# --------------------------------------------------------------------------- #
async def test_supervisor_research_expert_is_read_only():
    model = ScriptedModel(
        replies=[
            tool_call("transfer_to_research_expert", "t1"),
            tool_call("run_shell", "r1", command="rm -rf /srv/app"),  # 方案专家越权
            AIMessage("方案：交给 code_expert"),
            AIMessage("结束"),
        ]
    )
    app = build_supervisor(model, guarded_tools(), checkpointer=InMemorySaver())
    await run(app)
    assert EXECUTED == []


# --------------------------------------------------------------------------- #
# 交接：调研摘录直接交给实现者
# --------------------------------------------------------------------------- #
def _human_text(messages) -> str:
    from langchain_core.messages import HumanMessage

    return "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))


async def test_plan_context_is_handed_to_coder_but_not_reviewer():
    from fakes import RecordingScriptedModel

    excerpt = {
        "path": "shop/pricing.py",
        "content": "def apply_discount(price, discount):\n    ...",
        "note": "要修改的文件",
    }
    model = RecordingScriptedModel(
        replies=[
            tool_call(
                "Plan", "plan", needs_changes=True, goal="修 bug", steps=["改 apply_discount"], context=[excerpt]
            ),
            AIMessage("改好了"),
            review(True),
        ]
    )
    app = build_team(model, guarded_tools(), checkpointer=InMemorySaver())
    state = await run(app)

    planner_prompt, coder_prompt, reviewer_prompt = (_human_text(m) for m in model.seen)
    assert "调研摘录（原文）" in coder_prompt and "def apply_discount(price, discount)" in coder_prompt
    assert "def apply_discount(price, discount)" not in reviewer_prompt  # 审查要独立核实
    assert "附调研摘录 1 段" in reviewer_prompt
    assert state.values["plan"]["context"][0]["path"] == "shop/pricing.py"


def test_context_is_capped():
    from ai_agent_test.app.code_agent.agent.team import _format_context

    plan = {"context": [{"path": "a.py", "content": "x" * 100}, {"path": "b.py", "content": "y" * 100}]}
    text = _format_context(plan, limit=120)
    assert "x" * 100 in text and "y" * 20 in text and "y" * 21 not in text
    assert "截断" in text
    assert "还有摘录" in _format_context({"context": plan["context"] * 2}, limit=100)
