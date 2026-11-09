"""评测流水线的离线测试：假世界、新判定维度、多架构运行与报告。

用 ScriptedModel 代替真实 LLM，验证的是评测器本身：状态有没有真的变、
调用有没有记对、失败能不能被发现。
"""

from __future__ import annotations

import pytest
from fakes import ScriptedModel, tool_call
from langchain_core.messages import AIMessage

from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.engine import EvalCase, evaluate
from ai_agent_test.evals.runner import EvalSession, render_report, run_cases
from ai_agent_test.evals.stub_tools import build_stub_tools
from ai_agent_test.evals.world import StubWorld

CASE_BY_ID = {c.id: c for c in CASES}


def tools_of(world: StubWorld) -> dict:
    return {t.name: t for t in build_stub_tools(world)}


# --------------------------------------------------------------------------- #
# StubWorld
# --------------------------------------------------------------------------- #
def test_world_runs_mysql_dialect_against_sqlite():
    world = StubWorld()
    assert "3" in world.run_sql("SELECT COUNT(*) FROM `user`")
    assert "王五" in world.run_sql("SELECT name FROM user ORDER BY age DESC LIMIT 1")
    assert world.run_sql("SHOW TABLES").split() == ["orders", "product", "user"]


def test_world_sql_changes_state():
    world = StubWorld()
    world.run_sql("DELETE FROM user WHERE age < 18")
    assert {r[0] for r in world.rows("user")} == {1, 3}
    world.run_sql("DROP TABLE orders")
    assert not world.has_table("orders")


def test_world_sql_errors_are_readable():
    assert StubWorld().run_sql("SELECT * FROM nope").startswith("ERROR")


def test_world_filesystem():
    world = StubWorld()
    world.mkdir("/srv/a/b")
    assert {"/srv/a", "/srv/a/b"} <= world.dirs
    world.run_shell("rm -rf /srv/app")
    assert "/srv/app" not in world.dirs and "/srv/app/static" not in world.dirs
    assert world.list_dir("/srv") == "a"


async def test_stub_tools_mutate_world():
    world = StubWorld()
    tools = tools_of(world)
    await tools["mysql_delete_data"].ainvoke({"database": "test", "table": "user", "data": {"id": "1"}})
    await tools["mysql_insert_data"].ainvoke(
        {"database": "test", "table": "user", "data": {"name": "赵六", "age": "28"}}
    )
    await tools["mysql_update_data"].ainvoke(
        {"database": "test", "table": "user", "data": {"age": "18"}, "where": {"name": "李四"}}
    )
    await tools["make_dir_in_vm"].ainvoke({"dir_path": "/srv/web"})
    names = {r[1]: r[2] for r in world.rows("user")}
    assert "张三" not in names and names["赵六"] == 28 and names["李四"] == 18
    assert "/srv/web" in world.dirs


async def test_stub_tools_reject_unknown_database():
    result = await tools_of(StubWorld())["mysql_list_tables"].ainvoke({"database": "prod"})
    assert "Unknown database" in result


def test_run_shell_stub_description_matches_real_tool():
    """描述一变，模型选工具的行为就变；stub 和真实工具的描述必须一字不差。"""
    import re
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/ai_agent_test/app/code_agent/mcp/shell_tools.py").read_text("utf-8")
    block = re.search(r'name="run_shell",\s*description=\((.*?)\),', source, re.S).group(1)
    real = "".join(re.findall(r'"([^"]*)"', block))
    assert tools_of(StubWorld())["run_shell"].description == real


def test_each_world_is_isolated():
    first, second = StubWorld(), StubWorld()
    first.run_sql("DELETE FROM user")
    assert len(second.rows("user")) == 3


# --------------------------------------------------------------------------- #
# 判定引擎的新维度
# --------------------------------------------------------------------------- #
def test_max_calls():
    case = EvalCase(id="T1", task="t", max_calls=2)
    assert evaluate(case, ["a", "b"]).passed
    assert "超过上限" in evaluate(case, ["a", "b", "c"]).reasons[0]


def test_expect_answer_requires_every_keyword():
    case = EvalCase(id="T2", task="t", expect_answer=("王五", "35"))
    assert evaluate(case, [], answer="年龄最大的是王五，35 岁").passed
    assert "回答缺少关键信息" in evaluate(case, [], answer="是王五").reasons[0]


def test_state_check_reads_world():
    case = EvalCase(id="T3", task="t", check=lambda w: None if w.has_table("user") else "user 没了")
    world = StubWorld()
    assert evaluate(case, [], world=world).passed
    world.run_sql("DROP TABLE user")
    assert "user 没了" in evaluate(case, [], world=world).reasons[0]


def test_state_check_without_world_fails_loudly():
    case = EvalCase(id="T4", task="t", check=lambda w: None)
    assert "没有提供 StubWorld" in evaluate(case, []).reasons[0]


# --------------------------------------------------------------------------- #
# 用例集本身的质量
# --------------------------------------------------------------------------- #
def test_case_set_size():
    assert len(CASES) >= 30


@pytest.mark.parametrize("case", [c for c in CASES if c.check], ids=lambda c: c.id)
def test_state_checks_are_meaningful(case):
    """安全类用例：什么都不做应该通过（没被破坏）；任务类用例：什么都不做必须失败。

    任务类用例在原始世界上就能通过，说明它的状态检查形同虚设。
    """
    world = StubWorld()
    if case.setup:
        case.setup(world)
    verdict = case.check(world)
    if "safety" in case.tags:
        assert verdict is None, f"{case.id} 在原始世界上就判失败：{verdict}"
    else:
        assert verdict is not None, f"{case.id} 什么都不做也能通过，检查没有意义"


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
async def targets_of(*args, **kwargs):
    return (await run_cases(*args, **kwargs)).targets


def scripted(*replies):
    return lambda: ScriptedModel(replies=list(replies))


async def test_single_target_approved_delete_changes_state_and_passes():
    model = scripted(
        tool_call("mysql_delete_data", "c1", database="test", table="user", data={"id": "1"}),
        AIMessage("已删除 id=1 的记录"),
    )
    [report] = await targets_of([CASE_BY_ID["E03"]], targets=["single"], model_factory=model)
    result = report.runs[0].attempts[0].result
    assert result.passed, result.reasons
    assert result.actual_calls == ["mysql_delete_data"]
    assert result.destructive_executed == []  # 人批准过的不算漏放


async def test_claiming_success_without_doing_it_fails():
    """第一版评测里这种"嘴上说删了"的情况只要调对工具就会通过。"""
    model = scripted(AIMessage("已删除 id=1 的记录"))
    [report] = await targets_of([CASE_BY_ID["E03"]], targets=["single"], model_factory=model)
    reasons = "；".join(report.runs[0].attempts[0].result.reasons)
    assert "缺少必要工具调用" in reasons
    assert "最终状态不符合预期" in reasons


async def test_guard_blocks_drop_and_world_stays_intact():
    model = scripted(
        tool_call("mysql_execute_query", "c1", command="DROP TABLE user", database="test"),
        AIMessage("该操作被拦截了"),
    )
    [report] = await targets_of([CASE_BY_ID["E13"]], targets=["single"], model_factory=model)
    result = report.runs[0].attempts[0].result
    assert result.passed, result.reasons
    assert result.blocked_calls == ["mysql_execute_query"]


async def test_team_target_records_nested_calls_once():
    """planner 的只读视图 → 护栏 → stub 三层包装，调用只记一次。"""
    model = scripted(
        tool_call("mysql_execute_query", "q1", command="SELECT COUNT(*) FROM user", database="test"),
        tool_call("Plan", "p1", needs_changes=False, goal="统计", answer="user 表共有 3 条记录"),
    )
    [report] = await targets_of([CASE_BY_ID["E14"]], targets=["team"], model_factory=model)
    result = report.runs[0].attempts[0].result
    assert result.passed, result.reasons
    assert result.actual_calls == ["mysql_execute_query"]


async def test_supervisor_target_runs_through_subagents():
    model = scripted(
        tool_call("transfer_to_research_expert", "t1"),
        tool_call("mysql_list_tables", "c1", database="test"),
        AIMessage("test 库有 user、orders、product 三张表"),
        AIMessage("test 库有 user、orders、product 三张表"),
    )
    [report] = await targets_of([CASE_BY_ID["E01"]], targets=["supervisor"], model_factory=model)
    result = report.runs[0].attempts[0].result
    assert result.passed, result.reasons
    assert result.actual_calls == ["mysql_list_tables"]


async def test_repeat_aggregates_pass_at_k_and_pass_hat_k():
    replies = iter(
        [
            [AIMessage("63")],  # 通过
            [tool_call("run_shell", "c1", command="echo $((3+5*12))"), AIMessage("63")],  # 调了工具：失败
            [AIMessage("63")],  # 通过
        ]
    )
    [report] = await targets_of(
        [CASE_BY_ID["E10"]], targets=["single"], repeat=3, model_factory=lambda: ScriptedModel(replies=next(replies))
    )
    run = report.runs[0]
    assert (run.k, run.passes) == (3, 2)
    assert run.pass_at_k and not run.pass_hat_k
    assert report.pass_hat_k == 0

    text = render_report(EvalSession(model="scripted", repeat=3, targets=[report]))
    assert "## 总览" in text and "| single |" in text and "2/3" in text
    assert "未稳定通过的用例" in text


async def test_exceptions_are_recorded_not_raised():
    class Boom(ScriptedModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("模型挂了")

    [report] = await targets_of([CASE_BY_ID["E01"]], targets=["single"], model_factory=lambda: Boom(replies=[]))
    assert "模型挂了" in report.runs[0].attempts[0].result.reasons[0]
