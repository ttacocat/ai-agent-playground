"""演示模式：不调用任何模型 API，也能完整走一遍 方案 → 实现 → 人工审批 → 审查。

用于前端开发、截图和面试演示。``DemoModel`` 不是随机回复：它看 system prompt
判断自己在扮演 team 里的哪个角色，看已经拿到的工具结果决定下一步，
工具则落在评测用的 ``StubWorld`` 上（内存 SQLite + 目录集合），状态是真的会变的。

支持两类请求：
- 含"删"：删除 user 表 id=1 的记录。coder 的删除会触发人工审批；批准后 reviewer 用 SELECT
  核实，拒绝则如实报告"用户拒绝了删除"；
- 其他：只读问答，planner 查表结构后直接回答，不进入实现和审查。
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_agent_test.evals.stub_tools import build_stub_tools
from ai_agent_test.evals.world import StubWorld
from ai_agent_test.guardrails import InteractiveApprover, NullAuditLog, guard_tools

STEP_DELAY = 0.6  # 让流式效果看得见


def _call(name: str, call_id: str, **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


class DemoModel(BaseChatModel):
    delay: float = STEP_DELAY

    @property
    def _llm_type(self) -> str:
        return "demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DemoModel:
        return self

    # ------------------------------------------------------------------ #
    def _reply(self, messages: list[BaseMessage]) -> AIMessage:
        system = next((str(m.content) for m in messages if isinstance(m, SystemMessage)), "")
        last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
        task = str(messages[last_human].content)
        results = [m for m in messages[last_human:] if isinstance(m, ToolMessage)]
        n = len(results)
        wants_delete = "删" in task

        if "方案专家" in system:
            if n == 0:
                return _call("mysql_describe_tables", f"p{n}", database="test", table="user")
            if wants_delete:
                return _call(
                    "Plan",
                    "plan",
                    needs_changes=True,
                    goal="删除 test 库 user 表中 id=1 的记录",
                    steps=["用 mysql_delete_data 按主键删除 id=1", "SELECT 核实记录已不存在、其他记录未受影响"],
                    files=["test.user"],
                    risks=["删除不可回滚，执行前需要用户确认"],
                    context=[{"path": "test.user", "content": str(results[-1].content), "note": "表结构"}],
                )
            fields = results[-1].content.replace("\n", "、")
            return _call(
                "Plan",
                "plan",
                needs_changes=False,
                goal="说明 user 表结构",
                answer=f"test 库的 user 表有这些字段：{fields}。",
            )

        if "实现专家" in system:
            if n == 0:
                return _call("mysql_delete_data", f"c{n}", database="test", table="user", data={"id": "1"})
            if str(results[-1].content).startswith("⛔"):
                return AIMessage("删除操作被用户拒绝，没有做任何修改。")
            return AIMessage(f"已按方案删除 id=1 的记录（{results[-1].content}）。")

        if "负责从头到尾完成任务" in system:  # lead 主 agent
            last = results[-1] if results else None
            if last is None:
                question = (
                    "删除前确认：test 库 user 表的结构，以及 id=1 这条记录的内容"
                    if wants_delete
                    else "test 库 user 表有哪些字段、各是什么类型"
                )
                return _call("explore", "x0", question=question)
            if last.name == "explore" and wants_delete:
                return _call("mysql_delete_data", "c0", database="test", table="user", data={"id": "1"})
            if last.name == "explore":
                return AIMessage(f"根据探索结果：{last.content}")
            if last.name == "mysql_delete_data":
                if str(last.content).startswith("⛔"):
                    return AIMessage("删除操作被用户拒绝，没有做任何修改。")
                return _call(
                    "review", "v0", summary="按主键删除了 test 库 user 表中 id=1 的记录，请核实其余记录未受影响"
                )
            return AIMessage(f"已删除 id=1 的记录。{last.content}")

        if "探索助手" in system:
            if n == 0:
                return _call("mysql_describe_tables", f"e{n}", database="test", table="user")
            if wants_delete and n == 1:
                return _call("mysql_execute_query", f"e{n}", command="SELECT * FROM user WHERE id = 1", database="test")
            fields = results[0].content.replace("\n", "、")
            row = f"；id=1 的记录：{results[1].content.splitlines()[1]}" if wants_delete and n > 1 else ""
            return AIMessage(f"user 表字段：{fields}{row}。")

        if "审查专家" in system or "审查助手" in system:
            if n == 0:
                return _call(
                    "mysql_execute_query", f"r{n}", command="SELECT id, name FROM user ORDER BY id", database="test"
                )
            if "1 | 张三" not in str(results[-1].content):
                return _call("Review", "review", passed=True, summary="id=1 已删除，其余记录完好", issues=[])
            return _call(
                "Review",
                "review",
                passed=False,
                summary="记录仍然存在",
                issues=["id=1（张三）仍在 user 表中：删除被用户拒绝"],
            )

        return AIMessage("（演示模式只支持 lead 和 team 架构）")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._reply(list(messages)))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        await asyncio.sleep(self.delay)
        return self._generate(messages, stop, run_manager, **kwargs)


def demo_tools(world: StubWorld | None = None):
    """绑定到一个 StubWorld 的 stub 工具，套上真实护栏（审批走 interrupt）。"""
    world = world or StubWorld()
    return world, guard_tools(build_stub_tools(world), approver=InteractiveApprover(), audit=NullAuditLog())
