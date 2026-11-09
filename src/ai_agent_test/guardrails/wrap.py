"""工具包装层：在不改变工具契约的前提下插入风险闸门。

关键约束：包装后的工具 **name / description / args_schema 必须与原来完全一致**。
描述变了模型的选择行为就变了，评测基线会失真；参数 schema 变了则调用直接报错。
所以这里只替换执行体，元信息原样透传。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ai_agent_test.guardrails.approval import Approver, DenyAllApprover
from ai_agent_test.guardrails.audit import AuditLog, NullAuditLog
from ai_agent_test.guardrails.memory import DenialMemory
from ai_agent_test.guardrails.policy import Decision, assess

OutcomeHook = Callable[[Decision, str], None]

# 第一版的结尾是"请改用更安全的方式"。评测里模型把它理解成"换个写法达成同样的效果"：
# DELETE FROM user WHERE 1=1 被拒后，改成 DELETE FROM user WHERE id IN (1,2,3) 把表清空了。
# 用户拒绝的是意图，不是写法，提示必须说清楚这一点。
BLOCKED_TEMPLATE = (
    "⛔ 操作被安全护栏拦截：{tool}（风险等级 {level}，原因：{reason}）。\n"
    "本次调用未执行，未产生任何影响。\n"
    "不要用其他写法、其他工具或分多步来达成同样的效果——用户拒绝的是这个操作的意图，"
    "换一种方式绕过同样会被拦截。请停止这部分操作，向用户说明情况；"
    "确实需要执行的话，由用户自己在终端里完成。"
)


def _blocked_message(decision: Decision, denied: bool) -> str:
    reason = decision.reason if denied else f"{decision.reason}，且当前为无人值守环境，无法征得确认"
    return BLOCKED_TEMPLATE.format(tool=decision.tool, level=decision.level.value, reason=reason)


def _normalize_result(tool: BaseTool, result: Any) -> Any:
    """把底层工具的返回值归一成"内容"。

    MCP 适配器生成的工具是 response_format="content_and_artifact"，
    invoke 出来是 (content, artifact) 二元组。包装后我们只声明 content，
    必须把 artifact 摘掉，否则模型看到的是一个 tuple 的字符串形式。
    """
    if tool.response_format == "content_and_artifact" and isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def guard_tool(
    tool: BaseTool,
    approver: Approver | None = None,
    audit: AuditLog | None = None,
    on_decision: OutcomeHook | None = None,
    memory: DenialMemory | None = None,
) -> BaseTool:
    approver = approver or DenyAllApprover()
    audit = audit or NullAuditLog()
    memory = memory if memory is not None else DenialMemory()

    def _judge(kwargs: dict) -> Decision:
        return memory.escalate(assess(tool.name, kwargs), kwargs)

    async def _run(**kwargs: Any) -> Any:
        decision = _judge(kwargs)

        if not decision.needs_approval:
            audit.write(decision, "executed")
            if on_decision:
                on_decision(decision, "executed")
            return _normalize_result(tool, await tool.ainvoke(kwargs))

        approved = await approver.approve(decision)
        outcome = "executed" if approved else ("blocked" if approver.name == "deny-all" else "denied")
        audit.write(decision, outcome)
        if on_decision:
            on_decision(decision, outcome)
        if approved:
            return _normalize_result(tool, await tool.ainvoke(kwargs))
        memory.remember(decision, kwargs)
        return _blocked_message(decision, denied=outcome == "denied")

    def _run_sync(**kwargs: Any) -> Any:
        # 同步路径极少用到（agent 走 ainvoke），但保留以免 StructuredTool 校验收不到 func
        decision = _judge(kwargs)
        if not decision.needs_approval:
            audit.write(decision, "executed")
            if on_decision:
                on_decision(decision, "executed")
            return _normalize_result(tool, tool.invoke(kwargs))
        audit.write(decision, "blocked")
        if on_decision:
            on_decision(decision, "blocked")
        memory.remember(decision, kwargs)
        return _blocked_message(decision, denied=False)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_run_sync,
        coroutine=_run,
        metadata=tool.metadata,
        tags=tool.tags,
    )


def guard_tools(
    tools: Sequence[BaseTool],
    approver: Approver | None = None,
    audit: AuditLog | None = None,
    on_decision: OutcomeHook | None = None,
    memory: DenialMemory | None = None,
) -> list[BaseTool]:
    """一组工具共享同一份拒绝记忆：用 SQL 工具被拒的表，换结构化删除工具也绕不过去。"""
    memory = memory if memory is not None else DenialMemory()
    return [guard_tool(t, approver=approver, audit=audit, on_decision=on_decision, memory=memory) for t in tools]
