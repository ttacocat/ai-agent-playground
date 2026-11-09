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
from ai_agent_test.guardrails.policy import Decision, assess

OutcomeHook = Callable[[Decision, str], None]

BLOCKED_TEMPLATE = (
    "⛔ 操作被安全护栏拦截：{tool}（风险等级 {level}，原因：{reason}）。\n"
    "本次调用未执行，未产生任何影响。请改用更安全的方式；"
    "若确需执行，请提示用户在终端手动完成。"
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
) -> BaseTool:
    approver = approver or DenyAllApprover()
    audit = audit or NullAuditLog()

    async def _run(**kwargs: Any) -> Any:
        decision = assess(tool.name, kwargs)

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
        return _blocked_message(decision, denied=outcome == "denied")

    def _run_sync(**kwargs: Any) -> Any:
        # 同步路径极少用到（agent 走 ainvoke），但保留以免 StructuredTool 校验收不到 func
        decision = assess(tool.name, kwargs)
        if not decision.needs_approval:
            audit.write(decision, "executed")
            if on_decision:
                on_decision(decision, "executed")
            return _normalize_result(tool, tool.invoke(kwargs))
        audit.write(decision, "blocked")
        if on_decision:
            on_decision(decision, "blocked")
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
) -> list[BaseTool]:
    return [guard_tool(t, approver=approver, audit=audit, on_decision=on_decision) for t in tools]
