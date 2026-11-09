"""按角色收窄工具权限：只读角色在代码层面就拿不到写能力。

之前 research_expert 和 code_expert 拿的是同一套工具，"方案专家不写代码"
只是 system prompt 里的一句话。模型不听话时没有任何东西拦它。

``read_only(tools)`` 给只读角色（方案、审查）用：

- 默认等级为 read 的工具：原样保留；
- 等级由内容决定的工具（``run_shell``、``mysql_execute_query``）：保留，但每次调用
  先判定内容，不是 read 就直接拒绝并告诉模型交给谁——``ls`` 能用，``rm`` 不行；
- 其余工具（写文件、建目录、删数据……）：直接不给，模型根本看不到。

拒绝不走人工审批：这是角色边界，不是风险确认，问人"方案专家能不能删表"没有意义。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from ai_agent_test.guardrails.levels import RiskLevel
from ai_agent_test.guardrails.policy import assess, baseline_level, is_content_decided

ROLE_DENIED_TEMPLATE = (
    "⛔ 当前角色只有只读权限，{tool} 的这次调用未执行（{reason}）。"
    "需要修改时，请把具体操作写进方案或审查意见，由负责实现的角色执行。"
)


def _normalize(tool: BaseTool, result: Any) -> Any:
    if tool.response_format == "content_and_artifact" and isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


def _read_only_view(tool: BaseTool) -> BaseTool:
    async def _run(**kwargs: Any) -> Any:
        decision = assess(tool.name, kwargs)
        if decision.level is not RiskLevel.READ:
            return ROLE_DENIED_TEMPLATE.format(tool=tool.name, reason=decision.reason)
        return _normalize(tool, await tool.ainvoke(kwargs))

    def _run_sync(**kwargs: Any) -> Any:
        decision = assess(tool.name, kwargs)
        if decision.level is not RiskLevel.READ:
            return ROLE_DENIED_TEMPLATE.format(tool=tool.name, reason=decision.reason)
        return _normalize(tool, tool.invoke(kwargs))

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_run_sync,
        coroutine=_run,
        metadata=tool.metadata,
        tags=tool.tags,
    )


def read_only(tools: Sequence[BaseTool]) -> list[BaseTool]:
    """只读角色可用的工具子集。"""
    selected: list[BaseTool] = []
    for tool in tools:
        if is_content_decided(tool.name):
            selected.append(_read_only_view(tool))
        elif baseline_level(tool.name) is RiskLevel.READ:
            selected.append(tool)
    return selected
