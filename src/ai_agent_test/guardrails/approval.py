"""人工审批器：destructive 操作在执行前要拿到"是"。

三种实现对应三种运行场景：

- ``InteractiveApprover`` 人工审批，用于 `agent chat`（真实使用）
- ``DenyAllApprover``    一律拒绝，用于评测与 CI（保证零副作用、结果可重复）
- ``AllowAllApprover``   一律放行，仅用于本地已确认安全的批处理（显式开启）

``InteractiveApprover`` 第一版是在工具内部 ``input()`` 阻塞等终端输入，问题有三个：

1. 只能在终端里用，接不了 Web 前端，审批弹窗无从谈起；
2. 等待中进程一断，这次审批和图的执行进度一起丢失；
3. 审批是工具里的副作用，图的状态里看不到"正在等谁批什么"。

现在改用 LangGraph 的 ``interrupt()``：工具在执行前把审批请求抛给图，图连同
checkpoint 一起挂起；调用方读出待审批项、拿到人的决定后用 ``Command(resume=...)``
恢复。审批成了图状态的一部分：可以跨进程恢复，也可以由任何前端来回答。
驱动这套循环的代码在 ``hitl.py``。
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ai_agent_test.guardrails.policy import Decision, describe

#: 审批请求的类型标记，用来在图的所有 interrupt 中区分出护栏发起的那些
APPROVAL_KIND = "guard_approval"


class Approver:
    name = "base"

    async def approve(self, decision: Decision) -> bool:  # pragma: no cover - 接口
        raise NotImplementedError


class DenyAllApprover(Approver):
    name = "deny-all"

    async def approve(self, decision: Decision) -> bool:
        return False


class AllowAllApprover(Approver):
    name = "allow-all"

    async def approve(self, decision: Decision) -> bool:
        return True


def approval_request(decision: Decision) -> dict[str, Any]:
    """interrupt 的负载。必须可 JSON 序列化：它会随 checkpoint 落进 Redis / MongoDB。"""
    return {
        "kind": APPROVAL_KIND,
        "tool": decision.tool,
        "level": decision.level.value,
        "reason": decision.reason,
        "preview": decision.preview,
        "prompt": describe(decision),
    }


def parse_answer(answer: Any) -> bool:
    """把 resume 值归一成"批/不批"。认不出来的一律当作不批。"""
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, dict):
        return parse_answer(answer.get("approved", False))
    if isinstance(answer, str):
        return answer.strip().lower() in {"y", "yes", "true", "1", "approve", "approved"}
    return False


class InteractiveApprover(Approver):
    """通过 LangGraph ``interrupt()`` 请求人工审批。

    注意 interrupt 的语义：恢复时 **整个工具会从头再执行一遍**，这次 interrupt()
    直接返回人给的答案。所以 interrupt 之前不能有任何副作用——包装层保证了
    这一点：判定是纯函数，真实工具在拿到批准之后才会被调用。
    """

    name = "interactive"

    async def approve(self, decision: Decision) -> bool:
        try:
            answer = interrupt(approval_request(decision))
        except (RuntimeError, KeyError):
            # 不在图里运行（直接 tool.ainvoke）：没有 runnable 上下文时是 RuntimeError，
            # 有上下文但不是图的节点时是 KeyError（缺少 pregel scratchpad）。
            # 两种情况都没有人能回答，按拒绝处理，宁可不做。
            # 注意 GraphInterrupt 不是这两类异常，正常的挂起不会被吞掉。
            return False
        return parse_answer(answer)


def build_approver(mode: str) -> Approver:
    return {
        "interactive": InteractiveApprover,
        "deny": DenyAllApprover,
        "allow": AllowAllApprover,
    }.get(mode, InteractiveApprover)()
