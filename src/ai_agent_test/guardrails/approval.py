"""人工审批器：destructive 操作在执行前要拿到"是"。

三种实现对应三种运行场景：

- ``InteractiveApprover`` 终端里 y/n，用于 `agent chat`（真实使用）
- ``DenyAllApprover``    一律拒绝，用于评测与 CI（保证零副作用、结果可重复）
- ``AllowAllApprover``   一律放行，仅用于本地已确认安全的批处理（显式开启）
"""

from __future__ import annotations

import asyncio
import sys

from ai_agent_test.guardrails.policy import Decision, describe


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


class InteractiveApprover(Approver):
    name = "interactive"

    async def approve(self, decision: Decision) -> bool:
        banner = "\n" + "─" * 60 + f"\n⚠️  需要确认（{decision.level.value}）\n" + describe(decision) + "\n"
        # 打印到 stderr，避免污染标准输出里模型的消息流
        print(banner, file=sys.stderr, flush=True)
        answer = await asyncio.to_thread(input, "   执行吗？[y/N] ")
        return answer.strip().lower() in {"y", "yes"}


def build_approver(mode: str) -> Approver:
    return {
        "interactive": InteractiveApprover,
        "deny": DenyAllApprover,
        "allow": AllowAllApprover,
    }.get(mode, InteractiveApprover)()
