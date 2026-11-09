"""按配置装配护栏，供 chat / eval 调用。"""

from __future__ import annotations

from ai_agent_test.guardrails.approval import Approver, build_approver
from ai_agent_test.guardrails.audit import AuditLog


def build_guard(approver_mode: str | None = None, audit_path=None) -> tuple[Approver, AuditLog]:
    """返回 (审批器, 审计日志)。参数留空时取 .env 配置。"""
    from ai_agent_test.config import settings

    cfg = settings.guard
    return (
        build_approver(approver_mode or cfg.approver),
        AuditLog(audit_path if audit_path is not None else cfg.audit_log),
    )


def apply_guard(tools):
    """给工具列表套上护栏。GUARD_ENABLED=false 时原样返回，便于对比排障。"""
    from ai_agent_test.config import settings
    from ai_agent_test.guardrails.wrap import guard_tools

    if not settings.guard.enabled:
        return list(tools)
    approver, audit = build_guard()
    return guard_tools(tools, approver=approver, audit=audit)
