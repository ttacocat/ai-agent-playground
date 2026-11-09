"""可观测性：本地 trace + 上下文预算。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ai_agent_test.observability.context import ContextBudgetMiddleware, build_context_middleware, trim_messages
from ai_agent_test.observability.tracer import Span, Summary, Tracer, load_run, render, summarize

__all__ = [
    "ContextBudgetMiddleware",
    "Span",
    "Summary",
    "Tracer",
    "build_context_middleware",
    "build_tracer",
    "load_run",
    "render",
    "summarize",
    "trim_messages",
]


def build_tracer(name: str = "run") -> Tracer | None:
    """按配置创建一个 tracer；TRACE_ENABLED=false 时返回 None，调用方无需分支。"""
    from ai_agent_test.config import settings

    cfg = settings.trace
    if not cfg.enabled:
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Tracer(Path(cfg.dir) / f"{name}-{stamp}.jsonl")
