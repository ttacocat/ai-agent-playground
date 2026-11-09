"""风险等级定义。单独成模块，供 shell / sql 规则与 policy 共用，避免循环导入。"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"read": 0, "write": 1, "destructive": 2}[self.value]


Verdict = tuple[RiskLevel, str]


def worst(verdicts: Iterable[Verdict], default: Verdict) -> Verdict:
    """取风险最高的判定；同级保留先出现的那条原因。没有任何判定时返回 default。"""
    best: Verdict | None = None
    for verdict in verdicts:
        if best is None or verdict[0].rank > best[0].rank:
            best = verdict
    return best or default
