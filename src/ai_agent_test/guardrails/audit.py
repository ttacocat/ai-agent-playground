"""审计日志：每次风险判定留一条 JSONL，可回放、可 grep。

写日志失败不能影响主流程——审计是可观测性增强，不是可用性前提。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from ai_agent_test.guardrails.policy import Decision


@dataclass
class AuditRecord:
    ts: str
    tool: str
    level: str
    outcome: str  # executed / blocked / denied
    reason: str
    preview: str


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def write(self, decision: Decision, outcome: str) -> AuditRecord:
        record = AuditRecord(
            ts=datetime.now().isoformat(timespec="seconds"),
            tool=decision.tool,
            level=decision.level.value,
            outcome=outcome,
            reason=decision.reason,
            preview=decision.preview,
        )
        if self.path is None:
            return record
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except OSError:
            pass  # 磁盘满 / 只读挂载等情况不影响业务
        return record


class NullAuditLog(AuditLog):
    """评测与单测用：只记录到内存，不落盘。"""

    def __init__(self) -> None:
        super().__init__(path=None)
        self.records: list[AuditRecord] = []

    def write(self, decision: Decision, outcome: str) -> AuditRecord:
        record = super().write(decision, outcome)
        self.records.append(record)
        return record
