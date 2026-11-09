"""评测结果的持久化：每次运行追加一行 JSONL，中断后可以接着跑。

288 次运行要几十分钟、上百万 token。额度用完、网络断了、手动 Ctrl-C，
已经跑完的结果都不该丢。所以每完成一次就写一行，``--resume`` 时读回来，
只补跑缺的那几次。

两个防串味的设计：

- 文件按模型分开（``runs-<模型名>.jsonl``）。不同模型的结果混在一起比较没有意义，
  换了模型就是一组新数据；
- 每行带用例指纹。用例的任务描述或判定条件改了，旧结果自动作废，不会被续用。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from ai_agent_test.evals.engine import EvalCase


def case_fingerprint(case: EvalCase) -> str:
    """任务描述 + 全部判定条件的摘要；函数类条件取其限定名。"""
    parts = []
    for name, value in sorted(vars(case).items()):
        if name == "tags":
            continue
        if callable(value):
            value = getattr(value, "__qualname__", repr(value))
        parts.append(f"{name}={value!r}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "unknown"


def run_slug(model: str, tag: str | None = None) -> str:
    """结果文件名：模型名，加上可选的标签（例如修复前后各跑一组，互不覆盖）。"""
    return model_slug(model) + (f"-{model_slug(tag)}" if tag else "")


@dataclass
class RunRecord:
    model: str
    target: str
    case_id: str
    fingerprint: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    destructive_executed: list[str] = field(default_factory=list)
    answer: str = ""
    llm_calls: int = 0
    tokens: int = 0
    seconds: float = 0.0
    error: str | None = None  # 非空 = 接口问题，这次不算数
    tokens_by_model: dict = field(default_factory=dict)
    delegations: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ResultStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def for_model(cls, directory: Path, model: str, tag: str | None = None) -> ResultStore:
        return cls(directory / f"runs-{run_slug(model, tag)}.jsonl")

    def append(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def load(self) -> list[RunRecord]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(RunRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue  # 半行（写入时被打断）或旧格式：跳过
        return records

    def completed(self, model: str, target: str, case: EvalCase, limit: int) -> list[RunRecord]:
        """某架构、某用例已有的有效结果（不含接口错误），最多取最近 limit 条。"""
        fingerprint = case_fingerprint(case)
        valid = [
            r
            for r in self.load()
            if r.model == model
            and r.target == target
            and r.case_id == case.id
            and r.fingerprint == fingerprint
            and not r.error
        ]
        return valid[-limit:] if limit > 0 else []
