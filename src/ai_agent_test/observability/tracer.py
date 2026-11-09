"""本地 span 追踪：不依赖任何外部平台，跑完就有一份可分析的 JSONL。

为什么不用 Langfuse / LangSmith：这两者都需要账号 + 网络 + 额外容器，
而"能跑起来"是简历项目的第一道门槛。这里实现的是 LangChain
`BaseCallbackHandler` 的一个子集——真接了 Langfuse 也只是换一个 handler，
语义完全一致（llm span / tool span / token 用量）。

记录粒度刻意只到 **LLM 调用**和 **工具调用** 两级：
graph 节点的 chain span 数量是它的 5~10 倍，噪音远大于信息。
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class Span:
    span_id: str
    parent_id: str | None
    kind: str  # llm / tool
    name: str
    start: float  # 相对本次 run 起点的秒数
    duration: float
    status: str = "ok"
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str = ""


@dataclass
class TraceHeader:
    type: str = "run"
    started_at: str = ""
    finished_at: str = ""
    total: float = 0.0
    model: str = ""


class Tracer(BaseCallbackHandler):
    """收集一次运行里的 LLM / 工具 span，最后落一份 JSONL。"""

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = path
        self.spans: list[Span] = []
        self.started_at = datetime.now()
        self._t0 = time.perf_counter()
        self._open: dict[str, tuple[str, str, float, str]] = {}
        self._parent: dict[str, str | None] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _model_name(serialized: dict | None, metadata: dict | None = None) -> str:
        """真实模型名在 kwargs 或 metadata 里；serialized["name"] 只是类名 ChatOpenAI。"""
        kwargs = (serialized or {}).get("kwargs") or {}
        for key in ("model_name", "model"):
            if kwargs.get(key):
                return str(kwargs[key])
        if metadata and metadata.get("ls_model_name"):
            return str(metadata["ls_model_name"])
        return str((serialized or {}).get("name") or "model")

    def _is_duplicate_tool_span(self, parent_id: Any, kind: str, name: str) -> bool:
        """护栏包装层调 `tool.ainvoke()` 时会再触发一次 on_tool_start，
        形成同名工具套同名工具的空壳 span。真实嵌套（agent→tool）不会同名，可直接剪掉。"""
        if kind != "tool" or parent_id is None:
            return False
        parent = self._open.get(str(parent_id))
        return bool(parent and parent[0] == kind and parent[1] == name)

    def _begin(self, run_id: Any, parent_id: Any, kind: str, name: str, detail: str = "") -> None:
        with self._lock:
            if self._is_duplicate_tool_span(parent_id, kind, name):
                self._open[str(run_id)] = ("alias", name, time.perf_counter(), "")
                self._parent[str(run_id)] = str(parent_id) if parent_id else None
                return
            self._open[str(run_id)] = (kind, name, time.perf_counter(), detail)
            self._parent[str(run_id)] = str(parent_id) if parent_id else None

    def _end(self, run_id: Any, status: str = "ok", input_tokens: int = 0, output_tokens: int = 0) -> None:
        key = str(run_id)
        with self._lock:
            opened = self._open.pop(key, None)
            if opened is None:
                return
            kind, name, started, detail = opened
            parent = self._parent.pop(key, None)
            if kind == "alias":
                return
            self.spans.append(
                Span(
                    span_id=key,
                    parent_id=parent,
                    kind=kind,
                    name=name,
                    start=round(started - self._t0, 4),
                    duration=round(time.perf_counter() - started, 4),
                    status=status,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    detail=detail,
                )
            )

    @staticmethod
    def _usage(response: Any) -> tuple[int, int]:
        """token 用量：优先 llm_output，退回 usage_metadata。"""
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        if prompt or completion:
            return int(prompt), int(completion)

        for generation_list in getattr(response, "generations", None) or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                meta = getattr(message, "usage_metadata", None) or {}
                prompt += int(meta.get("input_tokens") or 0)
                completion += int(meta.get("output_tokens") or 0)
        return int(prompt), int(completion)

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        chars = sum(len(str(getattr(m, "content", ""))) for batch in messages for m in batch)
        self._begin(run_id, parent_run_id, "llm", self._model_name(serialized, metadata), f"prompt {chars} 字符")

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kwargs):
        chars = sum(len(p) for p in prompts)
        self._begin(run_id, parent_run_id, "llm", self._model_name(serialized, metadata), f"prompt {chars} 字符")

    def on_llm_end(self, response, *, run_id, **kwargs):
        prompt, completion = self._usage(response)
        self._end(run_id, input_tokens=prompt, output_tokens=completion)

    def on_llm_error(self, error: BaseException, *, run_id, **kwargs):
        self._end(run_id, status="error")

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = (serialized or {}).get("name") or "tool"
        self._begin(run_id, parent_run_id, "tool", str(name), str(input_str)[:200])

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id, **kwargs):
        self._end(run_id, status="error")

    # ------------------------------------------------------------------ #
    # 落盘
    # ------------------------------------------------------------------ #
    def save(self, path: Path | None = None) -> Path | None:
        target = path or self.path
        if target is None:
            return None
        header = TraceHeader(
            started_at=self.started_at.isoformat(timespec="seconds"),
            finished_at=datetime.now().isoformat(timespec="seconds"),
            total=round(time.perf_counter() - self._t0, 4),
            model=next((s.name for s in self.spans if s.kind == "llm"), ""),
        )
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(header), ensure_ascii=False) + "\n")
                for span in self.spans:
                    fh.write(json.dumps(asdict(span), ensure_ascii=False) + "\n")
        except OSError:
            return None
        return target


def load_run(path: Path) -> tuple[dict, list[Span]]:
    header: dict = {}
    spans: list[Span] = []
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if index == 0 and record.get("type") == "run":
                header = record
                continue
            spans.append(Span(**record))
    return header, spans


@dataclass
class Summary:
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    tool_seconds: float = 0.0
    errors: int = 0
    tool_breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_seconds(self) -> float:
        return self.llm_seconds + self.tool_seconds

    def cost(self, price_in_per_m: float, price_out_per_m: float) -> float:
        return self.input_tokens / 1_000_000 * price_in_per_m + self.output_tokens / 1_000_000 * price_out_per_m


def summarize(spans: list[Span]) -> Summary:
    summary = Summary()
    for span in spans:
        if span.status != "ok":
            summary.errors += 1
        if span.kind == "llm":
            summary.llm_calls += 1
            summary.llm_seconds += span.duration
            summary.input_tokens += span.input_tokens
            summary.output_tokens += span.output_tokens
        else:
            summary.tool_calls += 1
            summary.tool_seconds += span.duration
            summary.tool_breakdown[span.name] = summary.tool_breakdown.get(span.name, 0) + 1
    summary.llm_seconds = round(summary.llm_seconds, 3)
    summary.tool_seconds = round(summary.tool_seconds, 3)
    return summary


def render(header: dict, spans: list[Span], price_in: float = 0.0, price_out: float = 0.0) -> str:
    summary = summarize(spans)
    lines = [
        f"运行时间  {header.get('started_at', '-')}    总耗时 {header.get('total', 0):.2f}s",
        "",
    ]

    # span 树：按 start 时间排序，子调用缩进一层
    roots = [s for s in spans if not s.parent_id or s.parent_id not in {x.span_id for x in spans}]
    children: dict[str, list[Span]] = {}
    for span in spans:
        if span.parent_id:
            children.setdefault(span.parent_id, []).append(span)

    def emit(span: Span, depth: int) -> None:
        mark = "✔" if span.status == "ok" else "✘"
        head = "  " * depth + f"{mark} {span.kind:<5}{span.name:<28}{span.duration:>7.2f}s"
        if span.kind == "llm":
            head += f"   in {span.input_tokens:>6} / out {span.output_tokens:>5}"
        lines.append(head)
        for child in sorted(children.get(span.span_id, []), key=lambda s: s.start):
            emit(child, depth + 1)

    for root in sorted(roots, key=lambda s: s.start):
        emit(root, 0)

    lines += [
        "",
        f"调用   LLM {summary.llm_calls} 次，工具 {summary.tool_calls} 次"
        + (f"，失败 {summary.errors} 次" if summary.errors else ""),
        f"耗时   LLM {summary.llm_seconds:.2f}s / 工具 {summary.tool_seconds:.2f}s",
        f"Token  输入 {summary.input_tokens:,} / 输出 {summary.output_tokens:,} / 合计 {summary.total_tokens:,}",
    ]
    if price_in or price_out:
        lines.append(f"成本   ¥{summary.cost(price_in, price_out):.4f}（按配置单价）")
    else:
        lines.append("成本   未配置单价（MODEL_PRICE_INPUT / MODEL_PRICE_OUTPUT，单位：元/百万 token）")

    if summary.tool_breakdown:
        top = sorted(summary.tool_breakdown.items(), key=lambda kv: -kv[1])
        lines.append("工具   " + "，".join(f"{name}×{count}" for name, count in top))
    return "\n".join(lines) + "\n"
