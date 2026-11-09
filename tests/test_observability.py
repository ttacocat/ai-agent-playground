"""可观测性与上下文预算的单测。全部离线。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ai_agent_test.observability.context import total_chars, trim_messages, truncate_tool_output
from ai_agent_test.observability.tracer import Tracer, load_run, render, summarize


# --------------------------------------------------------------------------- #
# 上下文预算
# --------------------------------------------------------------------------- #
def _conversation(length: int = 3) -> list:
    """system + 若干轮 (human, ai 带 tool_call, tool)。"""
    messages = [SystemMessage(content="你是工程师助手")]
    for i in range(length):
        messages.append(HumanMessage(content=f"第 {i} 个任务：" + "很长的描述" * 40))
        messages.append(
            AIMessage(content="", tool_calls=[{"name": "run_shell", "args": {}, "id": f"call-{i}"}])
        )
        messages.append(ToolMessage(content="命令输出 " * 100, tool_call_id=f"call-{i}", name="run_shell"))
    return messages


def test_under_budget_is_untouched():
    messages = _conversation(1)
    assert trim_messages(messages, max_chars=10**6, keep_last=12) == messages


def test_over_budget_keeps_system_and_recent():
    messages = _conversation(5)
    trimmed = trim_messages(messages, max_chars=100, keep_last=3)

    assert len(trimmed) < len(messages)
    assert isinstance(trimmed[0], SystemMessage)
    assert any("已省略" in str(m.content) for m in trimmed if isinstance(m, SystemMessage))
    # 最近的消息必须在
    assert trimmed[-1].content == messages[-1].content


def test_tool_call_pair_is_never_split():
    """保留 ToolMessage 时必须连它发起调用的 AIMessage 一起保留，否则接口 400。"""
    messages = _conversation(5)
    trimmed = trim_messages(messages, max_chars=100, keep_last=3)

    for index, message in enumerate(trimmed):
        if isinstance(message, ToolMessage):
            assert index > 0 and isinstance(trimmed[index - 1], AIMessage), "ToolMessage 前面缺 AIMessage"
            assert trimmed[index - 1].tool_calls[0]["id"] == message.tool_call_id


def test_long_tool_output_is_truncated():
    huge = ToolMessage(content="x" * 5000, tool_call_id="c1", name="run_shell")
    shrunk = truncate_tool_output(huge, limit=1200)
    assert len(str(shrunk.content)) < 1400
    assert "省略" in str(shrunk.content)
    assert shrunk.tool_call_id == "c1"  # 身份信息不能丢


def test_non_tool_messages_are_not_truncated():
    human = HumanMessage(content="y" * 5000)
    assert truncate_tool_output(human) is human


def test_total_chars_counts_content():
    assert total_chars([HumanMessage(content="abc"), AIMessage(content="de")]) == 5


def test_no_trimming_when_message_count_below_keep_last():
    messages = _conversation(2)
    assert len(trim_messages(messages, max_chars=10, keep_last=50)) == len(messages)


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, prompt: int, completion: int) -> None:
        self.llm_output = {"token_usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
        self.generations = []


def _fake_run(tracer: Tracer) -> None:
    tracer.on_chat_model_start({}, [[HumanMessage(content="查一下表")]], run_id="llm-1")
    tracer.on_llm_end(_FakeResponse(120, 30), run_id="llm-1")

    tracer.on_tool_start({"name": "mysql_list_tables"}, '{"database": "test"}', run_id="tool-1", parent_run_id="llm-1")
    tracer.on_tool_end("user, orders", run_id="tool-1")

    tracer.on_chat_model_start({}, [[HumanMessage(content="总结")]], run_id="llm-2")
    tracer.on_llm_end(_FakeResponse(200, 50), run_id="llm-2")


def test_tracer_collects_spans_and_tokens(tmp_path=None):
    tracer = Tracer()
    _fake_run(tracer)

    assert len(tracer.spans) == 3
    summary = summarize(tracer.spans)
    assert summary.llm_calls == 2
    assert summary.tool_calls == 1
    assert summary.input_tokens == 320
    assert summary.output_tokens == 80
    assert summary.tool_breakdown == {"mysql_list_tables": 1}


def test_tracer_keeps_parent_relation():
    tracer = Tracer()
    _fake_run(tracer)
    tool_span = next(s for s in tracer.spans if s.kind == "tool")
    llm_span = next(s for s in tracer.spans if s.span_id == "llm-1")
    assert tool_span.parent_id == llm_span.span_id


def test_tracer_error_span_is_marked():
    tracer = Tracer()
    tracer.on_tool_start({"name": "run_shell"}, "rm -rf /", run_id="t-err")
    tracer.on_tool_error(RuntimeError("boom"), run_id="t-err")
    assert tracer.spans[0].status == "error"
    assert summarize(tracer.spans).errors == 1


def test_save_and_reload_roundtrip():
    root = Path(__file__).parent / "_tmp_trace"
    shutil.rmtree(root, ignore_errors=True)
    try:
        tracer = Tracer(root / "run-x.jsonl")
        _fake_run(tracer)
        path = tracer.save()
        assert path is not None and path.is_file()

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert json.loads(lines[0])["type"] == "run"
        assert len(lines) == 4  # header + 3 span

        header, spans = load_run(path)
        assert header["type"] == "run"
        assert len(spans) == 3
        assert summarize(spans).total_tokens == 400
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_cost_uses_per_million_price():
    tracer = Tracer()
    _fake_run(tracer)
    summary = summarize(tracer.spans)
    # 输入 320 token、输出 80 token；单价 10 元/百万与 30 元/百万
    expected = 320 / 1_000_000 * 10 + 80 / 1_000_000 * 30
    assert abs(summary.cost(10, 30) - expected) < 1e-9


def test_render_contains_tree_and_summary():
    tracer = Tracer()
    _fake_run(tracer)
    text = render({"started_at": "2026-11-09T12:00:00", "total": 1.23}, tracer.spans, 10, 30)

    assert "mysql_list_tables" in text
    assert "Token" in text
    assert "成本" in text
    assert "工具   mysql_list_tables×1" in text
