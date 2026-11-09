"""Web 服务的离线测试：演示模式跑真实的 team 图，验证 SSE 事件、两段式审批和刷新恢复。"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.store import ResultStore, RunRecord, case_fingerprint
from ai_agent_test.web.app import create_app
from ai_agent_test.web.events import EventTranslator, agent_of


def events_of(response) -> list[dict]:
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.fixture
def client(tmp_path):
    app = create_app(demo=True, demo_delay=0, reports_dir=tmp_path / "reports", traces_dir=tmp_path / "traces")
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["demo"] is True and body["modes"] == ["lead", "team"]


def test_read_only_question_streams_plan_and_final(client):
    events = events_of(client.post("/api/chat/t1/messages", json={"text": "user 表有哪些字段？", "mode": "team"}))
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done"
    assert "approval_required" not in kinds
    call = next(e for e in events if e["type"] == "tool_call")
    assert call["agent"] == "planner" and call["name"] == "mysql_describe_tables"
    assert call["risk"]["level"] == "read"
    assert not any(
        e["type"] == "tool_result" and e["name"] in {"Plan", "Review"} for e in events
    )  # 结构化输出回执不展示
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["plan"]["needs_changes"] is False
    final = next(e for e in events if e["type"] == "final")
    assert "name" in final["content"]


def test_delete_suspends_for_approval_then_resumes(client):
    events = events_of(client.post("/api/chat/t2/messages", json={"text": "删除 user 表 id=1 的记录", "mode": "team"}))
    approval = next(e for e in events if e["type"] == "approval_required")
    [item] = approval["items"]
    assert item["tool"] == "mysql_delete_data" and item["level"] == "destructive"
    assert not any(e["type"] == "review" for e in events)  # 还没执行，更没有审查

    # 刷新页面：从 checkpoint 恢复待审批项
    state = client.get("/api/chat/t2?mode=team").json()
    assert [p["id"] for p in state["pending"]] == [item["id"]]

    # 有待审批时不允许发新消息
    assert client.post("/api/chat/t2/messages", json={"text": "算了", "mode": "team"}).status_code == 409

    resumed = events_of(client.post("/api/chat/t2/approvals", json={"mode": "team", "answers": {item["id"]: True}}))
    result = next(e for e in resumed if e["type"] == "tool_result" and e["name"] == "mysql_delete_data")
    assert not result["blocked"]
    review = next(e for e in resumed if e["type"] == "review")
    assert review["review"]["passed"] is True
    assert resumed[-1]["type"] == "done"
    assert client.get("/api/chat/t2?mode=team").json()["pending"] == []


def test_denied_delete_is_reported_honestly(client):
    events = events_of(client.post("/api/chat/t3/messages", json={"text": "删除 user 表 id=1 的记录", "mode": "team"}))
    item = next(e for e in events if e["type"] == "approval_required")["items"][0]
    resumed = events_of(client.post("/api/chat/t3/approvals", json={"mode": "team", "answers": {item["id"]: False}}))
    result = next(e for e in resumed if e["type"] == "tool_result")
    assert result["blocked"]
    review = next(e for e in resumed if e["type"] == "review")
    assert review["review"]["passed"] is False


def test_approval_requires_answers_for_every_item(client):
    events = events_of(client.post("/api/chat/t4/messages", json={"text": "删除 user 表 id=1 的记录", "mode": "team"}))
    assert any(e["type"] == "approval_required" for e in events)
    assert client.post("/api/chat/t4/approvals", json={"mode": "team", "answers": {}}).status_code == 400
    assert client.post("/api/chat/none/approvals", json={"mode": "team", "answers": {"x": True}}).status_code == 409


def test_history_survives_reload(client):
    events_of(client.post("/api/chat/t5/messages", json={"text": "user 表有哪些字段？", "mode": "team"}))
    history = client.get("/api/chat/t5?mode=team").json()["events"]
    assert history[0] == {"type": "user", "content": "user 表有哪些字段？"}
    assert history[-1]["type"] == "final"


def test_demo_rejects_supervisor_mode(client):
    assert client.post("/api/chat/t6/messages", json={"text": "hi", "mode": "supervisor"}).status_code == 400


def test_evals_endpoint_reads_result_files(client, tmp_path):
    store = ResultStore(tmp_path / "reports" / "runs-kimi-k3.jsonl")
    e01, e23 = (next(c for c in CASES if c.id == i) for i in ("E01", "E23"))
    store.append(RunRecord("kimi-k3", "single", "E01", case_fingerprint(e01), True, tokens=100, seconds=1.0))
    store.append(
        RunRecord(
            "kimi-k3",
            "supervisor",
            "E23",
            case_fingerprint(e23),
            False,
            reasons=["最终状态不符合预期：user 表 id 应为 [1, 2, 3]，实际为 []"],
        )
    )
    store.append(RunRecord("kimi-k3", "single", "E01", "旧指纹", True))  # 用例改过：不计入

    [entry] = client.get("/api/evals").json()
    assert entry["name"] == "kimi-k3" and entry["runs"] == 3
    detail = client.get("/api/evals/kimi-k3").json()
    assert detail["stale"] == 1
    summary = {s["target"]: s for s in detail["summary"]}
    assert summary["single"]["success_rate"] == 1.0
    assert summary["supervisor"]["breaches"] == 1
    assert client.get("/api/evals/..%2Fsecret").status_code == 404


def test_traces_endpoint(client, tmp_path):
    from ai_agent_test.observability.tracer import Tracer

    tracer = Tracer(tmp_path / "traces" / "eval-1.jsonl")
    tracer.save()
    [entry] = client.get("/api/traces").json()
    assert entry["name"] == "eval-1"
    assert client.get("/api/traces/eval-1").json()["spans"] == []
    assert client.get("/api/traces/nope").status_code == 404


# --------------------------------------------------------------------------- #
# 事件翻译
# --------------------------------------------------------------------------- #
def test_agent_name_comes_from_outermost_subgraph():
    assert agent_of(("code_expert:abc", "model:def"), "model") == "code_expert"
    assert agent_of((), "supervisor") == "supervisor"
    assert agent_of((), "model") == "assistant"


def test_translator_dedups_messages_forwarded_by_parent():
    from langchain_core.messages import AIMessage

    message = AIMessage("结论", id="m1")
    translator = EventTranslator()
    first = translator.translate(("research_expert:1",), {"model": {"messages": [message]}})
    again = translator.translate((), {"research_expert": {"messages": [message]}})
    assert [e["type"] for e in first] == ["message"] and again == []


def test_handoff_and_transfer_receipts():
    from langchain_core.messages import AIMessage, ToolMessage

    translator = EventTranslator()
    handoff = AIMessage("", tool_calls=[{"name": "transfer_to_code_expert", "args": {}, "id": "t"}], id="a")
    receipt = ToolMessage("Successfully transferred to code_expert", tool_call_id="t")
    events = translator.translate((), {"supervisor": {"messages": [handoff, receipt]}})
    assert events == [{"type": "handoff", "agent": "supervisor", "to": "code_expert"}]


# --------------------------------------------------------------------------- #
# lead：主 agent + 子 agent 工具
# --------------------------------------------------------------------------- #
def test_lead_streams_subagent_steps_in_their_own_lane(client):
    events = events_of(client.post("/api/chat/l1/messages", json={"text": "user 表有哪些字段？", "mode": "lead"}))
    explore = next(e for e in events if e["type"] == "tool_call" and e["name"] == "explore")
    assert explore["agent"] == "lead" and explore["risk"]["level"] == "read"
    inner = next(e for e in events if e["type"] == "tool_call" and e["name"] == "mysql_describe_tables")
    assert inner["agent"] == "explorer"  # 子 agent 在工具内部运行，靠消息名字归到自己的泳道
    inner_result = next(e for e in events if e["type"] == "tool_result" and e["name"] == "mysql_describe_tables")
    assert inner_result["agent"] == "explorer"
    summary = next(e for e in events if e["type"] == "tool_result" and e["name"] == "explore")
    assert summary["agent"] == "lead" and "user 表字段" in summary["content"]
    assert events[-1]["type"] == "done"


def test_lead_delete_goes_through_approval_and_review(client):
    events = events_of(client.post("/api/chat/l2/messages", json={"text": "删除 user 表 id=1 的记录", "mode": "lead"}))
    item = next(e for e in events if e["type"] == "approval_required")["items"][0]
    assert item["tool"] == "mysql_delete_data"

    resumed = events_of(client.post("/api/chat/l2/approvals", json={"mode": "lead", "answers": {item["id"]: True}}))
    assert any(e["type"] == "tool_call" and e["agent"] == "reviewer" for e in resumed)
    review = next(e for e in resumed if e["type"] == "tool_result" and e["name"] == "review")
    assert review["content"].startswith("审查通过")

    history = client.get("/api/chat/l2?mode=lead").json()["events"]
    review_row = next(e for e in history if e["type"] == "tool_result" and e["name"] == "review")
    assert review_row["agent"] == "lead"  # 刷新后工具结果归到发起调用的 agent，而不是工具名
