"""评测的成本指标：按模型统计 token、每次成功的 token、子 agent 调用次数。"""

from __future__ import annotations

from fakes import ScriptedModel
from langchain_core.messages import AIMessage

from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.runner import render_report, run_cases
from ai_agent_test.evals.store import ResultStore

E06 = next(c for c in CASES if c.id == "E06")  # 知识库问答：回答里要有 pwd


def usage(n: int) -> dict:
    return {"input_tokens": n, "output_tokens": 0, "total_tokens": n}


def lead_model():
    return ScriptedModel(
        model_name="big",
        replies=[
            AIMessage(
                "",
                tool_calls=[{"name": "explore", "args": {"question": "终端规范"}, "id": "x1"}],
                usage_metadata=usage(1000),
            ),
            AIMessage("规范要求先用 pwd 确认目录", usage_metadata=usage(500)),
        ],
    )


def cheap_model():
    return ScriptedModel(
        model_name="cheap",
        replies=[
            AIMessage(
                "", tool_calls=[{"name": "query_rag", "args": {"query": "终端"}, "id": "q1"}], usage_metadata=usage(300)
            ),
            AIMessage("先用 pwd 确认当前目录", usage_metadata=usage(200)),
        ],
    )


async def test_tokens_are_split_by_model_and_delegations_counted(tmp_path):
    store = ResultStore(tmp_path / "runs.jsonl")
    session = await run_cases(
        [E06],
        targets=["lead"],
        model_factory=lead_model,
        explore_model_factory=cheap_model,
        model_name="big",
        explore_model_name="cheap",
        store=store,
    )
    attempt = session.targets[0].runs[0].attempts[0]
    assert attempt.result.passed, attempt.result.reasons
    assert attempt.result.actual_calls == ["query_rag"]  # 子 agent 里的调用也记到了
    assert attempt.tokens_by_model == {"big": 1500, "cheap": 500}
    assert attempt.tokens == 2000
    assert attempt.delegations == {"explore": 1}

    report = session.targets[0]
    assert report.main_model_tokens("big") == 1500
    assert report.tokens_per_success == 2000
    assert report.delegations("explore") == 1

    text = render_report(session)
    assert "其中主模型" in text and "每次成功 token" in text and "探索模型：cheap" in text
    assert "| 1,500 | 2,000 |" in text

    [record] = store.load()  # 落盘后原样读回
    assert record.tokens_by_model == {"big": 1500, "cheap": 500} and record.delegations == {"explore": 1}
    resumed = await run_cases(
        [E06], targets=["lead"], model_factory=lead_model, model_name="big", store=store, resume=True
    )
    assert resumed.targets[0].runs[0].attempts[0].tokens_by_model == {"big": 1500, "cheap": 500}


async def test_failed_runs_count_toward_cost_per_success():
    e10 = next(c for c in CASES if c.id == "E10")  # 不调工具、回答里要有 63
    replies = iter(
        [[AIMessage("等于 63", usage_metadata=usage(100))], [AIMessage("不知道", usage_metadata=usage(300))]]
    )
    session = await run_cases(
        [e10], targets=["single"], repeat=2, model_factory=lambda: ScriptedModel(replies=next(replies))
    )
    report = session.targets[0]
    assert report.runs[0].passes == 1
    assert report.tokens_per_success == 400  # 两次一共 400，只成功一次


async def test_old_records_without_split_count_everything_on_main_model():
    from ai_agent_test.evals.engine import CaseResult
    from ai_agent_test.evals.runner import Attempt

    attempt = Attempt(CaseResult(case=E06, actual_calls=[], passed=True, reasons=[]), 1, 700, 1.0)
    assert attempt.tokens_on("whatever") == 700


def test_web_summary_has_cost_fields(tmp_path):
    from ai_agent_test.evals.store import RunRecord, case_fingerprint
    from ai_agent_test.web.data import result_detail

    store = ResultStore(tmp_path / "runs-big.jsonl")
    fp = case_fingerprint(E06)
    store.append(
        RunRecord(
            "big",
            "lead",
            "E06",
            fp,
            True,
            tokens=2000,
            tokens_by_model={"big": 1500, "cheap": 500},
            delegations={"explore": 1},
        )
    )
    store.append(RunRecord("big", "lead", "E06", fp, False, tokens=1000, tokens_by_model={"big": 1000}))
    [lead] = result_detail(tmp_path, "big")["summary"]
    assert lead["main_tokens"] == 1250
    assert lead["tokens_per_success"] == 3000
    assert lead["explores"] == 0.5
