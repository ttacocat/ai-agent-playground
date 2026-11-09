"""评测省额度：单次运行的 token 上限（--run-cap）和整场预算（--budget）。"""

from __future__ import annotations

import argparse

import pytest
from fakes import ScriptedModel
from langchain_core.messages import AIMessage

from ai_agent_test.cli import parse_tokens
from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.runner import run_cases
from ai_agent_test.evals.store import ResultStore

E06 = next(c for c in CASES if c.id == "E06")  # 知识库问答：回答里要有 pwd


def usage(n: int) -> dict:
    return {"input_tokens": n, "output_tokens": 0, "total_tokens": n}


def search(i: int, tokens: int, name: str = "query_rag", **args) -> AIMessage:
    args = args or {"query": f"终端 {i}"}
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}], usage_metadata=usage(tokens))


def answer(tokens: int) -> AIMessage:
    return AIMessage("先用 pwd 确认当前目录", usage_metadata=usage(tokens))


def looping_model(tokens: int = 1000, steps: int = 20):
    """一直查知识库、不肯收尾的模型。"""
    return ScriptedModel(replies=[search(i, tokens) for i in range(steps)])


def cheap_run(tokens: int = 500):
    """一次检索 + 一次回答，总共 2 × tokens。"""
    return ScriptedModel(replies=[search(0, tokens), answer(tokens)])


async def test_run_cap_stops_runaway_run_and_counts_as_failure(tmp_path):
    store = ResultStore(tmp_path / "runs.jsonl")
    session = await run_cases([E06], model_factory=looping_model, model_name="m", store=store, token_limit=2_500)
    attempt = session.targets[0].runs[0].attempts[0]
    assert attempt.over_limit and not attempt.result.passed
    assert attempt.valid  # 不是接口错误：算一次失败
    assert attempt.tokens == 3_000  # 第三次调用后超线，第四次请求不会发出去
    assert "token 上限" in attempt.result.reasons[0]
    assert session.aborted is None
    [record] = store.load()
    assert not record.passed


async def test_run_cap_also_stops_inside_explore_subagent():
    def lead():
        return ScriptedModel(
            model_name="big",
            replies=[search(0, 1000, name="explore", question="终端规范"), answer(100)],
        )

    def explorer():
        return ScriptedModel(model_name="cheap", replies=[search(i, 1000) for i in range(20)])

    session = await run_cases(
        [E06],
        targets=["lead"],
        model_factory=lead,
        explore_model_factory=explorer,
        model_name="big",
        token_limit=2_500,
    )
    attempt = session.targets[0].runs[0].attempts[0]
    assert attempt.over_limit and not attempt.result.passed
    assert attempt.tokens_by_model == {"big": 1000, "cheap": 2000}


async def test_budget_stops_before_a_run_it_cannot_afford(tmp_path):
    store = ResultStore(tmp_path / "runs.jsonl")
    session = await run_cases([E06], repeat=3, model_factory=cheap_run, model_name="m", store=store, budget=2_500)
    run = session.targets[0].runs[0]
    assert [a.tokens for a in run.attempts] == [1000, 1000]  # 第三次预计 1000，只剩 500，不开跑
    assert session.new_tokens == 2000
    assert "预算" in session.aborted
    assert len(store.load()) == 2


async def test_budget_cut_mid_run_is_not_recorded(tmp_path):
    runs = iter([cheap_run(500), looping_model(1000)])
    store = ResultStore(tmp_path / "runs.jsonl")
    session = await run_cases(
        [E06], repeat=2, model_factory=lambda: next(runs), model_name="m", store=store, budget=2_500
    )
    run = session.targets[0].runs[0]
    assert len(run.attempts) == 1  # 第二次跑到一半预算用完：不落盘，也不算失败
    assert run.attempts[0].result.passed
    assert "已用完" in session.aborted
    assert len(store.load()) == 1


def test_parse_tokens():
    assert parse_tokens("300k") == 300_000
    assert parse_tokens("1.5m") == 1_500_000
    assert parse_tokens("200,000") == 200_000
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tokens("lots")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_tokens("0")
