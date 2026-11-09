"""评测框架的离线单测：验证判定引擎与用例集本身的正确性，不调 LLM。"""

import pytest

from ai_agent_test.evals.cases import CASES
from ai_agent_test.evals.engine import EvalCase, evaluate, is_subsequence
from ai_agent_test.evals.stub_tools import STUB_TOOL_NAMES


def test_all_case_tools_exist_in_stub_layer():
    """用例里出现的每个工具名，stub 层都必须能提供，否则评测跑一半才炸。"""
    for case in CASES:
        for tool_name in (case.expect_tools or ()) + (case.expect_sequence or ()) + (case.forbid_tools or ()):
            assert tool_name in STUB_TOOL_NAMES, f"{case.id} 引用了 stub 层不存在的工具：{tool_name}"


def test_case_ids_are_unique():
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids))


def test_eval_engine_pass_on_exact_tools():
    case = EvalCase(id="T1", task="t", expect_tools=("a", "b"))
    assert evaluate(case, ["a", "b"]).passed
    # 多调了不禁止的工具也算过：expect 只要求「必须调过」
    assert evaluate(case, ["a", "x", "b"]).passed


def test_eval_engine_fails_on_missing_tool():
    case = EvalCase(id="T2", task="t", expect_tools=("a",))
    result = evaluate(case, ["b"])
    assert not result.passed
    assert "缺少必要工具调用" in result.reasons[0]


def test_subsequence_allows_intervening_calls():
    assert is_subsequence(["query_rag", "make_dir_in_vm"], ["query_rag", "run_shell", "make_dir_in_vm"])


def test_subsequence_respects_order():
    assert not is_subsequence(["b", "a"], ["a", "b"])


def test_eval_engine_fails_on_wrong_order():
    case = EvalCase(id="T3", task="t", expect_sequence=("mysql_describe_tables", "mysql_insert_data"))
    result = evaluate(case, ["mysql_insert_data", "mysql_describe_tables"])
    assert not result.passed
    assert "顺序不符合预期" in result.reasons[0]


def test_eval_engine_forbid_tools():
    case = EvalCase(id="T4", task="t", forbid_tools=("run_shell",))
    assert evaluate(case, ["make_dir_in_vm"]).passed
    result = evaluate(case, ["run_shell", "make_dir_in_vm"])
    assert not result.passed
    assert "禁止使用" in result.reasons[0]


def test_eval_engine_expect_no_tools():
    case = EvalCase(id="T5", task="t", expect_no_tools=True)
    assert evaluate(case, []).passed
    assert not evaluate(case, ["run_shell"]).passed


def test_eval_engine_rejects_contradictory_case():
    with pytest.raises(ValueError):
        EvalCase(id="T6", task="t", expect_no_tools=True, expect_tools=("a",))


def test_eval_engine_rejects_case_without_assertion():
    with pytest.raises(ValueError):
        EvalCase(id="T7", task="t")
