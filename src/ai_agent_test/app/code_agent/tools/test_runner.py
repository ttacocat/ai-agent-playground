"""``run_tests``：在工作目录里跑 pytest，返回结果摘要。

之前 agent 要跑测试只能走 ``run_shell("pytest")``。shell 命令不在只读白名单里，
每次都要人确认；方案和审查这两个只读角色更是根本跑不了——而"审查专家自己跑测试
核实结果"恰恰是审查最有价值的部分。

这个工具只接受一个参数：测哪个文件或目录（必须在工作目录内）。命令本身是写死的，
模型没法借它执行任意命令，所以护栏把它归为只读（``guardrails/policy.py``）。
需要注意：测试代码本身是任意 Python，跑测试等于执行它。这里的边界是工作目录、
超时和精简过的环境变量，不是真正的沙箱。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from pydantic import Field

TEST_TIMEOUT = 60
OUTPUT_TAIL_LINES = 60
RUN_TESTS_DESCRIPTION = (
    "在工作目录中运行 pytest 测试并返回结果摘要。path 为要测试的文件或目录（相对工作目录），留空表示全部。"
)

_PYTEST_INI = "[pytest]\npythonpath = .\naddopts = -q -p no:cacheprovider\n"


def _clean_env() -> dict[str, str]:
    """只保留运行 Python 必需的环境变量：不把 API Key 之类的东西带进测试进程。"""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP")
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", NO_COLOR="1")
    return env


def run_pytest_in(root: Path, path: str = "", timeout: int = TEST_TIMEOUT) -> tuple[bool, str]:
    """在 root 下运行 pytest。返回 (是否全部通过, 输出摘要)。"""
    root = root.resolve()
    target = (root / path).resolve() if path else root
    if not target.is_relative_to(root):
        return False, f"Error: 只能测试工作目录内的路径：{path}"
    if not target.exists():
        return False, f"Error: 路径不存在：{path}"
    if not (root / "pytest.ini").exists():
        (root / "pytest.ini").write_text(_PYTEST_INI, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(target.relative_to(root)) if path else "."],
            cwd=root,
            env=_clean_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"测试运行超过 {timeout}s，已终止"
    output = (result.stdout + result.stderr).strip().splitlines()
    tail = "\n".join(output[-OUTPUT_TAIL_LINES:])
    # 退出码 5 = 没有收集到任何测试
    passed = result.returncode == 0
    if result.returncode == 5:
        tail = tail or "没有找到任何测试"
    return passed, tail


def build_run_tests_tool(root: Path):
    @tool("run_tests", description=RUN_TESTS_DESCRIPTION)
    def run_tests(
        path: Annotated[str, Field(description="要测试的文件或目录，相对工作目录；留空表示全部")] = "",
    ) -> str:
        passed, output = run_pytest_in(root, path)
        return ("✔ 全部通过\n" if passed else "✘ 有测试未通过\n") + output

    return run_tests
