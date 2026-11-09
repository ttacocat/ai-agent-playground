"""shell 工具的 MCP server。

这个 server 自己 **不做** 风险判定：判定集中在 agent 侧的护栏
（``guardrails/shell_rules.py``），在调用到达这里之前完成，危险命令需要人确认。
之前这里有一行 ``if "rm" in shlex.split(command)`` 的检查，看起来像防线，
实际上 ``find -delete``、``r''m`` 都能绕过，只会制造虚假的安全感，所以删掉了。

这里只负责把执行本身框住：

- 工作目录固定为 ``WORK_DIR``（默认项目下的 ``.temp``），不在项目根或 home 下执行；
- 每条命令有超时，防止 ``tail -f``、交互式命令把 agent 挂死；
- 输出截断，避免一次 ``cat`` 大文件撑爆上下文。

仍然使用 ``shell=True``：护栏允许 ``ls | grep x`` 这类只读管道，
它们需要 shell 才能执行。命令能不能执行由护栏的分段判定决定，而不是这里。
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parents[5]
load_dotenv(PROJECT_ROOT / ".env")

COMMAND_TIMEOUT = int(os.getenv("SHELL_TIMEOUT", "60"))
MAX_OUTPUT_CHARS = int(os.getenv("SHELL_MAX_OUTPUT", "20000"))


def work_dir() -> Path:
    raw = Path(os.getenv("WORK_DIR", ".temp")).expanduser()
    path = raw if raw.is_absolute() else PROJECT_ROOT / raw
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n…（输出过长，已截断 {len(text) - MAX_OUTPUT_CHARS} 字符）"


mcp = FastMCP(name="shell")


@mcp.tool(
    name="run_shell",
    description=(
        "在本机（不是虚拟机）的工作目录中执行一条 shell 命令并返回输出。"
        "虚拟机里的操作请用名字带 _in_vm 的工具。只读命令直接执行，其余命令需要用户确认。"
    ),
)
def run_shell_command(
    command: Annotated[str, Field(description="要执行的 shell 命令", examples=["ls -al"])],
) -> str:
    try:
        res = subprocess.run(
            command,
            shell=True,
            cwd=work_dir(),
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return f"命令执行超过 {COMMAND_TIMEOUT}s，已终止"
    except Exception as e:
        return f"执行失败：{type(e).__name__}: {e}"

    if res.returncode != 0:
        return _clip(f"[exit {res.returncode}]\n{res.stderr or res.stdout}")
    return _clip(res.stdout)


if __name__ == "__main__":
    mcp.run(transport="stdio")
