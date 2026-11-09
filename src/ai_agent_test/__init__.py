"""ai_agent_test —— 基于 LangGraph 的多智能体工程助手。

对外只暴露一个 CLI 入口，见 `ai_agent_test.cli:main`。
"""

from ai_agent_test.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
