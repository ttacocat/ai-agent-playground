from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from ai_agent_test.app.code_agent.mcp.browser_tools import search_in_duckduckgo
from ai_agent_test.app.code_agent.model.model import qwen_llm


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def web_search(query: str) -> str:
    """Search the web for the query and return the result text. Use it for current events."""
    result = search_in_duckduckgo(query)
    return result or "未搜索到相关结果"


# ---------------------------------------------------------------------------
# 子 Agent
# ---------------------------------------------------------------------------
math_agent = create_agent(
    model=qwen_llm,
    tools=[add, multiply],
    name="math_expert",
    system_prompt="你是一个数学专家，一次执行只使用一个工具"
)

research_agent = create_agent(
    model=qwen_llm,
    tools=[web_search],
    name="research_expert",
    system_prompt="你是一个世界级的调研专家，能够使用web_search工具，不要使用任何数学工具"
)

# ---------------------------------------------------------------------------
# Supervisor：负责把任务分派给合适的子 Agent
# ---------------------------------------------------------------------------
workflow = create_supervisor(
    agents=[math_agent, research_agent],
    model=qwen_llm,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# create_supervisor 返回的是 StateGraph，需要 compile 后才能运行
app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({
        "messages": [
            {"role": "user", "content": "北京今天的天气如何？另外帮我算一下 (3 + 5) * 12 等于多少？"}
        ]
    })

    for message in result["messages"]:
        message.pretty_print()