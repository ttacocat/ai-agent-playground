from langchain.agents import create_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from ai_agent_test.app.qwen.common import llm

tools = [PythonREPLTool()]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个能执行 Python 代码完成任务的助手。
需要写文件、跑代码时，请调用 Python 执行工具完成，不要只输出代码文本了事。
注意：调用工具时传入的 Python 代码不要加 ```python``` 或 ```py``` 这类 Markdown 代码块标记，直接传纯代码。""",
)

user_input = """
要求：
1. 向 /Users/wxy/ai-agent-test/.temp 目录下写入一个新文件，名称为：index.html
2. 写一个企业的官网
"""

res = agent.invoke({
    "messages": [
        {"role": "user", "content": user_input}
    ]
})

print(res["messages"][-1].content )
