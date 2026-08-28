from langchain.agents import create_agent
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from ai_agent_test.app.qwen.bailian_tools import tools
from ai_agent_test.app.qwen.common import llm

class CalcResult(BaseModel):
    result: int = Field(description="计算结果")
    expression: str = Field(description="原始计算表达式")

parser = JsonOutputParser(pydantic_object=CalcResult)
format_instructions = parser.get_format_instructions()

system_prompt = f"""你是一个数学计算助手，需要计算时请调用提供的工具，不要自己心算。
计算完成后，请严格按照以下格式输出最终答案，不要有多余文字：

{format_instructions}"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
)

res = agent.invoke({
    "messages": [
        {"role": "user", "content": "100 + 100 = ?"}
    ]
})

final_text = res["messages"][-1].content
parsed = parser.parse(final_text)

print(parsed)              # {'result': 200, 'expression': '100+100'}
print(parsed["result"])    # 200