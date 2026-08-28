# from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---------- 开发工具函数 ----------

class AddInput(BaseModel):
    a: int = Field(description="第一个加数")
    b: int = Field(description="第二个加数")


@tool(
    "add",
    description="add two numbers",
    args_schema=AddInput,
    return_direct=False
)
def add(a: int, b: int) -> int:
    """add two numbers"""
    return a + b


class MultiplyInput(BaseModel):
    a: int = Field(description="第一个乘数")
    b: int = Field(description="第二个乘数")


@tool(
    "multiply",
    description="multiply two numbers",
    args_schema=MultiplyInput,
    return_direct=False
)
def multiply(a: int, b: int) -> int:
    """multiply two numbers"""
    return a * b


tools = [add, multiply]

# 方法一
# add_tools = StructuredTool.from_function(
#     func=add,
#     name="add",
#     description="add two numbers",
# )
# tool_dict = {"add": add_tools}
# tool_dict = {"add": add}
# llm_with_tools = llm.bind_tools(
#     tools=[
#         add,
#     ])
# chain = chat_prompt_template | llm_with_tools
# res = chain.invoke(input={"role": "计算", "domain": "数学计算", "question": "100+200=?"})
# print(res)
# for tool_call in res.tool_calls:
#     tool_content = tool_dict[tool_call["name"]].invoke(tool_call["args"])
#     print(tool_content)  # 300