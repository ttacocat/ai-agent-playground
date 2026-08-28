from langchain_classic.output_parsers import BooleanOutputParser, DatetimeOutputParser
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ai_agent_test.app.qwen.common import chat_prompt_template, llm

parser = DatetimeOutputParser()
# parser = StrOutputParser()
# parser = CommaSeparatedListOutputParser()
# parser = BooleanOutputParser()
# parser = JsonOutputParser()

instruction = parser.get_format_instructions()
prompt = ChatPromptTemplate.from_messages([
    ("system", "必须按照以下格式返回时间{format_instructions}"),
    ("human", "请将以下自然语言转成标准时间格式: {text}")
]).partial(format_instructions=instruction)

chain = prompt | llm | parser
res = chain.invoke({"text": "二零二六年八月十日"})

print(res)