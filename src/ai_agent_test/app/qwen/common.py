from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ai_agent_test.config import settings

# 学习期示例代码：模型与目录一律复用配置层，避免这里再出现一套写死的值
llm = ChatOpenAI(
    model=settings.model.name,
    base_url=settings.model.base_url,
    api_key=SecretStr(settings.model.api_key),
    streaming=True,
)

system_prompt = ChatMessagePromptTemplate.from_template(
    role="system", template="你是{role}，擅长回答{domain}领域的问题。"
)
human_prompt = ChatMessagePromptTemplate.from_template(
    role="user", template="用户问题：{question}"
)
chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

file_toolkit = FileManagementToolkit(root_dir=str(settings.paths.work_dir))
file_tools = file_toolkit.get_tools()
