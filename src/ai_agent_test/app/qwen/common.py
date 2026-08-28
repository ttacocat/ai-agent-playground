from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import FileManagementToolkit
from pydantic import SecretStr

llm = ChatOpenAI(
    model="qwen3.8-max",
    base_url="https://.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    streaming=True,
)

system_prompt = ChatMessagePromptTemplate.from_template(
    role="system", template="你是{role}，擅长回答{domain}领域的问题。"
)
human_prompt = ChatMessagePromptTemplate.from_template(
    role="user", template="用户问题：{question}"
)
chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

file_toolkit = FileManagementToolkit(root_dir="/Users/wxy/ai-agent-test/.temp")
file_tools = file_toolkit.get_tools()
