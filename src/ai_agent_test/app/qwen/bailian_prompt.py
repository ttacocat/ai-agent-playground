from langchain_core.prompts import ChatPromptTemplate, ChatMessagePromptTemplate, FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

llm = ChatOpenAI(
    model="qwen3.8-max",
    base_url="https://.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(""),
    streaming=True,
)
#
# # 方法一：使用 ChatPromptTemplate.from_messages（最常用）
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "你是一个智能助手，专门回答关于{domain}的问题。"),
#     ("human", "请告诉我关于{topic}的基本信息。")
# ])
#
# # 填充变量
# messages = prompt.format_messages(domain="人工智能", topic="大语言模型")
#
# # 调用 LLM
# for chunk in llm.stream(messages):
#     # 每个 chunk 是一个 AIMessageChunk 对象，通过 .content 获取增量文本
#     print(chunk.content, end="", flush=True)
# print()  # 最后换行

# 方法二： ChatMessagePromptTemplate
# system_prompt = ChatMessagePromptTemplate.from_template(
#     role="system", template="你是{role}，你的知识截止到{date}。"
# )
# human_prompt = ChatMessagePromptTemplate.from_template(
#     role="human", template="请解释一下{topic}的概念。"
# )
# prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# 3. 少样本示例（分类情感的例子）
examples = [
    {
        "text": "这部电影太棒了，演员表演非常出色！",
        "label": "正面"
    },
    {
        "text": "服务很差，食物也难吃。",
        "label": "负面"
    },
    {
        "text": "天气晴朗，适合出去走走。",
        "label": "中性"
    }
]

# 2. 定义每个示例的格式化模板
example_prompt = PromptTemplate(
    input_variables=["text", "label"],
    template="文本: {text}\n情感: {label}\n"
)

# 3. 构建 FewShotPromptTemplate
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="请根据以下示例判断文本的情感类别（正面、负面、中性）：",
    suffix="文本: {input_text}\n情感:",
    input_variables=["input_text"],
    example_separator="\n"  # 示例之间的分隔符
)

# 4. 生成提示
prompt_text = few_shot_prompt.format(input_text="这家餐厅的装修很有特色，但菜品一般。")
print(prompt_text)

for chunk in llm.stream(prompt_text):
    print(chunk.content, end="", flush=True)
print()  # 最后换行


