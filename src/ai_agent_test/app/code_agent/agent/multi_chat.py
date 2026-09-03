import uuid

from langchain.agents import create_agent
from langchain_community.agent_toolkits.file_management import FileManagementToolkit
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.prompts.multi_chat_prompts import multi_chat_prompt


def get_session_history(session_id: str):
    return FileChatMessageHistory(f"{session_id}.json")


# --- 扩展点 1：以后加更多工具，就往这个列表里塞更多 toolkit.get_tools() 的结果 ---
file_toolkit = FileManagementToolkit(root_dir="/.temp")
file_tools = file_toolkit.get_tools()

# --- 扩展点 2：以后要加多步骤流程（人工审批、历史压缩等），
#     在这里给 create_agent 传 middleware=[...] 参数即可，
#     不需要改动下面 to_messages / extract_text 这两层转换逻辑 ---
agent = create_agent(model=qwen_llm, tools=file_tools)

# 把 multi_chat_prompt 渲染出的 ChatPromptValue，转成 code_agent 需要的 {"messages": [...]} 格式
# —— 这是 RunnableLambda 最典型的用法：把两个类型不兼容的 Runnable "粘"在一起
to_messages = RunnableLambda(
    lambda inputs: {"messages": multi_chat_prompt.invoke(inputs).to_messages()}
)

# 把 code_agent 返回的完整消息列表（dict），提取成最后一条 AI 回复的纯文本
extract_text = RunnableLambda(lambda result: result["messages"][-1].content)

# 三个 Runnable 用 "|" 拼成一个 RunnableSequence：
# prompt 渲染 → code_agent 执行（内部会按需循环调用工具）→ 提取文本
chain = to_messages | agent | extract_text

# 用 RunnableWithMessageHistory 包装原始 chain，
# 让它在每次调用前后自动读取/写入历史记录
chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="question",        # 对应 prompt 里的 {question}
    history_messages_key="chat_history",  # 对应 prompt 里的 MessagesPlaceholder("chat_history")
)

if __name__ == "__main__":
    demo_session_id = str(uuid.uuid4())

    run_config: RunnableConfig = {"configurable": {"session_id": demo_session_id}}

    print(f"多轮对话已启动（session_id: {demo_session_id}），输入内容开始对话，输入 exit / quit 退出。\n")

    while True:
        question = input("user：").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("对话结束。")
            break

        answer = chain_with_history.invoke({"question": question}, config=run_config)
        print(f"AI：{answer}\n")