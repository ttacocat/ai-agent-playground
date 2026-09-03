import uuid

from langchain.agents import create_agent
from langchain_community.agent_toolkits.file_management import FileManagementToolkit
from langgraph.checkpoint.memory import InMemorySaver

from ai_agent_test.app.code_agent.model.model import qwen_llm

file_toolkit = FileManagementToolkit(root_dir="/.temp")
file_tools = file_toolkit.get_tools()

# checkpointer 负责按 thread_id 自动保存/恢复对话历史，
# 替代之前手写的 RunnableWithMessageHistory + FileChatMessageHistory 组合
checkpointer = InMemorySaver()

agent = create_agent(
    model=qwen_llm,
    tools=file_tools,
    system_prompt="你是一位优秀的技术专家，擅长解决各种开发中的技术问题",
    checkpointer=checkpointer,
)

if __name__ == "__main__":
    demo_session_id = str(uuid.uuid4())  # 每次启动程序生成一个唯一会话 ID

    # LangGraph code_agent 用 thread_id 来区分/持久化不同会话，
    # 作用等价于之前的 session_id
    run_config = {"configurable": {"thread_id": demo_session_id}}

    print(f"多轮对话已启动（thread_id: {demo_session_id}），输入内容开始对话，输入 exit / quit 退出。\n")

    while True:
        question = input("user：").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("对话结束。")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=run_config,
        )
        answer = result["messages"][-1].content
        print(f"AI：{answer}\n")