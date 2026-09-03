from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# ========== 三种 checkpointer 的写法对比(学习记录) ==========

# --- 1. MemorySaver:纯内存,不落盘,进程结束/对象重建即丢失 ---
from langgraph.checkpoint.memory import MemorySaver
# memory = MemorySaver()

# --- 2. RedisSaver:需要 Redis >= 8.0(或 Redis Stack),setup() 建索引 ---
# from langgraph.checkpoint.redis import RedisSaver
# with RedisSaver.from_conn_string("redis://localhost:6379") as memory:
#     memory.setup()

# --- 3. MongoDBSaver:当前使用的方案 ---
from langgraph.checkpoint.mongodb import MongoDBSaver

from langchain.agents import create_agent

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.file_tools import file_tools

MONGODB_URL = "mongodb://localhost:27017/?directConnection=true"
MONGODB_DB = "chat"


def creat_agent(checkpointer):
    agent = create_agent(
        model=qwen_llm,
        tools=file_tools,
        checkpointer=checkpointer,  # 存储
        debug=True,
    )
    return agent


def run_agent(agent, user_input: str, config: RunnableConfig):
    res = agent.invoke(
        input={"messages": [HumanMessage(content=user_input)]},
        config=config,
    )
    final_message = res["messages"][-1]
    print(f">>> user: {user_input}")
    print(f"<<< agent: {final_message.content}")
    print(f"    当前 messages 总数: {len(res['messages'])}")
    return res


if __name__ == "__main__":
    config = RunnableConfig(configurable={"thread_id": "1"})

    # 所有逻辑都写在 with 块内,退出块时会自动调用 memory.close(),不需要手动管理
    with MongoDBSaver.from_conn_string(MONGODB_URL, MONGODB_DB) as memory:
        agent = creat_agent(memory)

        # run_agent(agent, "你好,我是wxy", config)
        # run_agent(agent, "我是谁?", config)
        run_agent(agent, "我们之前聊了什么", config)

        # 查询 checkpoint 里存的完整 state,验证 memory 确实落在了 checkpointer 里
        state = agent.get_state(config)
        print("\n=== checkpoint 中的完整 messages ===")
        for m in state.values["messages"]:
            print(f"[{m.type}] {m.content}")
        memory.close()