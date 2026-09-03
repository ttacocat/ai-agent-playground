import os
import pickle
import threading
from typing import Any, Optional, Sequence
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from ai_agent_test.app.code_agent.model.model import qwen_llm
from ai_agent_test.app.code_agent.tools.file_tools import file_tools
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    ChannelVersions,
)


class FileSaver(BaseCheckpointSaver[str]):
    def __init__(self, base_path: str = "/Users/wxy/ai-agent-test/.temp/checkpoint"):
        super().__init__()
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        # 简单起见用一把全局锁保护文件读写,避免同一进程内并发写同一线程文件时数据损坏
        self._lock = threading.Lock()

    # ---------- 内部工具方法 ----------

    def _thread_file(self, thread_id: str) -> str:
        return os.path.join(self.base_path, f"{thread_id}.pkl")

    def _load(self, thread_id: str) -> dict:
        path = self._thread_file(thread_id)
        if not os.path.exists(path):
            return {"checkpoints": {}, "writes": {}}
        with open(path, "rb") as f:
            return pickle.load(f)

    def _save(self, thread_id: str, data: dict) -> None:
        path = self._thread_file(thread_id)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    # ---------- BaseCheckpointSaver 核心方法 ----------

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")

        with self._lock:
            data = self._load(thread_id)

        checkpoints = data["checkpoints"]
        if not checkpoints:
            return None

        if checkpoint_id is None:
            # 没指定具体 checkpoint_id 时,取最新的一个
            checkpoint_id = max(checkpoints.keys())

        entry = checkpoints.get(checkpoint_id)
        if entry is None:
            return None

        # 收集挂在这个 checkpoint 下的 pending writes
        pending_writes = [
            (task_id, channel, value)
            for (cp_id, task_id), items in data["writes"].items()
            if cp_id == checkpoint_id
            for (_, channel, value) in items
        ]

        parent_config = None
        if entry["parent_checkpoint_id"] is not None:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": entry["parent_checkpoint_id"],
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=entry["checkpoint"],
            metadata=entry["metadata"],
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id = config["configurable"]["thread_id"]
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_id = checkpoint["id"]

        with self._lock:
            data = self._load(thread_id)
            data["checkpoints"][checkpoint_id] = {
                "checkpoint": checkpoint,
                "metadata": metadata,
                "parent_checkpoint_id": parent_checkpoint_id,
                "channel_versions": new_versions,
            }
            self._save(thread_id, data)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        with self._lock:
            data = self._load(thread_id)
            key = (checkpoint_id, task_id)
            existing = data["writes"].setdefault(key, [])
            start_idx = len(existing)
            for idx, (channel, value) in enumerate(writes):
                existing.append((start_idx + idx, channel, value))
            self._save(thread_id, data)


if __name__ == "__main__":
    saver = FileSaver()
    print(f"checkpoint 存储目录: {saver.base_path}")

    agent = create_agent(
        model=qwen_llm,
        tools=file_tools,
        checkpointer=saver,
        debug=True,
    )

    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    def chat(user_input: str):
        res = agent.invoke(
            input={"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        final_message = res["messages"][-1]
        print(f">>> user: {user_input}")
        print(f"<<< agent: {final_message.content}")
        print(f"    当前 messages 总数: {len(res['messages'])}")
        return res

    # 多轮对话:同一个 thread_id,验证 FileSaver 是否真的把历史存进了文件
    chat("我是谁")
    chat("你有哪些工具")

    # 直接查 checkpoint,验证 get_tuple 读出来的历史是完整的
    state = agent.get_state(config)
    print("\n=== checkpoint 中的完整 messages ===")
    for m in state.values["messages"]:
        print(f"[{m.type}] {m.content}")