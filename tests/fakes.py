"""测试用的假模型：按顺序吐出预先写好的回复，用来离线驱动真实的 LangGraph 图。

多个 agent 共用一个 ScriptedModel 时，回复按图的实际执行顺序依次消耗，
所以脚本本身就描述了"谁在第几步说了什么"。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class ScriptedModel(BaseChatModel):
    """最后一条之后一直重复最后一条，避免脚本写短了直接崩。"""

    replies: list[AIMessage]
    cursor: int = 0
    model_name: str = "scripted"  # 像真实模型一样通过 ls_model_name 上报，按模型统计 token 时用

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_model_name"] = self.model_name
        return params

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self.replies[min(self.cursor, len(self.replies) - 1)]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, call_id: str, **args: Any) -> AIMessage:
    """一条只包含单个工具调用的 AI 回复。"""
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])


class RecordingScriptedModel(ScriptedModel):
    """ScriptedModel，同时记下每次调用收到的消息，用来检查各角色拿到了什么。"""

    seen: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)
