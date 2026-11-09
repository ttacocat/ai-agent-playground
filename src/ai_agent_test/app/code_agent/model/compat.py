"""模型客户端的兼容层。不读配置，可以单独导入和测试。"""

from __future__ import annotations

from typing import Any, ClassVar

import openai
from langchain_openai import ChatOpenAI

_FORCED_CHOICES = {"any", "required"}


def _relaxed(kwargs: dict[str, Any]) -> dict[str, Any]:
    choice = kwargs.get("tool_choice")
    forced = isinstance(choice, dict) or choice in _FORCED_CHOICES
    return {**kwargs, "tool_choice": "auto"} if forced else kwargs


def _rejects_forced_choice(exc: openai.BadRequestError) -> bool:
    return "tool_choice" in str(exc)


class CompatChatOpenAI(ChatOpenAI):
    """兼容不支持"强制调用工具"的模型。

    结构化输出（ToolStrategy）会把 tool_choice 设成 required / 指定工具，逼模型必须调用
    Review 之类的工具。部分模型在思考模式下直接拒绝这个参数（qwen3.8：
    "The tool_choice parameter does not support being set to required or object in thinking mode"），
    整次运行因此崩掉。这里遇到这种 400 就降级成 auto 重发一次，并记住这个模型，
    之后直接用 auto。降级后模型仍然会按提示词调用工具，没调用时由调用方兜底。
    """

    _no_forced_choice: ClassVar[set[str]] = set()

    def _prepare(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        return _relaxed(kwargs) if self.model_name in self._no_forced_choice else kwargs

    def _should_retry(self, exc: openai.BadRequestError, kwargs: dict[str, Any]) -> bool:
        if _relaxed(kwargs) is kwargs or not _rejects_forced_choice(exc):
            return False
        self._no_forced_choice.add(self.model_name)
        return True

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._prepare(kwargs)
        try:
            return super()._generate(messages, stop, run_manager, **kwargs)
        except openai.BadRequestError as exc:
            if not self._should_retry(exc, kwargs):
                raise
            return super()._generate(messages, stop, run_manager, **_relaxed(kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._prepare(kwargs)
        try:
            return await super()._agenerate(messages, stop, run_manager, **kwargs)
        except openai.BadRequestError as exc:
            if not self._should_retry(exc, kwargs):
                raise
            return await super()._agenerate(messages, stop, run_manager, **_relaxed(kwargs))

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._prepare(kwargs)
        started = False
        try:
            for chunk in super()._stream(messages, stop, run_manager, **kwargs):
                started = True
                yield chunk
        except openai.BadRequestError as exc:
            if started or not self._should_retry(exc, kwargs):
                raise
            yield from super()._stream(messages, stop, run_manager, **_relaxed(kwargs))

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs = self._prepare(kwargs)
        started = False
        try:
            async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                started = True
                yield chunk
        except openai.BadRequestError as exc:
            if started or not self._should_retry(exc, kwargs):
                raise
            async for chunk in super()._astream(messages, stop, run_manager, **_relaxed(kwargs)):
                yield chunk
