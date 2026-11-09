from __future__ import annotations

from typing import Any, ClassVar

import httpx
import openai
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ai_agent_test.config import settings

_cfg = settings.model

# trust_env=False：忽略系统代理变量，避免本地 http_proxy 把请求劫持到
# 无法访问内网 MaaS 端点的代理上。
_SHARED_HTTP_CLIENT = httpx.Client(trust_env=False, timeout=_cfg.timeout)
_SHARED_HTTP_ASYNC_CLIENT = httpx.AsyncClient(trust_env=False, timeout=_cfg.timeout)


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


def build_chat_model(name: str | None = None) -> ChatOpenAI:
    """按配置建模型客户端；name 可以覆盖模型名（同一个端点、同一把 Key），评测换模型时用。"""
    return CompatChatOpenAI(
        model=name or _cfg.name,
        base_url=_cfg.base_url,
        api_key=SecretStr(_cfg.api_key),
        temperature=_cfg.temperature,
        timeout=_cfg.timeout,
        max_retries=_cfg.max_retries,
        streaming=True,
        # 流式响应默认不带 token 用量（部分模型，如 kimi，trace 里会记成 0），
        # 打开后最后一个 chunk 会附带 usage，成本统计和评测对比才准确
        stream_usage=True,
        http_client=_SHARED_HTTP_CLIENT,
        http_async_client=_SHARED_HTTP_ASYNC_CLIENT,
    )


qwen_llm = build_chat_model()
