from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ai_agent_test.app.code_agent.model.compat import CompatChatOpenAI
from ai_agent_test.config import settings

_cfg = settings.model

# trust_env=False：忽略系统代理变量，避免本地 http_proxy 把请求劫持到
# 无法访问内网 MaaS 端点的代理上。
_SHARED_HTTP_CLIENT = httpx.Client(trust_env=False, timeout=_cfg.timeout)
_SHARED_HTTP_ASYNC_CLIENT = httpx.AsyncClient(trust_env=False, timeout=_cfg.timeout)


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
