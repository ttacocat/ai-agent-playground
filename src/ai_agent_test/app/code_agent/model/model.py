import httpx
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ai_agent_test.config import settings

_cfg = settings.model

# trust_env=False：忽略系统代理变量，避免本地 http_proxy 把请求劫持到
# 无法访问内网 MaaS 端点的代理上。
_SHARED_HTTP_CLIENT = httpx.Client(trust_env=False, timeout=_cfg.timeout)
_SHARED_HTTP_ASYNC_CLIENT = httpx.AsyncClient(trust_env=False, timeout=_cfg.timeout)

qwen_llm = ChatOpenAI(
    model=_cfg.name,
    base_url=_cfg.base_url,
    api_key=SecretStr(_cfg.api_key),
    temperature=_cfg.temperature,
    timeout=_cfg.timeout,
    max_retries=_cfg.max_retries,
    streaming=True,
    http_client=_SHARED_HTTP_CLIENT,
    http_async_client=_SHARED_HTTP_ASYNC_CLIENT,
)
