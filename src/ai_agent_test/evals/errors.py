"""区分"agent 做错了"和"接口出问题了"。

第一次跑全量三架构评测时，免费额度在 supervisor 跑到一半时用完，之后每次调用都是 403。
评测器把这些 403 记成了用例失败，报告上 supervisor 15%、team 0%——看起来是架构
不行，实际上是没调到模型。还白白空转了上百次。

所以运行异常分三类：

- ``FATAL``     额度用完、鉴权失败、模型不存在：再跑也是同样的错，立即停止整场评测；
- ``INFRA``     限流、超时、网络、服务端 5xx：这一次不算数（不计入成功率），继续下一次，
                连续出现太多次也停止；
- ``None``      其他异常（递归超限、输出解析失败等）：是 agent 自己的问题，照常记为失败。
"""

from __future__ import annotations

from enum import StrEnum

import openai


class ErrorKind(StrEnum):
    FATAL = "fatal"
    INFRA = "infra"


_FATAL_CODES = {"insufficient_quota", "invalid_api_key", "model_not_found", "account_deactivated"}


def _chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _error_code(exc: openai.APIStatusError) -> str:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body)
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "")
    return ""


def classify(exc: BaseException) -> ErrorKind | None:
    for item in _chain(exc):
        if isinstance(item, openai.APIStatusError):
            code = _error_code(item)
            if code in _FATAL_CODES or "quota" in str(item).lower():
                return ErrorKind.FATAL
            if isinstance(item, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
                return ErrorKind.FATAL
            if item.status_code == 402:
                return ErrorKind.FATAL
            if isinstance(item, (openai.RateLimitError, openai.InternalServerError)) or item.status_code >= 500:
                return ErrorKind.INFRA
        if isinstance(item, (openai.APIConnectionError, openai.APITimeoutError, TimeoutError, ConnectionError)):
            return ErrorKind.INFRA
    return None


def short(exc: BaseException, limit: int = 160) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
