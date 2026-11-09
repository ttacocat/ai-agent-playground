"""全局配置。

约定：业务代码里 **不允许** 出现硬编码的路径、密钥、连接串、模型名，
一律通过 `from ai_agent_test.config import settings` 获取。

设计要点：
- 惰性求值：import 本模块不会因为缺少环境变量而崩溃，只有真正访问
  某个配置块（如 settings.mysql）时才校验，便于跑 lint / 单测。
- 相对路径统一基于仓库根目录解析，换台机器 clone 下来即可运行。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from dotenv import load_dotenv

# src/ai_agent_test/config.py -> 仓库根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parent
CODE_AGENT_ROOT = PACKAGE_ROOT / "app" / "code_agent"

load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """配置缺失或不合法。信息里必须包含修复指引，不要只说"未设置"。"""


def _require(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise ConfigError(f"缺少环境变量 {key}。请执行 `cp .env.example .env`，然后填写 {key} 后重试。")
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, "").strip() or default


def _as_bool(key: str, default: bool = False) -> bool:
    raw = _optional(key)
    if not raw:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _as_path(raw: str) -> Path:
    """支持 ~ 展开；相对路径按仓库根目录解析，保证在任何 cwd 下行为一致。"""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _as_int(key: str, default: int) -> int:
    raw = _optional(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"环境变量 {key} 必须是整数，当前值为 {raw!r}") from exc


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str | None
    api_key: str
    temperature: float
    timeout: float
    max_retries: int
    explore_name: str = ""  # lead 架构里探索子 agent 用的模型；留空 = 与主模型相同


@dataclass(frozen=True)
class MySQLConfig:
    host: str
    port: int
    user: str
    password: str
    charset: str
    database: str | None

    def connect_kwargs(self, database: str | None = None) -> dict:
        """pymysql.connect 的参数。库名留空时只连 server，不选库。"""
        db = database if database is not None else self.database
        kwargs = {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "charset": self.charset,
        }
        if db:
            kwargs["database"] = db
        return kwargs


@dataclass(frozen=True)
class BailianConfig:
    access_key_id: str
    access_key_secret: str
    endpoint: str
    workspace_id: str
    query_index_id: str
    upload_category_id: str
    parser: str


@dataclass(frozen=True)
class GuardConfig:
    """危险操作的拦截策略。enabled=False 时所有工具直通，仅用于排障对比。"""

    enabled: bool
    approver: str  # interactive / deny / allow
    audit_log: Path | None

    def validate(self) -> None:
        if self.approver not in {"interactive", "deny", "allow"}:
            raise ConfigError(f"GUARD_APPROVER 只支持 interactive / deny / allow，当前值为 {self.approver!r}")


@dataclass(frozen=True)
class TraceConfig:
    """本地 trace 与成本计算。价格为 0 时只统计 token，不算钱。"""

    enabled: bool
    dir: Path
    price_input_per_m: float
    price_output_per_m: float


@dataclass(frozen=True)
class ContextConfig:
    """模型单次请求的上下文预算。max_chars<=0 关闭。

    strategy=summarize：超预算时把早期对话压成摘要（多一次 LLM 调用，只在超预算时发生）；
    strategy=trim：直接丢弃中间的消息（零额外成本，但会丢信息）。
    """

    max_chars: int
    keep_last: int
    strategy: str = "summarize"
    summary_max_chars: int = 2000

    def validate(self) -> None:
        if self.strategy not in {"summarize", "trim"}:
            raise ConfigError(f"CONTEXT_STRATEGY 只支持 summarize / trim，当前值为 {self.strategy!r}")


@dataclass(frozen=True)
class CheckpointConfig:
    """对话状态的持久化后端。redis / mongodb / file，非法值直接报错。"""

    backend: str
    redis_url: str
    mongodb_url: str
    mongodb_db: str


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path
    work_dir: Path
    checkpoint_dir: Path
    chromedriver: Path | None

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class Settings:
    """配置聚合器。每个块都是 cached_property，首次访问时才校验。"""

    @cached_property
    def model(self) -> ModelConfig:
        return ModelConfig(
            name=_optional("MODEL_NAME", "qwen3.8-27b"),
            base_url=_optional("MODEL_BASE_URL") or None,
            api_key=_require("MODEL_API_KEY"),
            temperature=float(_optional("MODEL_TEMPERATURE", "0")),
            timeout=float(_optional("MODEL_TIMEOUT", "120")),
            max_retries=_as_int("MODEL_MAX_RETRIES", 2),
            explore_name=_optional("EXPLORE_MODEL_NAME"),
        )

    @cached_property
    def mysql(self) -> MySQLConfig:
        return MySQLConfig(
            host=_optional("MYSQL_HOST", "127.0.0.1"),
            port=_as_int("MYSQL_PORT", 3306),
            user=_optional("MYSQL_USER", "root"),
            password=_require("MYSQL_PASSWORD"),
            charset=_optional("MYSQL_CHARSET", "utf8mb4"),
            database=_optional("MYSQL_DATABASE") or None,
        )

    @cached_property
    def bailian(self) -> BailianConfig:
        return BailianConfig(
            access_key_id=_require("ALI_ACCESS_KEY_ID"),
            access_key_secret=_require("ALI_ACCESS_KEY_SECRET"),
            endpoint=_optional("BAILIAN_ENDPOINT", "bailian.cn-beijing.aliyuncs.com"),
            workspace_id=_require("WORKSPACE_ID"),
            query_index_id=_require("QUERY_INDEX_ID"),
            upload_category_id=_require("UPLOAD_CATEGORY_ID"),
            parser=_optional("BAILIAN_PARSER", "DASHSCOPE_DOCMIND"),
        )

    @cached_property
    def checkpoint(self) -> CheckpointConfig:
        backend = _optional("CHECKPOINT_BACKEND", "redis").lower()
        if backend not in {"redis", "mongodb", "file"}:
            raise ConfigError(f"CHECKPOINT_BACKEND 只支持 redis / mongodb / file，当前值为 {backend!r}")
        return CheckpointConfig(
            backend=backend,
            redis_url=_optional("REDIS_URL", "redis://localhost:6379"),
            mongodb_url=_optional("MONGODB_URL", "mongodb://localhost:27017/?directConnection=true"),
            mongodb_db=_optional("MONGODB_DB", "chat"),
        )

    @cached_property
    def paths(self) -> PathsConfig:
        chromedriver_raw = _optional("CHROMEDRIVER_PATH")
        chromedriver = _as_path(chromedriver_raw) if chromedriver_raw else None
        return PathsConfig(
            project_root=PROJECT_ROOT,
            work_dir=_as_path(_optional("WORK_DIR", ".temp")),
            checkpoint_dir=_as_path(_optional("CHECKPOINT_DIR", str(Path(".temp") / "checkpoint"))),
            chromedriver=chromedriver,
        )

    @cached_property
    def guard(self) -> GuardConfig:
        audit_raw = _optional("GUARD_AUDIT_LOG", ".temp/audit.jsonl")
        config = GuardConfig(
            enabled=_as_bool("GUARD_ENABLED", True),
            approver=_optional("GUARD_APPROVER", "interactive").lower(),
            audit_log=_as_path(audit_raw) if audit_raw else None,
        )
        config.validate()
        return config

    @cached_property
    def trace(self) -> TraceConfig:
        return TraceConfig(
            enabled=_as_bool("TRACE_ENABLED", True),
            dir=_as_path(_optional("TRACE_DIR", ".temp/traces")),
            price_input_per_m=float(_optional("MODEL_PRICE_INPUT", "0") or 0),
            price_output_per_m=float(_optional("MODEL_PRICE_OUTPUT", "0") or 0),
        )

    @cached_property
    def context(self) -> ContextConfig:
        return ContextConfig(
            max_chars=_as_int("CONTEXT_MAX_CHARS", 24000),
            keep_last=_as_int("CONTEXT_KEEP_LAST", 12),
            strategy=_optional("CONTEXT_STRATEGY", "summarize").lower(),
            summary_max_chars=_as_int("CONTEXT_SUMMARY_MAX_CHARS", 2000),
        )

    @cached_property
    def debug(self) -> bool:
        return _as_bool("DEBUG")


def mcp_script(relative: str) -> Path:
    """定位一个 MCP server 脚本，返回绝对路径。

    `relative` 相对 app/code_agent/，例如 "mcp/mysql_tools.py"。
    换成绝对路径是为了让子进程无论从哪个 cwd 启动都能找到依赖。
    """
    path = (CODE_AGENT_ROOT / relative).resolve()
    if not path.is_relative_to(CODE_AGENT_ROOT):
        raise ConfigError(f"MCP server 路径不允许逃出 {CODE_AGENT_ROOT}：{relative}")
    if not path.is_file():
        raise ConfigError(f"MCP server 脚本不存在：{path}")
    return path


def mcp_python() -> str:
    """启动 MCP 子进程用的解释器。

    必须用当前解释器而不是裸 "python"，否则在 uv/venv 里子进程
    找不到已安装的依赖，表现为工具全部加载失败。
    """
    return sys.executable


settings = Settings()
