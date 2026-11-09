"""工具调用的风险判定规则。

分三级：

- ``read``        只读，直接放行
- ``write``       写操作，放行但留审计
- ``destructive`` 可能造成不可逆影响，必须人工确认

判定看两件事：**工具本身的默认风险** + **参数内容**。

只看工具名是不够的：``run_shell("ls")`` 和 ``run_shell("rm -rf /")`` 风险差三个
数量级；``mysql_execute_query("SELECT 1")`` 和 ``...("DROP TABLE user")`` 同理。
反过来只看参数也不行：``mysql_delete_data`` 光看参数长得像写操作，但它是删数据。

本模块是纯函数、零 IO 依赖，便于单测与在 CI 里跑。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RiskLevel(StrEnum):
    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"

    @property
    def rank(self) -> int:
        return {"read": 0, "write": 1, "destructive": 2}[self.value]


class Action(StrEnum):
    ALLOW = "allow"
    CONFIRM = "confirm"


@dataclass(frozen=True)
class Decision:
    tool: str
    level: RiskLevel
    action: Action
    reason: str
    preview: str

    @property
    def needs_approval(self) -> bool:
        return self.action is Action.CONFIRM


# --------------------------------------------------------------------------- #
# 工具默认风险表
# --------------------------------------------------------------------------- #
_READ_TOOLS = {
    "mysql_list_databases",
    "mysql_list_tables",
    "mysql_describe_tables",
    "query_rag",
    "get_terminal_full_text",
    "list_terminal_windows",
    "list_files",
}

_WRITE_TOOLS = {
    "mysql_insert_data",
    "mysql_update_data",
    "mysql_create_table",
    "make_dir_in_vm",
    "write_file",
    "edit_file",
    # 下面几个的风险完全由参数内容决定，见 _CONTENT_DECIDED_TOOLS
    "run_shell",
    "mysql_execute_query",
    "execute_command",
}

_DESTRUCTIVE_TOOLS = {
    "mysql_delete_data",
    "mysql_drop_table",
    # 这两类会把内容直接敲进用户当前打开的终端窗口，不看内容也要确认
    "run_script_in_terminal",
    "send_terminal_keyboard_key",
    "execute_script_in_vm",
    "delete_file",
    "move_file",
}

# 名字看不出来时的兜底前缀规则（MCP server 以后加工具也能自动分类）
_READ_PREFIXES = ("list_", "get_", "read_", "search_", "show_", "describe_", "query_")

# 这几个工具参数本身就是一段可执行文本：等级 **由内容决定**，不叠加默认值。
# 否则 run_shell 会永远被判 destructive，`ls` 都要人确认一次，护栏就成了噪音。
_SHELL_LIKE_TOOLS = {"run_shell", "run_script_in_terminal", "execute_command", "execute_script_in_vm"}
_SQL_LIKE_TOOLS = {"mysql_execute_query"}
_CONTENT_DECIDED_TOOLS = {"run_shell", "execute_command", "mysql_execute_query"}


def baseline_level(tool_name: str) -> RiskLevel:
    if tool_name in _DESTRUCTIVE_TOOLS:
        return RiskLevel.DESTRUCTIVE
    if tool_name in _READ_TOOLS:
        return RiskLevel.READ
    if tool_name in _WRITE_TOOLS:
        return RiskLevel.WRITE
    if tool_name.startswith(_READ_PREFIXES):
        return RiskLevel.READ
    # 未知工具按写操作处理：不认识的东西不该被当成只读
    return RiskLevel.WRITE


# --------------------------------------------------------------------------- #
# 内容扫描
# --------------------------------------------------------------------------- #
_SHELL_RULES: list[tuple[str, str]] = [
    (r"\brm\b", "包含 rm 删除命令"),
    (r"\bmkfs(\.\w+)?\b", "格式化文件系统"),
    (r"\bdd\b[^\n]*\bof=", "dd 写裸设备"),
    (r">\s*/dev/(sd|nvme|disk)", "重定向写入磁盘设备"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "关机或重启"),
    (r"\bchmod\s+(-[a-zA-Z]+\s+)*777\b", "权限放大到 777"),
    (r"\bchown\s+-R\b", "递归改变属主"),
    (r"\bgit\s+clean\s+-[a-zA-Z]*[fx]", "git clean 会删除未跟踪文件"),
    (r"\bgit\s+reset\s+--hard\b", "git reset --hard 会丢弃改动"),
    (r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z)?sh\b", "下载脚本直接管道执行"),
    (r":\(\)\s*\{", "fork 炸弹"),
    (r"\btruncate\s+-s\s*0\b", "清空文件内容"),
    (r"\bkill(all)?\s+-9\b", "强杀进程"),
    (r"\bdrop\s+(table|database)\b", "SQL 删除表或库"),
]

_SHELL_PATTERNS = [(re.compile(p, re.I), why) for p, why in _SHELL_RULES]


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _statement_count(sql: str) -> int:
    body = sql.rstrip().rstrip(";").strip()
    return body.count(";") + 1 if body else 0


def scan_sql(sql: str) -> tuple[RiskLevel | None, str]:
    """按第一条语句的关键字判定 SQL 风险。"""
    body = _strip_sql_comments(sql).strip()
    if not body:
        return None, ""
    if _statement_count(body) > 1:
        return RiskLevel.DESTRUCTIVE, "包含多条语句（可能用分号接危险操作）"

    first = body.split(None, 1)[0].upper()
    if first in {"DROP", "TRUNCATE", "RENAME"}:
        return RiskLevel.DESTRUCTIVE, f"{first} 语句不可回滚"
    if first == "ALTER" and re.search(r"\bdrop\b", body, re.I):
        return RiskLevel.DESTRUCTIVE, "ALTER TABLE ... DROP 会丢列或索引"
    if first in {"DELETE", "UPDATE"} and not re.search(r"\bwhere\b", body, re.I):
        return RiskLevel.DESTRUCTIVE, f"{first} 缺少 WHERE，会影响整张表"
    if first in {"DELETE", "UPDATE", "INSERT", "REPLACE"}:
        return RiskLevel.WRITE, f"{first} 为带条件的写操作"
    return RiskLevel.READ, f"{first} 为只读查询"


def scan_shell(command: str) -> tuple[RiskLevel | None, str]:
    for pattern, why in _SHELL_PATTERNS:
        if pattern.search(command):
            return RiskLevel.DESTRUCTIVE, why
    return RiskLevel.WRITE, "未命中危险模式"


def _payload(args: Mapping[str, Any]) -> str:
    """取出需要扫描的文本：优先 command / script / sql 字段。"""
    for key in ("command", "script", "sql", "query"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _truncate(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def assess(tool_name: str, args: Mapping[str, Any] | None = None) -> Decision:
    args = dict(args or {})
    level = baseline_level(tool_name)
    reason = f"{tool_name} 默认等级为 {level.value}"
    payload = _payload(args)

    if payload and (tool_name in _SHELL_LIKE_TOOLS or tool_name in _SQL_LIKE_TOOLS):
        scanner = scan_shell if tool_name in _SHELL_LIKE_TOOLS else scan_sql
        found, why = scanner(payload)
        if found is not None:
            if tool_name in _CONTENT_DECIDED_TOOLS or found.rank > level.rank:
                level, reason = found, why
            else:
                reason = why

    # 结构化删除/更新工具：条件是空的等于全表操作
    if tool_name in {"mysql_delete_data", "mysql_update_data"}:
        condition = args.get("data") or args.get("where")
        if not condition:
            level, reason = RiskLevel.DESTRUCTIVE, "缺少过滤条件，等价于全表操作"

    action = Action.CONFIRM if level is RiskLevel.DESTRUCTIVE else Action.ALLOW
    preview = _truncate(payload) if payload else _truncate(str(args))
    return Decision(tool=tool_name, level=level, action=action, reason=reason, preview=preview)


def describe(decision: Decision) -> str:
    """给终端用户看的一行说明。"""
    return (
        f"即将执行 {decision.tool}（风险：{decision.level.value}）\n"
        f"    原因：{decision.reason}\n"
        f"    参数：{decision.preview}"
    )
