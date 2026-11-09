"""工具调用的风险判定规则。

分三级：

- ``read``        只读，直接放行
- ``write``       写操作，放行但留审计
- ``destructive`` 可能造成不可逆影响，必须人工确认

判定看两件事：**工具本身的默认风险** + **参数内容**。
参数内容的判定是白名单思路：shell 只放行认识的只读命令，SQL 只放行解析得出的
只读语句，其余一律要人确认（细节见 shell_rules.py / sql_rules.py）。

只看工具名是不够的：``run_shell("ls")`` 和 ``run_shell("rm -rf /")`` 风险差三个
数量级；``mysql_execute_query("SELECT 1")`` 和 ``...("DROP TABLE user")`` 同理。
反过来只看参数也不行：``mysql_delete_data`` 光看参数长得像写操作，但它是删数据。

本模块是纯函数、零 IO 依赖，便于单测与在 CI 里跑。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ai_agent_test.guardrails.levels import RiskLevel
from ai_agent_test.guardrails.shell_rules import scan_shell
from ai_agent_test.guardrails.sql_rules import scan_sql

__all__ = [
    "Action",
    "Decision",
    "RiskLevel",
    "assess",
    "baseline_level",
    "describe",
    "is_content_decided",
    "scan_shell",
    "scan_sql",
]


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
    "file_search",
    # 只接受"测哪个路径"一个参数，命令写死，模型没法借它执行任意命令
    "run_tests",
    # lead 架构的子 agent 工具：子 agent 只拿只读工具
    "explore",
    "review",
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
    "mysql_execute_command",
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
    # FileManagementToolkit 里删除工具的真实名字是 file_delete：第一版只写了 delete_file，
    # 真正的删除工具落进"未知工具 = write"，不经确认就执行了
    "file_delete",
    "move_file",
}

# 名字看不出来时的兜底前缀规则（MCP server 以后加工具也能自动分类）
_READ_PREFIXES = ("list_", "get_", "read_", "search_", "show_", "describe_", "query_")

# 这几个工具参数本身就是一段可执行文本：等级 **由内容决定**，不叠加默认值。
# 否则 run_shell 会永远被判 destructive，`ls` 都要人确认一次，护栏就成了噪音。
_SHELL_LIKE_TOOLS = {"run_shell", "run_script_in_terminal", "execute_command", "execute_script_in_vm"}
_SQL_LIKE_TOOLS = {"mysql_execute_query", "mysql_execute_command"}
_CONTENT_DECIDED_TOOLS = {"run_shell", "execute_command", "mysql_execute_query", "mysql_execute_command"}


def is_content_decided(tool_name: str) -> bool:
    """该工具的风险等级是否完全由参数内容决定（同一个工具既能只读也能删库）。"""
    return tool_name in _CONTENT_DECIDED_TOOLS


# 给人看的说明：会出现在审批面板和审计日志里
_BASELINE_REASONS = {
    RiskLevel.READ: "{tool} 是只读工具",
    RiskLevel.WRITE: "{tool} 会写入数据",
    RiskLevel.DESTRUCTIVE: "{tool} 可能删除数据或产生不可逆的影响",
}


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
# 内容扫描：规则见 shell_rules.py（白名单 + 词法分段）与 sql_rules.py（AST）
# --------------------------------------------------------------------------- #


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
    reason = _BASELINE_REASONS[level].format(tool=tool_name)
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
    preview = _truncate(payload) if payload else _truncate(json.dumps(args, ensure_ascii=False, default=str))
    return Decision(tool=tool_name, level=level, action=action, reason=reason, preview=preview)


def describe(decision: Decision) -> str:
    """给终端用户看的一行说明。"""
    return (
        f"即将执行 {decision.tool}（风险：{decision.level.value}）\n"
        f"    原因：{decision.reason}\n"
        f"    参数：{decision.preview}"
    )
