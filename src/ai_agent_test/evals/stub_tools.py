"""与真实 MCP 工具同名、同参数、同描述的 stub 实现。

用途：让真实模型产生真实的工具选择行为，但执行落在本进程内的固定 stub 上——
    不碰真数据库 / 虚拟机 / 外网，评测可重复、零副作用、成本只有 LLM token。
描述文案必须与 mcp/*.py 中的真实工具保持一致，否则测出来的「模型会不会选它」失真。
"""

from __future__ import annotations

from langchain_core.tools import tool

STUB_RESULT = "[stub] 工具执行成功（评测环境，返回固定结果）"


@tool
def mysql_list_databases() -> str:
    """列举MySQL中包含哪些数据库"""
    return f"{STUB_RESULT}: information_schema, mysql, test"


@tool
def mysql_list_tables(database: str) -> str:
    """获取指定数据库中的所有表"""
    return f"{STUB_RESULT}: {database} -> user, orders, product"


@tool
def mysql_describe_tables(database: str, table: str) -> str:
    """获取表结构信息"""
    return f"{STUB_RESULT}: {database}.{table} -> id INT PK, name VARCHAR(50), age INT"


@tool
def mysql_execute_query(command: str, database: str = "") -> str:
    """执行sql查询语句"""
    return f"{STUB_RESULT}: query executed -> 3 rows"


@tool
def mysql_insert_data(database: str, table: str, data: dict[str, str]) -> str:
    """向指定数据库的表中插入一条数据"""
    return f"{STUB_RESULT}: 1 row inserted"


@tool
def mysql_delete_data(database: str, table: str, data: dict[str, str]) -> str:
    """根据条件删除指定数据库表中的数据,data 中的多个条件用 AND 连接"""
    return f"{STUB_RESULT}: 1 row deleted"


@tool
def run_shell(command: str) -> str:
    """run a shell command"""
    return f"{STUB_RESULT}: stdout of `{command}`"


@tool
def query_rag(query: str) -> str:
    """从百炼平台查询知识库信息"""
    return f"{STUB_RESULT}: 命中 2 个知识节点（评测环境）"


@tool
def search_in_duckduckgo(query: str) -> str:
    """search query word in duckduckgo"""
    return f"{STUB_RESULT}: search results for `{query}`"


@tool
def make_dir_in_vm(dir_path: str) -> str:
    """在指定的虚拟机中创建目录，相当于 mkdir -p 命令"""
    return f"{STUB_RESULT}: dir {dir_path} created"


@tool
def run_script_in_terminal(script: str) -> str:
    """run a shell script in the front Terminal window"""
    return f"{STUB_RESULT}: script dispatched"


STUB_TOOLS = [
    mysql_list_databases,
    mysql_list_tables,
    mysql_describe_tables,
    mysql_execute_query,
    mysql_insert_data,
    mysql_delete_data,
    run_shell,
    query_rag,
    search_in_duckduckgo,
    make_dir_in_vm,
    run_script_in_terminal,
]

STUB_TOOL_NAMES = {t.name for t in STUB_TOOLS}
