"""与真实 MCP 工具同名、同参数、同描述的 stub 实现。

用途：让真实模型产生真实的工具选择行为，但执行落在本进程内的 ``StubWorld`` 上——
不碰真数据库 / 虚拟机 / 外网，评测可重复、零副作用、成本只有 LLM token。
描述文案必须与 mcp/*.py 中的真实工具保持一致，否则测出来的「模型会不会选它」失真。

每条用例用 ``build_stub_tools(world)`` 拿一套绑定到新世界的工具，互不干扰。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from ai_agent_test.app.code_agent.tools.test_runner import RUN_TESTS_DESCRIPTION
from ai_agent_test.evals.world import INVALID_PATH, StubWorld

DATABASES = ("information_schema", "mysql", "test")

# 与 mcp/shell_tools.py 里的描述保持一致（tests/test_eval_runner.py 有校验）
RUN_SHELL_DESCRIPTION = (
    "在本机（不是虚拟机）的工作目录中执行一条 shell 命令并返回输出。"
    "虚拟机里的操作请用名字带 _in_vm 的工具。只读命令直接执行，其余命令需要用户确认。"
)


def _check_db(database: str) -> str | None:
    if database and database != "test":
        return f"ERROR: Unknown database '{database}'"
    return None


def _quote(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def build_stub_tools(world: StubWorld) -> list[BaseTool]:
    @tool
    def mysql_list_databases() -> str:
        """列举MySQL中包含哪些数据库"""
        return "\n".join(DATABASES)

    @tool
    def mysql_list_tables(database: str) -> str:
        """获取指定数据库中的所有表"""
        return _check_db(database) or "\n".join(world.tables())

    @tool
    def mysql_describe_tables(database: str, table: str) -> str:
        """获取表结构信息"""
        return _check_db(database) or world.describe(table)

    @tool
    def mysql_execute_query(command: str, database: str = "") -> str:
        """执行sql查询语句"""
        return _check_db(database) or world.run_sql(command)

    @tool
    def mysql_insert_data(database: str, table: str, data: dict[str, str]) -> str:
        """向指定数据库的表中插入一条数据"""
        if error := _check_db(database):
            return error
        if not data:
            return "mysql insert data error: data is empty"
        columns = ", ".join(_quote(k) for k in data)
        marks = ", ".join("?" for _ in data)
        try:
            world.db.execute(f"INSERT INTO {_quote(table)} ({columns}) VALUES ({marks})", tuple(data.values()))
            world.db.commit()
        except Exception as exc:
            return f"mysql insert data error: {exc}"
        return "1 row inserted"

    @tool
    def mysql_update_data(database: str, table: str, data: dict[str, str], where: dict[str, str]) -> str:
        """根据条件更新指定数据库表中的数据,where 中的多个条件用 AND 连接"""
        if error := _check_db(database):
            return error
        if not data:
            return "mysql update data error: data is empty"
        if not where:
            return "mysql update data error: where is empty, refuse to update the whole table"
        sets = ", ".join(f"{_quote(k)} = ?" for k in data)
        conds = " AND ".join(f"{_quote(k)} = ?" for k in where)
        try:
            cursor = world.db.execute(
                f"UPDATE {_quote(table)} SET {sets} WHERE {conds}", (*data.values(), *where.values())
            )
            world.db.commit()
        except Exception as exc:
            return f"mysql update data error: {exc}"
        return f"{cursor.rowcount} rows updated"

    @tool
    def mysql_delete_data(database: str, table: str, data: dict[str, str]) -> str:
        """根据条件删除指定数据库表中的数据,data 中的多个条件用 AND 连接"""
        if error := _check_db(database):
            return error
        conds = " AND ".join(f"{_quote(k)} = ?" for k in data) or "1=1"
        try:
            cursor = world.db.execute(f"DELETE FROM {_quote(table)} WHERE {conds}", tuple((data or {}).values()))
            world.db.commit()
        except Exception as exc:
            return f"mysql delete data error: {exc}"
        return f"{cursor.rowcount} rows deleted"

    @tool("run_shell", description=RUN_SHELL_DESCRIPTION)
    def run_shell(command: str) -> str:
        return world.run_shell(command)

    @tool
    def query_rag(query: str) -> str:
        """从百炼平台查询知识库信息"""
        return world.rag_answer

    @tool
    def search_in_duckduckgo(query: str) -> str:
        """search query word in duckduckgo"""
        return f"搜索 `{query}` 的结果：\n{world.search_answer}"

    @tool
    def make_dir_in_vm(dir_path: str) -> str:
        """在指定的虚拟机中创建目录，相当于 mkdir -p 命令"""
        world.mkdir(dir_path)
        return ""

    @tool
    def list_files_in_vm(dir_path: str) -> str:
        """查看虚拟机中指定目录，相当于 ls -al 命令"""
        return world.list_dir(dir_path)

    @tool
    def run_script_in_terminal(script: str) -> str:
        """run a shell script in the front Terminal window"""
        world.terminal_log.append(script)
        return world.run_shell(script)

    # ---- 工作目录里的代码仓库：与 FileManagementToolkit 同名、同参数、同描述、同返回文案 ----
    def bad(arg: str, value: str) -> str:
        return INVALID_PATH.format(arg=arg, value=value)

    @tool
    def read_file(file_path: str) -> str:
        """Read file from disk"""
        rel = world.rel(file_path)
        if rel is None:
            return bad("file_path", file_path)
        if rel not in world.files:
            return f"Error: no such file or directory: {file_path}"
        return world.files[rel]

    @tool
    def write_file(file_path: str, text: str, append: bool = False) -> str:
        """Write file to disk"""
        rel = world.rel(file_path)
        if rel is None or rel == "":
            return bad("file_path", file_path)
        world.files[rel] = (world.files.get(rel, "") + text) if append else text
        return f"File written successfully to {file_path}."

    @tool
    def list_directory(dir_path: str = ".") -> str:
        """List files and directories in a specified folder"""
        rel = world.rel(dir_path)
        if rel is None:
            return bad("dir_path", dir_path)
        if not world.is_dir(rel):
            return f"Error: [Errno 2] No such file or directory: '{dir_path}'"
        return "\n".join(world.list_files(rel)) or f"No files found in directory {dir_path}"

    @tool
    def file_search(pattern: str, dir_path: str = ".") -> str:
        """Recursively search for files in a subdirectory that match the regex pattern"""
        rel = world.rel(dir_path)
        if rel is None:
            return bad("dir_path", dir_path)
        matches = world.search_files(rel, pattern)
        return "\n".join(matches) or f"No files found for pattern {pattern} in directory {dir_path}"

    @tool
    def copy_file(source_path: str, destination_path: str) -> str:
        """Create a copy of a file in a specified location"""
        src, dst = world.rel(source_path), world.rel(destination_path)
        if src is None or dst is None:
            return bad("source_path" if src is None else "destination_path", source_path)
        if src not in world.files:
            return f"Error: no such file or directory {source_path}"
        world.files[dst] = world.files[src]
        return f"File copied successfully from {source_path} to {destination_path}."

    @tool
    def move_file(source_path: str, destination_path: str) -> str:
        """Move or rename a file from one location to another"""
        src, dst = world.rel(source_path), world.rel(destination_path)
        if src is None or dst is None:
            return bad("source_path" if src is None else "destination_path", source_path)
        if src not in world.files:
            return f"Error: no such file or directory {source_path}"
        world.files[dst] = world.files.pop(src)
        return f"File moved successfully from {source_path} to {destination_path}."

    @tool
    def file_delete(file_path: str) -> str:
        """Delete a file"""
        rel = world.rel(file_path)
        if rel is None:
            return bad("file_path", file_path)
        if world.files.pop(rel, None) is None:
            return f"Error: no such file or directory: {file_path}"
        return f"File deleted successfully: {file_path}."

    @tool("run_tests", description=RUN_TESTS_DESCRIPTION)
    def run_tests(path: str = "") -> str:
        passed, output = world.run_tests(path)
        return ("✔ 全部通过\n" if passed else "✘ 有测试未通过\n") + output

    return [
        mysql_list_databases,
        mysql_list_tables,
        mysql_describe_tables,
        mysql_execute_query,
        mysql_insert_data,
        mysql_update_data,
        mysql_delete_data,
        run_shell,
        query_rag,
        search_in_duckduckgo,
        make_dir_in_vm,
        list_files_in_vm,
        run_script_in_terminal,
        read_file,
        write_file,
        list_directory,
        file_search,
        copy_file,
        move_file,
        file_delete,
        run_tests,
    ]


STUB_TOOLS = build_stub_tools(StubWorld())
STUB_TOOL_NAMES = {t.name for t in STUB_TOOLS}
