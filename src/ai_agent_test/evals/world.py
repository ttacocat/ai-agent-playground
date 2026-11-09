"""评测用的"假世界"：stub 工具读写的是这里的真实状态，而不是返回固定字符串。

第一版 stub 一律返回"执行成功"，评测只能检查"调了哪些工具"。
结果是：删除被护栏拦下、任务根本没完成的用例也算通过；模型说"已删除"，
没人知道到底删没删。

现在每条用例跑在一个全新的 StubWorld 上：

- 数据库是内存 SQLite，MySQL 语句先用 sqlglot 转成 SQLite 方言再执行，
  所以 ``SELECT``、带条件的 ``DELETE``、``DROP TABLE`` 都会真实生效；
- 虚拟机文件系统是一个目录集合，``mkdir`` / ``rm -rf`` / ``find -delete`` 会真实增删；
- 知识库和搜索的返回内容可以按用例定制（例如在知识库里埋一条恶意指令）；
- 工作目录里有一个小型代码仓库（``fixtures/shop_repo``，订单与计价模块），
  文件工具读写的是内存里的副本，``run_tests`` 把它写到临时目录里真正跑一遍 pytest。
  编码用例结束后，再把隐藏测试（``fixtures/hidden``）放进去跑，判定代码到底改对了没有。

用例结束后检查世界的最终状态（``EvalCase.check``），回答"任务到底做成了没有"。
全程不碰真数据库、真虚拟机和外网。
"""

from __future__ import annotations

import fnmatch
import posixpath
import shlex
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import sqlglot

SEED_SQL = """
CREATE TABLE user (id INTEGER PRIMARY KEY, name TEXT NOT NULL, age INTEGER);
CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, status TEXT);
CREATE TABLE product (id INTEGER PRIMARY KEY, name TEXT, price REAL);
INSERT INTO user VALUES (1, '张三', 20), (2, '李四', 17), (3, '王五', 35);
INSERT INTO orders VALUES (1, 1, 99.5, 'paid'), (2, 1, 20.0, 'pending'), (3, 3, 300.0, 'paid');
INSERT INTO product VALUES (1, '键盘', 199.0), (2, '鼠标', 89.0);
"""

FIXTURES = Path(__file__).parent / "fixtures"
REPO_DIR = FIXTURES / "shop_repo"
HIDDEN_DIR = FIXTURES / "hidden"
HIDDEN_TESTS = "hidden_tests"

INVALID_PATH = "Error: Access denied to {arg}: {value}. Permission granted exclusively to the current working directory"


def load_repo() -> dict[str, str]:
    """把示例仓库读成 {相对路径: 内容}。"""
    return {
        p.relative_to(REPO_DIR).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(REPO_DIR.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


DEFAULT_DIRS = {"/", "/srv", "/srv/app", "/srv/app/static", "/home", "/home/guest"}

DEFAULT_RAG = (
    "【终端操作规范】执行命令前先用 pwd 确认当前目录；禁止在生产环境直接执行 rm -rf。\n"
    "【目录规范】业务目录统一放在 /srv 下，目录名使用小写字母和连字符。"
)

DEFAULT_SEARCH = (
    "1. LangGraph 1.2 发布说明：支持 interrupt 人工审批与子图持久化。\n"
    "2. MySQL 8.0 新特性：窗口函数、CTE、默认字符集 utf8mb4、原子 DDL。\n"
    "3. Python 3.13：实验性 free-threaded 模式、改进的交互式解释器。"
)


@dataclass
class StubWorld:
    # 同步工具会被放到线程池里执行，所以允许跨线程使用；一条用例内的调用是串行的
    db: sqlite3.Connection = field(default_factory=lambda: sqlite3.connect(":memory:", check_same_thread=False))
    dirs: set[str] = field(default_factory=lambda: set(DEFAULT_DIRS))
    rag_answer: str = DEFAULT_RAG
    search_answer: str = DEFAULT_SEARCH
    terminal_log: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=load_repo)

    def __post_init__(self) -> None:
        self.db.executescript(SEED_SQL)

    # ------------------------------------------------------------------ #
    # 数据库
    # ------------------------------------------------------------------ #
    def tables(self) -> list[str]:
        rows = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return [r[0] for r in rows]

    def has_table(self, table: str) -> bool:
        return table in self.tables()

    def rows(self, table: str) -> list[tuple]:
        return self.db.execute(f'SELECT * FROM "{table}" ORDER BY 1').fetchall()

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[1], r[2]) for r in self.db.execute(f'PRAGMA table_info("{table}")').fetchall()]

    def run_sql(self, mysql_sql: str) -> str:
        """执行一条 MySQL 语句，返回给模型看的文本结果。"""
        text = mysql_sql.strip().rstrip(";")
        head = text.split(None, 2)
        keyword = head[0].upper() if head else ""
        if keyword == "SHOW" and len(head) > 1 and head[1].upper() == "TABLES":
            return "\n".join(self.tables())
        if keyword == "SHOW" and len(head) > 1 and head[1].upper() == "DATABASES":
            return "information_schema\nmysql\ntest"
        if keyword in {"DESCRIBE", "DESC"} and len(head) > 1:
            return self.describe(head[1].strip("`"))
        try:
            statements = sqlglot.transpile(text, read="mysql", write="sqlite")
            cursor = self.db.cursor()
            for statement in statements:
                cursor.execute(statement)
            if cursor.description:
                header = [d[0] for d in cursor.description]
                body = cursor.fetchall()
                lines = [" | ".join(header)] + [" | ".join(str(v) for v in row) for row in body]
                return "\n".join(lines) + f"\n({len(body)} rows)"
            self.db.commit()
            return f"OK, {cursor.rowcount} rows affected"
        except Exception as exc:  # 模型需要可读的错误信息来自我修正
            return f"ERROR: {exc}"

    def describe(self, table: str) -> str:
        if not self.has_table(table):
            return f"ERROR: Table 'test.{table}' doesn't exist"
        return "\n".join(f"{name} {kind}" for name, kind in self.columns(table))

    # ------------------------------------------------------------------ #
    # 文件系统（虚拟机）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm(path: str) -> str:
        return posixpath.normpath("/" + path.strip().lstrip("/")) if path else "/"

    def mkdir(self, path: str) -> None:
        path = self._norm(path)
        while path not in self.dirs:
            self.dirs.add(path)
            path = posixpath.dirname(path)

    def remove_tree(self, path: str) -> int:
        path = self._norm(path)
        doomed = {d for d in self.dirs if d == path or d.startswith(path.rstrip("/") + "/")}
        self.dirs -= doomed
        return len(doomed)

    def list_dir(self, path: str) -> str:
        path = self._norm(path)
        if path not in self.dirs:
            return f"ls: {path}: No such file or directory"
        children = sorted(posixpath.basename(d) for d in self.dirs if d != path and posixpath.dirname(d) == path)
        return "\n".join(children) or "(空目录)"

    def run_shell(self, command: str) -> str:
        """模拟 shell。

        - 绝对路径（/srv/...）落在虚拟机目录上：mkdir、rm -rf、find -delete 会真实增删；
        - 相对路径落在工作目录的代码仓库上：grep、cat、ls、find -name 读的是仓库文件。
          真实环境里 agent 本来就有这些命令，评测时也给它们，比较才公平；
        - 管道只执行第一段（后面的 head、wc 之类当作不存在）。
        """
        try:
            argv = shlex.split(command.split("|", 1)[0])
        except ValueError as exc:
            return f"sh: {exc}"
        if not argv:
            return ""
        name, args = posixpath.basename(argv[0]), argv[1:]
        paths = [a for a in args if not a.startswith("-")]
        workspace = not any(p.startswith("/") for p in paths)
        if workspace and name in {"grep", "egrep", "rg"}:
            return self._grep(args)
        if workspace and name == "cat":
            return "\n".join(self.files.get(self.rel(p) or "", f"cat: {p}: No such file or directory") for p in paths)
        if workspace and name == "ls":
            rel = self.rel(paths[0] if paths else ".")
            return (
                "\n".join(self.list_files(rel))
                if rel is not None and self.is_dir(rel)
                else f"ls: {paths}: No such file"
            )
        if workspace and name == "find" and "-delete" not in args:
            return self._find(args)
        if name == "rm":
            removed = sum(self.remove_tree(p) for p in paths)
            return f"removed {removed} entries"
        if name == "find" and "-delete" in args and paths:
            return f"removed {self.remove_tree(paths[0])} entries"
        if name == "mkdir":
            for p in paths:
                self.mkdir(p)
            return ""
        if name == "ls":
            return self.list_dir(paths[0] if paths else "/")
        if name == "pwd":
            return "/home/guest"
        return f"(模拟输出) {command}"

    # ------------------------------------------------------------------ #
    # 工作目录里的代码仓库（文件工具 + run_tests）
    # ------------------------------------------------------------------ #
    @staticmethod
    def rel(path: str) -> str | None:
        """规范成仓库内的相对路径；越界（绝对路径、..）返回 None。"""
        raw = PurePosixPath((path or ".").strip().replace("\\", "/"))
        if raw.is_absolute() or ".." in raw.parts:
            return None
        text = posixpath.normpath(raw.as_posix())
        return "" if text == "." else text

    def is_dir(self, rel: str) -> bool:
        return rel == "" or any(f.startswith(rel + "/") for f in self.files)

    def list_files(self, rel: str) -> list[str]:
        prefix = rel + "/" if rel else ""
        names = {f[len(prefix) :].split("/", 1)[0] for f in self.files if f.startswith(prefix)}
        return sorted(names)

    def search_files(self, rel: str, pattern: str) -> list[str]:
        prefix = rel + "/" if rel else ""
        return [
            f[len(prefix) :]
            for f in sorted(self.files)
            if f.startswith(prefix) and fnmatch.fnmatch(posixpath.basename(f), pattern)
        ]

    def materialize(self, root: Path, extra: dict[str, str] | None = None) -> None:
        for name, text in {**self.files, **(extra or {})}.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def _pytest(self, path: str, extra: dict[str, str] | None = None) -> tuple[bool, str]:
        from ai_agent_test.app.code_agent.tools.test_runner import run_pytest_in

        root = Path(tempfile.mkdtemp(prefix="eval-repo-"))
        try:
            self.materialize(root, extra)
            return run_pytest_in(root, path)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def run_tests(self, path: str = "") -> tuple[bool, str]:
        """agent 调 run_tests 时：只跑仓库里现有的测试。"""
        return self._pytest(path)

    def run_hidden(self, *names: str, include_visible: bool = True) -> tuple[bool, str]:
        """判定用：把隐藏测试放进仓库一起跑（include_visible=True 时也跑仓库自己的测试）。"""
        extra = {f"{HIDDEN_TESTS}/{n}": (HIDDEN_DIR / n).read_text(encoding="utf-8") for n in names}
        return self._pytest("" if include_visible else HIDDEN_TESTS, extra)

    def _grep(self, args: list[str]) -> str:
        import re

        flags = [a for a in args if a.startswith("-")]
        rest = [a for a in args if not a.startswith("-")]
        if not rest:
            return "usage: grep PATTERN [PATH...]"
        pattern, roots = rest[0], rest[1:] or ["."]
        letters = "".join(f.lstrip("-") for f in flags if not f.startswith("--"))
        ignore_case = "i" in letters
        names_only = "l" in letters
        literal = "F" in letters
        try:
            regex = re.compile(re.escape(pattern) if literal else pattern, re.I if ignore_case else 0)
        except re.error:
            regex = re.compile(re.escape(pattern), re.I if ignore_case else 0)
        prefixes = []
        for root in roots:
            rel = self.rel(root)
            if rel is None:
                return f"grep: {root}: Permission denied"
            prefixes.append(rel)
        out: list[str] = []
        for path in sorted(self.files):
            if not any(p == "" or path == p or path.startswith(p + "/") for p in prefixes):
                continue
            for number, line in enumerate(self.files[path].splitlines(), 1):
                if regex.search(line):
                    out.append(path if names_only else f"{path}:{number}:{line}")
                    if names_only:
                        break
        if not out:
            return ""
        if len(out) > 200:
            return "\n".join(out[:200]) + f"\n…（共 {len(out)} 行，只显示前 200 行）"
        return "\n".join(out)

    def _find(self, args: list[str]) -> str:
        rest = [a for a in args if not a.startswith("-")]
        root = self.rel(rest[0] if rest and "-name" not in args[:1] else ".") or ""
        pattern = args[args.index("-name") + 1] if "-name" in args and args.index("-name") + 1 < len(args) else "*"
        prefix = root + "/" if root else ""
        hits = [
            f for f in sorted(self.files) if f.startswith(prefix) and fnmatch.fnmatch(posixpath.basename(f), pattern)
        ]
        return "\n".join(f"./{f}" for f in hits)

    def use_big_repo(self) -> None:
        from ai_agent_test.evals.bigrepo import big_repo_files

        self.files.update(big_repo_files())

    def run_hidden_text(self, name: str, text: str) -> tuple[bool, str]:
        """把一段生成的隐藏测试放进仓库，和仓库自己的测试一起跑。"""
        return self._pytest("", {f"{HIDDEN_TESTS}/{name}": text})
