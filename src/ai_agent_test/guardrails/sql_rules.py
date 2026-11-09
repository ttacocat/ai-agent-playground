"""SQL 的风险判定：解析成语法树后按节点判，而不是看第一个关键字。

第一版取首个关键字判级，这些写法都能骗过它：

- ``WITH x AS (SELECT 1) DELETE FROM user``：首词是 WITH，实际是全表删除
- ``DELETE FROM user WHERE 1=1``：有 WHERE，但条件恒真
- ``GRANT ALL ON *.* TO ...``：不在任何分支里，落进了 read
- ``SELECT ... INTO OUTFILE '/tmp/x'``：首词是 SELECT，实际在写服务器文件

现在的做法：

1. 用 sqlglot（MySQL 方言）解析出 **全部** 语句，多于一条直接要确认；
2. 对每条语句按 AST 节点类型判级，遍历整棵树找 Drop / 危险函数等子节点；
3. ``UPDATE`` / ``DELETE`` 检查 WHERE 是否真的引用了列、是否有恒真的 OR 分支；
4. sqlglot 不支持、退化成 ``Command`` 的语句，以及解析失败的语句，**默认要确认**。

和 shell 一样是白名单思路：只有明确认识的只读结构才放行。
这是第一道防线；第二道是给 agent 用的 MySQL 账号只授予必要权限。
"""

from __future__ import annotations

import logging
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from ai_agent_test.guardrails.levels import RiskLevel, Verdict, worst

READ, WRITE, DESTRUCTIVE = RiskLevel.READ, RiskLevel.WRITE, RiskLevel.DESTRUCTIVE

# sqlglot 遇到不支持的语法会打 warning 再降级成 Command，这里由我们自己处理
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# 只读语句里出现这些函数同样危险：读服务器文件、拖垮数据库、加全局锁
_DANGEROUS_FUNCTIONS = {
    "LOAD_FILE": "LOAD_FILE 读取数据库服务器上的文件",
    "SLEEP": "SLEEP 会长时间占用连接",
    "BENCHMARK": "BENCHMARK 会长时间占用 CPU",
    "GET_LOCK": "GET_LOCK 会加全局锁",
    "SYS_EXEC": "SYS_EXEC 执行系统命令",
    "SYS_EVAL": "SYS_EVAL 执行系统命令",
}

# 解析不了也能先认出来的危险结构，给出更具体的原因
_PRECHECKS = [
    (re.compile(r"\binto\s+(out|dump)file\b", re.I), "INTO OUTFILE 会在数据库服务器上写文件"),
    (re.compile(r"\bload\s+data\b", re.I), "LOAD DATA 从文件批量导入"),
]

# sqlglot 降级为 Command 时，按首词判
_COMMAND_LEVELS: dict[str, Verdict] = {
    "REPLACE": (WRITE, "REPLACE INTO 为写操作"),
}

_READ_NODES = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Show, exp.Describe, exp.Use)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).upper()
    return node.sql_name().upper()


def _scan_functions(tree: exp.Expression) -> Verdict | None:
    for node in tree.find_all(exp.Func):
        name = _function_name(node)
        if name in _DANGEROUS_FUNCTIONS:
            return DESTRUCTIVE, _DANGEROUS_FUNCTIONS[name]
    return None


def _is_constant(node: exp.Expression) -> bool:
    """不引用任何列的表达式（1、TRUE、1=1、'a'='a'）对每一行结果都一样。"""
    return node.find(exp.Column) is None


def _judge_where(kind: str, tree: exp.Expression) -> Verdict:
    where = tree.args.get("where")
    if where is None:
        return DESTRUCTIVE, f"{kind} 缺少 WHERE，会影响整张表"
    condition = where.this
    if _is_constant(condition):
        return DESTRUCTIVE, f"{kind} 的 WHERE 条件不引用任何列（恒真），等价于全表操作"
    for node in condition.find_all(exp.Or):
        if _is_constant(node.left) or _is_constant(node.right):
            return DESTRUCTIVE, f"{kind} 的 WHERE 含有恒真的 OR 分支，等价于全表操作"
    return WRITE, f"{kind} 为带条件的写操作"


def _judge_alter(tree: exp.Alter) -> Verdict:
    sql = tree.sql(dialect="mysql").upper()
    if re.search(r"\bDROP\b", sql):
        return DESTRUCTIVE, "ALTER TABLE ... DROP 会丢列或索引"
    if re.search(r"\bRENAME\b", sql):
        return DESTRUCTIVE, "ALTER TABLE ... RENAME 会让依赖它的代码失效"
    return WRITE, "ALTER TABLE 修改表结构"


def judge_statement(tree: exp.Expression) -> Verdict:
    func_verdict = _scan_functions(tree)
    if func_verdict:
        return func_verdict

    if isinstance(tree, exp.Command):
        keyword = str(tree.this).upper()
        return _COMMAND_LEVELS.get(keyword, (DESTRUCTIVE, f"{keyword} 语句不在白名单内，需人工确认"))
    if isinstance(tree, exp.Drop):
        return DESTRUCTIVE, "DROP 语句不可回滚"
    if isinstance(tree, exp.TruncateTable):
        return DESTRUCTIVE, "TRUNCATE 语句不可回滚"
    if isinstance(tree, exp.Alter):
        return _judge_alter(tree)
    if isinstance(tree, exp.Delete):
        return _judge_where("DELETE", tree)
    if isinstance(tree, exp.Update):
        return _judge_where("UPDATE", tree)
    if isinstance(tree, exp.Insert):
        return WRITE, "INSERT 为写操作"
    if isinstance(tree, exp.Create):
        return WRITE, "CREATE 新建对象"
    if isinstance(tree, exp.Set):
        sql = tree.sql(dialect="mysql").upper()
        if re.search(r"\b(GLOBAL|PERSIST|PERSIST_ONLY)\b", sql):
            return DESTRUCTIVE, "SET GLOBAL 会修改整个数据库实例的配置"
        return WRITE, "SET 修改会话变量"
    if isinstance(tree, _READ_NODES):
        # 子查询里藏写操作（CTE 里的 DELETE 等）
        for node in tree.find_all(exp.Delete, exp.Update, exp.Insert, exp.Drop):
            if node is not tree:
                return judge_statement(node)
        return READ, f"{tree.key.upper()} 为只读查询"
    return DESTRUCTIVE, f"{tree.key.upper()} 语句不在白名单内，需人工确认"


def scan_sql(sql: str) -> tuple[RiskLevel | None, str]:
    """返回 (等级, 原因)；空语句返回 (None, "")。"""
    if not sql.strip():
        return None, ""
    for pattern, why in _PRECHECKS:
        if pattern.search(sql):
            return DESTRUCTIVE, why
    try:
        trees = [t for t in sqlglot.parse(sql, read="mysql") if t is not None]
    except SqlglotError as exc:
        return DESTRUCTIVE, f"SQL 无法解析（{type(exc).__name__}），按危险处理"
    if not trees:
        return None, ""
    if len(trees) > 1:
        return DESTRUCTIVE, "包含多条语句（可能用分号接危险操作）"
    return worst([judge_statement(trees[0])], default=(READ, "只读查询"))
