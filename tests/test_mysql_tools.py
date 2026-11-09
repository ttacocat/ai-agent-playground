"""MySQL 工具层的纯逻辑单测：不连数据库，只测 SQL 组装与转义。"""

from ai_agent_test.app.code_agent.mcp.mysql_tools import _quote_ident
from ai_agent_test.config import settings


def test_quote_ident_wraps_name():
    assert _quote_ident("user") == "`user`"


def test_quote_ident_escapes_backtick():
    # 反引号是 MySQL 标识符的包裹符，必须翻倍转义，否则就是注入入口
    assert _quote_ident("us`er") == "`us``er`"


def test_quote_ident_neutralizes_suffix_injection():
    assert _quote_ident("user`; DROP TABLE secrets; --") == "`user``; DROP TABLE secrets; --`"


def test_mysql_config_reads_from_env_not_source():
    """连接参数必须来自配置层：源码里不允许再出现明文口令。"""
    import inspect

    import pytest

    from ai_agent_test.app.code_agent.mcp import mysql_tools
    from ai_agent_test.config import ConfigError

    source = inspect.getsource(mysql_tools)
    assert 'password": "root"' not in source
    assert '"password":' not in source

    try:
        mysql_cfg = settings.mysql
    except ConfigError:
        pytest.skip("本地未配置 MYSQL_*（CI 环境），连接参数断言仅在配置齐全的环境生效")

    kwargs = mysql_cfg.connect_kwargs("test")
    assert kwargs["database"] == "test"
    assert kwargs["charset"] == "utf8mb4"

    # 不传库名时只连 server，不选库
    kwargs_no_db = mysql_cfg.connect_kwargs(None)
    assert not kwargs_no_db.get("database")


def test_sql_builder_uses_parametrized_values():
    """insert/update/delete 的值必须走 %s 占位符 + 参数元组，不允许 f-string 拼值。"""
    import inspect

    from ai_agent_test.app.code_agent.mcp import mysql_tools

    for func_name in ("mysql_insert_data", "mysql_update_data", "mysql_delete_data"):
        source = inspect.getsource(getattr(mysql_tools, func_name))
        assert "%s" in source, f"{func_name} 未使用参数化占位符"
