"""配置层单测：不依赖任何外部服务，可裸跑。"""

import sys

import pytest

from ai_agent_test.config import (
    CODE_AGENT_ROOT,
    PROJECT_ROOT,
    ConfigError,
    mcp_python,
    mcp_script,
)


def test_project_root_is_repo_root():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "src" / "ai_agent_test" / "config.py").is_file()


def test_mcp_python_is_current_interpreter():
    assert mcp_python() == sys.executable


def test_mcp_script_resolves_existing_server():
    path = mcp_script("mcp/mysql_tools.py")
    assert path == CODE_AGENT_ROOT / "mcp" / "mysql_tools.py"
    assert path.is_file()


def test_mcp_script_missing_raises_with_hint():
    with pytest.raises(ConfigError) as exc_info:
        mcp_script("mcp/definitely_not_exists.py")
    assert "不存在" in str(exc_info.value)


def test_mcp_script_rejects_path_escape():
    # ../ 逃出 code_agent 目录的路径也必须报错，而不是静默解析成功
    with pytest.raises(ConfigError):
        mcp_script("../../pyproject.toml")


def test_all_registered_mcp_servers_exist():
    from ai_agent_test.cli import MCP_SERVERS

    assert len(MCP_SERVERS) >= 7
    for name, rel in MCP_SERVERS.items():
        assert mcp_script(rel).is_file(), f"server {name} 的脚本缺失：{rel}"


def test_env_example_covers_all_required_keys():
    """没进 .env.example 的必填项，等于让用户猜配置。"""
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for key in [
        "MODEL_API_KEY",
        "MYSQL_PASSWORD",
        "ALI_ACCESS_KEY_ID",
        "WORKSPACE_ID",
        "QUERY_INDEX_ID",
        "UPLOAD_CATEGORY_ID",
        "CHECKPOINT_BACKEND",
    ]:
        assert key in example, f".env.example 缺少 {key}"
