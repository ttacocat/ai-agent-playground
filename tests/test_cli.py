"""CLI 层单测。"""

import pytest

from ai_agent_test.cli import MCP_SERVERS, cmd_serve_mcp, cmd_tools


class _Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_mcp_servers_registry_is_complete():
    assert {"mysql", "browser", "terminal", "shell", "vm", "rag", "self-rag"} <= set(MCP_SERVERS)


def test_tools_rejects_unknown_server(capsys):
    assert cmd_tools(_Args(server="nope")) == 2
    assert "未知 server" in capsys.readouterr().out


def test_serve_mcp_rejects_unknown_server(capsys):
    assert cmd_serve_mcp(_Args(name="nope")) == 2
    assert "未知 server" in capsys.readouterr().out


def test_help_lists_all_commands(capsys):
    from ai_agent_test.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("doctor", "chat", "tools", "serve-mcp", "rag-upload"):
        assert cmd in out
