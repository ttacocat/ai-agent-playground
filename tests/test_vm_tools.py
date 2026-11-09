"""VM 工具的命令组装单测：mock 掉 subprocess，只验证 argv 拼装是否安全。"""

import subprocess
from unittest.mock import patch

from ai_agent_test.app.code_agent.mcp import vm


def _fake_run(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _argv_of(mock_run):
    return mock_run.call_args.args[0]


def test_make_dir_builds_expected_argv():
    with patch.object(vm.subprocess, "run", return_value=_fake_run(stdout="")) as mock_run:
        vm.make_dir_in_vm("/srv/app")
    assert _argv_of(mock_run) == ["limactl", "shell", "default", "mkdir", "-p", "/srv/app"]


def test_path_with_space_survives_roundtrip():
    """路径含空格时，quote -> shlex.split 一来一回必须还是同一个 token。"""
    with patch.object(vm.subprocess, "run", return_value=_fake_run(stdout="")) as mock_run:
        vm.make_dir_in_vm("/srv/my app")
    assert _argv_of(mock_run) == ["limactl", "shell", "default", "mkdir", "-p", "/srv/my app"]


def test_semicolon_does_not_chain_commands():
    """shell=False + shlex.quote 下，整条注入串被降级成一个普通路径 token。"""
    with patch.object(vm.subprocess, "run", return_value=_fake_run(stdout="")) as mock_run:
        vm.make_dir_in_vm("/tmp/a; touch /tmp/pwned")
    argv = _argv_of(mock_run)
    assert argv == ["limactl", "shell", "default", "mkdir", "-p", "/tmp/a; touch /tmp/pwned"]
    assert mock_run.call_count == 1


def test_failed_command_returns_stderr_not_raise():
    with patch.object(vm.subprocess, "run", return_value=_fake_run(returncode=1, stderr="no such dir")):
        out = vm.make_dir_in_vm("/nope")
    assert out == "no such dir"


def test_run_error_is_swallowed_into_message():
    with patch.object(vm.subprocess, "run", side_effect=OSError("limactl not found")):
        out = vm.make_dir_in_vm("/tmp/x")
    assert "执行失败" in out


def test_upload_skips_node_modules_and_git():
    """目录上传必须跳过 node_modules / .git，否则会卡死在上传依赖上。"""
    copied = []

    def fake_run(argv, **kwargs):
        copied.append(argv)
        return _fake_run(stdout="ok")

    with (
        patch.object(vm.subprocess, "run", side_effect=fake_run),
        patch.object(vm.os.path, "exists", return_value=True),
        patch.object(vm.os.path, "isdir", return_value=True),
        patch.object(vm.os, "walk") as mock_walk,
    ):
        mock_walk.return_value = [
            ("/proj", ["node_modules", ".git", "src"], ["README.md"]),
            ("/proj/src", [], ["main.py"]),
        ]
        msg = vm.upload_directory_to_vm("/proj", "/srv/site")

    assert "成功" in msg
    # 没有任何 copy 命令涉及 node_modules 或 .git
    assert not any("node_modules" in " ".join(argv) or ".git/" in " ".join(argv) for argv in copied)
