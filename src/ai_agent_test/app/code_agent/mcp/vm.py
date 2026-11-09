import os
import shlex
import subprocess
import tempfile
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Lima 虚拟机名可通过环境变量覆盖，避免把本机环境写进代码
LIMA_VM_NAME = os.getenv("LIMA_VM_NAME", "default")
LIMACTL_BIN = os.getenv("LIMACTL_BIN", "limactl")
COMMAND_TIMEOUT = 120

mcp = FastMCP(name="vm")


def run_limavm_shell_command(command: str) -> str:
    """在虚拟机内执行命令。

    参数统一走 shlex 分词 + shell=False，不经过 shell 解释器，
    因此路径里的 `;` `|` `$()` 不会被当作命令分隔符。
    """
    argv = [LIMACTL_BIN, "shell", LIMA_VM_NAME, *shlex.split(command)]
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"执行失败：{exc}"
    return res.stderr if res.returncode != 0 else res.stdout


def run_limavm_command(command: str) -> str:
    """在宿主机执行 limactl 子命令（copy/start/stop 等）。"""
    argv = [LIMACTL_BIN, *shlex.split(command)]
    try:
        res = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT)
    except (subprocess.SubprocessError, OSError) as exc:
        return f"执行失败：{exc}"
    return res.stderr if res.returncode != 0 else res.stdout


@mcp.tool(name="make_dir_in_vm", description="在指定的虚拟机中创建目录，相当于 mkdir -p 命令")
def make_dir_in_vm(
    dir_path: Annotated[str, Field(description="要创建的目录路径", examples=["/home/user/uploads/test"])],
):
    """在虚拟机中创建目录"""
    print(f"make dir {dir_path}")
    return run_limavm_shell_command(f"mkdir -p {shlex.quote(dir_path)}")


@mcp.tool(name="list_files_in_vm", description="查看虚拟机中指定目录，相当于 ls -al 命令")
def list_files_in_vm(
    dir_path: Annotated[str, Field(description="要查看的目录路径", examples=["/home/user/uploads"])],
):
    """查看虚拟机指定目录下的文件"""
    print(f"list file {dir_path}")
    return run_limavm_shell_command(f"ls -al {shlex.quote(dir_path)}")


@mcp.tool(name="write_file_to_vm", description="向虚拟机中写入指定文件")
def write_file_to_vm(
    file_path: Annotated[
        str, Field(description="写入虚拟机中的文件地址", examples=["/home/user/uploads/index.html"])
    ],
    content: Annotated[str, Field(description="写入虚拟机中的文件内容", examples=["<div> hello </div>"])],
):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    print("本地临时文件已创建", tmp_file_path)
    run_limavm_command(f"copy {shlex.quote(tmp_file_path)} {LIMA_VM_NAME}:{shlex.quote(file_path)}")
    change_file_permission_in_vm(file_path, "755")


def change_file_permission_in_vm(file_path: str, mode: str) -> str:
    return run_limavm_shell_command(f"chmod {shlex.quote(mode)} {shlex.quote(file_path)}")


@mcp.tool(name="upload_directory_to_vm", description="将本地文件目录上传至虚拟机指定目录")
def upload_directory_to_vm(
    local_dir: Annotated[str, Field(description="本地文件目录", examples=["<仓库根目录>/.temp/site"])],
    vm_dest_dir: Annotated[str, Field(description="虚拟机文件目录", examples=["/home/user/uploads/site"])],
):
    if not os.path.exists(local_dir):
        msg = f"本地目录不存在： {local_dir}"
        print(f"[UPLOAD] {msg}")
        return msg
    if not os.path.isdir(local_dir):
        msg = f"指定路径不是文件夹： {local_dir}"
        print(f"[UPLOAD] {msg}")
        return msg
    make_dir_in_vm(vm_dest_dir)

    for root, dirs, files in os.walk(local_dir):
        # 跳过体积大且无意义的目录，否则上传会卡在 node_modules 里
        dirs[:] = [d for d in dirs if d not in {"node_modules", ".git"}]

        rel_path = os.path.relpath(root, local_dir)
        vm_subdir = os.path.join(vm_dest_dir, rel_path)
        make_dir_in_vm(vm_subdir)

        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            vm_file_path = os.path.join(vm_subdir, file_name)
            print(run_limavm_command(f"copy {shlex.quote(local_file_path)} {LIMA_VM_NAME}:{shlex.quote(vm_file_path)}"))
    return f"上传 [{local_dir}] 目录至 {LIMA_VM_NAME}:{vm_dest_dir} 成功"


if __name__ == "__main__":
    mcp.run(transport="stdio")
