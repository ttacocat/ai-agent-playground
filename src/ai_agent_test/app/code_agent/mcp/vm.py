import os
import tempfile
from typing import Annotated

from mcp.server.fastmcp import FastMCP
import shlex
import subprocess
from pydantic import Field

mcp = FastMCP()
def run_limavm_shell_command(command):
    try:
        wrapper_command = "limactl shell default " + command
        shell_command = shlex.split(wrapper_command)
        res =  subprocess.run(shell_command, shell=False , capture_output=True, text=True)

        if res.returncode != 0:
            return res.stderr
        return res.stdout
    except Exception as e:
        return str(e)

def run_limavm_command(command):
    try:
        wrapper_command = "limactl " + command
        shell_command = shlex.split(wrapper_command)
        res =  subprocess.run(shell_command, shell=False , capture_output=True, text=True)

        if res.returncode != 0:
            return res.stderr
        return res.stdout
    except Exception as e:
        return str(e)

@mcp.tool(name="make_dir_in_vm", description="在指定的虚拟机中创建目录，相当于 mkdir -p 命令")
def make_dir_in_vm(dir_path: Annotated[str, Field(description="要创建的目录路径", example="/home/wxy.guest/nginx/uploads/test")]):
    """在虚拟机中创建目录"""
    print("make dir {}".format(dir_path))
    return run_limavm_shell_command(f"mkdir -p {dir_path}")

@mcp.tool(name="list_files_in_vm", description="查看虚拟机中指定目录，相当于 ls -al 命令")
def list_files_in_vm(dir_path: Annotated[str, Field(description="要查看的目录路径", example="/home/wxy.guest/nginx/uploads")]):
    """查看虚拟机指定目录下的文件"""
    print("list file {}".format(dir_path))
    return run_limavm_shell_command(f"ls -al {dir_path}")

@mcp.tool(name="write_file_to_vm", description="向虚拟机中写入指定文件")
def write_file_to_vm(file_path: Annotated[str, Field(description="写入虚拟机中的文件地址", example="/home/wxy.guest/nginx/uploads/test1/index.html")],
                     content: Annotated[str, Field(description="写入虚拟机中的文件内容", example="<div> hello limavm </div>")]):
    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    print("本地临时文件已创建", tmp_file_path)
    run_limavm_command(f"""copy {tmp_file_path} default:{file_path}""")
    change_file_permission_in_vm(file_path, "755")

def change_file_permission_in_vm(file_path, mode):
    return run_limavm_shell_command(f"chmod {mode} {file_path}")

@mcp.tool(name="upload_directory_to_vm", description="将本地文件目录上传至虚拟机指定目录")
def upload_directory_to_vm(
    local_dir: Annotated[str, Field(description="本地文件目录", example="/Users/wxy/my-vue-project")],
    vm_dest_dir: Annotated[str, Field(description="虚拟机文件目录", example="/home/wxy.guest/nginx/uploads/vue-test")],
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
        if 'node_modules' in dirs:
            dirs.remove('node_modules')
        if '.git' in dirs:
            dirs.remove('.git')

        rel_path = os.path.relpath(root, local_dir)
        # 创建远程文件夹
        vm_subdir = os.path.join(vm_dest_dir, rel_path)
        make_dir_in_vm(vm_subdir)

        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            vm_file_path = os.path.join(vm_subdir, file_name)
            result = run_limavm_command(f"copy {local_file_path} default:{vm_file_path}")
            print(result)
    return f"上传 [{local_dir}] 目录至 default:{vm_dest_dir} 成功"


if __name__ == "__main__":
    mcp.run(transport="stdio")
    # make_dir_in_vm("/home/wxy.guest/nginx/uploads/test")
    # print(list_files_in_vm("/home/wxy.guest/nginx/uploads"))
    # write_file_to_vm("/home/wxy.guest/nginx/uploads/test1/index.html", "<div> hello limavm </div>")
    # print(upload_directory_to_vm(local_dir="/Users/wxy/my-vue-project", vm_dest_dir="/home/wxy.guest/nginx/uploads/vue-test"))