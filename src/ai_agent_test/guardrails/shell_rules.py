"""shell 命令的风险判定：白名单 + 词法解析。

第一版是黑名单正则（扫 ``\\brm\\b`` 之类），被下面这些写法轻松绕过：

- ``find . -delete`` / ``python3 -c "shutil.rmtree(...)"``：换个程序做同一件事
- ``r''m -rf x`` / ``/bin/rm``：引号拼接、绝对路径，正则看到的不是 ``rm``
- ``echo cm0g... | base64 -d | sh``：编码后交给解释器执行
- ``$(...)`` / 反引号：命令替换，静态文本里根本看不到真实命令

黑名单的问题是要枚举所有危险写法，而危险写法是无限的。所以反过来：

1. 用 ``shlex`` 按 shell 的规则做词法切分（引号会被正确还原，``r''m`` 就是 ``rm``）；
2. 按 ``; && || | &`` 和换行拆成多段，**每一段**独立判定，取最高风险；
3. 每段取真正的程序名（剥掉 ``VAR=x`` 前缀、``env`` / ``nohup`` / ``timeout`` 等包装、路径前缀）；
4. 只有白名单内的只读命令才放行，**不认识的命令一律要人确认**；
5. 静态无法判定的结构（命令替换、进程替换、解析失败）直接要确认。

代价是新命令第一次用时要多确认一次。这是有意的取舍：漏放一次 ``rm -rf``
的代价远大于多点一次 y。
"""

from __future__ import annotations

import os
import re
import shlex

from ai_agent_test.guardrails.levels import RiskLevel, Verdict, worst

READ, WRITE, DESTRUCTIVE = RiskLevel.READ, RiskLevel.WRITE, RiskLevel.DESTRUCTIVE

# --------------------------------------------------------------------------- #
# 命令表
# --------------------------------------------------------------------------- #
# fmt: off
# 只读命令：不修改文件系统、不联网、不执行任意代码。
# 刻意不放 awk（可 system()）、env/printenv（会打印密钥）、curl/wget（可外传数据）。
READ_COMMANDS = frozenset({
    "ls", "ll", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg",
    "pwd", "echo", "printf", "which", "whereis", "type", "whoami", "id",
    "date", "uname", "hostname", "uptime", "df", "du", "tree", "stat", "file",
    "ps", "sort", "uniq", "cut", "tr", "diff", "cmp", "basename", "dirname",
    "realpath", "readlink", "true", "false", "test", "[", "jq", "nl", "column", ":",
})

# 写命令：会创建内容，但不会删除或覆盖已有内容
WRITE_COMMANDS = frozenset({"mkdir", "touch"})

# 明确危险的命令，给出具体原因（不在表里的也会被拦，只是原因更笼统）
DESTRUCTIVE_COMMANDS: dict[str, str] = {
    "rm": "rm 删除文件",
    "rmdir": "rmdir 删除目录",
    "unlink": "unlink 删除文件",
    "shred": "shred 粉碎文件",
    "mv": "mv 可能覆盖或转移文件",
    "dd": "dd 可直接写裸设备",
    "truncate": "truncate 清空文件",
    "chmod": "chmod 修改权限",
    "chown": "chown 修改属主",
    "kill": "kill 结束进程",
    "killall": "killall 结束进程",
    "pkill": "pkill 结束进程",
    "shutdown": "关机",
    "reboot": "重启",
    "halt": "关机",
    "poweroff": "关机",
    "sudo": "sudo 提权执行",
    "su": "su 切换用户",
    "doas": "doas 提权执行",
    "eval": "eval 执行动态拼接的命令",
    "exec": "exec 替换当前进程",
    "source": "source 执行脚本文件",
    ".": "source 执行脚本文件",
    "xargs": "xargs 会把输入当成参数去执行命令",
    "crontab": "crontab 修改定时任务",
}
_INTERPRETERS = frozenset({
    "sh", "bash", "zsh", "fish", "dash", "ksh", "python", "python3", "perl",
    "ruby", "node", "php", "lua", "osascript", "awk", "gawk",
})
_NETWORK = frozenset({"curl", "wget", "nc", "ncat", "ssh", "scp", "rsync", "ftp", "telnet"})

# 只改变执行方式、真正干活的是后面那条命令
_WRAPPERS = frozenset({"env", "command", "builtin", "nohup", "time", "nice", "stdbuf"})

_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "|&", "(", ")", ";;"})
_REDIRECTS = frozenset({">", ">>", ">|", "&>", "&>>", "<", "<<", "<<<", "<>", ">&", "<&"})
_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# 只读命令读这些路径同样要确认：读出来以后可能被发到别处
_SENSITIVE_PATH = re.compile(
    r"(^|/)(\.ssh|\.aws|\.gnupg|\.kube|\.docker)(/|$)"
    r"|(^|/)(id_rsa|id_ed25519|id_ecdsa|\.netrc|\.git-credentials|\.pgpass|shadow)$"
    r"|(^|/)\.env(\.(?!example$)[\w-]+)?$"
    r"|\.(pem|key|p12|pfx)$",
)

# 子命令级白名单
_GIT_READ = frozenset({
    "status", "log", "diff", "show", "rev-parse", "ls-files", "blame", "shortlog",
    "describe", "grep", "reflog", "ls-tree", "cat-file",
})
_GIT_WRITE = frozenset({"add", "commit", "fetch", "init"})
_FIND_DANGEROUS = frozenset({
    "-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls",
})
# fmt: on


# --------------------------------------------------------------------------- #
# 词法
# --------------------------------------------------------------------------- #
def tokenize(command: str) -> list[str]:
    """按 shell 规则切词，操作符（``| ; && >`` 等）单独成 token。

    换行在 shell 里等价于 ``;``，但 shlex 会把它当空白吞掉，
    ``ls\\nrm -rf /`` 就会被误读成 ``ls`` 带了几个参数。所以先换成 ``;``：
    引号内的换行替换后仍在引号内，不会制造出新的分段。
    """
    lexer = shlex.shlex(command.replace("\n", " ; "), posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def split_segments(tokens: list[str]) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """拆成「命令段」和「重定向」两部分。

    返回 (segments, redirects)：redirects 是 (操作符, 目标) 列表。
    """
    segments: list[list[str]] = [[]]
    redirects: list[tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _SEPARATORS:
            segments.append([])
        elif token in _REDIRECTS:
            target = tokens[index + 1] if index + 1 < len(tokens) else ""
            redirects.append((token, target))
            index += 1
            # 2>&1 / 2>/dev/null 里的 fd 编号被切成了上一段的参数，摘掉
            current = segments[-1]
            if current and current[-1].isdigit():
                current.pop()
        else:
            segments[-1].append(token)
        index += 1
    return [s for s in segments if s], redirects


def unwrap(argv: list[str]) -> list[str]:
    """剥掉环境变量赋值和包装命令，返回真正执行的那条命令。"""
    argv = list(argv)
    while argv:
        head = os.path.basename(argv[0])
        if _ASSIGNMENT.match(argv[0]):
            argv.pop(0)
        elif head in _WRAPPERS:
            argv.pop(0)
            # 跳过包装命令自己的选项（env -i、nice -n 10 ...）
            while argv and (argv[0].startswith("-") or _ASSIGNMENT.match(argv[0]) or argv[0].isdigit()):
                argv.pop(0)
        elif head == "timeout":
            argv.pop(0)
            while argv and argv[0].startswith("-"):
                argv.pop(0)
            if argv:
                argv.pop(0)  # 时长参数
        else:
            break
    return argv


# --------------------------------------------------------------------------- #
# 判定
# --------------------------------------------------------------------------- #
def _judge_git(args: list[str]) -> Verdict:
    sub = next((a for a in args if not a.startswith("-")), "")
    flags = set(args)
    if sub == "push" and (
        flags & {"-f", "--force", "--force-with-lease", "--delete", "-d", "--mirror"}
        or any(a.startswith((":", "+")) for a in args[1:])
    ):
        return DESTRUCTIVE, "git push 强推或删除远端分支"
    if sub == "reset" and "--hard" in flags:
        return DESTRUCTIVE, "git reset --hard 会丢弃改动"
    if sub == "clean":
        return DESTRUCTIVE, "git clean 会删除未跟踪文件"
    if sub in _GIT_READ:
        return READ, f"git {sub} 为只读操作"
    if sub == "branch" and not flags & {"-d", "-D", "--delete", "-m", "-M", "--move", "-f", "--force"}:
        return READ, "git branch 为只读操作"
    if sub in _GIT_WRITE:
        return WRITE, f"git {sub} 只新增内容"
    return DESTRUCTIVE, f"git {sub or '(无子命令)'} 不在白名单内（可能改写历史或影响远端）"


def judge_argv(argv: list[str]) -> Verdict:
    """判定一段已拆分好的命令。"""
    argv = unwrap(argv)
    if not argv:
        return READ, "空命令"

    raw_name = argv[0]
    name = os.path.basename(raw_name)
    args = argv[1:]

    if raw_name.startswith("$") or "$" in name:
        return DESTRUCTIVE, f"程序名来自变量展开（{raw_name}），无法静态判定"
    if name in DESTRUCTIVE_COMMANDS:
        return DESTRUCTIVE, DESTRUCTIVE_COMMANDS[name]
    if name.startswith("mkfs"):
        return DESTRUCTIVE, "格式化文件系统"
    if name in _INTERPRETERS or re.fullmatch(r"python3?(\.\d+)?", name):
        return DESTRUCTIVE, f"{name} 会执行任意代码"
    if name in _NETWORK:
        return DESTRUCTIVE, f"{name} 会访问网络（可能下载或外传数据）"

    if name == "git":
        return _judge_git(args)
    if name == "find":
        hit = next((a for a in args if a in _FIND_DANGEROUS), None)
        if hit:
            return DESTRUCTIVE, f"find {hit} 会删除文件或执行命令"
        verdict: Verdict = (READ, "find 为只读查找")
    elif name == "sed":
        if any(a == "--in-place" or a.startswith("--in-place=") or re.fullmatch(r"-[a-zA-Z]*i.*", a) for a in args):
            return DESTRUCTIVE, "sed -i 原地修改文件"
        verdict = (READ, "sed 为只读输出")
    elif name in READ_COMMANDS:
        verdict = (READ, f"{name} 为只读命令")
    elif name in WRITE_COMMANDS:
        verdict = (WRITE, f"{name} 只新增内容")
    else:
        return DESTRUCTIVE, f"命令 {name} 不在白名单内，需人工确认"

    sensitive = next((a for a in args if _SENSITIVE_PATH.search(a.rstrip("/"))), None)
    if sensitive:
        return DESTRUCTIVE, f"读取敏感凭证文件 {sensitive}"
    return verdict


def judge_redirect(op: str, target: str) -> Verdict:
    if op in {"<", "<<", "<<<", "<&"} or (op == ">&" and target.isdigit()):
        if _SENSITIVE_PATH.search(target):
            return DESTRUCTIVE, f"读取敏感凭证文件 {target}"
        return READ, "输入重定向"
    if target in {"/dev/null", "/dev/stdout", "/dev/stderr"}:
        return READ, "重定向到空设备"
    if target.startswith("/dev/"):
        return DESTRUCTIVE, f"重定向写入设备 {target}"
    if op in {">>", "&>>"}:
        return WRITE, f"追加写入 {target}"
    return DESTRUCTIVE, f"重定向会覆盖文件 {target or '(未知)'}"


def scan_shell(command: str) -> Verdict:
    """返回 (等级, 原因)。整条命令的等级 = 所有分段与重定向里最高的那个。"""
    if re.search(r"\$\(|`|<\(|>\(", command):
        return DESTRUCTIVE, "包含命令替换或进程替换，无法静态判定真实命令"
    try:
        tokens = tokenize(command)
    except ValueError as exc:  # 引号不配对等
        return DESTRUCTIVE, f"命令无法解析（{exc}），按危险处理"

    segments, redirects = split_segments(tokens)
    if not segments:
        return READ, "空命令"

    verdicts = [judge_argv(seg) for seg in segments]
    verdicts += [judge_redirect(op, target) for op, target in redirects]
    return worst(verdicts, default=(READ, "只读命令"))
