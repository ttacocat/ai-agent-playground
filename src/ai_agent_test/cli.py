"""统一命令行入口。

    uv run agent doctor             环境自检（配置、依赖、网络、MCP 脚本）
    uv run agent chat               启动多智能体代码助手
    uv run agent tools              列出所有可加载的 MCP 工具
    uv run agent serve-mcp mysql    前台启动一个 MCP server（调试协议用）
    uv run agent rag-upload F.md    把文档上传到百炼知识库并挂到索引

设计意图：把原来 8 个各自为战的 `python xxx.py` 收成一个入口，
从此"怎么运行"只有一种答案。
"""

from __future__ import annotations

import argparse
import asyncio
import runpy
import socket
import sys

from ai_agent_test.config import PROJECT_ROOT, ConfigError, mcp_script, settings

MCP_SERVERS: dict[str, str] = {
    "mysql": "mcp/mysql_tools.py",
    "browser": "mcp/browser_tools.py",
    "terminal": "mcp/terminal_tools.py",
    "shell": "mcp/shell_tools.py",
    "vm": "mcp/vm.py",
    "rag": "rag/rag.py",
    "self-rag": "rag/self_rag.py",
}


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def _tcp_alive(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def cmd_doctor(_args: argparse.Namespace) -> int:
    failures = 0

    def report(ok: bool, label: str, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f"  {'✔' if ok else '✘'} {label:<14}{detail}")

    def warn(label: str, detail: str) -> None:
        print(f"  ⚠ {label:<14}{detail}")

    print("\n环境自检")
    print("─" * 60)

    py_ok = sys.version_info >= (3, 13)
    report(py_ok, "Python", f"{sys.version.split()[0]}（需要 >= 3.13）")

    env_file = PROJECT_ROOT / ".env"
    report(env_file.is_file(), ".env", str(env_file) if env_file.is_file()
           else "缺失，执行 cp .env.example .env 后填写")

    try:
        model = settings.model
        hint = model.base_url or "官方 OpenAI 端点"
        report(bool(model.api_key), "模型", f"{model.name} @ {hint}")
    except ConfigError as exc:
        report(False, "模型", str(exc))

    try:
        mysql = settings.mysql
    except ConfigError as exc:
        report(False, "MySQL", str(exc))
    else:
        if _tcp_alive(mysql.host, mysql.port):
            report(True, "MySQL", f"{mysql.host}:{mysql.port}")
        else:
            # 外部依赖连不上不阻断启动，只是对应工具本轮不可用
            warn("MySQL", f"{mysql.host}:{mysql.port} 连不上，mysql 工具暂不可用")

    try:
        ckpt = settings.checkpoint
        if ckpt.backend == "redis":
            if _tcp_alive("localhost", 6379):
                report(True, "Redis", f"{ckpt.redis_url}（检查点后端）")
            else:
                warn("Redis", f"{ckpt.redis_url} 连不上，先启动 Redis 或设 CHECKPOINT_BACKEND=file")
        elif ckpt.backend == "mongodb":
            if _tcp_alive("localhost", 27017):
                report(True, "MongoDB", ckpt.mongodb_url)
            else:
                warn("MongoDB", f"{ckpt.mongodb_url} 连不上")
        else:
            report(True, "检查点", "file（本地文件）")
    except ConfigError as exc:
        report(False, "检查点", str(exc))

    try:
        bailian = settings.bailian
        report(all([bailian.access_key_id, bailian.workspace_id, bailian.query_index_id]),
               "百炼 RAG", f"workspace={bailian.workspace_id} index={bailian.query_index_id}")
    except ConfigError as exc:
        report(False, "百炼 RAG", str(exc))

    try:
        paths = settings.paths
        paths.ensure_dirs()
        probe = paths.work_dir / ".doctor_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        report(True, "工作目录", str(paths.work_dir))
    except OSError as exc:
        report(False, "工作目录", f"不可写：{exc}")

    if settings.paths.chromedriver:
        report(settings.paths.chromedriver.is_file(), "chromedriver", str(settings.paths.chromedriver))
    else:
        warn("chromedriver", "未配置 CHROMEDRIVER_PATH，将由 Selenium Manager 自动管理")

    missing = [name for name, rel in MCP_SERVERS.items() if not mcp_script(rel).is_file()]
    report(not missing, "MCP 脚本", f"{len(MCP_SERVERS)} 个就绪" if not missing else f"缺失：{', '.join(missing)}")

    print("─" * 60)
    if failures:
        print(f"✘ {failures} 项未通过，先按提示修好再启动。\n")
        return 1
    print("✔ 一切就绪，运行 `uv run agent chat` 开始。\n")
    return 0


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #
def cmd_chat(args: argparse.Namespace) -> int:
    if args.mode == "supervisor":
        from ai_agent_test.app.code_agent.agent.langgraph_code_agent import run_agent

        asyncio.run(run_agent(thread_id=args.thread))
    else:
        from ai_agent_test.app.code_agent.agent.code_agent import run_agent as run_single

        asyncio.run(run_single())
    return 0


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
async def _load_tools(server: str | None) -> int:
    from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools

    targets = {server: MCP_SERVERS[server]} if server else MCP_SERVERS
    total = 0
    for name, rel in targets.items():
        try:
            tools = await get_stdio_tools(name, rel)
        except Exception as exc:  # noqa: BLE001 - 这里就是要展示每个 server 的真实失败原因
            print(f"✘ {name:<10} 加载失败：{type(exc).__name__}: {exc}")
            continue
        total += len(tools)
        print(f"✔ {name:<10} {len(tools)} 个工具")
        for tool in tools:
            desc = (tool.description or "").strip().splitlines()[0][:70]
            print(f"    · {tool.name:<32}{desc}")
    print(f"\n共 {total} 个工具")
    return 0 if total else 1


def cmd_tools(args: argparse.Namespace) -> int:
    if args.server and args.server not in MCP_SERVERS:
        print(f"未知 server：{args.server}，可选：{', '.join(MCP_SERVERS)}")
        return 2
    return asyncio.run(_load_tools(args.server))


# --------------------------------------------------------------------------- #
# serve-mcp
# --------------------------------------------------------------------------- #
def cmd_serve_mcp(args: argparse.Namespace) -> int:
    rel = MCP_SERVERS.get(args.name)
    if rel is None:
        print(f"未知 server：{args.name}，可选：{', '.join(MCP_SERVERS)}")
        return 2
    print(f"启动 MCP server：{args.name}（stdio）", file=sys.stderr)
    runpy.run_path(str(mcp_script(rel)), run_name="__main__")
    return 0


# --------------------------------------------------------------------------- #
# rag-upload
# --------------------------------------------------------------------------- #
def cmd_rag_upload(args: argparse.Namespace) -> int:
    from ai_agent_test.app.code_agent.rag.rag import (
        add_document_to_index,
        bailian,
        create_client,
        upload_rag_file_to_bailian,
    )

    cfg = bailian()
    client = create_client()
    file_id = upload_rag_file_to_bailian(client, cfg.workspace_id, cfg.upload_category_id, args.file)
    add_document_to_index(client, cfg.workspace_id, cfg.query_index_id, file_id)
    print(f"✔ 已上传并挂到索引 {cfg.query_index_id}：{file_id}")
    return 0


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def cmd_eval(args: argparse.Namespace) -> int:
    """行为评测：真实 LLM 产生真实的工具选择，执行落在 stub 工具上（零外部副作用）。"""
    from ai_agent_test.evals.cases import CASES
    from ai_agent_test.evals.runner import render_report, run_cases, save_report
    from ai_agent_test.observability import build_tracer, summarize

    selected = CASES
    if args.case:
        wanted = set(args.case)
        selected = [c for c in CASES if c.id in wanted]
        missing = wanted - {c.id for c in selected}
        if missing:
            print(f"未知用例：{', '.join(sorted(missing))}\n可用用例：{', '.join(c.id for c in CASES)}")
            return 2

    print(f"开始评测 {len(selected)} 条用例（真实 LLM + stub 工具）...")
    tracer = build_tracer("eval")
    results = asyncio.run(run_cases(selected, tracer))
    report = render_report(results)
    print(report)

    if tracer and tracer.spans:
        summary = summarize(tracer.spans)
        print(
            f"消耗：LLM {summary.llm_calls} 次 / 工具 {summary.tool_calls} 次，"
            f"token {summary.total_tokens:,}（入 {summary.input_tokens:,} / 出 {summary.output_tokens:,}），"
            f"累计耗时 {summary.total_seconds:.1f}s"
        )
        if settings.trace.price_input_per_m or settings.trace.price_output_per_m:
            cost = summary.cost(settings.trace.price_input_per_m, settings.trace.price_output_per_m)
            print(f"成本：¥{cost:.4f}")
        print(f"trace：{tracer.save()}")

    path = save_report(report)
    print(f"报告已写入：{path}")
    return 0 if all(r.passed for r in results) else 1


# --------------------------------------------------------------------------- #
# trace
# --------------------------------------------------------------------------- #
def cmd_trace(args: argparse.Namespace) -> int:
    """查看最近一次运行的 span 树与 token / 耗时统计。"""
    from ai_agent_test.observability import load_run, render

    cfg = settings.trace
    traces = sorted(cfg.dir.glob("*.jsonl")) if cfg.dir.is_dir() else []
    if not traces:
        print(f"还没有 trace：{cfg.dir}\n先跑一次 `uv run agent chat` 或 `uv run agent eval`。")
        return 1

    if args.list:
        for path in traces[-args.number :]:
            print(path.name)
        return 0

    index = min(args.number, len(traces))
    path = traces[-index]
    header, spans = load_run(path)
    print(render(header, spans, cfg.price_input_per_m, cfg.price_output_per_m))
    print(f"来源：{path}")
    return 0


# --------------------------------------------------------------------------- #
# entrypoint
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Agent 平台命令行入口",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="环境自检").set_defaults(func=cmd_doctor)

    p_chat = sub.add_parser("chat", help="启动多智能体代码助手")
    p_chat.add_argument("--thread", default="default", help="会话 ID，相同 ID 会续用历史上下文")
    p_chat.add_argument(
        "--mode",
        choices=["supervisor", "single"],
        default="supervisor",
        help="supervisor=方案专家+编码专家编排；single=单 agent",
    )
    p_chat.set_defaults(func=cmd_chat)

    p_tools = sub.add_parser("tools", help="列出所有可加载的 MCP 工具")
    p_tools.add_argument("server", nargs="?", help="只看某一个 server")
    p_tools.set_defaults(func=cmd_tools)

    p_serve = sub.add_parser("serve-mcp", help="前台启动一个 MCP server")
    p_serve.add_argument("name", help=f"server 名，可选：{', '.join(MCP_SERVERS)}")
    p_serve.set_defaults(func=cmd_serve_mcp)

    p_rag = sub.add_parser("rag-upload", help="上传文档到百炼知识库")
    p_rag.add_argument("file", help="本地文件路径")
    p_rag.set_defaults(func=cmd_rag_upload)

    p_eval = sub.add_parser("eval", help="跑 Agent 行为评测（真实 LLM + stub 工具，无外部副作用）")
    p_eval.add_argument("case", nargs="*", help="只跑指定用例 ID，如：eval E03 E11；留空跑全部")
    p_eval.set_defaults(func=cmd_eval)

    p_trace = sub.add_parser("trace", help="查看最近一次运行的 span 树、token 与耗时")
    p_trace.add_argument("-n", "--number", type=int, default=1, help="倒数第几次运行，默认 1（最近一次）")
    p_trace.add_argument("-l", "--list", action="store_true", help="只列出最近的 trace 文件")
    p_trace.set_defaults(func=cmd_trace)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
