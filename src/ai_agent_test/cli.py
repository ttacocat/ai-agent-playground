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
    report(
        env_file.is_file(), ".env", str(env_file) if env_file.is_file() else "缺失，执行 cp .env.example .env 后填写"
    )

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
        report(
            all([bailian.access_key_id, bailian.workspace_id, bailian.query_index_id]),
            "百炼 RAG",
            f"workspace={bailian.workspace_id} index={bailian.query_index_id}",
        )
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
    if args.mode in {"supervisor", "team", "lead"}:
        from ai_agent_test.app.code_agent.agent.langgraph_code_agent import run_agent

        asyncio.run(run_agent(thread_id=args.thread, mode=args.mode))
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
def parse_tokens(text: str) -> int:
    """把 300k / 1.5m / 200000 解析成 token 数。"""
    raw = text.strip().lower().replace(",", "").replace("_", "")
    scale = {"k": 1_000, "m": 1_000_000}.get(raw[-1:], 1)
    number = raw[:-1] if scale > 1 else raw
    try:
        value = int(float(number) * scale)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无法解析 token 数：{text!r}，示例：300k、1.5m、200000") from None
    if value <= 0:
        raise argparse.ArgumentTypeError("token 数必须大于 0")
    return value


def cmd_eval(args: argparse.Namespace) -> int:
    """行为评测：真实 LLM 产生真实的工具选择，执行落在 stub 工具上（零外部副作用）。"""
    from ai_agent_test.evals.cases import CASES
    from ai_agent_test.evals.runner import REPORT_DIR, render_report, run_cases, save_report
    from ai_agent_test.evals.store import ResultStore
    from ai_agent_test.observability import build_tracer, summarize

    selected = CASES
    if args.case:
        wanted = set(args.case)
        selected = [c for c in CASES if c.id in wanted]
        missing = wanted - {c.id for c in selected}
        if missing:
            print(f"未知用例：{', '.join(sorted(missing))}\n可用用例：{', '.join(c.id for c in CASES)}")
            return 2
    if args.group:
        selected = [c for c in selected if args.group in c.tags]
        if not selected:
            tags = sorted({t for c in CASES for t in c.tags})
            print(f"没有带标签 {args.group!r} 的用例。可用标签：{', '.join(tags)}")
            return 2
    if args.repeat < 1:
        print("--repeat 至少为 1")
        return 2

    model = args.model or settings.model.name
    explore_name = args.explore_model or settings.model.explore_name
    if explore_name == model:
        explore_name = ""
    # 探索用了别的模型 = 另一组数据，单独存一个结果文件，续跑时不会和"探索也用主模型"的结果混在一起
    tag = "-".join(t for t in (args.tag, f"explore-{explore_name}" if explore_name else "") if t) or None
    store = ResultStore.for_model(REPORT_DIR, model, tag)
    model_factory = None
    if args.model:
        from ai_agent_test.app.code_agent.model.model import build_chat_model

        client = build_chat_model(args.model)
        model_factory = lambda: client  # noqa: E731
    explore_factory = None
    if explore_name:
        from ai_agent_test.app.code_agent.model.model import build_chat_model

        explore_client = build_chat_model(explore_name)
        explore_factory = lambda: explore_client  # noqa: E731
    planned = len(selected) * args.repeat * len(args.target)
    print(
        f"模型 {model}：{len(selected)} 条用例 × {args.repeat} 次 × 架构 {', '.join(args.target)}，"
        f"计划 {planned} 次运行（真实 LLM + stub 工具）"
    )
    print(f"结果文件：{store.path}" + ("（续跑：已有结果不再重跑）" if args.resume else ""))
    done = 0

    def progress(target, case, attempt) -> None:
        nonlocal done
        done += 1
        passed_mark = "✔" if attempt.result.passed else "✘"
        mark = "⚠ 接口错误" if attempt.error else passed_mark
        print(f"  [{done}] {target:<10} {case.id} {mark}  {attempt.seconds:.1f}s  {attempt.tokens:,} tok")
        if attempt.error:
            print(f"      {attempt.error}")

    tracer = build_tracer("eval")
    session = asyncio.run(
        run_cases(
            selected,
            targets=args.target,
            repeat=args.repeat,
            tracer=tracer,
            on_attempt=progress,
            model_name=model,
            model_factory=model_factory,
            explore_model_factory=explore_factory,
            explore_model_name=explore_name or None,
            store=store,
            resume=args.resume,
            max_runs=args.max_runs,
            tag=tag,
            budget=args.budget,
            token_limit=args.run_cap,
        )
    )
    report = render_report(session)
    print(report)

    if tracer and tracer.spans:
        summary = summarize(tracer.spans)
        print(
            f"本次消耗：LLM {summary.llm_calls} 次 / 工具 {summary.tool_calls} 次，"
            f"token {summary.total_tokens:,}（入 {summary.input_tokens:,} / 出 {summary.output_tokens:,}），"
            f"累计耗时 {summary.total_seconds:.1f}s"
        )
        if settings.trace.price_input_per_m or settings.trace.price_output_per_m:
            cost = summary.cost(settings.trace.price_input_per_m, settings.trace.price_output_per_m)
            print(f"成本：¥{cost:.4f}")
        print(f"trace：{tracer.save()}")

    path = save_report(report, model, tag)
    print(f"报告已写入：{path}")
    if session.aborted:
        print(f"\n⚠️  {session.aborted}\n   恢复后运行同一命令并加上 --resume 继续。")
        return 3
    return 0 if session.all_passed else 1


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ai_agent_test.web.app import FRONTEND_DIST, create_app

    if not FRONTEND_DIST.is_dir():
        print("提示：还没有构建前端，只提供 API。构建方法：cd web && npm install && npm run build")
    mode = "演示模式（不调用 API）" if args.demo else "真实模式"
    print(f"{mode}：http://{args.host}:{args.port}")
    uvicorn.run(create_app(demo=args.demo), host=args.host, port=args.port, log_level="warning")
    return 0


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
        choices=["supervisor", "team", "lead", "single"],
        default="supervisor",
        help=(
            "supervisor=主管 LLM 分派方案/编码专家；"
            "team=固定的 方案→实现→审查 流程（审查不通过自动重试）；"
            "lead=主 agent 负责到底，探索和审查交给子 agent；single=单 agent"
        ),
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
    p_eval.add_argument(
        "-g",
        "--group",
        default=None,
        help="只跑带这个标签的用例，如 core（4 条最能拉开差距的编码题，省额度）、code、search、safety、mysql",
    )
    p_eval.add_argument(
        "-t",
        "--target",
        nargs="+",
        choices=["single", "supervisor", "team", "lead"],
        default=["single"],
        help="评测的架构，可多选做对比，如：--target single team",
    )
    p_eval.add_argument("-k", "--repeat", type=int, default=1, help="每条用例重复次数，用于计算 pass@k / pass^k")
    p_eval.add_argument(
        "--resume",
        action="store_true",
        help="续用结果文件里同一模型已跑完的结果，只补跑缺的（额度用完或中断后接着跑）",
    )
    p_eval.add_argument("--max-runs", type=int, default=None, help="本次最多新跑多少次，免费额度可以分几次跑完")
    p_eval.add_argument(
        "--budget",
        type=parse_tokens,
        default=None,
        help="本次最多花多少 token，如 300k、1.5m。预计不够再跑一次就停，跑到一半超了不落盘，下次 --resume 重跑",
    )
    p_eval.add_argument(
        "--run-cap",
        type=parse_tokens,
        default=None,
        help="单次运行的 token 上限，如 150k。超出即判失败，防止个别失控的运行吃掉额度",
    )
    p_eval.add_argument(
        "-m",
        "--model",
        default=None,
        help="评测用的模型名（同一个端点和 Key），默认取 MODEL_NAME。换便宜的小模型可以省额度",
    )
    p_eval.add_argument(
        "--explore-model",
        default=None,
        help="lead 架构里探索子 agent 用的模型，默认取 EXPLORE_MODEL_NAME，都没配则与主模型相同",
    )
    p_eval.add_argument(
        "--tag",
        default=None,
        help="给这组结果打标签，单独存一个结果文件（例如改了护栏后用 --tag v2 重跑，与修复前对比）",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_web = sub.add_parser("serve", help="启动 Web 界面（对话 / 评测对比 / trace）")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8000)
    p_web.add_argument("--demo", action="store_true", help="演示模式：脚本化模型 + 模拟环境，不调用任何 API")
    p_web.set_defaults(func=cmd_serve)

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
