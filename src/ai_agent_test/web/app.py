"""Web 服务：对话走 SSE，危险操作的审批走"挂起 → 提交决定 → 继续"两段式。

为什么不让一个 SSE 连接一直挂着等用户点"批准"：等多久不确定，连接随时可能断
（刷新页面、切网络、手机锁屏）。审批状态本来就在 checkpoint 里（interrupt），
所以每段请求只做一件事：

1. ``POST /api/chat/{thread}/messages``：运行到结束或挂起；挂起时最后推一条
   ``approval_required``，连接关闭；
2. ``POST /api/chat/{thread}/approvals``：提交 ``{interrupt_id: 是/否}``，用
   ``Command(resume=...)`` 继续运行，同样流式推送，可能再次挂起；
3. ``GET /api/chat/{thread}``：刷新页面后从 checkpoint 恢复历史和待审批项。

有待审批项时拒绝新消息（409），否则会话里会留下没有结果的 tool_call，接口直接 400。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from ai_agent_test.guardrails.hitl import pending_approvals
from ai_agent_test.web.data import list_result_files, list_traces, result_detail, trace_detail
from ai_agent_test.web.events import EventTranslator, approval_events, history_events

logger = logging.getLogger(__name__)

MODES = ("lead", "team", "supervisor")
DEMO_MODES = ("lead", "team")
RECURSION_LIMIT = 60
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class MessageIn(BaseModel):
    text: str
    mode: str = "lead"


class ApprovalsIn(BaseModel):
    mode: str = "lead"
    answers: dict[str, bool]


def sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n".encode()


class Runtime:
    """按架构缓存构建好的图；所有会话共用一个 checkpointer。"""

    def __init__(self, demo: bool, demo_delay: float | None = None) -> None:
        self.demo = demo
        self.demo_delay = demo_delay
        self.model_name = "demo（脚本化，不调用 API）" if demo else ""
        self.checkpointer: Any = None
        self.apps: dict[str, Any] = {}
        self.tools: list | None = None
        self._stack = AsyncExitStack()

    async def start(self) -> None:
        if self.demo:
            self.checkpointer = InMemorySaver()
            return
        from ai_agent_test.config import settings

        self.model_name = settings.model.name
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            saver = await self._stack.enter_async_context(
                AsyncRedisSaver.from_conn_string(settings.checkpoint.redis_url)
            )
            await saver.asetup()
            self.checkpointer = saver
        except Exception as exc:  # Redis 不可用时仍能用，只是重启后会话丢失
            logger.warning("Redis 不可用，会话只保存在内存里：%s", exc)
            self.checkpointer = InMemorySaver()

    async def stop(self) -> None:
        await self._stack.aclose()

    async def app(self, mode: str):
        if mode not in MODES:
            raise HTTPException(400, f"mode 只支持 {', '.join(MODES)}")
        if mode in self.apps:
            return self.apps[mode]
        from ai_agent_test.app.code_agent.agent.team import build_supervisor, build_team

        if self.demo:
            if mode not in DEMO_MODES:
                raise HTTPException(400, "演示模式只支持 lead 和 team 架构")
            from ai_agent_test.app.code_agent.agent.lead import build_lead
            from ai_agent_test.web.demo import DemoModel, demo_tools

            _, tools = demo_tools()
            model = DemoModel() if self.demo_delay is None else DemoModel(delay=self.demo_delay)
            if mode == "lead":
                app = build_lead(model, tools, checkpointer=self.checkpointer)
            else:
                app = build_team(model, tools, max_attempts=1, checkpointer=self.checkpointer)
        else:
            from ai_agent_test.app.code_agent.agent.langgraph_code_agent import load_tools
            from ai_agent_test.app.code_agent.model.model import qwen_llm
            from ai_agent_test.observability import build_context_middleware

            if self.tools is None:
                self.tools = await load_tools()
            middleware = build_context_middleware(qwen_llm)
            if mode == "lead":
                from ai_agent_test.app.code_agent.agent.langgraph_code_agent import explore_model
                from ai_agent_test.app.code_agent.agent.lead import build_lead

                app = build_lead(
                    qwen_llm,
                    self.tools,
                    explore_model=explore_model(),
                    middleware=middleware,
                    checkpointer=self.checkpointer,
                )
            else:
                builder = build_team if mode == "team" else build_supervisor
                app = builder(qwen_llm, self.tools, middleware=middleware, checkpointer=self.checkpointer)
        self.apps[mode] = app
        return app


def thread_config(thread: str, mode: str) -> dict:
    return {"configurable": {"thread_id": f"web:{mode}:{thread}"}, "recursion_limit": RECURSION_LIMIT}


async def stream_run(app: Any, payload: Any, config: dict) -> AsyncIterator[bytes]:
    translator = EventTranslator()
    try:
        async for namespace, update in app.astream(payload, config, stream_mode="updates", subgraphs=True):
            for event in translator.translate(namespace, update):
                yield sse(event)
        for event in approval_events(await pending_approvals(app, config)):
            yield sse(event)
        yield sse({"type": "done"})
    except Exception as exc:  # 把错误交给前端显示，而不是让连接无声断开
        logger.exception("运行失败")
        yield sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})


def _streaming(body: AsyncIterator[bytes]) -> StreamingResponse:
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    demo: bool = False,
    reports_dir: Path | None = None,
    traces_dir: Path | None = None,
    demo_delay: float | None = None,
) -> FastAPI:
    runtime = Runtime(demo, demo_delay)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await runtime.start()
        yield
        await runtime.stop()

    api = FastAPI(title="ai-agent-test", lifespan=lifespan)

    def reports() -> Path:
        if reports_dir is not None:
            return reports_dir
        from ai_agent_test.evals.runner import REPORT_DIR

        return REPORT_DIR

    def traces() -> Path:
        if traces_dir is not None:
            return traces_dir
        from ai_agent_test.config import settings

        return settings.trace.dir

    # ---- 基础 --------------------------------------------------------- #
    @api.get("/api/health")
    async def health() -> dict:
        return {
            "ok": True,
            "demo": runtime.demo,
            "model": runtime.model_name,
            "modes": list(DEMO_MODES) if demo else list(MODES),
        }

    # ---- 对话 --------------------------------------------------------- #
    @api.get("/api/chat/{thread}")
    async def chat_state(thread: str, mode: str = "lead") -> dict:
        app = await runtime.app(mode)
        config = thread_config(thread, mode)
        state = await app.aget_state(config)
        pending = approval_events(await pending_approvals(app, config))
        return {"events": history_events(state.values or {}), "pending": pending[0]["items"] if pending else []}

    @api.post("/api/chat/{thread}/messages")
    async def chat_send(thread: str, body: MessageIn) -> StreamingResponse:
        app = await runtime.app(body.mode)
        config = thread_config(thread, body.mode)
        if await pending_approvals(app, config):
            raise HTTPException(409, "还有待审批的操作，请先处理")
        if not body.text.strip():
            raise HTTPException(400, "消息不能为空")
        return _streaming(stream_run(app, {"messages": [HumanMessage(content=body.text)]}, config))

    @api.post("/api/chat/{thread}/approvals")
    async def chat_approve(thread: str, body: ApprovalsIn) -> StreamingResponse:
        app = await runtime.app(body.mode)
        config = thread_config(thread, body.mode)
        pending = {item.id for item in await pending_approvals(app, config)}
        if not pending:
            raise HTTPException(409, "没有待审批的操作")
        unknown = set(body.answers) - pending
        if unknown or pending - set(body.answers):
            raise HTTPException(400, "需要对每一项待审批操作都给出决定")
        return _streaming(stream_run(app, Command(resume=body.answers), config))

    # ---- 评测与 trace -------------------------------------------------- #
    @api.get("/api/evals")
    async def evals() -> list[dict]:
        return list_result_files(reports())

    @api.get("/api/evals/{name}")
    async def eval_detail(name: str) -> dict:
        detail = result_detail(reports(), name)
        if detail is None:
            raise HTTPException(404, "结果文件不存在")
        return detail

    @api.get("/api/traces")
    async def trace_list() -> list[dict]:
        return list_traces(traces())

    @api.get("/api/traces/{name}")
    async def trace_get(name: str) -> dict:
        detail = trace_detail(traces(), name)
        if detail is None:
            raise HTTPException(404, "trace 不存在")
        return detail

    # ---- 前端静态文件 --------------------------------------------------- #
    if FRONTEND_DIST.is_dir():
        api.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @api.get("/{path:path}", include_in_schema=False)
        async def spa(path: str) -> FileResponse:
            if path.startswith("api/"):
                raise HTTPException(404, "接口不存在")
            target = (FRONTEND_DIST / path).resolve()
            if path and target.is_file() and target.parent == FRONTEND_DIST.resolve():
                return FileResponse(target)
            return FileResponse(FRONTEND_DIST / "index.html")

    return api
