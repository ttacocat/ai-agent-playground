"""模型兼容层：不支持强制 tool_choice 的模型自动降级成 auto，而不是让整次运行崩掉。"""

from __future__ import annotations

import json

import httpx
from pydantic import BaseModel

from ai_agent_test.app.code_agent.model.compat import CompatChatOpenAI

REJECT = {
    "error": {
        "code": "invalid_parameter_error",
        "message": "The tool_choice parameter does not support being set to required or object in thinking mode",
    }
}


class Verdict(BaseModel):
    passed: bool


def completion(tool_call: bool = True) -> dict:
    message = {"role": "assistant", "content": "ok"}
    if tool_call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "Verdict", "arguments": '{"passed": true}'}}
            ],
        }
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def sse(tool_call: bool = True) -> bytes:
    body = completion(tool_call)
    message = body["choices"][0]["message"]
    delta = {"role": "assistant", "content": message["content"]}
    if tool_call:
        delta["tool_calls"] = [{"index": 0, **message["tool_calls"][0]}]
    chunk = {
        **body,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


def make_model(name: str, *, streaming: bool, seen: list):
    def handle(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload.get("tool_choice"))
        choice = payload.get("tool_choice")
        if choice == "required" or isinstance(choice, dict):
            return httpx.Response(400, json=REJECT)
        if payload.get("stream"):
            return httpx.Response(200, content=sse(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=completion())

    transport = httpx.MockTransport(handle)
    return CompatChatOpenAI(
        model=name,
        api_key="k",
        base_url="http://test/v1",
        streaming=streaming,
        max_retries=0,
        http_client=httpx.Client(transport=transport),
        http_async_client=httpx.AsyncClient(transport=transport),
    )


async def test_forced_tool_choice_falls_back_to_auto_and_is_remembered():
    seen: list = []
    model = make_model("thinking-a", streaming=False, seen=seen)
    bound = model.bind_tools([Verdict], tool_choice="any")
    reply = await bound.ainvoke("审查")
    assert reply.tool_calls[0]["name"] == "Verdict"
    assert seen == ["required", "auto"]

    await bound.ainvoke("再审查一次")  # 记住了：不再先试一次 required
    assert seen == ["required", "auto", "auto"]


async def test_fallback_also_works_when_streaming():
    seen: list = []
    model = make_model("thinking-b", streaming=True, seen=seen)
    reply = await model.bind_tools([Verdict], tool_choice="Verdict").ainvoke("审查")
    assert reply.tool_calls[0]["name"] == "Verdict"
    assert seen[0] == {"type": "function", "function": {"name": "Verdict"}} and seen[1] == "auto"


def test_sync_path_and_unrelated_errors_are_untouched():
    seen: list = []
    model = make_model("thinking-c", streaming=False, seen=seen)
    assert model.bind_tools([Verdict], tool_choice="any").invoke("审查").tool_calls
    assert seen == ["required", "auto"]

    plain: list = []
    ok = make_model("plain", streaming=False, seen=plain)
    ok.bind_tools([Verdict]).invoke("审查")  # 没有强制调用：原样发出
    assert plain == [None] or plain == ["auto"]
