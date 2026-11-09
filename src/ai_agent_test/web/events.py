"""把 LangGraph 的流式更新翻译成前端能直接渲染的事件。

``app.astream(..., stream_mode="updates", subgraphs=True)`` 产出 ``(namespace, {节点: 更新})``。
直接把它丢给前端，前端得懂 LangGraph 的内部结构；这里统一成几种事件：

==================  ==========================================================
``message``         某个 agent 说了一段话
``tool_call``       某个 agent 调用了工具（附带护栏预判的风险等级）
``tool_result``     工具返回（被护栏拦下的会标记 ``blocked``）
``handoff``         supervisor 把任务交给某个专家
``plan``            team 的方案（结构化）
``review``          team 的审查结论（结构化）
``final``           team 的最终答复
``approval_required`` 图挂起，等待人工审批
``done`` / ``error``
==================  ==========================================================

agent 名字优先取 AI 消息自带的 name（create_agent(name=...) 会写上），并记在它所在的
namespace 上：工具结果不带 agent 名字，按 namespace 归到最近一次说话的 agent。
lead 架构的子 agent 是在工具内部运行的，namespace 是 ``tools:<id>``，只能靠这个办法区分。
没有名字时退回子图 namespace（``code_expert:<uuid>`` → ``code_expert``）或节点名。
同一条消息可能既出现在子图里，又作为子图的输出出现在父图里，按消息 id 去重。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from ai_agent_test.guardrails.policy import assess

TOOL_RESULT_LIMIT = 4000
_INTERNAL_NODES = {"__interrupt__", "__start__"}
_TEAM_STAGES = {"planner", "coder", "reviewer", "finish"}
_STRUCTURED_TOOLS = {"Plan", "Review"}


def agent_of(namespace: tuple[str, ...], node: str) -> str:
    if namespace:
        return namespace[0].split(":", 1)[0]  # 最外层子图就是 agent；更深的是它内部的节点
    return "assistant" if node in {"model", "tools", "agent"} else node


def _clip(text: str, limit: int = TOOL_RESULT_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n…（省略 {len(text) - limit} 字符）"


def _content(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):  # 多段内容：只取文本
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def message_events(message: BaseMessage, agent: str) -> list[dict[str, Any]]:
    """单条消息 → 事件列表（一条 AI 消息可能既有文字又有多个工具调用）。"""
    events: list[dict[str, Any]] = []
    if isinstance(message, HumanMessage):
        return [{"type": "user", "content": _content(message)}]
    if isinstance(message, ToolMessage):
        text = _content(message)
        if text.startswith("Successfully transferred"):
            return []  # supervisor 的交接回执，handoff 事件已经表达了
        if message.name in _STRUCTURED_TOOLS:
            return []  # 结构化输出的回执（"Returning structured response..."），plan / review 事件已经表达了
        return [
            {
                "type": "tool_result",
                "agent": agent,
                "call_id": message.tool_call_id,
                "name": message.name or "",
                "content": _clip(text),
                "blocked": text.startswith("⛔"),
            }
        ]
    if isinstance(message, AIMessage):
        text = _content(message).strip()
        if text:
            events.append({"type": "message", "agent": message.name or agent, "content": text, "id": message.id})
        for call in message.tool_calls or []:
            name = call["name"]
            if name.startswith("transfer_to_"):
                events.append({"type": "handoff", "agent": agent, "to": name.removeprefix("transfer_to_")})
                continue
            if name in _STRUCTURED_TOOLS:
                continue  # 结构化输出，由 plan / review 事件表达
            decision = assess(name, call.get("args") or {})
            events.append(
                {
                    "type": "tool_call",
                    "agent": message.name or agent,
                    "call_id": call.get("id"),
                    "name": name,
                    "args": call.get("args") or {},
                    "risk": {"level": decision.level.value, "reason": decision.reason},
                }
            )
    return events


class EventTranslator:
    """有状态：记住已经发过的消息 id，避免父图重复转发子图的输出。"""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._ns_agent: dict[tuple[str, ...], str] = {}

    def _fresh(self, message: BaseMessage) -> bool:
        if message.id is None:
            return True
        if message.id in self._seen:
            return False
        self._seen.add(message.id)
        return True

    def translate(self, namespace: tuple[str, ...], update: dict[str, Any]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for node, data in update.items():
            if node in _INTERNAL_NODES or not isinstance(data, dict):
                continue
            top_level_team = not namespace and node in _TEAM_STAGES
            if top_level_team:
                events += self._team_stage(node, data)
                continue
            for message in data.get("messages") or []:
                if isinstance(message, HumanMessage):
                    continue  # 用户消息由前端自己显示；子 agent 收到的任务说明不重复展示
                if isinstance(message, AIMessage) and message.name:
                    self._ns_agent[namespace] = message.name
                agent = self._ns_agent.get(namespace) or agent_of(namespace, node)
                if self._fresh(message):
                    events += message_events(message, agent)
        return events

    def _team_stage(self, node: str, data: dict[str, Any]) -> list[dict[str, Any]]:
        # team 顶层节点的消息是阶段汇总；细节已经从子图流过来，这里只发结构化结果
        for message in data.get("messages") or []:
            self._fresh(message)
        if node == "planner" and data.get("plan"):
            return [{"type": "plan", "agent": "planner", "plan": data["plan"]}]
        if node == "coder":
            return [{"type": "stage", "agent": "coder", "attempt": data.get("attempts", 1)}]
        if node == "reviewer" and data.get("review"):
            return [{"type": "review", "agent": "reviewer", "review": data["review"]}]
        if node == "finish":
            messages = data.get("messages") or []
            text = _content(messages[-1]) if messages else ""
            return [{"type": "final", "agent": "team", "content": text}]
        return []


def history_events(values: dict[str, Any]) -> list[dict[str, Any]]:
    """刷新页面时，从会话状态重建事件。子 agent 的内部步骤不在父图状态里，只能看到汇总。"""
    events: list[dict[str, Any]] = []
    speaker = "assistant"
    for message in values.get("messages") or []:
        if isinstance(message, AIMessage):
            speaker = message.name or "assistant"
        # 工具结果的 name 是工具名，不是 agent 名：归到发起调用的那个 agent
        agent = speaker if isinstance(message, AIMessage | ToolMessage) else "user"
        if agent == "team" and isinstance(message, AIMessage):
            events.append({"type": "final", "agent": "team", "content": _content(message)})
        else:
            events += message_events(message, agent)
    return events


def approval_events(pending: Iterable[Any]) -> list[dict[str, Any]]:
    items = [
        {
            "id": item.id,
            "tool": item.value.get("tool"),
            "level": item.value.get("level"),
            "reason": item.value.get("reason"),
            "preview": item.value.get("preview"),
        }
        for item in pending
    ]
    return [{"type": "approval_required", "items": items}] if items else []
