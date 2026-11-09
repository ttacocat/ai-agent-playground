import { describe, expect, it } from "vitest";
import { parseSSE, type ServerEvent } from "../api";
import { emptyChat, lanesFor, reduce, rowsFromEvents } from "./model";

describe("parseSSE", () => {
  it("按空行分帧，半帧留到下一块", () => {
    const seen: ServerEvent[] = [];
    const rest = parseSSE('data: {"type":"done"}\n\ndata: {"type":"er', (e) => seen.push(e));
    expect(seen).toEqual([{ type: "done" }]);
    expect(rest).toBe('data: {"type":"er');
    parseSSE(rest + 'ror","message":"x"}\n\n', (e) => seen.push(e));
    expect(seen[1]).toEqual({ type: "error", message: "x" });
  });
});

describe("reduce", () => {
  it("工具结果合并到发起调用的那一行", () => {
    const rows = rowsFromEvents([
      { type: "tool_call", agent: "coder", call_id: "c1", name: "mysql_delete_data", args: { table: "user" }, risk: { level: "destructive", reason: "删数据" } },
      { type: "tool_result", agent: "coder", call_id: "c1", name: "mysql_delete_data", content: "⛔ 拦截", blocked: true },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ kind: "tool", result: { blocked: true } });
  });

  it("挂起时记录待审批项，结束时停止运行状态", () => {
    let state = reduce(emptyChat, { type: "running", running: true });
    state = reduce(state, {
      type: "event",
      event: { type: "approval_required", items: [{ id: "i1", tool: "run_shell", level: "destructive", reason: "rm", preview: "rm -rf x" }] },
    });
    expect(state.pending).toHaveLength(1);
    expect(state.rows.at(-1)).toMatchObject({ kind: "waiting", count: 1 });
    state = reduce(state, { type: "event", event: { type: "done" } });
    expect(state.running).toBe(false);
  });

  it("提交决定后，等待行变成决定记录", () => {
    let state = reduce(emptyChat, {
      type: "event",
      event: { type: "approval_required", items: [{ id: "i1", tool: "run_shell", level: "destructive", reason: "rm", preview: "rm" }] },
    });
    state = reduce(state, { type: "decided", approved: 0, denied: 1 });
    expect(state.pending).toEqual([]);
    expect(state.rows.at(-1)).toMatchObject({ kind: "decided", approved: 0, denied: 1 });
  });

  it("泳道按架构固定顺序，出现新 agent 时追加", () => {
    const rows = rowsFromEvents([{ type: "message", agent: "assistant", content: "hi" }]);
    expect(lanesFor("team", rows)).toEqual(["planner", "coder", "reviewer", "assistant"]);
    expect(lanesFor("unknown", [])).toEqual(["assistant"]);
  });
});
