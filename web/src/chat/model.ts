import type { ApprovalItem, Plan, Review, RiskLevel, ServerEvent } from "../api";

// 页面上的一行：工具调用和它的结果合并成一行，其余事件各占一行
export type Row =
  | { kind: "user"; key: string; content: string }
  | { kind: "message"; key: string; agent: string; content: string }
  | {
      kind: "tool";
      key: string;
      agent: string;
      name: string;
      args: Record<string, unknown>;
      risk: { level: RiskLevel; reason: string };
      result?: { content: string; blocked: boolean };
    }
  | { kind: "handoff"; key: string; agent: string; to: string }
  | { kind: "plan"; key: string; agent: string; plan: Plan }
  | { kind: "stage"; key: string; agent: string; attempt: number }
  | { kind: "review"; key: string; agent: string; review: Review }
  | { kind: "final"; key: string; content: string }
  | { kind: "waiting"; key: string; count: number }
  | { kind: "decided"; key: string; approved: number; denied: number }
  | { kind: "error"; key: string; message: string };

export interface ChatState {
  rows: Row[];
  pending: ApprovalItem[];
  running: boolean;
  error?: string;
}

export const emptyChat: ChatState = { rows: [], pending: [], running: false };

export type Action =
  | { type: "reset"; rows: Row[]; pending: ApprovalItem[] }
  | { type: "event"; event: ServerEvent }
  | { type: "local_user"; content: string }
  | { type: "decided"; approved: number; denied: number }
  | { type: "running"; running: boolean }
  | { type: "failed"; message: string };

let counter = 0;
const key = () => `r${++counter}`;

export function rowsFromEvents(events: ServerEvent[]): Row[] {
  return events.reduce<ChatState>((state, event) => reduce(state, { type: "event", event }), emptyChat).rows;
}

export function reduce(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "reset":
      return { rows: action.rows, pending: action.pending, running: false };
    case "local_user":
      return { ...state, error: undefined, rows: [...state.rows, { kind: "user", key: key(), content: action.content }] };
    case "running":
      return { ...state, running: action.running };
    case "decided": {
      // "等你确认"那一行换成你的决定，待审批项清空
      const rows = state.rows.slice();
      const index = rows.map((r) => r.kind).lastIndexOf("waiting");
      const decided: Row = { kind: "decided", key: key(), approved: action.approved, denied: action.denied };
      if (index === -1) rows.push(decided);
      else rows[index] = decided;
      return { ...state, rows, pending: [] };
    }
    case "failed":
      return { ...state, running: false, error: action.message };
    case "event":
      return applyEvent(state, action.event);
  }
}

function applyEvent(state: ChatState, event: ServerEvent): ChatState {
  const push = (row: Row) => ({ ...state, rows: [...state.rows, row] });
  switch (event.type) {
    case "user":
      return push({ kind: "user", key: key(), content: event.content });
    case "message":
      return push({ kind: "message", key: key(), agent: event.agent, content: event.content });
    case "tool_call":
      return push({
        kind: "tool",
        key: event.call_id ?? key(),
        agent: event.agent,
        name: event.name,
        args: event.args,
        risk: event.risk,
      });
    case "tool_result": {
      // 挂到发起调用的那一行上；找不到（例如刷新后只剩结果）就单独一行
      const index = state.rows.findIndex((r) => r.kind === "tool" && r.key === event.call_id && !r.result);
      const result = { content: event.content, blocked: event.blocked };
      if (index === -1) {
        return push({
          kind: "tool",
          key: key(),
          agent: event.agent,
          name: event.name,
          args: {},
          risk: { level: "read", reason: "" },
          result,
        });
      }
      const rows = state.rows.slice();
      rows[index] = { ...(rows[index] as Extract<Row, { kind: "tool" }>), result };
      return { ...state, rows };
    }
    case "handoff":
      return push({ kind: "handoff", key: key(), agent: event.agent, to: event.to });
    case "plan":
      return push({ kind: "plan", key: key(), agent: event.agent, plan: event.plan });
    case "stage":
      return push({ kind: "stage", key: key(), agent: event.agent, attempt: event.attempt });
    case "review":
      return push({ kind: "review", key: key(), agent: event.agent, review: event.review });
    case "final":
      return push({ kind: "final", key: key(), content: event.content });
    case "approval_required":
      return { ...push({ kind: "waiting", key: key(), count: event.items.length }), pending: event.items };
    case "error":
      return { ...push({ kind: "error", key: key(), message: event.message }), running: false };
    case "done":
      return { ...state, running: false };
  }
}

// ---- 泳道 -------------------------------------------------------------- //
export const AGENT_LABELS: Record<string, string> = {
  planner: "方案",
  coder: "实现",
  reviewer: "审查",
  supervisor: "主管",
  research_expert: "方案专家",
  code_expert: "编码专家",
  assistant: "助手",
  team: "团队",
  lead: "主 agent",
  explorer: "探索",
};

export const LANES_BY_MODE: Record<string, string[]> = {
  lead: ["lead", "explorer", "reviewer"],
  team: ["planner", "coder", "reviewer"],
  supervisor: ["supervisor", "research_expert", "code_expert"],
};

export function lanesFor(mode: string, rows: Row[]): string[] {
  const lanes = [...(LANES_BY_MODE[mode] ?? [])];
  for (const row of rows) {
    if ("agent" in row && !lanes.includes(row.agent)) lanes.push(row.agent);
  }
  return lanes.length ? lanes : ["assistant"];
}

export const RISK_LABELS: Record<RiskLevel, string> = {
  read: "只读",
  write: "写入",
  destructive: "需确认",
};
