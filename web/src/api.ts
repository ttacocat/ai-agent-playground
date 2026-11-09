// 与后端 src/ai_agent_test/web/events.py 的事件一一对应

export type RiskLevel = "read" | "write" | "destructive";

export interface Plan {
  needs_changes: boolean;
  goal: string;
  steps: string[];
  files: string[];
  risks: string[];
  answer: string;
  context?: { path: string; content: string; note?: string }[];
}

export interface Review {
  passed: boolean;
  issues: string[];
  summary: string;
}

export interface ApprovalItem {
  id: string;
  tool: string;
  level: RiskLevel;
  reason: string;
  preview: string;
}

export type ServerEvent =
  | { type: "user"; content: string }
  | { type: "message"; agent: string; content: string; id?: string }
  | {
      type: "tool_call";
      agent: string;
      call_id: string;
      name: string;
      args: Record<string, unknown>;
      risk: { level: RiskLevel; reason: string };
    }
  | { type: "tool_result"; agent: string; call_id: string; name: string; content: string; blocked: boolean }
  | { type: "handoff"; agent: string; to: string }
  | { type: "plan"; agent: string; plan: Plan }
  | { type: "stage"; agent: string; attempt: number }
  | { type: "review"; agent: string; review: Review }
  | { type: "final"; agent: string; content: string }
  | { type: "approval_required"; items: ApprovalItem[] }
  | { type: "done" }
  | { type: "error"; message: string };

export interface Health {
  ok: boolean;
  demo: boolean;
  model: string;
  modes: string[];
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* 不是 JSON：用状态文本 */
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

export const getJSON = <T>(url: string) => fetch(url).then((r) => json<T>(r));

/**
 * 把 SSE 文本切成事件。EventSource 只支持 GET，这里的对话接口是 POST，
 * 所以自己解析：按空行分帧，取 `data:` 行。返回值是没解析完的半帧，留给下一块数据。
 */
export function parseSSE(buffer: string, onEvent: (event: ServerEvent) => void): string {
  const frames = buffer.split("\n\n");
  const rest = frames.pop() ?? "";
  for (const frame of frames) {
    const data = frame
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => line.slice(6))
      .join("\n");
    if (data) onEvent(JSON.parse(data) as ServerEvent);
  }
  return rest;
}

export async function streamPost(
  url: string,
  body: unknown,
  onEvent: (event: ServerEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body) {
    await json(response); // 抛出带后端说明的错误
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = parseSSE(buffer + decoder.decode(value, { stream: true }), onEvent);
  }
  parseSSE(buffer + "\n\n", onEvent);
}

// ---- 评测与 trace ------------------------------------------------------ //
export interface ResultFile {
  name: string;
  models: string[];
  targets: string[];
  runs: number;
  updated: number;
}

export interface TargetSummary {
  target: string;
  cases: number;
  runs: number;
  errors: number;
  success_rate: number;
  stable: number;
  breaches: number;
  tokens: number;
  main_tokens: number;
  tokens_per_success: number | null;
  explores: number;
  calls: number;
  seconds: number;
}

export interface Cell {
  target: string;
  case: string;
  runs: number;
  passes: number;
  errors: number;
  breaches: number;
  tokens: number;
  calls: number;
  seconds: number;
  failures: { reasons: string[]; calls: string[]; answer: string }[];
}

export interface EvalDetail {
  name: string;
  models: string[];
  targets: string[];
  summary: TargetSummary[];
  cells: Cell[];
  cases: { id: string; task: string; tags: string[] }[];
  stale: number;
}

export interface TraceEntry {
  name: string;
  started_at: string | null;
  model: string | null;
  total: number | null;
  spans: number;
  llm_calls: number;
  tool_calls: number;
  tokens: number;
}

export interface Span {
  id: string;
  parent: string | null;
  kind: "llm" | "tool" | string;
  name: string;
  start: number;
  duration: number;
  status: string;
  tokens: number;
  detail: string;
}

export interface TraceDetail {
  name: string;
  header: { started_at?: string; model?: string; total?: number };
  truncated: boolean;
  spans: Span[];
}
