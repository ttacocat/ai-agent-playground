import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { ApiError, getJSON, streamPost, type ApprovalItem, type Health, type ServerEvent } from "../api";
import { ApprovalPanel } from "./ApprovalPanel";
import { emptyChat, lanesFor, reduce, rowsFromEvents, type Row } from "./model";
import { Swimlanes } from "./Rows";

interface Thread {
  id: string;
  mode: string;
  title: string;
}

const STORE_KEY = "agent-console-threads";

function loadThreads(): Thread[] {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) ?? "[]") as Thread[];
  } catch {
    return [];
  }
}

function saveThreads(threads: Thread[]) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(threads.slice(0, 30)));
  } catch {
    /* 隐私模式等：只是不记住会话列表 */
  }
}

const newThread = (mode: string): Thread => ({ id: Date.now().toString(36), mode, title: "新对话" });

const MODE_LABELS: Record<string, string> = { lead: "Lead", team: "Team", supervisor: "Supervisor" };

const MODE_INTROS: Record<string, string> = {
  lead: "主 agent 从头做到尾；大范围搜索交给探索助手，改完可以请审查助手核实。危险操作会停下来等你确认。",
  team: "方案专家先调研，实现专家动手，审查专家核实结果。危险操作会停下来等你确认。",
  supervisor: "主管把任务分派给方案专家和编码专家。危险操作会停下来等你确认。",
};

const SUGGESTIONS = ["test 库的 user 表有哪些字段？", "删除 test 库 user 表里 id=1 的记录"];

function Progress({ rows }: { rows: Row[] }) {
  const planned = rows.some((r) => r.kind === "plan");
  const coded = rows.some((r) => r.kind === "stage");
  const review = [...rows].reverse().find((r) => r.kind === "review");
  const steps = [
    { name: "方案", done: planned },
    { name: "实现", done: coded },
    { name: "审查", done: Boolean(review) },
  ];
  return (
    <ol className="progress" aria-label="当前进度">
      {steps.map((s) => (
        <li key={s.name} className={s.done ? "done" : ""}>
          {s.name}
        </li>
      ))}
    </ol>
  );
}

export function ChatView({ health }: { health: Health }) {
  const [threads, setThreads] = useState<Thread[]>(() => {
    const saved = loadThreads();
    return saved.length ? saved : [newThread(health.modes[0] ?? "team")];
  });
  const [current, setCurrent] = useState<string>(() => threads[0].id);
  const [draft, setDraft] = useState("");
  const [state, dispatch] = useReducer(reduce, emptyChat);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  const thread = threads.find((t) => t.id === current) ?? threads[0];

  useEffect(() => saveThreads(threads), [threads]);

  useEffect(() => {
    let cancelled = false;
    abort.current?.abort();
    getJSON<{ events: ServerEvent[]; pending: ApprovalItem[] }>(
      `/api/chat/${thread.id}?mode=${encodeURIComponent(thread.mode)}`,
    )
      .then((s) => !cancelled && dispatch({ type: "reset", rows: rowsFromEvents(s.events), pending: s.pending }))
      .catch((e: Error) => !cancelled && dispatch({ type: "failed", message: e.message }));
    return () => {
      cancelled = true;
    };
  }, [thread.id, thread.mode]);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: "end", behavior: "smooth" });
  }, [state.rows.length]);

  const run = useCallback(async (url: string, body: unknown) => {
    abort.current = new AbortController();
    dispatch({ type: "running", running: true });
    try {
      await streamPost(url, body, (event) => dispatch({ type: "event", event }), abort.current.signal);
    } catch (e) {
      if ((e as Error).name === "AbortError") return;
      const message = e instanceof ApiError && e.status === 409 ? "还有操作等你确认，先在右侧处理" : (e as Error).message;
      dispatch({ type: "failed", message });
    }
  }, []);

  const send = (text: string) => {
    const content = text.trim();
    if (!content || state.running) return;
    setDraft("");
    if (thread.title === "新对话") {
      setThreads((ts) => ts.map((t) => (t.id === thread.id ? { ...t, title: content.slice(0, 24) } : t)));
    }
    dispatch({ type: "local_user", content });
    void run(`/api/chat/${thread.id}/messages`, { text: content, mode: thread.mode });
  };

  const decide = (answers: Record<string, boolean>) => {
    const values = Object.values(answers);
    dispatch({ type: "decided", approved: values.filter(Boolean).length, denied: values.filter((v) => !v).length });
    void run(`/api/chat/${thread.id}/approvals`, { mode: thread.mode, answers });
  };

  const create = (mode: string) => {
    const t = newThread(mode);
    setThreads((ts) => [t, ...ts]);
    setCurrent(t.id);
  };

  const lanes = lanesFor(thread.mode, state.rows);
  const blocked = state.pending.length > 0;

  return (
    <div className="chat">
      <aside className="threads" aria-label="会话">
        <div className="new-thread">
          {health.modes.map((m) => (
            <button key={m} type="button" onClick={() => create(m)}>
              新建 {MODE_LABELS[m] ?? m} 对话
            </button>
          ))}
        </div>
        <ul>
          {threads.map((t) => (
            <li key={t.id}>
              <button type="button" className={t.id === thread.id ? "active" : ""} onClick={() => setCurrent(t.id)}>
                <span>{t.title}</span>
                <small>{t.mode}</small>
              </button>
            </li>
          ))}
        </ul>
      </aside>

      <main className="conversation">
        {state.rows.length === 0 ? (
          <div className="empty">
            <h2>让团队开始干活</h2>
            <p>{MODE_INTROS[thread.mode] ?? MODE_INTROS.lead}</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <Swimlanes rows={state.rows} lanes={lanes} awaiting={blocked} />
        )}
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            send(draft);
          }}
        >
          <textarea
            value={draft}
            rows={2}
            placeholder={blocked ? "先处理右侧等待确认的操作" : "描述要做的事，Enter 发送，Shift+Enter 换行"}
            disabled={blocked}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                send(draft);
              }
            }}
          />
          <button type="submit" className="primary" disabled={state.running || blocked || !draft.trim()}>
            {state.running ? "运行中" : "发送"}
          </button>
        </form>
        {state.error && <p className="banner-error">{state.error}</p>}
        <div ref={bottom} className="scroll-end" />
      </main>

      <aside className={`side${blocked ? " has-approval" : ""}`}>
        {blocked ? (
          <ApprovalPanel items={state.pending} busy={state.running} onSubmit={decide} />
        ) : (
          <section className="status">
            <h2>状态</h2>
            {thread.mode === "team" && <Progress rows={state.rows} />}
            <dl>
              <div>
                <dt>架构</dt>
                <dd>{thread.mode}</dd>
              </div>
              <div>
                <dt>模型</dt>
                <dd>{health.model || "未配置"}</dd>
              </div>
              <div>
                <dt>运行</dt>
                <dd>{state.running ? "进行中" : "空闲"}</dd>
              </div>
            </dl>
          </section>
        )}
      </aside>
    </div>
  );
}
