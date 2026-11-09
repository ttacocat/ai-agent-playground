import { useEffect, useState } from "react";
import { getJSON, type Span, type TraceDetail, type TraceEntry } from "../api";

const PAGE = 120;

/** 一段调用的时间线。时间轴按这一段自己的起止缩放：一小时的评测 trace 整体铺开，每个调用都只剩一条线。 */
function Gantt({ spans }: { spans: Span[] }) {
  const shown = spans;
  const begin = Math.min(...shown.map((s) => s.start));
  const end = Math.max(...shown.map((s) => s.start + s.duration), begin + 0.001);
  const ROW = 22,
    LABEL = 190,
    W = 900;
  const x = (t: number) => LABEL + ((t - begin) / (end - begin)) * (W - LABEL - 12);
  const H = shown.length * ROW + 28;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => begin + f * (end - begin));
  return (
    <div className="gantt-scroll">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="调用时间线">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={H - 20} className="grid" />
            <text x={x(t)} y={H - 6} textAnchor="middle" className="tick">
              {t.toFixed(end - begin < 10 ? 1 : 0)}s
            </text>
          </g>
        ))}
        {shown.map((s, i) => (
          <g key={s.id} className={`bar kind-${s.kind}${s.status !== "ok" ? " failed" : ""}`}>
            <title>
              {`${s.name}  ${s.duration.toFixed(2)}s${s.tokens ? `  ${s.tokens} token` : ""}\n${s.detail}`}
            </title>
            <text x={LABEL - 8} y={i * ROW + 15} textAnchor="end" className="bar-label">
              {s.name.length > 24 ? `${s.name.slice(0, 23)}…` : s.name}
            </text>
            <rect x={x(s.start)} y={i * ROW + 4} width={Math.max(x(s.start + s.duration) - x(s.start), 2)} height={ROW - 8} rx={2} />
          </g>
        ))}
      </svg>
    </div>
  );
}

export function TraceView() {
  const [list, setList] = useState<TraceEntry[] | null>(null);
  const [name, setName] = useState("");
  const [page, setPage] = useState(0);
  const [detail, setDetail] = useState<TraceDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<TraceEntry[]>("/api/traces")
      .then((items) => {
        setList(items);
        if (items[0]) setName(items[0].name);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!name) return;
    setPage(0);
    getJSON<TraceDetail>(`/api/traces/${encodeURIComponent(name)}`)
      .then(setDetail)
      .catch((e: Error) => setError(e.message));
  }, [name]);

  if (error) return <p className="banner-error">读取 trace 失败：{error}</p>;
  if (!list) return <p className="loading">读取 trace</p>;
  if (list.length === 0)
    return (
      <div className="empty">
        <h2>还没有 trace</h2>
        <p>
          运行一次 <code>uv run agent chat</code> 或 <code>uv run agent eval</code>，每次运行都会留下一份记录。
        </p>
      </div>
    );

  const spans = detail?.spans ?? [];
  const llm = spans.filter((s) => s.kind === "llm");
  const tools = spans.filter((s) => s.kind === "tool");
  const tokens = spans.reduce((n, s) => n + s.tokens, 0);

  return (
    <div className="traces">
      <aside className="trace-list" aria-label="运行记录">
        <ul>
          {list.map((t) => (
            <li key={t.name}>
              <button type="button" className={t.name === name ? "active" : ""} onClick={() => setName(t.name)}>
                <span>{t.name}</span>
                <small>
                  {t.model || "未知模型"}，{t.llm_calls} 次 LLM，{t.tool_calls} 次工具
                </small>
              </button>
            </li>
          ))}
        </ul>
      </aside>
      <section className="trace-main">
        {detail && (
          <>
            <dl className="stats">
              <div>
                <dt>总耗时</dt>
                <dd>{(detail.header.total ?? 0).toFixed(1)}s</dd>
              </div>
              <div>
                <dt>LLM 调用</dt>
                <dd>{llm.length}</dd>
              </div>
              <div>
                <dt>工具调用</dt>
                <dd>{tools.length}</dd>
              </div>
              <div>
                <dt>token</dt>
                <dd>{tokens ? tokens.toLocaleString("zh-CN") : "未记录"}</dd>
              </div>
            </dl>
            <ul className="legend">
              <li className="kind-llm">LLM</li>
              <li className="kind-tool">工具</li>
              <li className="failed">出错</li>
            </ul>
            {spans.length > PAGE && (
              <div className="pager">
                <button type="button" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                  上一段
                </button>
                <span>
                  第 {page * PAGE + 1}–{Math.min((page + 1) * PAGE, spans.length)} 个调用，共 {spans.length} 个
                </span>
                <button
                  type="button"
                  disabled={(page + 1) * PAGE >= spans.length}
                  onClick={() => setPage((p) => p + 1)}
                >
                  下一段
                </button>
              </div>
            )}
            {spans.length > 0 && <Gantt spans={spans.slice(page * PAGE, (page + 1) * PAGE)} />}
            {detail.truncated && <p className="note">记录太长，只读取了前 {spans.length} 个调用。</p>}
          </>
        )}
      </section>
    </div>
  );
}
