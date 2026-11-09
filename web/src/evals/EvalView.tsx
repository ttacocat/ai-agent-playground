import { useEffect, useMemo, useState } from "react";
import { getJSON, type Cell, type EvalDetail, type ResultFile, type TargetSummary } from "../api";

const TARGET_LABELS: Record<string, string> = {
  single: "单 agent",
  supervisor: "Supervisor",
  team: "Team",
  lead: "Lead",
};
const TAG_FILTERS: { label: string; tag: string | null }[] = [
  { label: "全部", tag: null },
  { label: "安全", tag: "safety" },
  { label: "数据库", tag: "mysql" },
  { label: "虚拟机", tag: "vm" },
  { label: "编码", tag: "code" },
  { label: "大仓库", tag: "search" },
  { label: "检索", tag: "rag" },
  { label: "基线", tag: "baseline" },
];

const pct = (v: number) => `${Math.round(v * 100)}%`;
const fmt = (v: number) => Math.round(v).toLocaleString("zh-CN");

/** 成本与成功率：每个架构一个点。往左上越好。 */
function niceMax(v: number) {
  const step = 10 ** Math.floor(Math.log10(Math.max(v, 1)));
  return Math.ceil(v / step) * step;
}

function CostChart({ summary }: { summary: TargetSummary[] }) {
  // 固定像素尺寸：随容器等比放大会把字也放大
  const W = 400,
    H = 240,
    pad = { l: 44, r: 20, t: 14, b: 44 };
  const maxTok = niceMax(Math.max(...summary.map((s) => s.tokens), 1) * 1.1);
  const minRate = Math.min(0.5, Math.floor(Math.min(...summary.map((s) => s.success_rate)) * 10) / 10);
  const x = (t: number) => pad.l + (t / maxTok) * (W - pad.l - pad.r);
  const y = (r: number) => pad.t + ((1 - r) / (1 - minRate || 1)) * (H - pad.t - pad.b);
  const yTicks = [minRate, (minRate + 1) / 2, 1];
  const xTicks = [0, maxTok / 2, maxTok];
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label="各架构的平均 token 与成功率">
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={pad.l - 6} y={y(t) + 4} textAnchor="end" className="tick">
              {pct(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={x(t)} y={H - pad.b + 16} textAnchor="middle" className="tick">
            {fmt(t)}
          </text>
        ))}
        <text x={(W + pad.l) / 2} y={H - 6} textAnchor="middle" className="axis">
          平均 token / 次
        </text>
        {summary.map((s) => {
          const cx = x(s.tokens);
          const right = cx > W * 0.7; // 靠右的点把标签放左边，免得出界
          return (
            <g key={s.target} className={`dot target-${s.target}`}>
              <circle cx={cx} cy={y(s.success_rate)} r={6} />
              <text x={right ? cx - 10 : cx + 10} y={y(s.success_rate) + 4} textAnchor={right ? "end" : "start"}>
                {TARGET_LABELS[s.target] ?? s.target}
              </text>
            </g>
          );
        })}
      </svg>
      <figcaption>越靠左上越好：花得少，成功率高。</figcaption>
    </figure>
  );
}

function cellState(cell?: Cell) {
  if (!cell || cell.runs === 0) return "untested";
  if (cell.breaches > 0) return "breach";
  if (cell.passes === cell.runs) return "pass";
  return cell.passes > 0 ? "partial" : "fail";
}

export function EvalView() {
  const [files, setFiles] = useState<ResultFile[] | null>(null);
  const [name, setName] = useState<string>("");
  const [detail, setDetail] = useState<EvalDetail | null>(null);
  const [tag, setTag] = useState<string | null>(null);
  const [selected, setSelected] = useState<Cell | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getJSON<ResultFile[]>("/api/evals")
      .then((fs) => {
        setFiles(fs);
        if (fs[0]) setName(fs[0].name);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!name) return;
    setSelected(null);
    getJSON<EvalDetail>(`/api/evals/${encodeURIComponent(name)}`)
      .then(setDetail)
      .catch((e: Error) => setError(e.message));
  }, [name]);

  const cells = useMemo(() => {
    const map = new Map<string, Cell>();
    detail?.cells.forEach((c) => map.set(`${c.target}/${c.case}`, c));
    return map;
  }, [detail]);

  if (error) return <p className="banner-error">读取评测结果失败：{error}</p>;
  if (!files) return <p className="loading">读取评测结果</p>;
  if (files.length === 0)
    return (
      <div className="empty">
        <h2>还没有评测结果</h2>
        <p>
          运行 <code>uv run agent eval -t single supervisor team -k 1</code> 后，结果会出现在这里。
        </p>
      </div>
    );

  const cases = detail?.cases.filter((c) => !tag || c.tags.includes(tag)) ?? [];
  const caseTask = (id: string) => detail?.cases.find((c) => c.id === id)?.task ?? "";

  return (
    <div className="evals">
      <header className="evals-head">
        <label>
          结果
          <select value={name} onChange={(e) => setName(e.target.value)}>
            {files.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name}（{f.runs} 次运行）
              </option>
            ))}
          </select>
        </label>
        {detail && detail.stale > 0 && (
          <p className="note">有 {detail.stale} 条结果对应的用例已经改过，没有计入。</p>
        )}
      </header>

      {detail && (
        <>
          <section className="overview">
            <table className="summary">
              <caption>架构对比（{detail.models.join("、")}）</caption>
              <thead>
                <tr>
                  <th scope="col">架构</th>
                  <th scope="col">平均成功率</th>
                  <th scope="col">稳定通过</th>
                  <th scope="col">安全失守</th>
                  <th scope="col">平均 token</th>
                  <th scope="col" title="花在主模型上的 token；lead 架构的探索可以交给更便宜的模型">其中主模型</th>
                  <th scope="col" title="总 token ÷ 成功次数，失败的运行也算进成本">每次成功 token</th>
                  <th scope="col">工具调用</th>
                  <th scope="col">耗时</th>
                </tr>
              </thead>
              <tbody>
                {detail.summary.map((s) => (
                  <tr key={s.target}>
                    <th scope="row">{TARGET_LABELS[s.target] ?? s.target}</th>
                    <td>
                      <span className="rate">
                        <span className="rate-track" aria-hidden>
                          <span className="rate-bar" style={{ width: pct(s.success_rate) }} />
                        </span>
                        <span>{pct(s.success_rate)}</span>
                      </span>
                    </td>
                    <td>
                      {s.stable}/{s.cases}
                    </td>
                    <td className={s.breaches ? "bad" : ""}>{s.breaches}</td>
                    <td>{fmt(s.tokens)}</td>
                    <td>{fmt(s.main_tokens)}</td>
                    <td>{s.tokens_per_success == null ? "-" : fmt(s.tokens_per_success)}</td>
                    <td>{s.calls.toFixed(1)}</td>
                    <td>{s.seconds.toFixed(1)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <CostChart summary={detail.summary} />
          </section>

          <section className="matrix-wrap">
            <div className="matrix-head">
              <h2>逐条用例</h2>
              <div className="filters" role="group" aria-label="按类别筛选">
                {TAG_FILTERS.map((f) => (
                  <button
                    key={f.label}
                    type="button"
                    aria-pressed={tag === f.tag}
                    className={tag === f.tag ? "active" : ""}
                    onClick={() => setTag(f.tag)}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
              <ul className="legend">
                <li className="pass">全部通过</li>
                <li className="partial">部分通过</li>
                <li className="fail">未通过</li>
                <li className="breach">安全失守</li>
                <li className="untested">未测</li>
              </ul>
            </div>
            <table className="matrix">
              <thead>
                <tr>
                  <th scope="col">用例</th>
                  {detail.targets.map((t) => (
                    <th key={t} scope="col">
                      {TARGET_LABELS[t] ?? t}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => (
                  <tr key={c.id}>
                    <th scope="row">
                      <span className="case-id">{c.id}</span>
                      <span className="case-task">{c.task}</span>
                    </th>
                    {detail.targets.map((t) => {
                      const cell = cells.get(`${t}/${c.id}`);
                      const st = cellState(cell);
                      return (
                        <td key={t}>
                          <button
                            type="button"
                            className={`cell ${st}${selected === cell && cell ? " selected" : ""}`}
                            disabled={!cell || st === "pass"}
                            onClick={() => setSelected(cell ?? null)}
                            aria-label={`${c.id} ${TARGET_LABELS[t] ?? t}：${cell ? `${cell.passes}/${cell.runs} 通过` : "未测"}`}
                          >
                            {cell && cell.runs ? `${cell.passes}/${cell.runs}` : "—"}
                          </button>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          {selected && (
            <section className="failure" aria-live="polite">
              <h2>
                {selected.case} · {TARGET_LABELS[selected.target] ?? selected.target}
              </h2>
              <p className="case-task">{caseTask(selected.case)}</p>
              {selected.failures.map((f, i) => (
                <div key={i} className="failure-item">
                  <ul>
                    {f.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                  <p className="calls">调用：{f.calls.length ? f.calls.join("，") : "无"}</p>
                  {f.answer && <blockquote>{f.answer}</blockquote>}
                </div>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}
