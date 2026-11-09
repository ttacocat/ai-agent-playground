import type { CSSProperties } from "react";
import { AGENT_LABELS, RISK_LABELS, type Row } from "./model";

const label = (agent: string) => AGENT_LABELS[agent] ?? agent;

function laneStyle(lanes: string[], agent: string, row: number): CSSProperties {
  const col = Math.max(lanes.indexOf(agent), 0) + 1;
  return { "--col": `${col} / span 1`, gridRow: row } as CSSProperties;
}

const wide = (row: number): CSSProperties => ({ "--col": "1 / -1", gridRow: row }) as CSSProperties;

/** 命令、SQL 这类参数单独成块显示；其余参数紧凑地列出来 */
function Args({ args }: { args: Record<string, unknown> }) {
  const code = ["command", "script", "sql", "query"].map((k) => args[k]).find((v) => typeof v === "string") as
    | string
    | undefined;
  const rest = Object.entries(args).filter(([k, v]) => !(typeof v === "string" && v === code && k !== "query"));
  return (
    <>
      {code && <pre className="code">{code}</pre>}
      {rest.length > 0 && (
        <dl className="args">
          {rest.map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{typeof v === "string" ? v : JSON.stringify(v)}</dd>
            </div>
          ))}
        </dl>
      )}
    </>
  );
}

function ToolRow({ row, awaiting }: { row: Extract<Row, { kind: "tool" }>; awaiting: boolean }) {
  const lines = row.result?.content.split("\n") ?? [];
  const long = lines.length > 4 || (row.result?.content.length ?? 0) > 280;
  return (
    <div className={`tool risk-${row.risk.level}${row.result?.blocked ? " blocked" : ""}`}>
      <header>
        <span className="tool-name">{row.name}</span>
        <span className="risk" title={row.risk.reason}>
          {RISK_LABELS[row.risk.level]}
        </span>
      </header>
      <Args args={row.args} />
      {!row.result && <p className="tool-wait">{awaiting ? "等待你确认" : "执行中"}</p>}
      {row.result?.blocked && <p className="tool-blocked">已被安全护栏拦截，没有执行</p>}
      {row.result &&
        !row.result.blocked &&
        (long ? (
          <details className="result">
            <summary>{lines.slice(0, 2).join(" ").slice(0, 120)}</summary>
            <pre>{row.result.content}</pre>
          </details>
        ) : (
          <pre className="result">{row.result.content || "（无输出）"}</pre>
        ))}
    </div>
  );
}

interface RowProps {
  row: Row;
  lanes: string[];
  index: number;
  awaiting: boolean;
}

export function RowView({ row, lanes, index, awaiting }: RowProps) {
  const gridRow = index + 2; // 第 1 行是泳道表头
  switch (row.kind) {
    case "user":
      return (
        <div className="item user" style={wide(gridRow)}>
          <p>{row.content}</p>
        </div>
      );
    case "message":
      return (
        <div className="item message" style={laneStyle(lanes, row.agent, gridRow)}>
          <span className="who">{label(row.agent)}</span>
          <p>{row.content}</p>
        </div>
      );
    case "tool":
      return (
        <div className="item" style={laneStyle(lanes, row.agent, gridRow)}>
          <span className="who">{label(row.agent)}</span>
          <ToolRow row={row} awaiting={awaiting} />
        </div>
      );
    case "handoff": {
      const from = Math.max(lanes.indexOf(row.agent), 0);
      const to = Math.max(lanes.indexOf(row.to), 0);
      const style = {
        "--col": `${Math.min(from, to) + 1} / ${Math.max(from, to) + 2}`,
        gridRow,
      } as CSSProperties;
      return (
        <div className={`item handoff ${to < from ? "leftward" : ""}`} style={style}>
          <span>
            {label(row.agent)}把任务交给{label(row.to)}
          </span>
        </div>
      );
    }
    case "plan":
      return (
        <div className="item" style={laneStyle(lanes, row.agent, gridRow)}>
          <span className="who">{label(row.agent)}</span>
          <div className="card plan">
            <h3>{row.plan.needs_changes ? "方案" : "无需改动"}</h3>
            <p>{row.plan.goal}</p>
            {row.plan.steps.length > 0 && (
              <ol>
                {row.plan.steps.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ol>
            )}
            {row.plan.risks.length > 0 && <p className="note">风险：{row.plan.risks.join("；")}</p>}
            {row.plan.context && row.plan.context.length > 0 && (
              <details className="excerpts">
                <summary>交给实现者的调研摘录（{row.plan.context.length} 段）</summary>
                {row.plan.context.map((c) => (
                  <div key={c.path}>
                    <p className="excerpt-path">
                      {c.path}
                      {c.note && <span>（{c.note}）</span>}
                    </p>
                    <pre className="code">{c.content}</pre>
                  </div>
                ))}
              </details>
            )}
          </div>
        </div>
      );
    case "stage":
      return (
        <div className="item stage" style={laneStyle(lanes, row.agent, gridRow)}>
          <span>第 {row.attempt} 轮实现完成</span>
        </div>
      );
    case "review":
      return (
        <div className="item" style={laneStyle(lanes, row.agent, gridRow)}>
          <span className="who">{label(row.agent)}</span>
          <div className={`card review ${row.review.passed ? "passed" : "failed"}`}>
            <h3>{row.review.passed ? "审查通过" : "审查未通过"}</h3>
            {row.review.summary && <p>{row.review.summary}</p>}
            {row.review.issues.length > 0 && (
              <ul>
                {row.review.issues.map((i) => (
                  <li key={i}>{i}</li>
                ))}
              </ul>
            )}
          </div>
        </div>
      );
    case "final":
      return (
        <div className="item final" style={wide(gridRow)}>
          <p>{row.content}</p>
        </div>
      );
    case "waiting":
      return (
        <div className="item waiting" style={wide(gridRow)}>
          <span>暂停：有 {row.count} 个操作等你确认</span>
        </div>
      );
    case "decided": {
      const parts = [row.approved && `批准了 ${row.approved} 个`, row.denied && `拒绝了 ${row.denied} 个`].filter(Boolean);
      return (
        <div className="item decided" style={wide(gridRow)}>
          <span>你{parts.join("，")}操作</span>
        </div>
      );
    }
    case "error":
      return (
        <div className="item error" style={wide(gridRow)}>
          <span>运行出错：{row.message}</span>
        </div>
      );
  }
}

export function Swimlanes({ rows, lanes, awaiting }: { rows: Row[]; lanes: string[]; awaiting: boolean }) {
  const style = { "--lanes": lanes.length } as CSSProperties;
  return (
    <div className="lanes" style={style}>
      {lanes.map((lane, i) => (
        <div key={lane} className="lane-head" style={{ gridColumn: i + 1, gridRow: 1 }}>
          {label(lane)}
        </div>
      ))}
      {lanes.map((lane, i) => (
        <div
          key={`${lane}-rail`}
          className="lane-rail"
          aria-hidden
          style={{ gridColumn: i + 1, gridRow: `1 / ${rows.length + 3}` }}
        />
      ))}
      {rows.map((row, i) => (
        <RowView key={row.key} row={row} lanes={lanes} index={i} awaiting={awaiting} />
      ))}
    </div>
  );
}
