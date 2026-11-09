import { useEffect, useState } from "react";
import type { ApprovalItem } from "../api";

interface Props {
  items: ApprovalItem[];
  busy: boolean;
  onSubmit: (answers: Record<string, boolean>) => void;
}

export function ApprovalPanel({ items, busy, onSubmit }: Props) {
  const [answers, setAnswers] = useState<Record<string, boolean>>({});
  useEffect(() => setAnswers({}), [items]);

  const decided = items.every((i) => i.id in answers);
  const set = (id: string, value: boolean) => setAnswers((a) => ({ ...a, [id]: value }));

  return (
    <section className="approval" aria-label="等待确认的操作">
      <div className="hazard" aria-hidden />
      <h2>{items.length === 1 ? "这个操作需要你确认" : `这 ${items.length} 个操作需要你确认`}</h2>
      <p className="approval-hint">确认之前不会执行。拒绝后，助手会停止这部分操作，不会换个写法再试。</p>
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <div className="approval-tool">{item.tool}</div>
            <p className="approval-reason">{item.reason}</p>
            <pre className="code">{item.preview}</pre>
            <div className="choice" role="group" aria-label={`${item.tool} 的决定`}>
              <button
                type="button"
                className={answers[item.id] === false ? "chosen deny" : "deny"}
                aria-pressed={answers[item.id] === false}
                onClick={() => set(item.id, false)}
                disabled={busy}
              >
                拒绝
              </button>
              <button
                type="button"
                className={answers[item.id] === true ? "chosen allow" : "allow"}
                aria-pressed={answers[item.id] === true}
                onClick={() => set(item.id, true)}
                disabled={busy}
              >
                批准执行
              </button>
            </div>
          </li>
        ))}
      </ul>
      <button type="button" className="primary" disabled={!decided || busy} onClick={() => onSubmit(answers)}>
        {decided ? "提交决定并继续" : "先对每个操作做出选择"}
      </button>
    </section>
  );
}
