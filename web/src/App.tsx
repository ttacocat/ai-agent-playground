import { useEffect, useState } from "react";
import { getJSON, type Health } from "./api";
import { ChatView } from "./chat/ChatView";
import { EvalView } from "./evals/EvalView";
import { TraceView } from "./traces/TraceView";

const TABS = [
  { id: "chat", label: "对话" },
  { id: "evals", label: "评测对比" },
  { id: "traces", label: "运行记录" },
] as const;
type Tab = (typeof TABS)[number]["id"];

const tabFromHash = (): Tab => {
  const id = window.location.hash.slice(1);
  return (TABS.find((t) => t.id === id)?.id ?? "chat") as Tab;
};

export function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    getJSON<Health>("/api/health")
      .then(setHealth)
      .catch(() => setError("连不上后端服务。先运行 uv run agent serve（或加 --demo 进入演示模式）。"));
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <h1>工程助手控制台</h1>
        <nav aria-label="页面">
          {TABS.map((t) => (
            <a key={t.id} href={`#${t.id}`} aria-current={tab === t.id ? "page" : undefined}>
              {t.label}
            </a>
          ))}
        </nav>
        {health && <p className={`mode-pill${health.demo ? " demo" : ""}`}>{health.demo ? "演示模式" : health.model}</p>}
      </header>
      {error && <p className="banner-error">{error}</p>}
      {tab === "chat" && health && <ChatView health={health} />}
      {tab === "evals" && <EvalView />}
      {tab === "traces" && <TraceView />}
    </div>
  );
}
