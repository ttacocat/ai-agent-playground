import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// 开发时把 /api 转发给 `uv run agent serve`（默认 8000 端口）
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: { environment: "node" },
} as never);
