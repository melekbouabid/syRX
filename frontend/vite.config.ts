import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The browser only ever talks to this dev server; /api, the Swagger UI (/docs) and its
// schema (/openapi.json) are forwarded to FastAPI. Same origin means the session cookie
// just works and the API needs no CORS setup. nginx.conf does the same in production.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_TARGET ?? "http://127.0.0.1:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": target,
        "/docs": target,
        "/openapi.json": target,
      },
    },
  };
});
