import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { loadEnv } from "vite";
import { defineConfig } from "vitest/config";

// The browser only ever talks to this dev server; /api, the Swagger UI (/docs), its schema
// (/openapi.json) and the model routes of the course guide (/health, /model/metadata) are
// forwarded to FastAPI. Same origin means the session cookie just works and the API needs no
// CORS setup. nginx.conf does the same in production.
// Tailwind CSS v4 through its Vite plugin (src/index.css, docs/decisions.md D106); Vitest
// runs the unit tests of src/**/*.test.ts(x) in Node, without a browser.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  const target = env.VITE_API_TARGET ?? "http://127.0.0.1:8000";
  return {
    plugins: [react(), tailwindcss()],
    // Strictly local (docs/infra.md): the dev and preview servers listen on 127.0.0.1 only.
    server: {
      host: "127.0.0.1",
      port: 5173,
      strictPort: true,
      proxy: {
        "/api": target,
        "/docs": target,
        "/openapi.json": target,
        "/health": target,
        "/model/metadata": target,
      },
    },
    preview: {
      host: "127.0.0.1",
      port: 4173,
      strictPort: true,
    },
    test: {
      include: ["src/**/*.test.{ts,tsx}"],
      environment: "node",
    },
  };
});
