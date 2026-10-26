import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The browser only ever talks to this dev server; /api is forwarded to FastAPI.
// Same origin means the session cookie just works and the API needs no CORS setup.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "VITE_");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/api": env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
      },
    },
  };
});
