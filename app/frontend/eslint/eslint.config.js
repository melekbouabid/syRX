// ESLint for the React app (app/frontend). This folder is its own npm package because
// typescript-eslint needs the TypeScript JavaScript API, which TypeScript 7 (used by the app's
// build) no longer ships: the parser runs on TypeScript 6.0 here, type checking stays `tsc`.
// Run from app/frontend: `npm --prefix eslint ci` once, then `npm run lint`.
import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  // Paths are relative to app/frontend, where `npm run lint` runs ESLint with this config.
  { ignores: ["dist/", "node_modules/", "eslint/", "public/"] },
  {
    files: ["src/**/*.{ts,tsx}", "vite.config.ts"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: { globals: globals.browser },
    plugins: { "react-hooks": reactHooks },
    rules: {
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
    },
  },
  {
    // Node scripts (screenshots, report builds); the callbacks they give Playwright's
    // page.evaluate run in the browser.
    files: ["scripts/**/*.mjs"],
    extends: [js.configs.recommended],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
);
