import { describe, expect, it } from "vitest";
import { readSavedTheme, resolveTheme, saveTheme, THEME_KEY } from "./theme";

function memoryStorage() {
  const data = new Map<string, string>();
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
  };
}

const blocked = {
  getItem: () => {
    throw new Error("SecurityError");
  },
  setItem: () => {
    throw new Error("QuotaExceededError");
  },
};

describe("theme", () => {
  it("follows the system until a choice is saved", () => {
    expect(resolveTheme(null, true)).toBe("dark");
    expect(resolveTheme(null, false)).toBe("light");
    expect(resolveTheme("light", true)).toBe("light");
    expect(resolveTheme("dark", false)).toBe("dark");
  });

  it("remembers the choice under its key", () => {
    const storage = memoryStorage();
    expect(saveTheme(storage, "dark")).toBe(true);
    expect(storage.getItem(THEME_KEY)).toBe("dark");
    expect(readSavedTheme(storage)).toBe("dark");
  });

  it("ignores anything that is not a theme", () => {
    expect(readSavedTheme({ getItem: () => "sepia" })).toBeNull();
  });

  it("works without storage: private window or blocked site data", () => {
    expect(readSavedTheme(blocked)).toBeNull();
    expect(saveTheme(blocked, "dark")).toBe(false);
    expect(readSavedTheme(undefined)).toBeNull();
    expect(saveTheme(undefined, "light")).toBe(false);
  });
});
