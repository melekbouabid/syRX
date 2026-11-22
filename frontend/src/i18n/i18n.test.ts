import { describe, expect, it } from "vitest";
import en from "./en";
import fr from "./fr";
import { browserLang, getLang, isLang, messages, readSavedLang, setLang } from "./index";

/** Every key path of a messages object, with "fn/<arity>" for the texts that take values. */
function shape(value: unknown, path = ""): string[] {
  if (typeof value === "function") return [`${path}:fn/${value.length}`];
  if (Array.isArray(value)) return [`${path}:array/${value.length}`, ...value.flatMap((v, i) => shape(v, `${path}[${i}]`))];
  if (value && typeof value === "object") {
    return Object.entries(value).flatMap(([key, v]) => shape(v, path ? `${path}.${key}` : key));
  }
  return [`${path}:${typeof value}`];
}

describe("translations", () => {
  it("French has exactly the texts of English, same lists and same parameters", () => {
    expect(shape(fr)).toEqual(shape(en));
  });

  it("no French text is left empty or copied from a missing translation", () => {
    const empty = shape(fr).filter((entry) => entry.endsWith(":string"));
    const values = (obj: unknown): string[] =>
      typeof obj === "string" ? [obj] : obj && typeof obj === "object" ? Object.values(obj).flatMap(values) : [];
    expect(empty.length).toBeGreaterThan(300);
    expect(values(fr).filter((text) => text.trim() === "")).toEqual([]);
  });

  it("keeps the safety wording of the project rules in both languages", () => {
    expect(fr.common.disclaimer).toBe("Information uniquement : consultez un pharmacien ou un médecin.");
    expect(en.common.disclaimer).toBe("Information only: consult a pharmacist or doctor.");
    expect(fr.intro.rules[1].text).toContain("aucune interaction connue dans cette base de données");
    expect(en.intro.rules[1].text).toContain("no known interaction in this database");
  });

  it("puts the start page title in both languages", () => {
    expect(`${en.intro.title1} ${en.intro.title2}`).toBe("Know your pills. Check before you mix.");
    expect(fr.intro.title1).toBe("Connaissez vos comprimés.");
  });
});

describe("language choice", () => {
  it("accepts only English and French", () => {
    expect(isLang("en")).toBe(true);
    expect(isLang("fr")).toBe(true);
    expect(isLang("ru")).toBe(false);
  });

  it("follows the browser's first language, English otherwise", () => {
    expect(browserLang(["fr-FR", "en"])).toBe("fr");
    expect(browserLang(["en-GB", "fr"])).toBe("en");
    expect(browserLang(["ru-RU"])).toBe("en");
    expect(browserLang(undefined)).toBe("en");
  });

  it("reads a saved choice, and survives blocked storage", () => {
    expect(readSavedLang({ getItem: () => "fr" })).toBe("fr");
    expect(readSavedLang({ getItem: () => "de" })).toBeNull();
    expect(
      readSavedLang({
        getItem: () => {
          throw new Error("SecurityError");
        },
      }),
    ).toBeNull();
    expect(readSavedLang(undefined)).toBeNull();
  });

  it("switches every text at once", () => {
    expect(getLang()).toBe("en"); // Node has no browser language: English
    setLang("fr");
    expect(messages().scan.title).toBe(fr.scan.title);
    expect(messages().admin.users.title).toBe("Utilisateurs");
    setLang("en");
    expect(messages().admin.users.title).toBe("Users");
  });
});
