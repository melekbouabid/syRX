import { describe, expect, it } from "vitest";
import { docTypeLabel, facetOptions, formatDocDate, sectionLabel, shortHash } from "./documents";

describe("documents helpers", () => {
  it("formats openFDA dates and keeps the others", () => {
    expect(formatDocDate("20260910")).toBe("2026-09-10");
    expect(formatDocDate("1999")).toBe("1999");
    expect(formatDocDate("")).toBe("n/a");
  });

  it("names types and sections", () => {
    expect(docTypeLabel("abstract")).toBe("PubMed abstract");
    expect(docTypeLabel("other")).toBe("other");
    expect(sectionLabel("drug_interactions")).toBe("Drug interactions");
  });

  it("shortens hashes and builds facet options", () => {
    expect(shortHash("a".repeat(64))).toBe("aaaaaaaaaaaa…");
    expect(shortHash("")).toBe("n/a");
    expect(facetOptions({ label: 82, abstract: 1570 })).toEqual([
      ["label", "Drug label (82)"],
      ["abstract", "PubMed abstract (1,570)"],
    ]);
    expect(facetOptions(undefined)).toEqual([]);
  });
});
