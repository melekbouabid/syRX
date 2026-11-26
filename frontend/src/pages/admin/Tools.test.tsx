import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import en from "../../i18n/en";
import fr from "../../i18n/fr";
import { toolLinks } from "../../lib/tools";
import { ToolsSectionView } from "./Tools";

function hrefs(html: string): string[] {
  return [...html.matchAll(/<a [^>]*href="([^"]*)"/g)].map((match) => match[1]);
}

describe("toolLinks", () => {
  it("builds each address from the page host and the compose ports", () => {
    expect(toolLinks("localhost").map((tool) => tool.url)).toEqual([
      "http://localhost:5000",
      "http://localhost:6333/dashboard",
      "http://localhost:8081",
      "/docs",
      "http://localhost:8090",
    ]);
    expect(toolLinks("192.168.1.20")[0].url).toBe("http://192.168.1.20:5000");
    expect(toolLinks("::1")[2].url).toBe("http://[::1]:8081");
  });
});

describe("ToolsSectionView", () => {
  it("shows one card per tool to an admin, each opening in a new tab", () => {
    const html = renderToStaticMarkup(<ToolsSectionView role="admin" hostname="localhost" t={en} />);
    expect(html).toContain("Tools");
    for (const name of ["MLflow", "Qdrant", "Adminer / PostgreSQL", "Swagger", "Spark UI"]) {
      expect(html).toContain(name);
    }
    expect(hrefs(html)).toEqual(toolLinks("localhost").map((tool) => tool.url));
    const anchors = html.match(/<a [^>]*>/g) ?? [];
    expect(anchors).toHaveLength(5);
    for (const anchor of anchors) {
      expect(anchor).toContain('target="_blank"');
      expect(anchor).toContain('rel="noopener noreferrer"');
    }
    expect(html).toContain("Only with --profile spark");
  });

  it("gives the Adminer sign-in settings but never a password", () => {
    const html = renderToStaticMarkup(<ToolsSectionView role="admin" hostname="localhost" t={en} />);
    for (const value of ["PostgreSQL", "postgres", "tujya", "mlflow"]) {
      expect(html).toContain(`<code>${value}</code>`);
    }
    expect(html.toLowerCase()).not.toContain("password");
    expect(html.toLowerCase()).not.toContain("mot de passe");
  });

  it("is translated", () => {
    const html = renderToStaticMarkup(<ToolsSectionView role="admin" hostname="localhost" t={fr} />);
    expect(html).toContain("Outils");
    expect(html).toContain("Ouvrir");
    expect(html).toContain("Seulement avec --profile spark");
  });

  it("shows nothing to a user or a visitor", () => {
    expect(renderToStaticMarkup(<ToolsSectionView role="user" hostname="localhost" t={en} />)).toBe("");
    expect(renderToStaticMarkup(<ToolsSectionView role={undefined} hostname="localhost" t={en} />)).toBe("");
  });
});
