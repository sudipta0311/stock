import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { load } from "js-yaml";
import { describe, expect, it } from "vitest";
import { bannedVisibleTerms } from "./plain-language";

type HelpEntry = { term: string; meaning: string };
type HelpArticle = {
  id: string;
  title: string;
  summary: string;
  steps: string[];
  example: string;
  related: string[];
  entries?: HelpEntry[];
};
type HelpCopy = { groups: Array<{ title: string; articles: HelpArticle[] }> };

function helpCopy(): HelpCopy {
  return load(
    readFileSync(resolve(process.cwd(), "../../copy/help.en.yaml"), "utf8"),
  ) as HelpCopy;
}

function words(value: string): number {
  return value.trim().split(/\s+/).filter(Boolean).length;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function generated(sourcePath: string, outputPath: string): void {
  const source = load(
    readFileSync(fileURLToPath(new URL(sourcePath, import.meta.url)), "utf8"),
  );
  const expected = `${JSON.stringify(source, null, 2)}\n`;
  const actual = readFileSync(
    fileURLToPath(new URL(outputPath, import.meta.url)),
    "utf8",
  );
  expect(actual).toBe(expected);
}

describe("generated copy", () => {
  it("keeps the PWA English copy in sync with the YAML source", () => {
    generated("../../../copy/en.yaml", "../src/copy/en.json");
  });

  it("keeps banned terms in sync with the YAML source", () => {
    generated(
      "../../../copy/banned_terms.yaml",
      "../src/copy/banned-terms.json",
    );
  });

  it("keeps help articles in sync with the YAML source", () => {
    generated("../../../copy/help.en.yaml", "../src/copy/help.en.json");
  });

  it("keeps help copy plain and short", () => {
    const copy = helpCopy();
    const banned = load(
      readFileSync(
        resolve(process.cwd(), "../../copy/banned_terms.yaml"),
        "utf8",
      ),
    ) as { terms: Array<{ term: string }> };
    const articles = copy.groups.flatMap((group) => group.articles);
    const articleIds = new Set(articles.map((article) => article.id));

    for (const article of articles) {
      expect(words(article.title), article.title).toBeLessThanOrEqual(8);
      expect(article.summary.split(/[.!?]+/).filter(Boolean)).toHaveLength(1);
      expect(article.steps.length).toBeGreaterThanOrEqual(3);
      expect(article.steps.length).toBeLessThanOrEqual(7);
      for (const related of article.related) {
        expect(
          articleIds.has(related),
          `${article.id} links to ${related}`,
        ).toBe(true);
      }

      const visibleCopy = [
        article.title,
        article.summary,
        ...article.steps,
        article.example,
        ...(article.entries ?? []).map((entry) => entry.meaning),
      ].join(" ");
      for (const { term } of banned.terms) {
        expect(
          new RegExp(`(?<!\\w)${escapeRegExp(term)}(?!\\w)`, "i").test(
            visibleCopy,
          ),
          `${article.id} uses ${term}`,
        ).toBe(false);
      }

      const sentences = [
        article.summary,
        ...article.steps,
        article.example,
        ...(article.entries ?? []).map((entry) => entry.meaning),
      ];
      for (const sentence of sentences) {
        for (const part of sentence.split(/(?<=[.!?])\s+/).filter(Boolean)) {
          expect(words(part), `${article.id}: ${part}`).toBeLessThanOrEqual(20);
        }
      }
    }
  });

  it("includes every in-app term in the glossary", () => {
    const glossary = helpCopy()
      .groups.flatMap((group) => group.articles)
      .find((article) => article.id === "glossary");
    const terms = new Set((glossary?.entries ?? []).map((entry) => entry.term));
    for (const term of [
      "Net worth",
      "Money mix",
      "Safe money",
      "Monthly payments",
      "Monthly payment",
      "ISIN",
      "W-8BEN",
      "CAS",
    ]) {
      expect(terms.has(term), term).toBe(true);
    }
  });

  it("skips security data and term explanations in the screen scan", () => {
    const ideas = document.createElement("div");
    ideas.dataset.copyScreen = "ideas";
    ideas.innerHTML =
      "<span data-security>TCS ETF UCITS</span><span data-term-copy>ISIN</span><p>Exchange traded fund</p>";
    expect(bannedVisibleTerms(ideas)).toEqual([]);
  });

  it("allows the statement label only on the import screen", () => {
    const home = document.createElement("div");
    home.dataset.copyScreen = "home";
    home.textContent = "CAS statement";
    expect(bannedVisibleTerms(home)).toEqual(["CAS"]);

    const importer = document.createElement("div");
    importer.dataset.copyScreen = "import";
    importer.textContent = "CAS statement";
    expect(bannedVisibleTerms(importer)).toEqual([]);
  });
});
