import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { load } from "js-yaml";
import { describe, expect, it } from "vitest";
import { bannedVisibleTerms } from "./plain-language";

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
