import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Link, MemoryRouter } from "react-router";
import { Button, buttonClassName } from "../src/ui/primitives";

interface CssRule {
  prelude: string;
  body?: string;
  statement: boolean;
}

const sourceRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../src");

function normalizeCss(value: string): string {
  return value
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function skipComment(source: string, index: number): number {
  if (source[index] !== "/" || source[index + 1] !== "*") return index;
  const end = source.indexOf("*/", index + 2);
  if (end < 0) throw new Error("Unclosed CSS comment.");
  return end + 2;
}

function findClosingBrace(source: string, opening: number): number {
  let depth = 1;
  let quote: string | null = null;
  let escaped = false;

  for (let index = opening + 1; index < source.length; index += 1) {
    const next = skipComment(source, index);
    if (next !== index) {
      index = next - 1;
      continue;
    }
    const character = source[index]!;
    if (quote) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === quote) quote = null;
      continue;
    }
    if (character === '"' || character === "'") {
      quote = character;
    } else if (character === "{") {
      depth += 1;
    } else if (character === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
  }

  throw new Error("Unclosed CSS rule.");
}

function topLevelRules(source: string): CssRule[] {
  const rules: CssRule[] = [];
  let index = 0;

  while (index < source.length) {
    while (index < source.length) {
      const next = skipComment(source, index);
      if (next !== index) index = next;
      else if (/\s/.test(source[index]!)) index += 1;
      else break;
    }
    if (index >= source.length) break;

    const start = index;
    let parentheses = 0;
    let quote: string | null = null;
    let escaped = false;
    let foundRuleEnd = false;

    for (; index < source.length; index += 1) {
      const next = skipComment(source, index);
      if (next !== index) {
        index = next - 1;
        continue;
      }
      const character = source[index]!;
      if (quote) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === quote) quote = null;
        continue;
      }
      if (character === '"' || character === "'") {
        quote = character;
      } else if (character === "(") {
        parentheses += 1;
      } else if (character === ")") {
        parentheses -= 1;
      } else if (parentheses === 0 && character === ";") {
        rules.push({
          prelude: normalizeCss(source.slice(start, index)),
          statement: true,
        });
        index += 1;
        foundRuleEnd = true;
        break;
      } else if (parentheses === 0 && character === "{") {
        const closing = findClosingBrace(source, index);
        rules.push({
          prelude: normalizeCss(source.slice(start, index)),
          body: source.slice(index + 1, closing),
          statement: false,
        });
        index = closing + 1;
        foundRuleEnd = true;
        break;
      } else if (parentheses === 0 && character === "}") {
        throw new Error("Unexpected closing CSS brace.");
      }
    }

    if (!foundRuleEnd) throw new Error("CSS rule has no ending delimiter.");
  }

  return rules;
}

function isAllowedUnlayeredRule(rule: CssRule): boolean {
  if (rule.statement) return rule.prelude.startsWith("@import ");
  if (
    rule.prelude === ":root" ||
    rule.prelude === ':root[data-theme="dark"]' ||
    rule.prelude.startsWith("@theme") ||
    rule.prelude === "body[data-scroll-locked]" ||
    rule.prelude === ":focus-visible" ||
    rule.prelude === "summary:focus-visible"
  ) {
    return true;
  }

  if (rule.prelude === "@media (prefers-color-scheme: dark)") {
    const children = topLevelRules(rule.body ?? "");
    return (
      children.length === 1 &&
      children[0]?.prelude === ":root:not([data-theme])" &&
      !children[0].statement
    );
  }

  if (rule.prelude === "@media (prefers-reduced-motion: reduce)") {
    const children = topLevelRules(rule.body ?? "");
    return (
      children.length === 1 &&
      children[0]?.prelude === "*, *::before, *::after" &&
      !children[0].statement
    );
  }

  return false;
}

describe("CSS cascade rules", () => {
  it("keeps app and card rules in layers", () => {
    const css = readFileSync(resolve(sourceRoot, "styles.css"), "utf8");
    const unexpected = topLevelRules(css)
      .filter((rule) => !rule.prelude.startsWith("@layer"))
      .filter((rule) => !isAllowedUnlayeredRule(rule))
      .map((rule) => rule.prelude);

    expect(
      unexpected,
      "Unlayered rules outrank Tailwind utilities. Put app-wide styles in @layer base and card styles in @layer components.",
    ).toEqual([]);
  });

  it("uses Button classes for primary links", () => {
    render(
      createElement(
        MemoryRouter,
        null,
        createElement(Button, null, "Primary action"),
        createElement(
          Link,
          { to: "/money", className: buttonClassName() },
          "Add holdings",
        ),
      ),
    );

    expect(
      screen.getByRole("button", { name: "Primary action" }).className,
    ).toBe(buttonClassName());
    expect(screen.getByRole("link", { name: "Add holdings" }).className).toBe(
      buttonClassName(),
    );

    const homePage = readFileSync(
      resolve(sourceRoot, "home/HomePage.tsx"),
      "utf8",
    );
    expect(homePage).toMatch(
      /to="\/money\?add=1"[\s\S]{0,140}className=\{buttonClassName\(\)\}/,
    );
  });
});
