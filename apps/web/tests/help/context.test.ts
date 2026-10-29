import helpCopy from "../../src/copy/help.en.json";
import {
  helpArticleForAssetType,
  helpArticleForPath,
} from "../../src/help/context";
import { expect, it } from "vitest";

const ids = new Set(
  helpCopy.groups.flatMap((group) =>
    group.articles.map((article) => article.id),
  ),
);

it("points each signed-in screen to an existing help article", () => {
  const screenPaths = ["/", "/money", "/plan", "/ideas", "/inbox", "/settings"];
  for (const path of screenPaths) {
    expect(ids.has(helpArticleForPath(path)), path).toBe(true);
  }
});

it("points forms to help for their asset group", () => {
  for (const type of [
    "fd",
    "ppf",
    "epf",
    "po_scheme",
    "insurance",
    "gold",
    "stocks",
    "property",
    "us_investments",
  ]) {
    expect(ids.has(helpArticleForAssetType(type)), type).toBe(true);
  }
});
