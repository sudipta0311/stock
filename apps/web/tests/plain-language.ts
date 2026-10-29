import bannedTerms from "../src/copy/banned-terms.json";

type BannedTerm = { term: string };

export function visibleText(root: HTMLElement): string {
  const clone = root.cloneNode(true) as HTMLElement;
  clone
    .querySelectorAll("[data-security], [data-term-copy]")
    .forEach((element) => element.remove());
  clone.querySelectorAll("details:not([open])").forEach((details) => {
    const summary = details.querySelector(":scope > summary")?.cloneNode(true);
    details.replaceChildren(...(summary ? [summary] : []));
  });
  return clone.textContent ?? "";
}

export function bannedVisibleTerms(root: HTMLElement): string[] {
  const route = root.dataset.copyScreen;
  const text = visibleText(root);
  const scanText =
    route === "import" ? text.replace(/\bCAS statement\b/gi, "") : text;
  return (bannedTerms.terms as BannedTerm[])
    .filter(({ term }) =>
      new RegExp(`(?<!\\w)${escapeRegExp(term)}(?!\\w)`, "i").test(scanText),
    )
    .map(({ term }) => term);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
