import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router";
import { Search } from "lucide-react";
import { Button, Card, Term } from "../ui";
import helpCopy from "../copy/help.en.json";
import * as englishCopy from "../copy/sections/index";

type HelpEntry = { id: string; term: string; meaning: string };
type HelpArticle = {
  id: string;
  title: string;
  summary: string;
  steps: string[];
  example: string;
  related: string[];
  entries?: HelpEntry[];
};
type HelpGroup = { title: string; articles: HelpArticle[] };
type HelpDocument = { groups: HelpGroup[] };

const documentCopy = helpCopy as HelpDocument;

function searchableText(article: HelpArticle): string {
  return [
    article.title,
    article.summary,
    ...article.steps,
    article.example,
    ...(article.entries ?? []).flatMap((entry) => [entry.term, entry.meaning]),
  ]
    .join(" ")
    .toLocaleLowerCase();
}

function showTourAgain(): void {
  window.dispatchEvent(new Event("wealth:show-tour"));
}

function articleForAnchor(anchor: string): string | undefined {
  const articles = documentCopy.groups.flatMap((group) => group.articles);
  return articles.find(
    (article) =>
      article.id === anchor ||
      article.entries?.some((entry) => `term-${entry.id}` === anchor),
  )?.id;
}

export function HelpPage() {
  const location = useLocation();
  const origin = (location.state as { from?: unknown } | null)?.from;
  const hasOrigin =
    typeof origin === "string" &&
    origin.startsWith("/") &&
    !origin.startsWith("/help");
  const backTo = hasOrigin ? origin : "/";
  const [search, setSearch] = useState("");
  const filteredGroups = useMemo(() => {
    const query = search.trim().toLocaleLowerCase();
    return documentCopy.groups
      .map((group) => ({
        ...group,
        articles: group.articles.filter(
          (article) => !query || searchableText(article).includes(query),
        ),
      }))
      .filter((group) => group.articles.length > 0);
  }, [search]);
  const articleTitles = useMemo(
    () =>
      new Map(
        documentCopy.groups
          .flatMap((group) => group.articles)
          .map((article) => [article.id, article.title]),
      ),
    [],
  );

  useEffect(() => {
    const id = decodeURIComponent(location.hash.slice(1));
    if (!id) return;
    const articleId = articleForAnchor(id);
    if (!articleId) return;
    setSearch("");
    const frame = window.requestAnimationFrame(() => {
      const article = document.getElementById(articleId);
      const details = article?.querySelector("details");
      if (details) details.open = true;
      const heading = article?.querySelector<HTMLElement>(
        "[data-help-heading]",
      );
      heading?.focus({ preventScroll: true });
      heading?.scrollIntoView?.({ block: "start" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [location.hash]);

  return (
    <div data-copy-screen="help" className="grid gap-5">
      <header className="grid gap-2">
        <Link
          to={backTo}
          className="inline-flex min-h-11 w-fit items-center text-sm font-medium text-brand underline underline-offset-2"
        >
          {hasOrigin
            ? englishCopy.household_ui.help_back
            : englishCopy.household_ui.back_to_home}
        </Link>
        <h2 className="text-xl font-semibold">Help and guide</h2>
        <p className="text-sm text-muted-foreground">
          Find clear steps for using your family plan.
        </p>
      </header>

      <Card className="grid gap-3">
        <label htmlFor="help-search" className="text-sm font-medium">
          Search help articles
        </label>
        <div className="flex items-center gap-2 rounded-xl border border-line bg-background px-3 focus-within:ring-2 focus-within:ring-brand/30">
          <Search aria-hidden="true" className="size-4 text-muted-foreground" />
          <input
            id="help-search"
            type="search"
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
            placeholder="Try deposits or loans"
            className="min-h-12 min-w-0 flex-1 bg-transparent text-base outline-none"
          />
        </div>
        <Button type="button" variant="secondary" onClick={showTourAgain}>
          Show the tour again
        </Button>
      </Card>

      {filteredGroups.length ? (
        filteredGroups.map((group) => (
          <section key={group.title} className="grid gap-3">
            <h2 className="text-lg font-semibold">{group.title}</h2>
            {group.articles.map((article) => (
              <article
                key={article.id}
                id={article.id}
                className="scroll-mt-20"
              >
                <details className="rounded-2xl border border-line bg-surface px-3">
                  <summary className="flex min-h-14 cursor-pointer list-none items-center justify-between gap-3 py-3 [&::-webkit-details-marker]:hidden">
                    <span className="min-w-0">
                      <h3
                        data-help-heading
                        tabIndex={-1}
                        className="text-base font-semibold focus:outline-none"
                      >
                        {article.title}
                      </h3>
                      <span className="mt-1 block text-sm text-muted-foreground">
                        {article.summary}
                      </span>
                    </span>
                  </summary>
                  <div className="grid gap-3 border-t border-line py-3">
                    <ol className="grid list-decimal gap-2 pl-5 text-sm">
                      {article.steps.map((step, index) => (
                        <li key={`${article.id}-${index}`}>{step}</li>
                      ))}
                    </ol>
                    <p className="rounded-xl bg-muted p-3 text-sm">
                      {article.example}
                    </p>
                    {article.entries?.length ? (
                      <dl className="grid gap-3 border-t border-line pt-3">
                        {article.entries.map((entry) => (
                          <div
                            id={`term-${entry.id}`}
                            key={entry.id}
                            className="grid gap-1"
                          >
                            <dt>
                              <Term
                                explanation={entry.meaning}
                                prompt="What's this?"
                                learnMoreHref={`/help#term-${entry.id}`}
                              >
                                {entry.term}
                              </Term>
                            </dt>
                            <dd className="text-sm text-muted-foreground">
                              {entry.meaning}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    ) : null}
                    <div className="grid gap-2 border-t border-line pt-3">
                      <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        Related
                      </h4>
                      <ul className="flex flex-wrap gap-x-4 gap-y-2">
                        {article.related.map((relatedId) => (
                          <li key={relatedId}>
                            <Link
                              to={`/help#${relatedId}`}
                              state={hasOrigin ? { from: origin } : undefined}
                              className="inline-flex min-h-11 items-center text-sm font-medium text-brand underline underline-offset-2"
                            >
                              {articleTitles.get(relatedId) ?? relatedId}
                            </Link>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </details>
              </article>
            ))}
          </section>
        ))
      ) : (
        <Card className="grid justify-items-start gap-3">
          <p className="text-sm text-muted-foreground">
            No guide matches that search.
          </p>
          <Button
            type="button"
            variant="secondary"
            onClick={() => setSearch("")}
          >
            Clear search
          </Button>
        </Card>
      )}
    </div>
  );
}
