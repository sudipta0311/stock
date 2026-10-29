import { Button, Card } from "../ui";
import { CircleHelp, Info, TriangleAlert } from "lucide-react";
import englishCopy from "../copy/en.json";

export type HighlightCardModel = {
  key: string;
  screen: "home" | "plan" | "ideas" | "money";
  title: string;
  body: string;
  learn_more_key: string;
  severity: "info" | "attention";
};

export function Highlights({
  items,
  onDismiss,
}: {
  items?: HighlightCardModel[];
  onDismiss?(key: string): void;
}) {
  if (!items?.length) return null;
  const learnMore = englishCopy.highlights.learn_more as Record<string, string>;
  const visibleItems = items.slice(0, 3);
  return (
    <Card className="grid gap-3" aria-label="Helpful notes">
      <h2 className="text-sm font-semibold">Helpful notes</h2>
      <ul className="grid gap-3">
        {visibleItems.map((item) => {
          const attention = item.severity === "attention";
          const SeverityIcon = attention ? TriangleAlert : Info;
          return (
            <li
              key={item.key}
              className={`grid gap-2 rounded-xl border p-3 ${attention ? "border-warning/30 bg-warning-soft text-warning-foreground" : "border-line bg-muted/60"}`}
            >
              <div className="flex items-start gap-3">
                <span
                  className={`mt-0.5 grid size-8 shrink-0 place-items-center rounded-full ${attention ? "bg-warning/10 text-warning-foreground" : "bg-surface text-muted-foreground"}`}
                >
                  <SeverityIcon aria-hidden="true" className="size-4" />
                </span>
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold">{item.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {item.body}
                  </p>
                </div>
              </div>
              <div className="flex min-h-12 items-center justify-between gap-3 border-t border-line/70 pt-1">
                <details className="min-w-0">
                  <summary className="flex min-h-12 cursor-pointer list-none items-center gap-1.5 py-1 text-sm font-medium text-brand">
                    <CircleHelp aria-hidden="true" className="size-4" />
                    {englishCopy.highlights.why}
                  </summary>
                  <p className="pb-2 text-base text-muted-foreground">
                    {learnMore[item.learn_more_key] ?? learnMore.plan_details}
                  </p>
                </details>
                {onDismiss ? (
                  <Button
                    type="button"
                    variant="ghost"
                    className="min-h-12 px-2 py-2 text-sm font-medium text-muted-foreground"
                    aria-label={`${englishCopy.highlights.dismiss}: ${item.title}`}
                    onClick={() => onDismiss(item.key)}
                  >
                    {englishCopy.highlights.dismiss}
                  </Button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
