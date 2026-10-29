import { Button, Card } from "../ui";
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
  return (
    <Card className="grid gap-3" aria-label="Helpful notes">
      <h2 className="text-sm font-semibold">Helpful notes</h2>
      <ul className="grid gap-3">
        {items.map((item) => (
          <li
            key={item.key}
            className={`grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0 ${item.severity === "attention" ? "border-l-2 border-l-warning pl-3" : ""}`}
          >
            <div>
              <h3 className="text-sm font-semibold">{item.title}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{item.body}</p>
            </div>
            <div className="flex items-center justify-between gap-3">
              <details>
                <summary className="min-h-10 cursor-pointer py-2 text-sm font-medium text-brand">
                  {englishCopy.highlights.why}
                </summary>
                <p className="pb-2 text-sm leading-6 text-muted-foreground">
                  {learnMore[item.learn_more_key] ?? learnMore.plan_details}
                </p>
              </details>
              {onDismiss ? (
                <Button
                  type="button"
                  variant="ghost"
                  className="min-h-10"
                  aria-label={`${englishCopy.highlights.dismiss}: ${item.title}`}
                  onClick={() => onDismiss(item.key)}
                >
                  {englishCopy.highlights.dismiss}
                </Button>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </Card>
  );
}
