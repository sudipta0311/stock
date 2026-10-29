import { CircleAlert, CircleCheck, CircleHelp, Eye } from "lucide-react";
import { useParams } from "react-router";
import { formatInr } from "../lib/money";
import { exitLabel, labelForCode, triggerKeyLabel } from "../lib/labels";
import { asArray, asRecord, reasonCopy, textValue } from "../features/data";
import englishCopy from "../copy/en.json";
import {
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import { getNetworth } from "./api";
import type { NetworthResponse } from "./api";
import { getPlan } from "../planning/api";
import type { WeeklyPlan } from "../planning/api";
import { Badge, Card } from "../ui";
import { quantityLabel } from "./quantity";

function statusIcon(status: string) {
  if (status === "ok")
    return <CircleCheck aria-hidden="true" className="size-4 text-brand" />;
  if (status === "watch")
    return <Eye aria-hidden="true" className="size-4 text-muted-foreground" />;
  if (status === "triggered")
    return (
      <CircleAlert
        aria-hidden="true"
        className="size-4 text-warning-foreground"
      />
    );
  return (
    <CircleHelp aria-hidden="true" className="size-4 text-muted-foreground" />
  );
}

export function ExitCard({ value }: { value: unknown }) {
  const card = asRecord(value);
  const triggers = asRecord(card.triggers ?? card.trigger_statuses);
  const tax = asRecord(card.tax_estimate);
  const action = textValue(card.action, "");
  const taxValue = textValue(
    tax.tax_due ?? tax.tax_estimate ?? tax.estimated_tax ?? tax.total_tax,
  );
  const longTermDate = textValue(
    tax.long_term_date ??
      tax.long_term_since ??
      asRecord(card.params).long_term_date,
  );
  const taxNotes = new Set(asArray(tax.notes).map((note) => textValue(note)));
  return (
    <Card className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <h2 className="text-base font-semibold">Exit Card</h2>
        {textValue(card.urgency) ? (
          <Badge>{plainUrgency(textValue(card.urgency))}</Badge>
        ) : null}
      </div>
      <div className="grid gap-2">
        {Object.entries(triggers).map(([key, raw]) => {
          const trigger = asRecord(raw);
          const status = textValue(trigger.status, "unknown");
          const valueText = textValue(trigger.value);
          const threshold = textValue(trigger.threshold);
          return (
            <div
              key={key}
              className="flex min-h-12 items-start justify-between gap-3 border-t border-line pt-2 first:border-0 first:pt-0"
            >
              <div className="flex items-start gap-2">
                <span className="mt-0.5">{statusIcon(status)}</span>
                <div>
                  <p className="text-sm font-medium">{triggerKeyLabel(key)}</p>
                  <p className="text-xs text-muted-foreground">
                    {labelForCode(status)}
                  </p>
                </div>
              </div>
              <p className="text-right text-sm tabular-nums">
                {valueText || "—"}
                {threshold ? (
                  <span className="block text-xs text-muted-foreground">
                    Threshold {threshold}
                  </span>
                ) : null}
              </p>
            </div>
          );
        })}
      </div>
      <div className="border-t border-line pt-3">
        <p className="text-sm text-muted-foreground">Suggested action</p>
        <p className="mt-1 text-base font-semibold">
          {action ? exitLabel(action) : "Not enough data"}
        </p>
      </div>
      {asArray(card.reason_codes).length ? (
        <ul className="grid gap-1 border-t border-line pt-3 text-sm text-muted-foreground">
          {asArray(card.reason_codes).map((code, index) => (
            <li key={`${textValue(code)}-${index}`}>
              {reasonCopy(textValue(code))}
            </li>
          ))}
        </ul>
      ) : null}
      {taxValue ? (
        <p className="text-sm">
          Estimated tax if sold:{" "}
          <strong className="tabular-nums">
            {formatInr(taxValue, "compact")}
          </strong>
        </p>
      ) : (
        <p className="text-sm text-muted-foreground">
          Tax estimate is unavailable for this holding.
        </p>
      )}
      {taxNotes.has("acquisition_fx_missing") ? (
        <p className="text-sm text-muted-foreground">
          {englishCopy.imports.tax_fx_missing}
        </p>
      ) : null}
      {taxNotes.has("acquisition_fx_approximation") ? (
        <p className="text-sm text-muted-foreground">
          {englishCopy.imports.tax_fx_approximation}
        </p>
      ) : null}
      {longTermDate ? (
        <p className="text-sm text-muted-foreground">
          Long-term date: {longTermDate}
        </p>
      ) : null}
    </Card>
  );
}

function plainUrgency(value: string): string {
  return value.toLowerCase() === "critical"
    ? "Review soon"
    : `${value.toLowerCase()} priority`;
}

function exitCardFor(
  plan: WeeklyPlan,
  securityId: string,
  symbol: string,
): unknown {
  return asArray(plan.exit_cards).find((raw) => {
    const card = asRecord(raw);
    return (
      textValue(card.security_id) === securityId ||
      textValue(card.symbol) === symbol
    );
  });
}

export function HoldingPage() {
  const { id = "" } = useParams();
  const networth = useFeatureQuery(["c22", "networth"], "/v1/networth", () =>
    getNetworth(),
  );
  const plan = useFeatureQuery(["c22", "plan"], "/v1/plan", getPlan);
  if (
    !networth.supported ||
    !plan.supported ||
    isMissingFeature(networth.error) ||
    isMissingFeature(plan.error)
  )
    return <ComingSoon title="Holding details are coming soon" />;
  if (networth.isPending || plan.isPending)
    return <ScreenLoading label="Loading holding" />;
  if (
    (networth.error && !isMissingFeature(networth.error)) ||
    (plan.error && !isMissingFeature(plan.error))
  )
    return <ScreenError title="This holding couldn’t load" />;
  const networthData = networth.data as NetworthResponse | undefined;
  const planData = plan.data as WeeklyPlan | undefined;
  if (!networthData || !planData)
    return <ScreenError title="This holding couldn’t load" />;
  const holding = networthData.holdings.find(
    (candidate) => candidate.security_id === id,
  );
  if (!holding)
    return (
      <Card>
        <h2 className="font-semibold">Holding not found</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          This security may no longer be in your Money Map.
        </p>
      </Card>
    );
  const extra = asRecord(holding);
  const costKnown = textValue(extra.cost_basis ?? extra.cost_value);
  const pnl = textValue(extra.profit_loss ?? extra.pnl);
  const longTermDate = textValue(extra.long_term_date ?? extra.long_term_since);
  const card = exitCardFor(planData, holding.security_id, holding.symbol);
  return (
    <div className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">
          {holding.name || holding.symbol}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {holding.symbol} ·{" "}
          {quantityLabel(holding.quantity, holding.security_type)}
        </p>
      </header>
      <Card className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-muted-foreground">Value</p>
          <p className="mt-1 font-semibold tabular-nums">
            {holding.value
              ? formatInr(holding.value, "compact")
              : "Not available"}
          </p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Profit or loss</p>
          <p className="mt-1 font-semibold tabular-nums">
            {pnl && costKnown
              ? formatInr(pnl, "compact")
              : "Cost not available"}
          </p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Long-term date</p>
          <p className="mt-1 text-sm">
            {longTermDate ||
              textValue(
                asRecord(asRecord(card).tax_estimate).long_term_date,
                "Not available",
              )}
          </p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Price as of</p>
          <p className="mt-1 text-sm">
            {holding.price_as_of
              ? new Intl.DateTimeFormat("en-GB", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                  timeZone: "UTC",
                }).format(new Date(`${holding.price_as_of}T00:00:00Z`))
              : "Not available"}
          </p>
        </div>
      </Card>
      {/* Without a review there is nothing to show; a placeholder only adds noise. */}
      {card ? <ExitCard value={card} /> : null}
    </div>
  );
}

export { exitCardFor };
