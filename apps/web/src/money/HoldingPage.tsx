import { Link, useParams } from "react-router";
import { formatInr } from "../lib/money";
import { asArray, asRecord, textValue } from "../features/data";
import * as englishCopy from "../copy/sections/index";
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
import { Card } from "../ui";
import { MoneyValue } from "../household/currency-view";
import { exactQuantity, quantityLabel } from "./quantity";

function reviewFor(plan: WeeklyPlan, securityId: string): unknown {
  const priority: Record<string, number> = {
    sell: 0,
    review: 1,
    sell_some: 2,
    add_more: 3,
    keep: 4,
    not_checked: 5,
  };
  return asArray(plan.holding_reviews)
    .map(asRecord)
    .filter((item) => textValue(item.security_id) === securityId)
    .sort(
      (left, right) =>
        (priority[textValue(left.verdict)] ?? 6) -
        (priority[textValue(right.verdict)] ?? 6),
    )[0];
}

function checkedDate(value: unknown): string {
  const isoDate = textValue(value);
  if (!isoDate) return "";
  const parsed = new Date(`${isoDate}T00:00:00Z`);
  return Number.isNaN(parsed.getTime())
    ? isoDate
    : new Intl.DateTimeFormat("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
        timeZone: "UTC",
      }).format(parsed);
}

function plainPercent(value: string): string {
  return value.replace(/^-/, "").replace(/(?:\.0+|(\.\d*?)0+)$/, "$1");
}

function reasonText(raw: unknown): string {
  const reason = asRecord(raw);
  const code = textValue(reason.code, "default");
  const copy = englishCopy.holding_review.reasons as Record<string, string>;
  const params = asRecord(reason.params);
  const copyKey =
    code === "revenue_down_quarters" && textValue(params.count) === "1"
      ? "revenue_down_quarter"
      : code;
  const template = copy[copyKey] ?? copy.default ?? "";
  return template.replace(/\{([a-z_]+)\}/g, (_match, key: string) => {
    const value = textValue(params[key]);
    return key === "drawdown_pct" && value ? plainPercent(value) : value;
  });
}

function taxText(raw: unknown): string {
  const tax = asRecord(raw);
  const copy = englishCopy.holding_review.tax as Record<string, string>;
  const noteCode = textValue(tax.note_code);
  if (noteCode === "ask_same_tax" || noteCode === "tax_not_estimated") {
    return copy[noteCode] ?? "";
  }
  if (noteCode === "tax_wait") {
    const saved = textValue(tax.tax_saved);
    return saved
      ? (copy.tax_wait ?? "")
          .replace("{date}", checkedDate(tax.wait_until))
          .replace("{amount}", formatInr(saved, "compact"))
      : (copy.tax_wait_plain ?? "").replace(
          "{date}",
          checkedDate(tax.wait_until),
        );
  }
  const estimate = textValue(tax.estimated_tax);
  return estimate
    ? (copy.estimated_tax ?? "").replace(
        "{amount}",
        formatInr(estimate, "compact"),
      )
    : "";
}

export function OurView({ value }: { value: unknown }) {
  const review = asRecord(value);
  const copy = englishCopy.holding_review;
  const verdict = textValue(review.verdict, "not_checked");
  const verdicts = copy.verdicts as Record<string, string>;
  const checkLabels = copy.checks as Record<string, string>;
  const statuses = copy.statuses as Record<string, string>;
  const reviewChecks = asArray(review.checks);
  const knownChecks = reviewChecks.filter((raw) =>
    ["ok", "watch", "triggered"].includes(
      textValue(asRecord(raw).status, "unknown"),
    ),
  );
  const waitingLabels = reviewChecks
    .filter(
      (raw) =>
        !["ok", "watch", "triggered"].includes(
          textValue(asRecord(raw).status, "unknown"),
        ),
    )
    .map((raw) => {
      const key = textValue(asRecord(raw).key);
      return checkLabels[key] ?? key;
    });
  const ranChecks = knownChecks.length;
  const allKnownChecksAreOk =
    knownChecks.length > 0 &&
    knownChecks.every(
      (raw) => textValue(asRecord(raw).status, "unknown") === "ok",
    );
  const reasons = asArray(review.reasons)
    .filter((reason) => textValue(asRecord(reason).code) !== "tax_wait")
    .slice(0, 3);
  const taxLine = taxText(review.tax);
  const checkedOn = checkedDate(review.checked_on);

  return (
    <Card className="grid gap-3">
      <h2 className="text-card-title font-semibold">{copy.title}</h2>
      {verdict === "not_checked" ? (
        <p className="text-sm text-muted-foreground">{copy.not_checked}</p>
      ) : (
        <>
          <p className="text-lg font-semibold">
            {verdicts[verdict] ?? verdicts.keep}
          </p>
          {reasons.length ? (
            <ul className="grid gap-2 text-sm text-muted-foreground">
              {reasons.slice(0, 1).map((reason, index) => (
                <li key={`${textValue(asRecord(reason).code)}-${index}`}>
                  {reasonText(reason)}
                </li>
              ))}
            </ul>
          ) : null}
          {!reasons.length && reviewChecks.length && allKnownChecksAreOk ? (
            <p className="text-sm text-muted-foreground">
              {(ranChecks === reviewChecks.length
                ? copy.summary_all
                : copy.summary_partial
              )
                .replace("{ran}", String(ranChecks))
                .replace("{total}", String(reviewChecks.length))}
            </p>
          ) : null}
        </>
      )}
      {taxLine ? <p className="text-sm">{taxLine}</p> : null}
      {checkedOn ? (
        <p className="text-sm text-muted-foreground">
          {copy.checked_on.replace("{date}", checkedOn)}
        </p>
      ) : null}
      <details className="border-t border-line pt-2">
        <summary className="min-h-12 cursor-pointer py-2 text-sm font-semibold">
          {copy.why_suggestion}
        </summary>
        {reasons.length > 1 ? (
          <ul className="grid gap-2 border-t border-line py-2 text-sm text-muted-foreground">
            {reasons.slice(1).map((reason, index) => (
              <li key={`${textValue(asRecord(reason).code)}-${index + 1}`}>
                {reasonText(reason)}
              </li>
            ))}
          </ul>
        ) : null}
        <h3 className="mt-2 text-sm font-semibold">{copy.how_we_check}</h3>
        <ul className="grid gap-2 border-t border-line pt-2">
          {knownChecks.map((raw, index) => {
            const check = asRecord(raw);
            const key = textValue(check.key);
            const status = textValue(check.status, "unknown");
            const symbol =
              status === "ok"
                ? "✓"
                : status === "watch" || status === "triggered"
                  ? "!"
                  : "—";
            return (
              <li
                key={`${key}-${index}`}
                className="flex min-h-10 items-start gap-2 text-sm"
              >
                <span aria-label={statuses[status] ?? statuses.unknown}>
                  {symbol}
                </span>
                <span className="flex-1">
                  {checkLabels[key] ?? key}{" "}
                  <span className="ml-2 text-muted-foreground">
                    {statuses[status] ?? statuses.unknown}
                  </span>
                </span>
              </li>
            );
          })}
        </ul>
        {waitingLabels.length ? (
          <p className="text-sm text-muted-foreground">
            {copy.waiting_for_data.replace(
              "{checks}",
              waitingLabels.join(", "),
            )}
          </p>
        ) : null}
      </details>
    </Card>
  );
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
    return <ScreenError title="This holding couldn't load" />;
  const networthData = networth.data as NetworthResponse | undefined;
  const planData = plan.data as WeeklyPlan | undefined;
  if (!networthData || !planData)
    return <ScreenError title="This holding couldn't load" />;
  const holding = networthData.holdings.find(
    (candidate) => candidate.security_id === id,
  );
  if (!holding?.security_id)
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
  const review = reviewFor(planData, holding.security_id);
  const copy = englishCopy.holding_review;
  return (
    <div className="grid gap-4">
      <Link
        to="/money"
        className="inline-flex min-h-11 w-fit items-center text-sm font-medium text-brand underline underline-offset-2"
      >
        {copy.back_to_money}
      </Link>
      <header>
        <h2 className="text-xl font-semibold">{holding.name || "Holding"}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {quantityLabel(holding.quantity, holding.security_type)}
        </p>
      </header>
      <Card className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-muted-foreground">Value</p>
          <p className="mt-1 font-semibold tabular-nums">
            {holding.value ? (
              <MoneyValue amountInr={holding.value} />
            ) : (
              "Not available"
            )}
          </p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Profit or loss</p>
          <p
            className={`mt-1 ${pnl && costKnown ? "font-semibold tabular-nums" : "text-sm font-normal text-muted-foreground"}`}
          >
            {pnl && costKnown ? (
              <MoneyValue amountInr={pnl} />
            ) : (
              copy.buying_price_not_recorded
            )}
          </p>
        </div>
        <div>
          <p className="text-sm text-muted-foreground">Price as of</p>
          <p className="mt-1 text-sm">
            {holding.price_as_of
              ? checkedDate(holding.price_as_of)
              : "Not available"}
          </p>
        </div>
      </Card>
      {review ? <OurView value={review} /> : null}
      <details className="rounded-xl border border-line px-3">
        <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
          {copy.holding_details}
        </summary>
        {!costKnown ? (
          <p className="border-t border-line py-3 text-sm text-muted-foreground">
            {copy.buying_price_help}
          </p>
        ) : null}
        <dl className="grid gap-2 border-t border-line py-3 text-sm">
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Code</dt>
            <dd className="text-right">{holding.symbol}</dd>
          </div>
          <div className="flex justify-between gap-3">
            <dt className="text-muted-foreground">Quantity</dt>
            <dd className="text-right">{exactQuantity(holding.quantity)}</dd>
          </div>
          {textValue(extra.isin) ? (
            <div className="flex justify-between gap-3">
              <dt className="text-muted-foreground">Other code</dt>
              <dd className="text-right">{textValue(extra.isin)}</dd>
            </div>
          ) : null}
        </dl>
      </details>
    </div>
  );
}

export { reviewFor };
