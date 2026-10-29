import englishCopy from "../copy/en.json";
import { formatHouseholdAmount, formatSecondaryAmount } from "./currency";
import { useHousehold } from "./HouseholdContext";
import { formatDkk, dkkToInr } from "./fx";

export function MoneyValue({
  amountInr,
  compact = true,
}: {
  amountInr: string;
  compact?: boolean;
}) {
  const { fxRate, primaryCurrency } = useHousehold();
  const primary = formatHouseholdAmount(
    amountInr,
    primaryCurrency,
    fxRate,
    compact,
  );
  const secondary = formatSecondaryAmount(amountInr, primaryCurrency, fxRate);
  return (
    <>
      <span className="tabular-nums">{primary}</span>
      {secondary ? (
        <span className="block text-xs font-normal tabular-nums text-muted-foreground">
          {secondary}
        </span>
      ) : null}
    </>
  );
}

export function DkkMoneyValue({ amountDkk }: { amountDkk: string }) {
  const { fxRate, primaryCurrency } = useHousehold();
  const amountInr = fxRate ? dkkToInr(amountDkk, fxRate) : null;
  const primary =
    primaryCurrency === "INR" && amountInr !== null
      ? formatHouseholdAmount(amountInr, "INR", undefined)
      : formatDkk(amountDkk);
  const secondary =
    amountInr === null
      ? null
      : primaryCurrency === "INR"
        ? formatDkk(amountDkk)
        : formatHouseholdAmount(amountInr, "INR", undefined);
  return (
    <>
      <span className="tabular-nums">{primary}</span>
      {secondary ? (
        <span className="block text-xs font-normal tabular-nums text-muted-foreground">
          {secondary}
        </span>
      ) : null}
    </>
  );
}

export function ExchangeRateNote() {
  const { fxRate } = useHousehold();
  if (!fxRate) return null;
  const date = new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${fxRate.date}T00:00:00Z`));
  return (
    <p className="text-xs text-muted-foreground">
      {englishCopy.household_ui.exchange_rate
        .replace("{rate}", fxRate.rate)
        .replace("{date}", date)}
    </p>
  );
}
