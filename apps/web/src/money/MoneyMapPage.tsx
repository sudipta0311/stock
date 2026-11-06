import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { FormEvent, ReactNode } from "react";
import {
  ArrowRight,
  Building2,
  CalendarDays,
  ChevronRight,
  WalletCards,
} from "lucide-react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router";
import { ApiError } from "../api/client";
import { formatInr, sumInr } from "../lib/money";
import {
  Badge,
  Button,
  Card,
  EmptyState,
  Progress,
  Sheet,
  Stat,
  Term,
} from "../ui";
import {
  getNetworth,
  listAssets,
  listLoans,
  quickAdd,
  updateChecklist,
} from "./api";
import type { AssetResponse, LoanListResponse, NetworthResponse } from "./api";
import {
  addPathForCategory,
  categoryLabel,
  CHECKLIST_PENDING_KEY,
  CHECKLIST_STORAGE_KEY,
  checklistTiles,
} from "./config";
import type { ChecklistWorkflow } from "./config";
import { listImports } from "../imports/api";
import { ImportHistorySheet } from "../imports/ImportHistorySheet";
import { markValueMoment } from "../pwa/install";
import * as englishCopy from "../copy/sections/index";
import { apiPathAvailable } from "../api/feature-support";
import { useHousehold } from "../household/HouseholdContext";
import { asArray, asRecord, textValue } from "../features/data";
import { quantityLabel } from "./quantity";
import { assetTitles } from "./titles";
import { getPlan, removeRecordedBuy, updateRecordedBuy } from "../planning/api";
import type { RecordedBuyPatch, WeeklyPlan } from "../planning/api";
import {
  formatHouseholdAmount,
  formatSecondaryAmount,
} from "../household/currency";
import {
  formatDkk,
  formatRateToTwoDecimals,
  type DkkInrRate,
} from "../household/fx";
import { FundSipFineTune } from "../settings/FundSipFineTune";

const FundCheckCard = lazy(() =>
  import("./FundCheckCard").then((module) => ({
    default: module.FundCheckCard,
  })),
);

const ASSET_KIND_ORDER = [
  englishCopy.money_ui.deposits,
  englishCopy.money_ui.savings,
  englishCopy.money_ui.other,
];

function readChecklist(): string[] {
  try {
    const parsed: unknown = JSON.parse(
      localStorage.getItem(CHECKLIST_STORAGE_KEY) ?? "[]",
    );
    return Array.isArray(parsed)
      ? parsed.filter((item): item is string => typeof item === "string")
      : [];
  } catch {
    return [];
  }
}

function toMinorUnits(value: string): bigint {
  const negative = value.startsWith("-");
  const [whole = "0", fraction = ""] = value.replace(/^-/, "").split(".");
  const amount = BigInt(whole) * 100n + BigInt((fraction + "00").slice(0, 2));
  return negative ? -amount : amount;
}

function fromMinorUnits(minor: bigint): string {
  const sign = minor < 0n ? "-" : "";
  const magnitude = minor < 0n ? -minor : minor;
  return `${sign}${magnitude / 100n}.${(magnitude % 100n).toString().padStart(2, "0")}`;
}

/** Several kinds share one plain name ("Other holdings"), so they show as one row. */
function sleeveRows(
  sleeves: readonly { sleeve: string; amount: string; pct: string }[],
): { label: string; amount: string; pct: number }[] {
  const rows = new Map<string, { minor: bigint; pct: number }>();
  for (const sleeve of sleeves) {
    const label = displaySleeveName(sleeve.sleeve);
    const row = rows.get(label) ?? { minor: 0n, pct: 0 };
    rows.set(label, {
      minor: row.minor + toMinorUnits(sleeve.amount),
      pct: row.pct + Number(sleeve.pct),
    });
  }
  // An empty kind ("Gold ₹0 · 0%") says nothing, so it isn't listed.
  return [...rows]
    .filter(([, row]) => row.minor !== 0n)
    .map(([label, row]) => ({
      label,
      amount: fromMinorUnits(row.minor),
      pct: row.pct,
    }));
}

const HOLDINGS_SHOWN = 8;

function verdictForHolding(
  reviews: unknown,
  securityId: string | null,
): string {
  if (!securityId) return "";
  const priority: Record<string, number> = {
    sell: 0,
    review: 1,
    sell_some: 2,
    add_more: 3,
    keep: 4,
    not_checked: 5,
  };
  const matching = asArray(reviews)
    .map(asRecord)
    .filter((review) => textValue(review.security_id) === securityId)
    .sort(
      (left, right) =>
        (priority[textValue(left.verdict)] ?? 6) -
        (priority[textValue(right.verdict)] ?? 6),
    );
  return textValue(matching[0]?.verdict);
}

/** "2026-10-28" → "28 Oct 2026". */
function plainDate(isoDate: string): string {
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

function isPositiveDecimalText(value: string): boolean {
  const trimmed = value.trim();
  return /^\d+(?:\.\d+)?$/.test(trimmed) && /[1-9]/.test(trimmed);
}

/** Largest first; holdings without a value go last, then by name. */
function byValue<T extends { value?: string | null; name: string }>(
  holdings: readonly T[],
): T[] {
  return [...holdings].sort((a, b) => {
    const left = a.value ? toMinorUnits(a.value) : -1n;
    const right = b.value ? toMinorUnits(b.value) : -1n;
    if (left !== right) return right > left ? 1 : -1;
    return a.name.localeCompare(b.name);
  });
}

function moneyKindForAsset(asset: AssetResponse): string {
  switch (asset.asset_type) {
    case "savings":
      return englishCopy.money_ui.savings;
    case "fd":
    case "rd":
    case "ppf":
    case "epf":
    case "nps":
    case "po_scheme":
    case "bond":
      return englishCopy.money_ui.deposits;
    default:
      return englishCopy.money_ui.other;
  }
}

function displaySleeveName(value: string): string {
  const key = value.toLowerCase().replaceAll(" ", "_");
  if (key.includes("equity_global")) return englishCopy.household_ui.abroad;
  if (key.includes("equity_india"))
    return englishCopy.household_ui.indian_shares;
  if (key === "gold") return "Gold";
  if (key === "debt") return englishCopy.household_ui.safe_money;
  if (key === "cash") return englishCopy.household_ui.cash;
  if (["property", "real_estate", "protection", "liabilities"].includes(key)) {
    return englishCopy.household_ui.other_holdings;
  }
  return value;
}

/** Whole percentages read better than "46.88%"; tiny shares show as "under 1%". */
function plainPercent(value: string | number | null | undefined): string {
  const number = Number(value ?? 0);
  if (number > 0 && number < 1) return "under 1%";
  return `${Math.round(number)}%`;
}

function assetLabel(asset: AssetResponse): string {
  const label = asset.details.label;
  if (typeof label === "string" && label.trim()) return label;
  const bank =
    asset.details.bank ?? asset.details.insurer ?? asset.details.institution;
  const typeLabel = assetTitles[asset.asset_type] ?? asset.asset_type;
  return typeof bank === "string" && bank
    ? `${typeLabel} at ${bank}`
    : typeLabel;
}

function maturityStatus(asset: AssetResponse): string {
  if (asset.valuation_status === "price_needed") return "Price needed";
  if (asset.valuation_status === "rate_needed") return "Rate needed";
  if (asset.valuation_status === "fx_needed") return "Rate needed";
  const maturity = asset.details.maturity;
  if (typeof maturity === "string" && /^\d{4}-\d{2}$/.test(maturity)) {
    const [yearText = "", monthText = ""] = maturity.split("-");
    const year = Number(yearText);
    const month = Number(monthText);
    const date = new Date(year, month - 1, 1);
    return `Matures ${new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric" }).format(date)}`;
  }
  return (
    asset.valuation_note ||
    (asset.value_cached ? "Estimate" : "Value not added yet")
  );
}

function formatNativeLoanAmount(amount: string, currency: string): string {
  if (currency === "INR") return formatInr(amount, "compact", 0);
  if (currency === "DKK") return formatDkk(amount);
  return `${currency} ${new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(Number(amount))}`;
}

function formatSavingsAmount(
  asset: AssetResponse,
  primaryCurrency: "INR" | "DKK",
  fxQuote: DkkInrRate | undefined,
  showSecondary: boolean,
): ReactNode {
  const currency = String(asset.details.currency ?? "INR");
  const balance = asset.details.balance;
  if (
    currency === "INR" ||
    (typeof balance !== "string" && typeof balance !== "number")
  ) {
    return asset.value_cached ? (
      <AmountLabel
        amount={asset.value_cached}
        primaryCurrency={primaryCurrency}
        fxQuote={fxQuote}
        showSecondary={showSecondary}
      />
    ) : (
      "Rate needed"
    );
  }
  return (
    <span className="tabular-nums">
      {formatNativeLoanAmount(String(balance), currency)}
      <span className="block text-xs font-normal text-muted-foreground">
        {asset.value_cached ? (
          <AmountLabel
            amount={asset.value_cached}
            primaryCurrency={primaryCurrency}
            fxQuote={fxQuote}
            showSecondary={false}
          />
        ) : (
          "Rate needed"
        )}
      </span>
    </span>
  );
}

function LoansGroup({
  summary,
  shownAmount,
}: {
  summary: LoanListResponse | null;
  shownAmount(amount: string): string;
}) {
  if (!summary?.loans.length) return null;
  return (
    <Card className="grid gap-1">
      <div className="mb-1 grid gap-1">
        <div className="flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-muted-foreground">Loans</h3>
          <span className="shrink-0 text-right text-sm font-semibold tabular-nums">
            {shownAmount(summary.remaining_inr)}
          </span>
        </div>
        <p className="text-xs text-muted-foreground">
          Monthly payments {shownAmount(summary.monthly_emis_inr)}
        </p>
      </div>
      {summary.loans.map((loan) => {
        const payoff = loan.payoff_month
          ? new Intl.DateTimeFormat("en-IN", {
              month: "short",
              year: "numeric",
            }).format(
              new Date(loan.payoff_month.year, loan.payoff_month.month - 1, 1),
            )
          : null;
        return (
          <Link
            key={loan.id}
            to={`/money/loan/${loan.id}`}
            className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2 hover:text-brand"
          >
            <span className="flex w-0 min-w-0 flex-1 items-center gap-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <WalletCards aria-hidden="true" className="size-4" />
              </span>
              <span className="min-w-0 flex-1">
                <strong className="block truncate text-sm">{loan.name}</strong>
                {loan.lender ? (
                  <span className="block truncate text-xs text-muted-foreground">
                    {loan.lender}
                  </span>
                ) : null}
                <span className="block truncate text-xs text-muted-foreground">
                  {loan.never_pays_off
                    ? "Monthly payment may not cover interest"
                    : payoff
                      ? `Ends ${payoff}`
                      : "End date not available"}
                </span>
                {loan.percent_repaid !== null ? (
                  <Progress
                    value={Number(loan.percent_repaid)}
                    label="Amount repaid"
                    quiet
                  />
                ) : null}
              </span>
            </span>
            <span className="flex shrink-0 items-center gap-2 text-right text-sm font-semibold tabular-nums">
              {loan.outstanding_inr
                ? shownAmount(loan.outstanding_inr)
                : loan.fx_missing
                  ? "Rate needed"
                  : formatNativeLoanAmount(loan.outstanding, loan.currency)}
              <ChevronRight
                aria-hidden="true"
                className="size-4 text-muted-foreground"
              />
            </span>
          </Link>
        );
      })}
    </Card>
  );
}

function sumKnownValues(
  values: readonly (string | null | undefined)[],
): string | null {
  const knownValues = values.filter(
    (value): value is string => typeof value === "string",
  );
  return knownValues.length ? sumInr(knownValues) : null;
}

function GroupAmount({
  amount,
  primaryCurrency,
  fxQuote,
  showSecondary,
}: {
  amount: string | null;
  primaryCurrency: "INR" | "DKK";
  fxQuote: DkkInrRate | undefined;
  showSecondary: boolean;
}) {
  return (
    <span className="shrink-0 text-right text-sm font-semibold tabular-nums">
      {amount ? (
        <AmountLabel
          amount={amount}
          primaryCurrency={primaryCurrency}
          fxQuote={fxQuote}
          showSecondary={showSecondary}
        />
      ) : (
        englishCopy.money_ui.group_total_unavailable
      )}
    </span>
  );
}

function HoldingsGroup({
  title,
  holdings,
  showAll,
  onToggleShowAll,
  primaryCurrency,
  fxQuote,
  showSecondary,
  onSelectRecordedBuy,
  holdingReviews,
}: {
  title: string;
  holdings: NetworthResponse["holdings"];
  showAll: boolean;
  onToggleShowAll(): void;
  primaryCurrency: "INR" | "DKK";
  fxQuote: DkkInrRate | undefined;
  showSecondary: boolean;
  onSelectRecordedBuy(holding: NetworthResponse["holdings"][number]): void;
  holdingReviews?: unknown;
}) {
  if (!holdings.length) return null;
  return (
    <section className="grid gap-2" aria-label={title}>
      <Card className="grid gap-2">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold">{title}</h2>
          <GroupAmount
            amount={sumKnownValues(holdings.map((holding) => holding.value))}
            primaryCurrency={primaryCurrency}
            fxQuote={fxQuote}
            showSecondary={showSecondary}
          />
        </div>
        {byValue(holdings)
          .slice(0, showAll ? undefined : HOLDINGS_SHOWN)
          .map((holding, index) => {
            const verdict =
              holding.security_type === "equity"
                ? verdictForHolding(holdingReviews, holding.security_id)
                : "";
            const verdictLabels = englishCopy.holding_review
              .money_labels as Record<string, string>;
            const verdictTone =
              verdict === "add_more"
                ? "success"
                : verdict === "review" || verdict === "sell_some"
                  ? "warning"
                  : "neutral";
            const row = (
              <>
                <span className="flex w-0 min-w-0 flex-1 items-center gap-3">
                  <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                    <Building2 aria-hidden="true" className="size-4" />
                  </span>
                  <span className="min-w-0">
                    <strong className="block line-clamp-2 text-sm [overflow-wrap:anywhere]">
                      <span data-security>{holding.name}</span>
                    </strong>
                    <span className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      {holding.added_by_you
                        ? englishCopy.money_ui.added_by_you
                        : quantityLabel(
                            holding.quantity,
                            holding.security_type,
                          )}
                      {verdict &&
                      verdict !== "keep" &&
                      verdict !== "not_checked" ? (
                        <Badge tone={verdictTone}>
                          {verdictLabels[verdict] ?? verdictLabels.review}
                        </Badge>
                      ) : null}
                    </span>
                  </span>
                </span>
                <span className="flex shrink-0 items-center gap-2 text-right text-sm font-semibold tabular-nums">
                  {holding.value ? (
                    <AmountLabel
                      amount={holding.value}
                      primaryCurrency={primaryCurrency}
                      fxQuote={fxQuote}
                      showSecondary={showSecondary}
                    />
                  ) : (
                    "Not available"
                  )}
                  {holding.recorded_buy_id ? (
                    <span className="text-xs font-medium text-brand">
                      {englishCopy.money_ui.details_buy}
                    </span>
                  ) : (
                    <ChevronRight
                      aria-hidden="true"
                      className="size-4 text-muted-foreground"
                    />
                  )}
                </span>
              </>
            );
            return holding.recorded_buy_id ? (
              <button
                key={holding.recorded_buy_id}
                type="button"
                onClick={() => onSelectRecordedBuy(holding)}
                className="flex min-h-16 w-full items-center justify-between gap-3 border-t border-line py-2 text-left first:border-0 first:pt-0 hover:text-brand"
              >
                {row}
              </button>
            ) : (
              <Link
                key={holding.security_id ?? `${holding.name}:${index}`}
                to={`/money/holding/${holding.security_id}`}
                className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2 hover:text-brand"
              >
                {row}
              </Link>
            );
          })}
        {holdings.length > HOLDINGS_SHOWN ? (
          <Button
            variant="secondary"
            aria-expanded={showAll}
            onClick={onToggleShowAll}
          >
            {showAll ? "Show fewer" : `Show all ${holdings.length}`}
          </Button>
        ) : null}
      </Card>
    </section>
  );
}

function RecordedBuySheet({
  holding,
  saving,
  removing,
  error,
  onClose,
  onSave,
  onRemove,
}: {
  holding: NetworthResponse["holdings"][number];
  saving: boolean;
  removing: boolean;
  error: boolean;
  onClose(): void;
  onSave(payload: RecordedBuyPatch): void;
  onRemove(): void;
}) {
  const [amount, setAmount] = useState(holding.amount ?? "");
  const [units, setUnits] = useState(holding.quantity);
  const [unitsChanged, setUnitsChanged] = useState(false);
  const [boughtOn, setBoughtOn] = useState(holding.bought_on ?? "");
  const [editing, setEditing] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [message, setMessage] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!isPositiveDecimalText(amount)) {
      setMessage(englishCopy.money_ui.buy_amount_error);
      return;
    }
    if (unitsChanged && units.trim() && !isPositiveDecimalText(units)) {
      setMessage(englishCopy.money_ui.buy_units_error);
      return;
    }
    onSave({
      amount: amount.trim(),
      bought_on: boughtOn || undefined,
      ...(unitsChanged ? { units: units.trim() || null } : {}),
    });
  }

  return (
    <Sheet
      open
      onOpenChange={(open) => !open && onClose()}
      title={
        confirmRemove
          ? englishCopy.money_ui.remove_buy_title
          : englishCopy.money_ui.added_by_you
      }
    >
      {confirmRemove ? (
        <div className="grid gap-4">
          <p className="text-sm">{englishCopy.money_ui.remove_buy_confirm}</p>
          <div className="flex flex-wrap gap-2">
            <Button
              variant="secondary"
              onClick={() => setConfirmRemove(false)}
              disabled={removing}
            >
              {englishCopy.money_ui.cancel}
            </Button>
            <Button data-primary="true" loading={removing} onClick={onRemove}>
              {englishCopy.money_ui.remove_buy}
            </Button>
          </div>
        </div>
      ) : editing ? (
        <form className="grid gap-4" onSubmit={submit}>
          <p className="text-sm text-muted-foreground">
            {englishCopy.money_ui.buy_details}
          </p>
          <label className="grid gap-1 text-sm">
            <span>
              {englishCopy.money_ui.buy_amount} ({holding.amount_currency})
            </span>
            <input
              type="text"
              inputMode="decimal"
              value={amount}
              onChange={(event) => setAmount(event.currentTarget.value)}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
              required
            />
          </label>
          <label className="grid gap-1 text-sm">
            <span>{englishCopy.money_ui.buy_units_optional}</span>
            <input
              type="text"
              inputMode="decimal"
              value={units}
              onChange={(event) => {
                setUnitsChanged(true);
                setUnits(event.currentTarget.value);
              }}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
            />
          </label>
          <label className="grid gap-1 text-sm">
            <span>{englishCopy.money_ui.buy_date}</span>
            <input
              type="date"
              value={boughtOn}
              onChange={(event) => setBoughtOn(event.currentTarget.value)}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base"
              required
            />
          </label>
          {message || error ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {message || englishCopy.money_ui.buy_save_error}
            </p>
          ) : null}
          <div className="flex flex-wrap gap-2">
            <Button type="submit" data-primary="true" loading={saving}>
              {englishCopy.money_ui.save_buy}
            </Button>
            <Button
              variant="ghost"
              onClick={() => setEditing(false)}
              disabled={saving}
            >
              {englishCopy.money_ui.cancel}
            </Button>
          </div>
        </form>
      ) : (
        <div className="grid gap-4">
          <p className="text-sm text-muted-foreground">
            {englishCopy.money_ui.buy_details}
          </p>
          <dl className="grid gap-2 text-sm">
            <div className="flex justify-between gap-4 border-t border-line pt-2">
              <dt>{englishCopy.money_ui.buy_amount}</dt>
              <dd className="text-right tabular-nums">
                {holding.amount} {holding.amount_currency}
              </dd>
            </div>
            <div className="flex justify-between gap-4 border-t border-line pt-2">
              <dt>{englishCopy.money_ui.buy_units_optional}</dt>
              <dd className="text-right tabular-nums">
                {holding.quantity || englishCopy.money_ui.buy_no_units}
              </dd>
            </div>
            <div className="flex justify-between gap-4 border-t border-line pt-2">
              <dt>{englishCopy.money_ui.buy_date}</dt>
              <dd className="text-right">{holding.bought_on}</dd>
            </div>
          </dl>
          <div className="flex flex-wrap gap-2">
            <Button data-primary="true" onClick={() => setEditing(true)}>
              {englishCopy.money_ui.edit_buy}
            </Button>
            <Button variant="ghost" onClick={() => setConfirmRemove(true)}>
              {englishCopy.money_ui.remove_buy}
            </Button>
          </div>
        </div>
      )}
    </Sheet>
  );
}

function QuickAddBar({
  onDraft,
}: {
  onDraft(result: Awaited<ReturnType<typeof quickAdd>>): void;
}) {
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!text.trim()) return;
    setLoading(true);
    setError("");
    try {
      onDraft(await quickAdd(text.trim()));
    } catch (reason) {
      if (reason instanceof ApiError && reason.code === "not_understood") {
        setError(
          "Try a short line such as “FD 5L SBI 7.1% till Mar 2027” or “PPF 6.5 lakh”.",
        );
      } else {
        setError(
          "We couldn't read that just now. You can add the asset from the tiles below.",
        );
      }
    } finally {
      setLoading(false);
    }
  }
  return (
    <Card>
      <form className="grid gap-3" onSubmit={(event) => void submit(event)}>
        <label htmlFor="quick-add" className="sr-only">
          {englishCopy.money_ui.quick_add_summary}
        </label>
        <div className="flex flex-col gap-2 sm:flex-row">
          <input
            id="quick-add"
            value={text}
            onChange={(event) => setText(event.currentTarget.value)}
            placeholder="FD 5L SBI 7.1% till Mar 2027"
            className="min-h-12 min-w-0 flex-1 rounded-xl border border-line bg-background px-3 text-base outline-none focus-visible:border-brand"
            aria-describedby={
              error ? "quick-add-help quick-add-error" : "quick-add-help"
            }
          />
          <Button
            type="submit"
            variant="secondary"
            loading={loading}
            disabled={!text.trim()}
          >
            Add details
          </Button>
        </div>
        <p
          id="quick-add-help"
          className="min-h-5 text-sm text-muted-foreground"
        >
          A quick note fills what it can. You can review every field.
        </p>
        {error ? (
          <p
            id="quick-add-error"
            className="text-sm text-danger-foreground"
            role="status"
          >
            {error}
          </p>
        ) : null}
      </form>
    </Card>
  );
}

function AmountLabel({
  amount,
  primaryCurrency,
  fxQuote,
  showSecondary,
}: {
  amount: string;
  primaryCurrency: "INR" | "DKK";
  fxQuote: DkkInrRate | undefined;
  showSecondary: boolean;
}) {
  const main = formatHouseholdAmount(amount, primaryCurrency, fxQuote);
  const secondary = showSecondary
    ? formatSecondaryAmount(amount, primaryCurrency, fxQuote)
    : null;
  return (
    <span className="tabular-nums">
      {main}
      {secondary ? (
        <span className="block text-xs font-normal text-muted-foreground">
          {secondary}
        </span>
      ) : null}
    </span>
  );
}

function HouseholdInsights({
  networth,
  primaryCurrency,
  fxQuote,
  showSecondary,
}: {
  networth: NetworthResponse;
  primaryCurrency: "INR" | "DKK";
  fxQuote: DkkInrRate | undefined;
  showSecondary: boolean;
}) {
  const liquidity = networth.liquidity_ladder;
  const ladderTotal = sumInr([
    liquidity.instant,
    liquidity.within_1y,
    liquidity.locked,
  ]);
  const toPercent = (value: string) => {
    const total = toMinorUnits(ladderTotal);
    if (total <= 0n) return 0;
    const pctValue = (toMinorUnits(value) * 100n) / total;
    return Number(pctValue > 100n ? 100n : pctValue);
  };
  // Count from midnight so something due today still shows today.
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const oneYear = new Date(today);
  oneYear.setFullYear(oneYear.getFullYear() + 1);
  const upcoming = networth.maturity_calendar.filter((event) => {
    const date = new Date(`${event.date}T00:00:00`);
    return date >= today && date <= oneYear;
  });
  const debtPercent = networth.debt_insight.debt_pct_including_locked;
  return (
    <div className="grid gap-4">
      <Card className="grid gap-2">
        <h2 className="text-base font-semibold">What your Money Map says</h2>
        <p className="text-base text-muted-foreground">
          Including EPF, PPF and fixed deposits,{" "}
          <strong className="text-foreground">
            {plainPercent(debtPercent)}
          </strong>{" "}
          {englishCopy.household_ui.safe_money_share}
        </p>
      </Card>
      <Card className="grid gap-4">
        <div>
          <h2 className="text-base font-semibold">
            {englishCopy.household_ui.accessible_money}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {englishCopy.household_ui.accessible_money_help}
          </p>
        </div>
        {[
          ["Available today", liquidity.instant],
          ["Within 1 year", liquidity.within_1y],
          ["Locked beyond 1 year", liquidity.locked],
        ].map(([label, amount]) => (
          <div key={String(label)} className="grid gap-1.5">
            <div className="flex justify-between gap-3 text-sm">
              <span>{label}</span>
              <span className="tabular-nums">
                {formatHouseholdAmount(
                  String(amount),
                  primaryCurrency,
                  fxQuote,
                )}
              </span>
            </div>
            <Progress
              value={toPercent(String(amount))}
              label={`${label}: ${formatHouseholdAmount(String(amount), primaryCurrency, fxQuote)}`}
            />
          </div>
        ))}
      </Card>
      <Card className="grid gap-3">
        <div className="flex items-center gap-2">
          <CalendarDays aria-hidden="true" className="size-5 text-brand" />
          <h2 className="text-base font-semibold">Coming up in 12 months</h2>
        </div>
        {upcoming.length ? (
          upcoming.slice(0, 8).map((event) => (
            <Link
              key={`${event.asset_id}-${event.date}-${event.kind}`}
              to={`/money/asset/${event.asset_id}`}
              className="flex min-h-12 items-center justify-between gap-3 border-t border-line pt-2 text-sm"
            >
              <span>
                <strong>{event.label}</strong>
                <span className="block text-muted-foreground">
                  {new Intl.DateTimeFormat("en-IN", {
                    day: "numeric",
                    month: "short",
                    year: "numeric",
                  }).format(new Date(`${event.date}T00:00:00`))}
                </span>
              </span>
              {event.amount ? (
                <AmountLabel
                  amount={event.amount}
                  primaryCurrency={primaryCurrency}
                  fxQuote={fxQuote}
                  showSecondary={showSecondary}
                />
              ) : null}
            </Link>
          ))
        ) : (
          <p className="text-sm text-muted-foreground">
            No maturities or renewals are due in the next 12 months.
          </p>
        )}
      </Card>
      {networth.unvalued_assets.length ? (
        <Card className="grid gap-3 border-warning/50">
          <div className="flex items-center gap-2">
            <Building2 aria-hidden="true" className="size-5 text-warning" />
            <h2 className="text-base font-semibold">Waiting for a value</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            These items stay visible and are not counted as ₹0.
          </p>
          {networth.unvalued_assets.map((asset) => (
            <Link
              key={asset.id}
              to={
                asset.asset_type === "loan"
                  ? `/money/loan/${asset.id}`
                  : `/money/asset/${asset.id}`
              }
              className="flex min-h-12 items-center justify-between border-t border-line pt-2 text-sm"
            >
              <span>
                {asset.label}
                <span className="block text-muted-foreground">
                  {asset.reason === "price_needed"
                    ? "We'll value this once today's price is in."
                    : asset.reason === "fx_needed"
                      ? "We'll include this once today's exchange rate is in."
                      : "Add the rate from your certificate."}
                </span>
              </span>
              <ArrowRight aria-hidden="true" className="size-4" />
            </Link>
          ))}
        </Card>
      ) : null}
    </div>
  );
}

export function MoneyMapPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const {
    activeMemberId,
    selectedMemberId,
    selectedHouseholdId,
    meMemberId,
    members,
    primaryCurrency,
    isViewer,
    fxRate,
  } = useHousehold();
  const [showAllHoldings, setShowAllHoldings] = useState(false);
  const [checklist, setChecklist] = useState<string[]>(readChecklist);
  const [checklistError, setChecklistError] = useState("");
  const [showFundCheck, setShowFundCheck] = useState(false);
  const [showMonthlyAmounts, setShowMonthlyAmounts] = useState(false);
  const [addSheetOpen, setAddSheetOpen] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const saveQueued = Boolean(
    (location.state as { saveQueued?: boolean } | null)?.saveQueued,
  );
  const chooserRef = useRef<HTMLDetailsElement>(null);
  const chooserSummaryRef = useRef<HTMLElement>(null);
  const [editingRecordedBuy, setEditingRecordedBuy] = useState<
    NetworthResponse["holdings"][number] | null
  >(null);
  const assetsQuery = useQuery({
    queryKey: ["c27b", "assets", selectedHouseholdId, activeMemberId],
    queryFn: () => listAssets(activeMemberId),
    staleTime: 0,
  });
  const networthQuery = useQuery({
    queryKey: ["c27b", "networth", selectedHouseholdId, activeMemberId],
    queryFn: () => getNetworth(activeMemberId),
    staleTime: 0,
  });
  const planQuery = useQuery({
    queryKey: ["c27b", "plan", selectedHouseholdId, activeMemberId],
    queryFn: getPlan,
    enabled: apiPathAvailable("/v1/plan"),
    staleTime: 0,
  });
  const loansQuery = useQuery({
    queryKey: ["c30", "loans", selectedHouseholdId],
    queryFn: listLoans,
    enabled: apiPathAvailable("/v1/loans"),
    staleTime: 0,
  });
  const importsQuery = useQuery({
    queryKey: ["c36", "imports", selectedHouseholdId],
    queryFn: listImports,
    staleTime: 0,
  });
  const saveRecordedBuy = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: RecordedBuyPatch }) =>
      updateRecordedBuy(id, payload),
    onSuccess: async () => {
      setEditingRecordedBuy(null);
      await queryClient.invalidateQueries({ queryKey: ["c27b"] });
    },
  });
  const deleteRecordedBuy = useMutation({
    mutationFn: removeRecordedBuy,
    onSuccess: async () => {
      setEditingRecordedBuy(null);
      await queryClient.invalidateQueries({ queryKey: ["c27b"] });
    },
  });
  const assets = useMemo(() => assetsQuery.data ?? [], [assetsQuery.data]);
  const importJobs = importsQuery.data ?? [];
  const networth = networthQuery.data ?? null;
  const listedItemCount =
    assetsQuery.data && networth
      ? assets.length + networth.holdings.length
      : null;
  const loanSummary = loansQuery.data ?? null;
  const plan = (planQuery.data as WeeklyPlan | undefined) ?? null;
  const loading = networthQuery.isPending;
  const pageError =
    networthQuery.error && !networth
      ? "We couldn't load your Money Map. Please try again."
      : "";
  const offlineSummaryOnly = Boolean(
    networth && assetsQuery.error && !assetsQuery.data,
  );
  const noListedItems = Boolean(
    !offlineSummaryOnly &&
    listedItemCount === 0 &&
    assetsQuery.data &&
    networth &&
    apiPathAvailable("/v1/loans") &&
    loansQuery.data &&
    (loanSummary?.loans.length ?? 0) === 0,
  );

  useEffect(() => {
    if (assets.length >= 3) markValueMoment();
  }, [assets.length]);

  useEffect(() => {
    function syncChecklist() {
      try {
        const pending = localStorage.getItem(CHECKLIST_PENDING_KEY);
        if (!pending) return;
        const categories: unknown = JSON.parse(pending);
        if (!Array.isArray(categories)) return;
        void updateChecklist(categories as string[])
          .then(() => {
            localStorage.removeItem(CHECKLIST_PENDING_KEY);
            window.dispatchEvent(new Event("wealth:pending-changes"));
          })
          .catch(() => undefined);
      } catch {
        // Local storage is optional; the current screen remains usable without it.
      }
    }
    window.addEventListener("online", syncChecklist);
    return () => window.removeEventListener("online", syncChecklist);
  }, []);

  const orderedSelected = useMemo(
    () =>
      checklistTiles
        .filter((tile) => checklist.includes(tile.category))
        .map((tile) => tile.category),
    [checklist],
  );
  const totals = networth
    ? {
        netWorth: networth.net_worth,
        assets: networth.assets_total,
        liabilities: networth.liabilities_total,
      }
    : null;
  const hasDanishMember = members.some(
    (member) => member.residency.country === "DK",
  );
  const fxQuote = fxRate;
  const familyMembers = plan?.members ?? [];

  function shownAmount(amount: string): string {
    return formatHouseholdAmount(amount, primaryCurrency, fxQuote);
  }

  function secondaryAmount(amount: string): string | null {
    return hasDanishMember
      ? formatSecondaryAmount(amount, primaryCurrency, fxQuote)
      : null;
  }
  const fxLine =
    hasDanishMember && fxQuote
      ? englishCopy.household.fx_line
          .replace("{rate}", formatRateToTwoDecimals(fxQuote.rate))
          .replace("{date}", plainDate(fxQuote.date))
      : null;

  const tickedCategories = networth?.completeness.ticked ?? [];
  const nextCategory = checklistTiles.find(
    (tile) =>
      tickedCategories.includes(tile.category) &&
      !networth?.completeness.done.includes(tile.category),
  )?.category;

  const openManualChooser = useCallback(() => {
    setAddSheetOpen(false);
    requestAnimationFrame(() => {
      const chooser = chooserRef.current;
      if (!chooser) return;
      chooser.open = true;
      chooser.scrollIntoView?.({ block: "start" });
      requestAnimationFrame(() => chooserSummaryRef.current?.focus());
    });
  }, []);

  useEffect(() => {
    const add = searchParams.get("add");
    if (add !== "1" && add !== "manual") return;
    if (add === "1") setAddSheetOpen(true);
    else openManualChooser();
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("add");
    setSearchParams(nextParams, { replace: true });
  }, [openManualChooser, searchParams, setSearchParams]);
  async function toggleCategory(category: string) {
    if (isViewer) return;
    const updated = checklist.includes(category)
      ? checklist.filter((item) => item !== category)
      : [...checklist, category];
    setChecklist(updated);
    setChecklistError("");
    try {
      localStorage.setItem(CHECKLIST_STORAGE_KEY, JSON.stringify(updated));
      if (!navigator.onLine) {
        localStorage.setItem(CHECKLIST_PENDING_KEY, JSON.stringify(updated));
        window.dispatchEvent(new Event("wealth:pending-changes"));
        setChecklistError(
          "Saved on this device. We'll sync when you're back online.",
        );
      } else {
        await updateChecklist(updated);
        localStorage.removeItem(CHECKLIST_PENDING_KEY);
        window.dispatchEvent(new Event("wealth:pending-changes"));
      }
    } catch {
      setChecklistError(
        "Your selection is saved on this device. We'll sync it when you're back online.",
      );
      try {
        localStorage.setItem(CHECKLIST_PENDING_KEY, JSON.stringify(updated));
      } catch {
        /* optional storage */
      }
      window.dispatchEvent(new Event("wealth:pending-changes"));
    }
  }

  function startChecklist() {
    if (!orderedSelected.length) return;
    const workflow: ChecklistWorkflow = {
      categories: orderedSelected,
      index: 0,
    };
    navigate(addPathForCategory(orderedSelected[0] ?? ""), {
      state: { workflow, memberId: activeMemberId ?? meMemberId },
    });
  }

  function startQuickDraft(result: Awaited<ReturnType<typeof quickAdd>>) {
    navigate(`/money/add/${result.asset_type}`, {
      state: {
        draft: result.draft,
        missing: result.missing,
        memberId: activeMemberId ?? meMemberId,
      },
    });
  }

  const visibleGroups = useMemo(() => {
    const groups = new Map<string, AssetResponse[]>();
    for (const asset of assets) {
      if (asset.asset_type === "loan") continue;
      const kind = moneyKindForAsset(asset);
      groups.set(kind, [...(groups.get(kind) ?? []), asset]);
    }
    return ASSET_KIND_ORDER.filter((kind) => groups.has(kind)).map(
      (kind) => [kind, groups.get(kind) ?? []] as const,
    );
  }, [assets]);

  const sortedHoldings = useMemo(
    () => byValue(networth?.holdings ?? []),
    [networth?.holdings],
  );
  const fundHoldings = sortedHoldings.filter((holding) =>
    ["mf", "etf"].includes(holding.security_type),
  );
  const shareHoldings = sortedHoldings.filter(
    (holding) => holding.security_type === "equity",
  );
  const bondsAndGoldBonds = sortedHoldings.filter((holding) =>
    ["bond", "sgb"].includes(holding.security_type),
  );
  const otherHoldings = sortedHoldings.filter(
    (holding) =>
      !["equity", "etf", "mf", "bond", "sgb"].includes(holding.security_type),
  );

  return (
    <div data-copy-screen="money" className="grid gap-4">
      {saveQueued ? (
        <p role="status" className="text-sm text-muted-foreground">
          {englishCopy.money_ui.saved_offline}
        </p>
      ) : null}
      {pageError ? (
        <Card role="alert" className="border-danger/50 text-sm">
          {pageError}
        </Card>
      ) : null}
      <Card className="grid gap-4">
        {loading || !totals ? (
          <div
            className="grid gap-3"
            role="status"
            aria-label="Loading your net worth"
          >
            <div className="h-5 w-32 animate-pulse rounded bg-muted" />
            <div className="h-9 w-48 animate-pulse rounded bg-muted" />
            <div className="h-4 w-full animate-pulse rounded bg-muted" />
          </div>
        ) : (
          <>
            <div className="flex flex-wrap items-start justify-between gap-3">
              <Stat
                label={
                  <Term
                    explanation="The value of everything you own, minus what you owe."
                    prompt="What's this?"
                    learnMoreHref="/help#term-net-worth"
                  >
                    Net worth
                  </Term>
                }
                value={shownAmount(totals.netWorth)}
                detail={
                  <span className="grid gap-0.5">
                    {secondaryAmount(totals.netWorth) ? (
                      <span>{secondaryAmount(totals.netWorth)}</span>
                    ) : null}
                    <span>
                      Updated{" "}
                      {networth?.as_of
                        ? new Intl.DateTimeFormat("en-IN", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                          }).format(new Date(`${networth.as_of}T00:00:00`))
                        : "recently"}
                    </span>
                  </span>
                }
              />
            </div>
            {!noListedItems &&
            selectedMemberId === "family" &&
            familyMembers.length > 0 ? (
              <details className="border-t border-line pt-2">
                <summary className="min-h-12 cursor-pointer py-2 text-sm font-semibold">
                  {englishCopy.household.money.member_breakdown}
                </summary>
                <div className="grid gap-2 pb-2">
                  {familyMembers.map((member) => (
                    <div
                      key={member.id}
                      className="flex items-center justify-between gap-3 text-sm"
                    >
                      <span>
                        {member.id === meMemberId
                          ? englishCopy.household.you
                          : member.name}
                      </span>
                      <AmountLabel
                        amount={member.net_worth}
                        primaryCurrency={primaryCurrency}
                        fxQuote={fxQuote}
                        showSecondary={hasDanishMember}
                      />
                    </div>
                  ))}
                </div>
              </details>
            ) : null}
            {hasDanishMember ? (
              <p className="text-xs text-muted-foreground">
                {fxLine ?? englishCopy.household.dkk_unavailable}
              </p>
            ) : null}
          </>
        )}
      </Card>

      {!offlineSummaryOnly && !noListedItems ? (
        <Button
          data-primary="true"
          onClick={() => setAddSheetOpen(true)}
          disabled={isViewer}
          className="w-full"
        >
          {englishCopy.money_ui.add_update}
        </Button>
      ) : null}

      <section className="grid gap-3" aria-labelledby="assets-heading">
        <div className="flex items-center justify-between">
          <h2 id="assets-heading" className="text-lg font-semibold">
            Your money
          </h2>
          {!offlineSummaryOnly && listedItemCount !== null ? (
            <Badge>{listedItemCount} items</Badge>
          ) : null}
        </div>
        {offlineSummaryOnly ? (
          <Card>
            <p className="text-sm text-muted-foreground">
              {englishCopy.offline.money_map_summary}
            </p>
          </Card>
        ) : null}
        <HoldingsGroup
          title={englishCopy.money_ui.funds}
          holdings={fundHoldings}
          showAll={showAllHoldings}
          onToggleShowAll={() => setShowAllHoldings((shown) => !shown)}
          primaryCurrency={primaryCurrency}
          fxQuote={fxQuote}
          showSecondary={hasDanishMember}
          onSelectRecordedBuy={(holding) => {
            saveRecordedBuy.reset();
            deleteRecordedBuy.reset();
            setEditingRecordedBuy(holding);
          }}
        />
        {!noListedItems ? (
          <details
            onToggle={(event) =>
              setShowMonthlyAmounts(event.currentTarget.open)
            }
          >
            <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
              {englishCopy.money_ui.monthly_amounts}
            </summary>
            {showMonthlyAmounts ? <FundSipFineTune /> : null}
          </details>
        ) : null}
        <HoldingsGroup
          title={englishCopy.money_ui.shares}
          holdings={shareHoldings}
          holdingReviews={plan?.holding_reviews}
          showAll={showAllHoldings}
          onToggleShowAll={() => setShowAllHoldings((shown) => !shown)}
          primaryCurrency={primaryCurrency}
          fxQuote={fxQuote}
          showSecondary={hasDanishMember}
          onSelectRecordedBuy={(holding) => {
            saveRecordedBuy.reset();
            deleteRecordedBuy.reset();
            setEditingRecordedBuy(holding);
          }}
        />
        <HoldingsGroup
          title={englishCopy.money_ui.bonds_and_gold_bonds}
          holdings={bondsAndGoldBonds}
          showAll={showAllHoldings}
          onToggleShowAll={() => setShowAllHoldings((shown) => !shown)}
          primaryCurrency={primaryCurrency}
          fxQuote={fxQuote}
          showSecondary={hasDanishMember}
          onSelectRecordedBuy={(holding) => {
            saveRecordedBuy.reset();
            deleteRecordedBuy.reset();
            setEditingRecordedBuy(holding);
          }}
        />
        <HoldingsGroup
          title={englishCopy.money_ui.other}
          holdings={otherHoldings}
          showAll={showAllHoldings}
          onToggleShowAll={() => setShowAllHoldings((shown) => !shown)}
          primaryCurrency={primaryCurrency}
          fxQuote={fxQuote}
          showSecondary={hasDanishMember}
          onSelectRecordedBuy={(holding) => {
            saveRecordedBuy.reset();
            deleteRecordedBuy.reset();
            setEditingRecordedBuy(holding);
          }}
        />
        {visibleGroups.length ? (
          visibleGroups.map(([sleeve, items]) => (
            <Card key={sleeve} className="grid gap-1">
              <div className="mb-1 flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-muted-foreground">
                  {sleeve}
                </h3>
                <GroupAmount
                  amount={sumKnownValues(
                    items.map((asset) => asset.value_cached),
                  )}
                  primaryCurrency={primaryCurrency}
                  fxQuote={fxQuote}
                  showSecondary={hasDanishMember}
                />
              </div>
              {items.map((asset) => (
                <Link
                  key={asset.id}
                  to={`/money/asset/${asset.id}`}
                  className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2 hover:text-brand"
                >
                  <span className="flex w-0 min-w-0 flex-1 items-center gap-3">
                    <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                      <Building2 aria-hidden="true" className="size-4" />
                    </span>
                    <span className="min-w-0">
                      <strong
                        className="block line-clamp-2 text-sm [overflow-wrap:anywhere]"
                        data-security
                      >
                        {assetLabel(asset)}
                      </strong>
                      <span className="block truncate text-xs text-muted-foreground">
                        {maturityStatus(asset)}
                      </span>
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-2 text-right text-sm font-semibold tabular-nums">
                    {asset.asset_type === "savings" ? (
                      formatSavingsAmount(
                        asset,
                        primaryCurrency,
                        fxQuote,
                        hasDanishMember,
                      )
                    ) : asset.valuation_status === "price_needed" ||
                      !asset.value_cached ? (
                      "—"
                    ) : (
                      <AmountLabel
                        amount={asset.value_cached}
                        primaryCurrency={primaryCurrency}
                        fxQuote={fxQuote}
                        showSecondary={hasDanishMember}
                      />
                    )}
                    <ChevronRight
                      aria-hidden="true"
                      className="size-4 text-muted-foreground"
                    />
                  </span>
                </Link>
              ))}
            </Card>
          ))
        ) : noListedItems ? (
          <EmptyState
            title={englishCopy.household_ui.money_empty_title}
            description={englishCopy.household_ui.money_empty_description}
            icon={<WalletCards aria-hidden="true" className="size-5" />}
            action={
              <Button
                type="button"
                data-primary="true"
                onClick={() => setAddSheetOpen(true)}
              >
                {englishCopy.household_ui.money_empty_action}
              </Button>
            }
          />
        ) : null}
        <LoansGroup summary={loanSummary} shownAmount={shownAmount} />
      </section>
      <details
        id="money-checklist"
        ref={chooserRef}
        className="rounded-xl border border-line px-3"
      >
        <summary
          ref={chooserSummaryRef}
          tabIndex={-1}
          className="min-h-12 cursor-pointer py-3 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand"
        >
          {englishCopy.money_ui.add_by_hand}
        </summary>
        <div className="grid gap-4">
          <Card className="grid gap-4">
            <p className="text-sm text-muted-foreground">
              Tap everything that applies. Rough numbers are fine; you can edit
              anytime.
            </p>
            {networth ? (
              <div className="grid gap-1 text-xs text-muted-foreground">
                {nextCategory ? (
                  <p>Add {categoryLabel(nextCategory)} to finish</p>
                ) : tickedCategories.length ? (
                  <p>Everything is up to date</p>
                ) : null}
              </div>
            ) : null}
            <div className="grid grid-cols-2 gap-2 [@media(min-width:360px)]:grid-cols-3">
              {checklistTiles.map((tile) => {
                const selected = checklist.includes(tile.category);
                return (
                  <button
                    key={tile.category}
                    type="button"
                    aria-pressed={selected}
                    disabled={isViewer}
                    onClick={() => void toggleCategory(tile.category)}
                    className={`min-h-16 rounded-xl border px-2 py-3 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand ${selected ? "border-brand bg-brand/10 text-brand" : "border-line bg-background hover:bg-muted"}`}
                  >
                    {selected ? <span aria-hidden="true">✓ </span> : null}
                    {tile.shortLabel}
                  </button>
                );
              })}
            </div>
            {checklistError ? (
              <p className="text-sm text-danger-foreground" role="status">
                {checklistError}
              </p>
            ) : null}
            <Button
              onClick={startChecklist}
              disabled={isViewer || !orderedSelected.length}
            >
              Continue ({orderedSelected.length} selected)
            </Button>
          </Card>
          {!isViewer ? (
            <details className="rounded-xl border border-line px-3">
              <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand">
                {englishCopy.money_ui.quick_add_summary}
              </summary>
              <div className="pb-3">
                <QuickAddBar onDraft={startQuickDraft} />
              </div>
            </details>
          ) : null}
        </div>
      </details>
      <ImportHistorySheet jobs={importJobs} />
      <details
        className="rounded-xl border border-line px-3"
        onToggle={(event) => setShowFundCheck(event.currentTarget.open)}
      >
        <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
          {englishCopy.money_ui.check_investment}
        </summary>
        {showFundCheck ? (
          <Suspense fallback={<p className="py-3 text-sm">Loading…</p>}>
            <FundCheckCard />
          </Suspense>
        ) : null}
      </details>
      {networth ? (
        <details className="rounded-xl border border-line px-3">
          <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
            {englishCopy.money_ui.more_details}
          </summary>
          <div className="grid gap-4">
            {networth ? (
              <HouseholdInsights
                networth={networth}
                primaryCurrency={primaryCurrency}
                fxQuote={fxQuote}
                showSecondary={hasDanishMember}
              />
            ) : null}
            {networth?.by_sleeve.length ? (
              <Card className="grid gap-3">
                <h2 className="text-base font-semibold">Split by type</h2>
                {sleeveRows(networth.by_sleeve).map((row) => (
                  <div
                    key={row.label}
                    className="flex min-h-10 items-center justify-between gap-3 border-t border-line pt-2 text-sm"
                  >
                    <span>{row.label}</span>
                    <span className="flex gap-3 tabular-nums">
                      <span>{shownAmount(row.amount)}</span>
                      <span className="text-muted-foreground">
                        {plainPercent(row.pct)}
                      </span>
                    </span>
                  </div>
                ))}
              </Card>
            ) : null}
          </div>
        </details>
      ) : null}
      <Sheet
        open={addSheetOpen}
        onOpenChange={setAddSheetOpen}
        title={englishCopy.money_ui.add_update}
      >
        <div className="grid gap-2">
          <div className="grid gap-1">
            <Link
              to="/import"
              onClick={() => setAddSheetOpen(false)}
              data-primary="true"
              className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
            >
              {englishCopy.money_ui.upload_statement}
            </Link>
            <p className="text-sm text-muted-foreground">
              {englishCopy.money_ui.upload_description}
            </p>
          </div>
          <Button variant="secondary" onClick={openManualChooser}>
            {englishCopy.money_ui.add_by_hand}
          </Button>
          <Link
            to="/settings/gmail"
            onClick={() => setAddSheetOpen(false)}
            className="inline-flex min-h-11 items-center text-sm font-medium text-brand"
          >
            {englishCopy.money_ui.gmail_statements}
          </Link>
        </div>
      </Sheet>
      {editingRecordedBuy ? (
        <RecordedBuySheet
          holding={editingRecordedBuy}
          saving={saveRecordedBuy.isPending}
          removing={deleteRecordedBuy.isPending}
          error={saveRecordedBuy.isError || deleteRecordedBuy.isError}
          onClose={() => setEditingRecordedBuy(null)}
          onSave={(payload) => {
            const id = editingRecordedBuy.recorded_buy_id;
            if (id) saveRecordedBuy.mutate({ id, payload });
          }}
          onRemove={() => {
            const id = editingRecordedBuy.recorded_buy_id;
            if (id) deleteRecordedBuy.mutate(id);
          }}
        />
      ) : null}
    </div>
  );
}
