import { Suspense, lazy, useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import {
  ArrowRight,
  Building2,
  CalendarDays,
  ChevronRight,
  CloudUpload,
  WalletCards,
} from "lucide-react";
import { Link, useNavigate } from "react-router";
import { ApiError } from "../api/client";
import { formatInr, sumInr } from "../lib/money";
import { Badge, Button, Card, EmptyState, Progress, Stat, Term } from "../ui";
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
import type { ImportJob } from "../imports/api";
import { ImportHistorySheet } from "../imports/ImportHistorySheet";
import { markValueMoment } from "../pwa/install";
import englishCopy from "../copy/en.json";
import { apiPathAvailable } from "../api/feature-support";
import { useHousehold } from "../household/HouseholdContext";
import { quantityLabel } from "./quantity";
import { assetTitles } from "./titles";
import { dismissHighlight, getPlan } from "../planning/api";
import type { WeeklyPlan } from "../planning/api";
import { Highlights } from "../features/Highlights";
import {
  formatHouseholdAmount,
  formatSecondaryAmount,
} from "../household/currency";
import {
  formatDkk,
  formatRateToTwoDecimals,
  type DkkInrRate,
} from "../household/fx";

const FundCheckCard = lazy(() =>
  import("./FundCheckCard").then((module) => ({
    default: module.FundCheckCard,
  })),
);

const SLEEVE_ORDER = [
  englishCopy.household_ui.indian_shares,
  "Deposits & PPF",
  "Gold",
  englishCopy.household_ui.abroad,
  englishCopy.household_ui.other_holdings,
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

function sleeveForAsset(asset: AssetResponse): string {
  switch (asset.asset_type) {
    case "gold":
      return "Gold";
    case "property":
    case "term_insurance":
    case "health_insurance":
      return englishCopy.household_ui.other_holdings;
    case "loan":
      return "Loans";
    case "fd":
    case "rd":
    case "ppf":
    case "epf":
    case "nps":
    case "po_scheme":
    case "bond":
    case "savings":
    case "traditional_policy":
    case "ulip":
      return "Deposits & PPF";
    default:
      return englishCopy.household_ui.indian_shares;
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
    ? `${typeLabel} · ${bank}`
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
      <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold text-muted-foreground">Loans</h3>
        <p className="text-xs text-muted-foreground">
          Total left {shownAmount(summary.remaining_inr)} ·{" "}
          {shownAmount(summary.monthly_emis_inr)} per month
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
            <span className="flex min-w-0 flex-1 items-center gap-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <WalletCards aria-hidden="true" className="size-4" />
              </span>
              <span className="min-w-0 flex-1">
                <strong className="block truncate text-sm">
                  {loan.name}
                  {loan.lender ? ` · ${loan.lender}` : ""}
                </strong>
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
        <label htmlFor="quick-add" className="text-sm font-semibold">
          Quick add
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
  const now = new Date();
  const oneYear = new Date(now);
  oneYear.setFullYear(oneYear.getFullYear() + 1);
  const upcoming = networth.maturity_calendar.filter((event) => {
    const date = new Date(`${event.date}T00:00:00`);
    return date >= now && date <= oneYear;
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
  const navigate = useNavigate();
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
  const [assets, setAssets] = useState<AssetResponse[]>([]);
  const [showAllHoldings, setShowAllHoldings] = useState(false);
  const [importJobs, setImportJobs] = useState<ImportJob[]>([]);
  const [networth, setNetworth] = useState<NetworthResponse | null>(null);
  const [loanSummary, setLoanSummary] = useState<LoanListResponse | null>(null);
  const [plan, setPlan] = useState<WeeklyPlan | null>(null);
  const [checklist, setChecklist] = useState<string[]>(readChecklist);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState("");
  const [offlineSummaryOnly, setOfflineSummaryOnly] = useState(false);
  const [checklistError, setChecklistError] = useState("");
  const [showFundCheck, setShowFundCheck] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setPageError("");
    setOfflineSummaryOnly(false);
    setAssets([]);
    setNetworth(null);
    setLoanSummary(null);
    setPlan(null);
    setImportJobs([]);
    const planRequest = apiPathAvailable("/v1/plan")
      ? getPlan()
      : Promise.resolve(null);
    const loanRequest = apiPathAvailable("/v1/loans")
      ? listLoans()
      : Promise.resolve(null);
    Promise.allSettled([
      listAssets(activeMemberId),
      getNetworth(activeMemberId),
      planRequest,
      loanRequest,
    ])
      .then(([assetResult, networthResult, planResult, loanResult]) => {
        if (!active) return;
        if (networthResult.status === "rejected") throw networthResult.reason;
        const loadedNetworth = networthResult.value;
        setNetworth(loadedNetworth);
        if (planResult.status === "fulfilled") setPlan(planResult.value);
        if (loanResult.status === "fulfilled") setLoanSummary(loanResult.value);
        if (assetResult.status === "rejected") {
          setAssets([]);
          setOfflineSummaryOnly(true);
          return;
        }
        const visibleAssets = assetResult.value;
        setAssets(visibleAssets);
        setOfflineSummaryOnly(false);
        if (visibleAssets.length >= 3) markValueMoment();
      })
      .catch(() => {
        if (active)
          setPageError("We couldn't load your Money Map. Please try again.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [activeMemberId, selectedHouseholdId]);

  useEffect(() => {
    let active = true;
    listImports()
      .then((jobs) => {
        if (active) setImportJobs(jobs);
      })
      .catch(() => {
        if (active) setImportJobs([]);
      })
      .finally(() => {});
    return () => {
      active = false;
    };
  }, [selectedHouseholdId]);

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

  const completenessPct = Math.max(
    0,
    Math.min(100, Number(networth?.completeness.pct ?? 0)),
  );
  const tickedCategories = networth?.completeness.ticked ?? [];
  const nextCategory = checklistTiles.find(
    (tile) =>
      tickedCategories.includes(tile.category) &&
      !networth?.completeness.done.includes(tile.category),
  )?.category;
  const completenessText = `Money Map ${completenessPct}% complete${nextCategory ? ` · add ${categoryLabel(nextCategory)} to finish` : tickedCategories.length ? " · everything is up to date" : ""}`;
  const checklistCompleted = (networth?.completeness.done.length ?? 0) > 0;

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

  async function hideHighlight(key: string) {
    try {
      await dismissHighlight(key);
      setPlan(await getPlan());
    } catch {
      setPageError("We couldn't hide this note. Please try again.");
    }
  }

  const visibleGroups = useMemo(() => {
    const groups = new Map<string, AssetResponse[]>();
    for (const asset of assets) {
      if (asset.asset_type === "loan") continue;
      const sleeve = sleeveForAsset(asset);
      groups.set(sleeve, [...(groups.get(sleeve) ?? []), asset]);
    }
    return SLEEVE_ORDER.filter((sleeve) => groups.has(sleeve)).map(
      (sleeve) => [sleeve, groups.get(sleeve) ?? []] as const,
    );
  }, [assets]);

  return (
    <div data-copy-screen="money" className="grid gap-4">
      {pageError ? (
        <Card role="alert" className="border-danger/50 text-sm">
          {pageError}
        </Card>
      ) : null}
      {!offlineSummaryOnly ? (
        <Card className="grid gap-3 border-brand/40 bg-brand/5">
          <div className="flex items-center gap-3">
            <span className="grid size-11 place-items-center rounded-xl bg-brand/10 text-brand">
              <CloudUpload aria-hidden="true" className="size-5" />
            </span>
            <div>
              <h2 className="text-base font-semibold">Upload your statement</h2>
              <p className="text-sm text-muted-foreground">
                Add account and investment details in one go.
              </p>
            </div>
          </div>
          <Link
            to={isViewer ? "/plan" : "/import"}
            data-primary="true"
            className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
          >
            {isViewer ? "View your plan" : "Upload statement"}
          </Link>
        </Card>
      ) : null}
      {!isViewer ? <QuickAddBar onDraft={startQuickDraft} /> : null}
      <Highlights
        items={plan?.highlights_by_screen?.money}
        onDismiss={(key) => void hideHighlight(key)}
      />
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
                detail={`${secondaryAmount(totals.netWorth) ? `${secondaryAmount(totals.netWorth)} · ` : ""}Updated ${networth?.as_of ? new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric" }).format(new Date(`${networth.as_of}T00:00:00`)) : "recently"}`}
              />
              <div className="grid grid-cols-2 gap-4 text-right">
                <Stat
                  label={englishCopy.household_ui.money_you_own}
                  value={shownAmount(totals.assets)}
                  detail={secondaryAmount(totals.assets) ?? undefined}
                />
                <Stat
                  label={englishCopy.household_ui.money_you_owe}
                  value={shownAmount(totals.liabilities)}
                  detail={secondaryAmount(totals.liabilities) ?? undefined}
                />
              </div>
            </div>
            <div className="grid gap-2">
              <div className="flex justify-between gap-3 text-sm">
                <span>{completenessText}</span>
                <span className="tabular-nums">{completenessPct}%</span>
              </div>
              <Progress value={completenessPct} label={completenessText} />
            </div>
            {selectedMemberId === "family" && familyMembers.length > 0 ? (
              <div className="grid gap-2 border-t border-line pt-3">
                <h2 className="text-sm font-semibold">
                  {englishCopy.household.money.member_breakdown}
                </h2>
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
            ) : null}
            {hasDanishMember ? (
              <p className="text-xs text-muted-foreground">
                {fxLine ?? englishCopy.household.dkk_unavailable}
              </p>
            ) : null}
          </>
        )}
      </Card>

      <ImportHistorySheet jobs={importJobs} />

      <details onToggle={(event) => setShowFundCheck(event.currentTarget.open)}>
        <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
          Check an investment
        </summary>
        {showFundCheck ? (
          <Suspense fallback={<p className="py-3 text-sm">Loading…</p>}>
            <FundCheckCard />
          </Suspense>
        ) : null}
      </details>

      <details id="money-checklist" open={!checklistCompleted}>
        <summary className="min-h-12 cursor-pointer py-3 text-base font-semibold">
          What else do you have?
        </summary>
        <Card className="grid gap-4">
          <p className="text-sm text-muted-foreground">
            Tap everything that applies. Rough numbers are fine; you can edit
            anytime.
          </p>
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
      </details>

      <section className="grid gap-3" aria-labelledby="assets-heading">
        <div className="flex items-center justify-between">
          <h2 id="assets-heading" className="text-lg font-semibold">
            Your money
          </h2>
          {!offlineSummaryOnly ? <Badge>{assets.length} items</Badge> : null}
        </div>
        {offlineSummaryOnly ? (
          <Card>
            <p className="text-sm text-muted-foreground">
              {englishCopy.offline.money_map_summary}
            </p>
          </Card>
        ) : null}
        <LoansGroup summary={loanSummary} shownAmount={shownAmount} />
        {visibleGroups.length ? (
          visibleGroups.map(([sleeve, items]) => (
            <Card key={sleeve} className="grid gap-1">
              <h3 className="mb-1 text-sm font-semibold text-muted-foreground">
                {sleeve}
              </h3>
              {items.map((asset) => (
                <Link
                  key={asset.id}
                  to={`/money/asset/${asset.id}`}
                  className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2 hover:text-brand"
                >
                  <span className="flex min-w-0 flex-1 items-center gap-3">
                    <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                      <Building2 aria-hidden="true" className="size-4" />
                    </span>
                    <span className="min-w-0">
                      <strong className="block truncate text-sm" data-security>
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
        ) : offlineSummaryOnly ? null : (
          <EmptyState
            title={englishCopy.household_ui.money_empty_title}
            description="Choose one item below to add it to your Money Map."
            icon={<WalletCards aria-hidden="true" className="size-5" />}
            action={
              <Button
                type="button"
                variant="secondary"
                onClick={() =>
                  document
                    .getElementById("money-checklist")
                    ?.scrollIntoView({ block: "start" })
                }
              >
                {englishCopy.household_ui.money_empty_action}
              </Button>
            }
          />
        )}
      </section>

      {networth?.holdings.length ? (
        <section className="grid gap-3" aria-labelledby="holdings-heading">
          <div className="flex items-center justify-between gap-3">
            <h2 id="holdings-heading" className="text-lg font-semibold">
              Shares and funds
            </h2>
            <Badge>{networth.holdings.length} holdings</Badge>
          </div>
          <Card className="grid gap-2">
            {byValue(networth.holdings)
              .slice(0, showAllHoldings ? undefined : HOLDINGS_SHOWN)
              .map((holding) => (
                <Link
                  key={holding.security_id}
                  to={`/money/holding/${holding.security_id}`}
                  className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2 first:border-0 first:pt-0 hover:text-brand"
                >
                  <span className="flex min-w-0 flex-1 items-center gap-3">
                    <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                      <Building2 aria-hidden="true" className="size-4" />
                    </span>
                    <span className="min-w-0">
                      <strong className="block truncate text-sm">
                        <span data-security>{holding.name}</span>
                      </strong>
                      <span className="block truncate text-xs text-muted-foreground">
                        <span data-security>{holding.symbol}</span> ·{" "}
                        {quantityLabel(holding.quantity, holding.security_type)}
                      </span>
                    </span>
                  </span>
                  <span className="flex shrink-0 items-center gap-2 text-right text-sm font-semibold tabular-nums">
                    <span>
                      {holding.value ? (
                        <AmountLabel
                          amount={holding.value}
                          primaryCurrency={primaryCurrency}
                          fxQuote={fxQuote}
                          showSecondary={hasDanishMember}
                        />
                      ) : (
                        "Not available"
                      )}
                    </span>
                    <ChevronRight
                      aria-hidden="true"
                      className="size-4 text-muted-foreground"
                    />
                  </span>
                </Link>
              ))}
            {networth.holdings.length > HOLDINGS_SHOWN ? (
              <Button
                variant="secondary"
                aria-expanded={showAllHoldings}
                onClick={() => setShowAllHoldings((shown) => !shown)}
              >
                {showAllHoldings
                  ? "Show fewer"
                  : `Show all ${networth.holdings.length}`}
              </Button>
            ) : null}
          </Card>
        </section>
      ) : null}

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
              <span className="tabular-nums">
                {shownAmount(row.amount)} · {plainPercent(row.pct)}
              </span>
            </div>
          ))}
        </Card>
      ) : null}
    </div>
  );
}
