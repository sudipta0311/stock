import { Link } from "react-router";
import { WalletCards } from "lucide-react";
import { labelForCode, SLEEVE_LABELS } from "../lib/labels";
import {
  asArray,
  asRecord,
  friendlyRouteReason,
  friendlyRouteWarning,
  instrumentLabel,
  isPositiveAmount,
  textValue,
} from "../features/data";
import {
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import { getPlan } from "../planning/api";
import { addPathForCategory } from "../money/config";
import { getNetworth, listAssets, listGoals, listLoans } from "../money/api";
import type {
  AssetResponse,
  GoalResponse,
  LoanListResponse,
} from "../money/api";
import type { WeeklyPlan } from "../planning/api";
import { Card, EmptyState, Progress, Term } from "../ui";
import { useHousehold } from "../household/HouseholdContext";
import { formatHouseholdAmount } from "../household/currency";
import {
  DkkMoneyValue,
  ExchangeRateNote,
  MoneyValue,
} from "../household/currency-view";
import type { MemberResponse, NetworthResponse } from "../money/api";
import * as englishCopy from "../copy/sections/index";

const SLEEVES = [
  "equity_india",
  "equity_global",
  "debt",
  "gold",
  "cash",
] as const;
const FOUNDATION = [
  ["term_cover", "Term cover"],
  ["health_cover", "Health cover"],
  ["emergency_fund", "Emergency fund"],
  ["high_interest_debt", "Costly debt"],
] as const;

interface AllocationRow {
  key: string;
  label: string;
  actualPct: string;
  targetPct: string;
  actualAmount: string;
  targetAmount: string;
  gapAmount: string;
}

type PlanGapKey = "equity_india" | "equity_global" | "debt" | "cash" | "gold";

interface PlanGap {
  row: AllocationRow;
  amount: string;
}

const PLAN_GAP_COPY: Record<PlanGapKey, { button: string; name: string }> = {
  equity_india: {
    button: englishCopy.plan_ui.gaps.equity_india.button,
    name: englishCopy.plan_ui.gaps.equity_india.name,
  },
  equity_global: {
    button: englishCopy.plan_ui.gaps.equity_global.button,
    name: englishCopy.plan_ui.gaps.equity_global.name,
  },
  debt: {
    button: englishCopy.plan_ui.gaps.debt.button,
    name: englishCopy.plan_ui.gaps.debt.name,
  },
  cash: {
    button: englishCopy.plan_ui.gaps.cash.button,
    name: englishCopy.plan_ui.gaps.cash.name,
  },
  gold: {
    button: englishCopy.plan_ui.gaps.gold.button,
    name: englishCopy.plan_ui.gaps.gold.name,
  },
};

const PLAN_GAP_KEYS = new Set<PlanGapKey>([
  "equity_india",
  "equity_global",
  "debt",
  "cash",
  "gold",
]);

type PlanMember = NonNullable<WeeklyPlan["members"]>[number];

function LoansPlanDetails({
  summary,
}: {
  summary: LoanListResponse | undefined;
}) {
  if (!summary?.loans.length) return null;
  return (
    <details className="rounded-xl border border-line p-3">
      <summary className="min-h-11 cursor-pointer py-2 text-sm font-semibold">
        Loans
      </summary>
      <div className="grid gap-2 border-t border-line pt-3 text-sm">
        <p>
          Total left: <MoneyValue amountInr={summary.remaining_inr} />
        </p>
        <div className="flex flex-wrap items-center gap-1">
          <Term
            explanation="The total amount due on your loans each month."
            prompt="What's this?"
            learnMoreHref="/help#term-monthly-payments"
          >
            Monthly payments
          </Term>
          : <MoneyValue amountInr={summary.monthly_emis_inr} />
        </div>
        {summary.debt_load_pct !== null ? (
          <p>
            Monthly payments are {summary.debt_load_pct}% of monthly income.
          </p>
        ) : null}
        <p className="text-muted-foreground">{summary.advice}</p>
      </div>
    </details>
  );
}

const DISPLAY_SLEEVES: Record<string, string> = {
  equity_india: englishCopy.household_ui.indian_shares,
  debt: englishCopy.household_ui.safe_money,
  cash: englishCopy.household_ui.cash,
  equity_global: englishCopy.household_ui.abroad,
  gold: englishCopy.labels.sleeves.gold,
  property: englishCopy.household_ui.other_holdings,
  protection: englishCopy.household_ui.other_holdings,
  liabilities: englishCopy.household_ui.other_holdings,
};

function memberDisplayName(
  id: string,
  fallback: string,
  meMemberId: string | undefined,
): string {
  return id === meMemberId ? englishCopy.household_ui.you : fallback;
}

function limitsForMember(
  planMember: PlanMember | undefined,
  person: MemberResponse | undefined,
): { ppf: string | null; lrs: string | null; ask: string | null } {
  return {
    ppf: planMember?.limits_room?.ppf_annual_inr ?? null,
    lrs: person?.investing.lrs_used_this_fy ?? null,
    ask: planMember?.limits_room?.ask_deposit_dkk ?? null,
  };
}

function placementNoteText(note: string): string {
  if (note === "ask_start_value_missing") {
    return englishCopy.denmark.ask_start_value_missing;
  }
  if (/clubbing(?:_check)?/i.test(note)) {
    return englishCopy.household_ui.clubbing_check;
  }
  if (/after[_ -]?tax[_ -]?yield/i.test(note)) {
    return englishCopy.household_ui.placement_yield_note.replace(
      "{name}",
      englishCopy.household_ui.this_person,
    );
  }
  if (/^[a-z0-9_]+$/i.test(note)) {
    return englishCopy.household_ui.placement_general;
  }
  return /\s/.test(note) ? note : englishCopy.household_ui.placement_general;
}

function placementNotes(plan: WeeklyPlan): string[] {
  const notes = [
    ...(plan.placement_notes ?? []),
    ...(plan.residency_notes ?? []),
    ...(plan.members ?? []).flatMap((member) => member.notes ?? []),
  ];
  const denmarkCodes = notes.filter((note) =>
    /^(?:dk_|ask_|denmark_)/i.test(note),
  );
  const preferredDenmarkCode =
    denmarkCodes.find((note) => note === "ask_start_value_missing") ??
    denmarkCodes[0];
  const otherNotes = notes.filter(
    (note) => !/^(?:dk_|ask_|denmark_)/i.test(note),
  );
  const friendly = [
    ...(preferredDenmarkCode ? [preferredDenmarkCode] : []),
    ...otherNotes,
  ].map(placementNoteText);
  const hasClubbingAction = asArray(plan.actions).some((action) => {
    const record = asRecord(action);
    const payload = asRecord(record.payload);
    const flag = record.clubbing_check ?? payload.clubbing_check;
    return flag === true || (typeof flag === "string" && flag.length > 0);
  });
  if (hasClubbingAction) friendly.push(englishCopy.household_ui.clubbing_check);
  const unique = [...new Set(friendly)];
  unique.sort((left, right) =>
    left === englishCopy.household_ui.clubbing_check
      ? -1
      : right === englishCopy.household_ui.clubbing_check
        ? 1
        : 0,
  );
  return unique.slice(0, 2);
}

function PlacementNotesCard({ plan }: { plan: WeeklyPlan }) {
  const notes = placementNotes(plan);
  if (!notes.length) return null;
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">
        {englishCopy.household_ui.placement_title}
      </h2>
      <details>
        <summary className="min-h-11 cursor-pointer py-2 text-sm font-semibold">
          {englishCopy.household_ui.placement_details}
        </summary>
        <ul className="mt-2 grid gap-2 text-base">
          {notes.map((note) => (
            <li key={note} className="rounded-xl bg-muted p-3">
              {note}
            </li>
          ))}
        </ul>
      </details>
    </Card>
  );
}

function MemberLimits({
  planMember,
  person,
}: {
  planMember: PlanMember | undefined;
  person: MemberResponse | undefined;
}) {
  const limits = limitsForMember(planMember, person);
  const rows = [
    limits.ppf === null
      ? null
      : {
          key: "ppf",
          label: englishCopy.household_ui.ppf_room,
          value: <MoneyValue amountInr={limits.ppf} />,
        },
    limits.lrs === null
      ? null
      : {
          key: "lrs",
          label: englishCopy.household_ui.sent_abroad,
          value: <MoneyValue amountInr={limits.lrs} />,
        },
    limits.ask === null
      ? null
      : {
          key: "ask",
          label: englishCopy.household_ui.ask_room,
          value: <DkkMoneyValue amountDkk={limits.ask} />,
        },
  ].filter((row): row is NonNullable<typeof row> => row !== null);
  if (!rows.length) return null;
  return (
    <div className="grid gap-2 border-t border-line pt-3">
      <h4 className="text-sm font-semibold">
        {englishCopy.household_ui.limits_title}
      </h4>
      <dl className="grid gap-2">
        {rows.map((row) => (
          <div
            key={row.key}
            className="flex items-start justify-between gap-3 text-sm"
          >
            <dt className="text-muted-foreground">{row.label}</dt>
            <dd className="text-right font-medium tabular-nums">{row.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function MemberHoldingsCard({
  members,
  planMembers,
  meMemberId,
}: {
  members: MemberResponse[];
  planMembers: PlanMember[];
  meMemberId: string | undefined;
}) {
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">
          {englishCopy.household_ui.holds_title}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.household_ui.family_plan_summary}
        </p>
      </div>
      <div className="grid gap-4">
        {planMembers.map((member) => {
          const person = members.find(
            (candidate) => candidate.id === member.id,
          );
          const name = memberDisplayName(member.id, member.name, meMemberId);
          const sleeves = Object.entries(member.sleeves ?? {}).filter(
            ([, value]) => isPositiveAmount(value),
          );
          return (
            <section
              key={member.id}
              className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold">{name}</h3>
                <span className="text-right text-sm font-semibold tabular-nums">
                  <MoneyValue amountInr={member.net_worth} />
                </span>
              </div>
              <details>
                <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
                  {englishCopy.household_ui.details}
                </summary>
                <div className="mt-3 grid gap-3">
                  {sleeves.length ? (
                    <dl className="grid gap-1 text-sm">
                      {sleeves.map(([key, value]) => (
                        <div key={key} className="flex justify-between gap-3">
                          <dt className="text-muted-foreground">
                            {DISPLAY_SLEEVES[key] ??
                              englishCopy.household_ui.other_holdings}
                          </dt>
                          <dd className="tabular-nums">
                            <MoneyValue amountInr={value} />
                          </dd>
                        </div>
                      ))}
                    </dl>
                  ) : null}
                  <MemberLimits planMember={member} person={person} />
                </div>
              </details>
            </section>
          );
        })}
      </div>
    </Card>
  );
}

function SelectedMemberCard({
  member,
  person,
  networth,
  meMemberId,
}: {
  member: PlanMember | undefined;
  person: MemberResponse | undefined;
  networth: NetworthResponse | undefined;
  meMemberId: string | undefined;
}) {
  const id = member?.id ?? person?.id ?? "";
  const fallback = member?.name ?? person?.display_name ?? "";
  const name = memberDisplayName(id, fallback, meMemberId);
  const sleeves = member?.sleeves
    ? Object.entries(member.sleeves)
    : (networth?.by_sleeve ?? []).map(
        (item) => [item.sleeve, item.amount] as const,
      );
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">{name}</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.household_ui.selected_member_plan.replace(
            "{name}",
            name,
          )}
        </p>
      </div>
      <div className="flex items-center justify-between gap-3 text-sm">
        <span className="text-muted-foreground">
          {englishCopy.household_ui.member_net_worth}
        </span>
        <span className="text-right font-semibold tabular-nums">
          <MoneyValue
            amountInr={member?.net_worth ?? networth?.net_worth ?? "0"}
          />
        </span>
      </div>
      {sleeves.length ? (
        <dl className="grid gap-2 border-t border-line pt-3 text-sm">
          {sleeves.map(([key, value]) => (
            <div key={key} className="flex justify-between gap-3">
              <dt className="text-muted-foreground">
                {DISPLAY_SLEEVES[key] ??
                  englishCopy.household_ui.other_holdings}
              </dt>
              <dd className="text-right tabular-nums">
                <MoneyValue amountInr={value} />
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      <MemberLimits planMember={member} person={person} />
    </Card>
  );
}

function allocationRows(plan: WeeklyPlan | undefined): AllocationRow[] {
  const allocation = asRecord(plan?.allocation);
  return SLEEVES.map((key) => {
    const sleeve = asRecord(allocation[key]);
    return {
      key,
      label: DISPLAY_SLEEVES[key] ?? SLEEVE_LABELS[key],
      actualPct: textValue(sleeve.current_pct, "0"),
      targetPct: textValue(sleeve.target_pct, "0"),
      actualAmount: textValue(sleeve.current, "0"),
      targetAmount: textValue(sleeve.target, "0"),
      gapAmount: textValue(sleeve.gap, "0"),
    };
  });
}

function safePercent(value: string): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? Math.min(100, Math.max(0, parsed)) : 0;
}

function amountMinorUnits(value: string): bigint {
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return 0n;
  const fraction = (match[3] ?? "").slice(0, 2).padEnd(2, "0");
  const amount = BigInt(match[2] ?? "0") * 100n + BigInt(fraction || "0");
  return match[1] === "-" ? -amount : amount;
}

function amountFromMinorUnits(value: bigint): string {
  const sign = value < 0n ? "-" : "";
  const magnitude = value < 0n ? -value : value;
  return `${sign}${magnitude / 100n}.${(magnitude % 100n)
    .toString()
    .padStart(2, "0")}`;
}

function percentageParts(value: string): { value: bigint; places: number } {
  const match = /^(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return { value: 0n, places: 0 };
  const fraction = match[2] ?? "";
  return {
    value: BigInt(`${match[1] ?? "0"}${fraction}`),
    places: fraction.length,
  };
}

function isAtLeastFivePointsBelowPlan(row: AllocationRow): boolean {
  const actual = percentageParts(row.actualPct);
  const target = percentageParts(row.targetPct);
  const places = Math.max(actual.places, target.places);
  const scale = 10n ** BigInt(places);
  const difference =
    target.value * 10n ** BigInt(places - target.places) -
    actual.value * 10n ** BigInt(places - actual.places);
  return difference >= 5n * scale;
}

function largestPlanGap(rows: AllocationRow[]): PlanGap | null {
  let largest: { row: AllocationRow; amount: string; minor: bigint } | null =
    null;
  for (const row of rows) {
    if (!PLAN_GAP_KEYS.has(row.key as PlanGapKey)) continue;
    if (!isAtLeastFivePointsBelowPlan(row)) continue;
    const minor =
      amountMinorUnits(row.targetAmount) - amountMinorUnits(row.actualAmount);
    if (minor > 0n && (!largest || minor > largest.minor)) {
      largest = { row, amount: amountFromMinorUnits(minor), minor };
    }
  }
  return largest;
}

function addPathForPlanGap(key: string): string {
  if (key === "equity_india") return "/import";
  const category = {
    equity_global: "us_investments",
    debt: "fd",
    gold: "gold",
    cash: "savings",
  }[key];
  return category ? addPathForCategory(category) : "/money";
}

function AllocationCard({
  rows,
  empty,
}: {
  rows: AllocationRow[];
  empty: boolean;
}) {
  const { fxRate, primaryCurrency } = useHousehold();
  if (empty) {
    return (
      <EmptyState
        title={englishCopy.home.plan_empty}
        icon={<WalletCards aria-hidden="true" className="size-5" />}
        action={
          <Link
            to="/money?add=1"
            data-primary="true"
            className="motion-safe:active:scale-[.98] inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground transition"
          >
            {englishCopy.money_ui.add_update}
          </Link>
        }
      />
    );
  }

  const colors = [
    "var(--chart-india)",
    "var(--chart-global)",
    "var(--chart-debt)",
    "var(--chart-gold)",
    "var(--chart-cash)",
  ];
  const gap = largestPlanGap(rows);
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">Your money vs your plan</h2>
        <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
          <Term
            explanation="Your money mix compares each kind of holding with its target."
            prompt="What's this?"
            learnMoreHref="/help#term-money-mix"
          >
            Money mix
          </Term>
        </div>
      </div>
      <div className="grid gap-2">
        {[
          { label: englishCopy.plan_ui.now, field: "actualPct" as const },
          { label: englishCopy.plan_ui.plan, field: "targetPct" as const },
        ].map(({ label, field }) => (
          <div key={label} className="grid gap-1">
            <span className="text-xs font-medium text-muted-foreground">
              {label}
            </span>
            <div
              role="img"
              aria-label={`${label}: ${rows
                .map((row) => `${row.label} ${safePercent(row[field])}%`)
                .join(", ")}`}
              className="flex h-2 gap-0.5 overflow-hidden rounded-full bg-muted"
            >
              {rows.map((row, index) => (
                <span
                  key={row.key}
                  className="min-w-0 rounded-full"
                  title={row.label}
                  style={{
                    flexGrow: safePercent(row[field]),
                    flexBasis: 0,
                    backgroundColor: colors[index] ?? "var(--chart-debt)",
                  }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      <p className="text-sm">
        {gap
          ? englishCopy.plan_ui.gap_sentence.replace(
              "{name}",
              PLAN_GAP_COPY[gap.row.key as PlanGapKey].name,
            )
          : englishCopy.plan_ui.close_to_plan}
      </p>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
        {rows.map((row, index) => (
          <span key={row.key} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="size-2 rounded-full"
              style={{ backgroundColor: colors[index] ?? "var(--chart-debt)" }}
            />
            {row.label}
          </span>
        ))}
      </div>
      <details>
        <summary className="min-h-11 cursor-pointer py-2 text-sm font-medium text-brand">
          {englishCopy.plan_ui.compare_amounts}
        </summary>
        {gap ? (
          <p className="mt-3 text-sm">
            {englishCopy.plan_ui.gap_amount_sentence
              .replace("{name}", PLAN_GAP_COPY[gap.row.key as PlanGapKey].name)
              .replace(
                "{amount}",
                formatHouseholdAmount(gap.amount, primaryCurrency, fxRate),
              )}
          </p>
        ) : null}
        <dl className="mt-3 grid gap-2 text-sm">
          {rows.map((row) => (
            <div
              key={row.key}
              className="grid gap-1 border-t border-line pt-2 first:border-0 first:pt-0"
            >
              <dt className="font-medium">{row.label}</dt>
              <dd className="flex flex-wrap justify-between gap-2 text-muted-foreground">
                <span>
                  {englishCopy.plan_ui.now}:{" "}
                  <MoneyValue amountInr={row.actualAmount} />
                  {` (${row.actualPct}%)`}
                </span>
                <span>
                  {englishCopy.plan_ui.plan}:{" "}
                  <MoneyValue amountInr={row.targetAmount} />
                  {` (${row.targetPct}%)`}
                </span>
              </dd>
            </div>
          ))}
        </dl>
      </details>
    </Card>
  );
}
function foundationMap(
  plan: WeeklyPlan | undefined,
): Map<string, Record<string, unknown>> {
  return new Map(
    asArray(plan?.foundation).map((entry) => {
      const row = asRecord(entry);
      return [textValue(row.key), row];
    }),
  );
}

function FoundationCard({ plan }: { plan: WeeklyPlan }) {
  const items = foundationMap(plan);
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">Your financial foundation</h2>
      {FOUNDATION.map(([key, label]) => {
        const item = items.get(key) ?? {};
        const status = textValue(item.status, "unknown");
        const gap = textValue(item.gap);
        return (
          <div
            key={key}
            className="flex min-h-12 items-center justify-between gap-3 border-t border-line pt-2 first:border-0 first:pt-0"
          >
            <div>
              <p className="text-sm font-medium">{label}</p>
              <p className="text-xs text-muted-foreground">
                {status === "ok"
                  ? "On track"
                  : status === "gap"
                    ? "Needs a look"
                    : "Not enough data"}
              </p>
            </div>
            {gap && isPositiveAmount(gap) ? (
              <span className="text-sm tabular-nums">
                Gap <MoneyValue amountInr={gap} />
              </span>
            ) : null}
          </div>
        );
      })}
    </Card>
  );
}

function routeWarnings(plan: WeeklyPlan): string[] {
  const route = asRecord(plan.route);
  return asArray(route.warnings).filter(
    (warning): warning is string => typeof warning === "string",
  );
}

function GlobalCard({ plan }: { plan: WeeklyPlan }) {
  const route = asRecord(plan.route);
  const allocation = asRecord(plan.allocation);
  const global = asRecord(allocation.equity_global);
  const vehicle = textValue(route.vehicle, "");
  const warnings = routeWarnings(plan);
  const visibleWarnings = warnings.slice(0, 2);
  const reasons = asArray(route.reasons).filter(
    (reason): reason is string => typeof reason === "string",
  );
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">
          {englishCopy.household_ui.abroad_shares}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.household_ui.abroad_shares_description}
        </p>
      </div>
      <p className="text-sm">
        {englishCopy.household_ui.abroad_mix
          .replace("{target}", textValue(global.target_pct, "—"))
          .replace("{current}", textValue(global.current_pct, "—"))}
      </p>
      {vehicle ? (
        <p className="text-sm">
          {englishCopy.household_ui.global_option.replace(
            "{instrument}",
            instrumentLabel(vehicle),
          )}
        </p>
      ) : (
        <p className="text-sm text-muted-foreground">
          {englishCopy.household_ui.no_abroad_choice}
        </p>
      )}
      {reasons.length ? (
        <details>
          <summary className="min-h-11 cursor-pointer py-2 text-sm font-semibold">
            {englishCopy.household_ui.why_global_choice}
          </summary>
          <ul
            className="mt-2 grid gap-1 text-sm text-muted-foreground"
            aria-label={englishCopy.household_ui.global_choice_reasons}
          >
            {reasons.map((reason) => (
              <li key={reason}>{friendlyRouteReason(reason)}</li>
            ))}
          </ul>
        </details>
      ) : null}
      {visibleWarnings.length ? (
        <ul
          className="grid gap-2"
          aria-label={englishCopy.household_ui.global_warnings_label}
        >
          {visibleWarnings.map((warning) => (
            <li key={warning} className="rounded-xl bg-muted p-3 text-sm">
              {friendlyRouteWarning(warning)}
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}

function DebtAnswerCard({
  current,
  target,
}: {
  current: string;
  target: string;
}) {
  return (
    <Card className="grid gap-2">
      <h2 className="text-base font-semibold">
        <Term
          explanation="Savings meant for near-term needs and steady access."
          prompt="What's this?"
          learnMoreHref="/help#term-safe-money"
        >
          Safe money
        </Term>
      </h2>
      <dl className="grid gap-2 text-sm">
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">
            {englishCopy.plan_ui.safe_money_current}
          </dt>
          <dd className="text-right font-semibold tabular-nums">
            {current ? <MoneyValue amountInr={current} /> : "—"}
          </dd>
        </div>
        <div className="flex items-center justify-between gap-3">
          <dt className="text-muted-foreground">
            {englishCopy.plan_ui.safe_money_target}
          </dt>
          <dd className="text-right font-semibold tabular-nums">
            {target ? <MoneyValue amountInr={target} /> : "—"}
          </dd>
        </div>
      </dl>
    </Card>
  );
}

function SafeMoneyDetails({ plan }: { plan: WeeklyPlan }) {
  const debt = asRecord(plan.debt_answer);
  const plannedDebtAction = asArray(plan.actions)
    .map(asRecord)
    .find((action) => textValue(action.sleeve) === "debt");
  const debtPick = asRecord(debt.top_debt_pick ?? debt.top_pick);
  const unrankedOptions = asArray(plan.unranked_options)
    .map(asRecord)
    .filter((option) => textValue(option.reason) === "yield_unavailable");
  const destination = textValue(
    debtPick.label ??
      debtPick.instrument ??
      plannedDebtAction?.instrument_category,
  );
  const afterTaxYield = textValue(
    debtPick.after_tax_yield_pct ?? debtPick.after_tax_yield,
  );
  const timePeriod = textValue(debtPick.tenure ?? debtPick.tenure_years);
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">Safe money details</h2>
      <dl className="grid grid-cols-2 gap-3 text-sm">
        <div>
          <dt className="text-muted-foreground">Current amount</dt>
          <dd className="mt-1 font-semibold tabular-nums">
            {debt.current_debt ? (
              <MoneyValue amountInr={textValue(debt.current_debt)} />
            ) : (
              "—"
            )}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Target amount</dt>
          <dd className="mt-1 font-semibold tabular-nums">
            {debt.target_debt ? (
              <MoneyValue amountInr={textValue(debt.target_debt)} />
            ) : (
              "—"
            )}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Hard-to-reach money</dt>
          <dd className="mt-1 tabular-nums">
            {debt.locked_debt ? (
              <MoneyValue amountInr={textValue(debt.locked_debt)} />
            ) : (
              "—"
            )}
          </dd>
        </div>
        <div>
          <dt className="text-muted-foreground">Suggested option</dt>
          <dd className="mt-1">
            {destination ? instrumentLabel(destination) : "Not available yet"}
          </dd>
        </div>
      </dl>
      {unrankedOptions.length ? (
        <ul
          className="grid gap-1 border-t border-line pt-3 text-sm"
          aria-label="Options without a current rate"
        >
          {unrankedOptions.map((option) => {
            const instrument = textValue(option.instrument);
            return (
              <li key={instrument}>
                {instrumentLabel(instrument)}: Rate not available: check your
                bank.
              </li>
            );
          })}
        </ul>
      ) : null}
      {afterTaxYield || timePeriod ? (
        <p className="text-sm text-muted-foreground">
          {afterTaxYield ? `Estimated return after tax ${afterTaxYield}%` : ""}
          {afterTaxYield && timePeriod ? " · " : ""}
          {timePeriod ? `Time period ${timePeriod}` : ""}
        </p>
      ) : null}
    </Card>
  );
}
function policyDetails(asset: AssetResponse): Record<string, unknown> {
  return asRecord(asset.details);
}

function InsuranceCard({ assets }: { assets: AssetResponse[] }) {
  const policies = assets.filter(
    (asset) => asset.asset_type === "traditional_policy",
  );
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">Your LIC / insurance policies</h2>
      {policies.length ? (
        policies.map((policy) => {
          const details = policyDetails(policy);
          const review = asRecord(details.review ?? details.insurance_review);
          const label = textValue(
            details.label,
            textValue(details.plan, "Traditional policy"),
          );
          const irr = textValue(review.irr_pct ?? review.estimated_return_pct);
          const comparisons = asRecord(review.comparison);
          const ppfRate = textValue(
            asRecord(comparisons.ppf).benchmark_pct ?? details.ppf_rate_pct,
          );
          const fdRate = textValue(
            asRecord(comparisons.fd).benchmark_pct ?? details.fd_rate_pct,
          );
          const continueValue = textValue(review.continue_value);
          const paidUpValue = textValue(review.paid_up_value);
          const surrenderValue = textValue(
            details.surrender_value ?? review.surrender_value,
          );
          return (
            <div
              key={policy.id}
              className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <h3 className="text-sm font-semibold">{label}</h3>
              {irr ? (
                <p className="text-sm">
                  Estimated return {irr}% compared with PPF
                  {ppfRate ? ` at ${ppfRate}%` : ""} and fixed deposits
                  {fdRate ? ` at ${fdRate}%` : ""}.
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  Return comparison with PPF and fixed deposits is not available
                  yet.
                </p>
              )}
              <dl className="grid gap-1 text-sm text-muted-foreground">
                {continueValue ? (
                  <div className="flex justify-between gap-3">
                    <dt>Continue value</dt>
                    <dd className="tabular-nums">
                      <MoneyValue amountInr={continueValue} />
                    </dd>
                  </div>
                ) : null}
                {paidUpValue ? (
                  <div className="flex justify-between gap-3">
                    <dt>Paid-up value</dt>
                    <dd className="tabular-nums">
                      <MoneyValue amountInr={paidUpValue} />
                    </dd>
                  </div>
                ) : (
                  <div>Paid-up value is not available.</div>
                )}
                {surrenderValue ? (
                  <div className="flex justify-between gap-3">
                    <dt>Insurer’s surrender value</dt>
                    <dd className="tabular-nums">
                      <MoneyValue amountInr={surrenderValue} />
                    </dd>
                  </div>
                ) : (
                  <div>Ask your insurer for the current surrender value.</div>
                )}
              </dl>
            </div>
          );
        })
      ) : (
        <p className="text-sm text-muted-foreground">
          Add a traditional policy to compare its options.
        </p>
      )}
      <p className="rounded-xl bg-muted p-3 text-sm">
        Compare options before deciding. Confirm the surrender value with your
        insurer and check how each option affects cover, charges and tax.
      </p>
    </Card>
  );
}

function GoalsCard({ goals }: { goals: GoalResponse[] }) {
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">Your goals</h2>
      {goals.length ? (
        goals.map((goal) => {
          const extra = asRecord(goal);
          const progress = textValue(
            extra.progress_pct ?? extra.progress_percent,
          );
          return (
            <div
              key={goal.id}
              className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium">{goal.name}</span>
                <span className="text-xs text-muted-foreground">
                  By {goal.target_year}
                </span>
              </div>
              <div className="flex justify-between gap-3 text-sm text-muted-foreground">
                <span>
                  Target <MoneyValue amountInr={goal.target_amount} />
                </span>
                {progress ? <span>{progress}% complete</span> : null}
              </div>
              {progress ? (
                <Progress
                  value={safePercent(progress)}
                  label={`${goal.name} progress`}
                />
              ) : null}
            </div>
          );
        })
      ) : (
        <p className="text-sm text-muted-foreground">
          Add a goal to see its progress here.
        </p>
      )}
    </Card>
  );
}

export function PlanPage() {
  const household = useHousehold();
  const {
    activeMemberId,
    selectedHouseholdId,
    selectedMemberId,
    meMemberId,
    members,
    fxRate,
  } = household;
  const familyView = selectedMemberId === "family";
  const plan = useFeatureQuery(
    ["c27b", "plan", selectedHouseholdId, activeMemberId],
    "/v1/plan",
    getPlan,
  );
  const assets = useFeatureQuery(
    ["c27b", "assets", selectedHouseholdId, activeMemberId],
    "/v1/assets",
    () => listAssets(activeMemberId),
  );
  const goals = useFeatureQuery(
    ["c27b", "goals", selectedHouseholdId],
    "/v1/goals",
    listGoals,
  );
  const loans = useFeatureQuery(
    ["c30", "loans", selectedHouseholdId],
    "/v1/loans",
    listLoans,
  );
  const networth = useFeatureQuery(
    ["c27b", "networth", selectedHouseholdId, activeMemberId],
    "/v1/networth",
    () => getNetworth(activeMemberId),
  );
  const planValue = plan.data as WeeklyPlan | undefined;
  const rows = allocationRows(planValue);
  const gap = largestPlanGap(rows);
  const selectedPlanMember = planValue?.members?.find(
    (member) => member.id === activeMemberId,
  );
  const debtAnswer = asRecord(planValue?.debt_answer);
  const selectedName = selectedPlanMember
    ? memberDisplayName(
        selectedPlanMember.id,
        selectedPlanMember.name,
        meMemberId,
      )
    : members.find((member) => member.id === activeMemberId)?.display_name;
  // Statement imports land in holdings, not in the manual assets list.
  const noAssets = Boolean(
    assets.supported &&
    networth.supported &&
    loans.supported &&
    assets.data &&
    networth.data &&
    loans.data &&
    !assets.isPending &&
    !assets.error &&
    assets.data.length === 0 &&
    !networth.isPending &&
    !networth.error &&
    networth.data.holdings.length === 0 &&
    !loans.isPending &&
    !loans.error &&
    loans.data.loans.length === 0,
  );

  if (!plan.supported || isMissingFeature(plan.error))
    return <ComingSoon title="Your plan is coming soon" />;
  if (
    (plan.error && !plan.data && !isMissingFeature(plan.error)) ||
    (assets.error && !assets.data && !isMissingFeature(assets.error)) ||
    (networth.error && !networth.data && !isMissingFeature(networth.error))
  )
    return (
      <ScreenError
        title="Your plan couldn’t load"
        onRetry={() => {
          void Promise.all([
            plan.refetch(),
            assets.refetch(),
            networth.refetch(),
          ]);
        }}
      />
    );
  if (
    plan.isPending ||
    (assets.supported && assets.isPending) ||
    (networth.supported && networth.isPending)
  )
    return <ScreenLoading label={englishCopy.home.loading_plan} />;
  if (!planValue)
    return (
      <ScreenError
        title="Your plan couldn’t load"
        onRetry={() => void plan.refetch()}
      />
    );

  return (
    <div data-copy-screen="plan" className="grid gap-4">
      {!familyView ? (
        <h2 className="text-sm font-medium text-muted-foreground">
          Plan for {selectedName || englishCopy.household.family}
        </h2>
      ) : null}
      {!noAssets ? (
        <DebtAnswerCard
          current={textValue(debtAnswer.current_debt)}
          target={textValue(debtAnswer.target_debt)}
        />
      ) : null}
      <AllocationCard rows={rows} empty={noAssets} />
      {!noAssets ? (
        <div className="grid gap-2">
          <Link
            to={
              gap?.row.key === "debt"
                ? "/money?add=1"
                : gap
                  ? addPathForPlanGap(gap.row.key)
                  : "/money"
            }
            data-primary="true"
            className="motion-safe:active:scale-[.98] inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground transition"
          >
            {gap?.row.key === "debt"
              ? englishCopy.money_ui.add_update
              : gap
                ? PLAN_GAP_COPY[gap.row.key as PlanGapKey].button
                : englishCopy.plan_ui.update_money}
          </Link>
          {gap?.row.key === "debt" ? (
            <p className="text-sm text-muted-foreground">
              {englishCopy.plan_ui.add_existing_items}
            </p>
          ) : null}
        </div>
      ) : null}
      <details>
        <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
          Details
        </summary>
        <div className="grid gap-4 border-t border-line pt-3">
          <LoansPlanDetails summary={loans.data} />
          <SafeMoneyDetails plan={planValue} />
          <ExchangeRateNote />
          {!fxRate &&
          members.some((member) => member.residency.country === "DK") ? (
            <p className="text-xs text-muted-foreground">
              {englishCopy.household_ui.no_fx_rate}
            </p>
          ) : null}
          {familyView ? (
            <>
              <GlobalCard plan={planValue} />
              <FoundationCard plan={planValue} />
            </>
          ) : (
            <SelectedMemberCard
              member={selectedPlanMember}
              person={members.find((member) => member.id === activeMemberId)}
              networth={networth.data}
              meMemberId={meMemberId}
            />
          )}
          {familyView && planValue.members?.length ? (
            <MemberHoldingsCard
              members={members}
              planMembers={planValue.members}
              meMemberId={meMemberId}
            />
          ) : null}
          <PlacementNotesCard plan={planValue} />
          <section aria-labelledby="insurance-heading">
            <OptionalAssetsSection
              supported={assets.supported}
              loading={assets.isPending}
              error={assets.error}
              assets={assets.data}
            />
          </section>
          {familyView ? (
            <OptionalGoalsSection
              supported={goals.supported}
              loading={goals.isPending}
              error={goals.error}
              goals={goals.data}
            />
          ) : null}
          {networth.data?.completeness ? (
            <p className="text-xs text-muted-foreground">
              Money Map {networth.data.completeness.pct}% complete.
            </p>
          ) : null}
          <p className="text-xs text-muted-foreground">
            {englishCopy.disclaimers.personal}
          </p>
        </div>
      </details>
    </div>
  );
}

function OptionalAssetsSection({
  supported,
  loading,
  error,
  assets,
}: {
  supported: boolean;
  loading: boolean;
  error: unknown;
  assets?: AssetResponse[];
}) {
  if (!supported || isMissingFeature(error))
    return <ComingSoon title="Insurance details are coming soon" />;
  if (loading)
    return (
      <div
        className="h-36 animate-pulse rounded-2xl bg-muted"
        aria-label="Loading insurance details"
      />
    );
  if (error) return <ScreenError title="Insurance details couldn’t load" />;
  return <InsuranceCard assets={assets ?? []} />;
}

function OptionalGoalsSection({
  supported,
  loading,
  error,
  goals,
}: {
  supported: boolean;
  loading: boolean;
  error: unknown;
  goals?: GoalResponse[];
}) {
  if (!supported || isMissingFeature(error))
    return <ComingSoon title="Goals are coming soon" />;
  if (loading)
    return (
      <div
        className="h-28 animate-pulse rounded-2xl bg-muted"
        aria-label="Loading goals"
      />
    );
  if (error) return <ScreenError title="Goals couldn’t load" />;
  return <GoalsCard goals={goals ?? []} />;
}

export { allocationRows, friendlyRouteWarning, labelForCode };
