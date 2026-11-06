import { useEffect, useState, type FormEvent } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CalendarDays,
  CheckCircle2,
  ChevronRight,
  MoveRight,
} from "lucide-react";
import { Link, useLocation } from "react-router";
import { ApiError } from "../api/client";
import { refreshActionQueries } from "../api/query-client";
import { getNetworth } from "../money/api";
import { roundActionAmount } from "../lib/money";
import {
  actionLabel,
  exitLabel,
  labelForCode,
  triggerKeyLabel,
} from "../lib/labels";
import {
  asArray,
  asRecord,
  instrumentLabel,
  isPositiveAmount,
  reasonCopy,
  routeHowTo,
  textValue,
} from "../features/data";
import {
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import {
  dismissHighlight,
  getActions,
  getPlan,
  recordBought,
  updateActionStatus,
} from "../planning/api";
import type {
  ActionResponse,
  RecordBuyRequest,
  WeeklyPlan,
} from "../planning/api";
import { Button, buttonClassName, Card, Sheet } from "../ui";
import { Highlights } from "../features/Highlights";
import * as englishCopy from "../copy/sections/index";
import { useHousehold } from "../household/HouseholdContext";
import { MoneyValue } from "../household/currency-view";
import { formatHouseholdAmount } from "../household/currency";
import { dkkToInr, inrToDkk } from "../household/fx";
import type { DkkInrRate } from "../household/fx";
import { listAssets, listLoans } from "../money/api";
import { quantityLabel } from "../money/quantity";
import type { MemberResponse } from "../money/api";
import { getInvestingPrefs, saveInvestingPrefs } from "../settings/api";
import type { InvestingPrefs, InvestingPrefsInput } from "../settings/api";

function payloadFor(action: ActionResponse): Record<string, unknown> {
  const payload = asRecord(action.payload);
  const nested = asRecord(payload.action);
  return Object.keys(nested).length ? nested : payload;
}

function actionTitle(action: ActionResponse): string {
  const payload = payloadFor(action);
  const kind = textValue(payload.kind, action.kind);
  const exitAction: Record<string, string> = {
    SELL_EXIT: "EXIT",
    SELL_TRIM: "TRIM_TO_TARGET",
    REVIEW_EXIT: "REVIEW_THEN_EXIT",
  };
  const tranche = asRecord(payload.tranche);
  if (kind === "BUY_STOCK_TRANCHE") {
    const part = textValue(tranche.n, "1");
    const of = textValue(tranche.of, "3");
    return `${actionLabel(kind)} (part ${part} of ${of})`;
  }
  if (exitAction[kind]) return exitLabel(exitAction[kind] ?? kind);
  return actionLabel(kind);
}

function actionCategory(action: ActionResponse): string {
  const category = textValue(payloadFor(action).instrument_category);
  return category ? instrumentLabel(category) : "";
}

function rawActionAmount(action: ActionResponse): string {
  return textValue(payloadFor(action).amount, "0.00");
}

function actionAmount(action: ActionResponse): string {
  return roundActionAmount(rawActionAmount(action));
}

function isBuyAction(action: ActionResponse): boolean {
  return ["BUY_LUMPSUM", "BUY_STP", "BUY_ABROAD", "BUY_STOCK_TRANCHE"].includes(
    textValue(payloadFor(action).kind, action.kind),
  );
}

function isSellAction(action: ActionResponse): boolean {
  return ["SELL_EXIT", "SELL_TRIM"].includes(
    textValue(payloadFor(action).kind, action.kind),
  );
}

function isReviewExitAction(action: ActionResponse): boolean {
  return textValue(payloadFor(action).kind, action.kind) === "REVIEW_EXIT";
}

function sellStepName(
  action: ActionResponse,
  holdings: NonNullable<Awaited<ReturnType<typeof getNetworth>>["holdings"]>,
): string {
  const payload = payloadFor(action);
  const params = asRecord(payload.params);
  const securityId = textValue(
    payload.security_id ?? params.security_id ?? asRecord(action).security_id,
  );
  const holdingName = securityId
    ? textValue(
        holdings.find((holding) => holding.security_id === securityId)?.name,
      )
    : "";
  return holdingName || textValue(payload.symbol ?? params.symbol);
}

function buyStepName(action: ActionResponse): string {
  const payload = payloadFor(action);
  if (textValue(payload.kind, action.kind) === "BUY_STOCK_TRANCHE") {
    const params = asRecord(payload.params);
    return textValue(
      payload.name ??
        payload.security_name ??
        params.name ??
        params.symbol ??
        payload.symbol,
      actionCategory(action) || actionTitle(action),
    );
  }
  return actionCategory(action) || actionTitle(action);
}

function todayForDateInput(): string {
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60_000);
  return local.toISOString().slice(0, 10);
}

type BuyLineDraft = {
  line: number;
  title: string;
  accountLabel?: string;
  amount: string;
  currency: string;
  units: string;
  checked: boolean;
  hasUnitsField: boolean;
};

function buyLineDrafts(action: ActionResponse): BuyLineDraft[] {
  const payload = payloadFor(action);
  const params = asRecord(payload.params);
  if (textValue(payload.kind, action.kind) === "BUY_ABROAD") {
    return asArray(params.abroad_buy_list).map((rawLine, line) => {
      const item = asRecord(rawLine);
      const instrument = asRecord(item.instrument);
      const hasDkkAmount = item.amount_dkk !== undefined;
      const reasons = asArray(item.reason_codes);
      const isStock =
        reasons.includes("approved_us_stock_idea") ||
        textValue(instrument.type).toLowerCase() === "equity";
      const account = textValue(item.account).toLowerCase();
      const accountLabel =
        account === "ask"
          ? englishCopy.denmark.account_ask
          : account === "ordinary"
            ? englishCopy.denmark.account_ordinary
            : "";
      const instrumentName = textValue(
        instrument.name ?? instrument.ticker ?? instrument.symbol,
        instrumentLabel(textValue(item.instrument_category)),
      );
      return {
        line,
        title: instrumentName,
        accountLabel: accountLabel || undefined,
        amount: textValue(item.amount_dkk ?? item.amount ?? item.amount_inr),
        currency: hasDkkAmount
          ? "DKK"
          : textValue(item.currency, "DKK").toUpperCase(),
        units: "",
        checked: true,
        hasUnitsField: !isStock,
      };
    });
  }
  return [
    {
      line: 0,
      title: buyStepName(action),
      amount: rawActionAmount(action),
      currency: textValue(payload.currency, "INR").toUpperCase(),
      units: textValue(payload.quantity ?? params.quantity),
      checked: true,
      hasUnitsField:
        textValue(payload.kind, action.kind) !== "BUY_STOCK_TRANCHE",
    },
  ];
}

function isPositiveDecimalText(value: string): boolean {
  const trimmed = value.trim();
  return /^\d+(?:\.\d+)?$/.test(trimmed) && /[1-9]/.test(trimmed);
}

function BuyRecordSheet({
  action,
  saving,
  error,
  onClose,
  onSubmit,
}: {
  action: ActionResponse;
  saving: boolean;
  error: boolean;
  onClose(): void;
  onSubmit(payload: RecordBuyRequest): void;
}) {
  const [lines, setLines] = useState(() => buyLineDrafts(action));
  const [boughtOn, setBoughtOn] = useState(todayForDateInput);
  const [message, setMessage] = useState("");
  const isStockTranche =
    textValue(payloadFor(action).kind, action.kind) === "BUY_STOCK_TRANCHE";

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const selected = lines.filter((line) => line.checked);
    if (!selected.length) {
      setMessage(englishCopy.home.buy_choose_one);
      return;
    }
    if (
      selected.some(
        (line) =>
          !isPositiveDecimalText(line.amount) ||
          (line.units.length > 0 && !isPositiveDecimalText(line.units)),
      )
    ) {
      setMessage(englishCopy.home.buy_amount_error);
      return;
    }
    onSubmit({
      bought_on: boughtOn || undefined,
      lines: selected.map((line) => ({
        line: line.line,
        amount: line.amount.trim(),
        currency: line.currency,
        units: line.units.trim() || undefined,
      })),
    });
  }

  return (
    <Sheet
      open
      onOpenChange={(open) => !open && onClose()}
      title={englishCopy.home.buy_sheet_title}
    >
      <form className="grid gap-4" onSubmit={submit}>
        <p className="text-sm text-muted-foreground">
          {lines.length > 1
            ? englishCopy.home.buy_sheet_multiple_description
            : englishCopy.home.buy_sheet_description}
        </p>
        {lines.map((line, index) => (
          <fieldset
            key={line.line}
            aria-label={
              line.accountLabel
                ? `${line.title}, ${line.accountLabel}`
                : line.title
            }
            data-testid={`buy-row-${line.line}`}
            className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
          >
            <label className="flex min-w-0 items-start gap-3 text-sm font-medium">
              {lines.length > 1 ? (
                <>
                  <input
                    type="checkbox"
                    checked={line.checked}
                    onChange={(event) => {
                      const checked = event.currentTarget.checked;
                      setLines((current) =>
                        current.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, checked } : entry,
                        ),
                      );
                    }}
                  />
                  <span className="shrink-0">
                    {englishCopy.home.buy_bought}
                  </span>
                </>
              ) : null}
              <span className="min-w-0 flex-1">
                <span className="block line-clamp-2 [overflow-wrap:anywhere]">
                  {line.title}
                </span>
                {line.accountLabel ? (
                  <span className="mt-0.5 block text-xs font-normal text-muted-foreground">
                    {line.accountLabel}
                  </span>
                ) : null}
              </span>
            </label>
            <label className="grid gap-1 text-sm">
              <span>
                {englishCopy.home.buy_amount} ({line.currency})
              </span>
              <input
                type="text"
                inputMode="decimal"
                value={line.amount}
                onChange={(event) => {
                  const amount = event.currentTarget.value;
                  setLines((current) =>
                    current.map((entry, entryIndex) =>
                      entryIndex === index ? { ...entry, amount } : entry,
                    ),
                  );
                }}
                className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
                required={line.checked}
              />
            </label>
            {isStockTranche && line.units ? (
              <p className="text-sm text-muted-foreground">
                {englishCopy.home.buy_units_from_step.replace(
                  "{units}",
                  line.units,
                )}
              </p>
            ) : null}
          </fieldset>
        ))}
        <details>
          <summary className="cursor-pointer text-sm font-semibold">
            {englishCopy.home.buy_more_details}
          </summary>
          <div className="grid gap-3 pt-3">
            {lines.map((line, index) =>
              line.hasUnitsField ? (
                <label key={line.line} className="grid gap-1 text-sm">
                  <span>{englishCopy.home.buy_units_optional}</span>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={line.units}
                    onChange={(event) => {
                      const units = event.currentTarget.value;
                      setLines((current) =>
                        current.map((entry, entryIndex) =>
                          entryIndex === index ? { ...entry, units } : entry,
                        ),
                      );
                    }}
                    className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
                  />
                </label>
              ) : null,
            )}
            <label className="grid gap-1 text-sm">
              <span>{englishCopy.home.buy_date_optional}</span>
              <input
                type="date"
                value={boughtOn}
                onChange={(event) => setBoughtOn(event.currentTarget.value)}
                className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base"
              />
            </label>
          </div>
        </details>
        {message || error ? (
          <p role="alert" className="text-sm text-danger-foreground">
            {message || englishCopy.home.buy_save_error}
          </p>
        ) : null}
        <div className="flex flex-wrap gap-2">
          <Button type="submit" data-primary="true" loading={saving}>
            {englishCopy.home.buy_save}
          </Button>
          <Button
            type="button"
            variant="ghost"
            onClick={onClose}
            disabled={saving}
          >
            {englishCopy.home.buy_cancel}
          </Button>
        </div>
      </form>
    </Sheet>
  );
}

function sellDetails(action: ActionResponse): {
  quantity: string;
  tax: string;
} {
  const payload = payloadFor(action);
  const taxEstimate = asRecord(payload.tax_estimate);
  const tax = textValue(
    taxEstimate.estimated_tax ?? taxEstimate.tax_due ?? payload.estimated_tax,
  );
  const rawQuantity = textValue(payload.quantity ?? payload.quantity_to_sell);
  return {
    quantity: rawQuantity ? quantityLabel(rawQuantity, "equity") : "",
    tax: tax && isPositiveAmount(tax) ? tax : "",
  };
}

function actionMemberId(action: ActionResponse): string {
  return textValue(asRecord(action).member_id ?? payloadFor(action).member_id);
}

function memberForAction(
  action: ActionResponse,
  meMemberId: string | undefined,
  members: MemberResponse[],
  plan: WeeklyPlan | undefined,
): { id: string; name: string; isYou: boolean } | undefined {
  const id = actionMemberId(action);
  if (!id) return undefined;
  const payload = payloadFor(action);
  const planMember = plan?.members?.find((member) => member.id === id);
  const member = members.find((candidate) => candidate.id === id);
  const apiName = textValue(
    asRecord(action).member_name ?? payload.member_name,
  );
  const isYou = id === meMemberId;
  return {
    id,
    name: isYou
      ? englishCopy.household.you
      : apiName ||
        member?.display_name ||
        planMember?.name ||
        englishCopy.household.family,
    isYou,
  };
}

function actionHowTo(
  action: ActionResponse,
  person: ReturnType<typeof memberForAction>,
  members: MemberResponse[],
  plan: WeeklyPlan | undefined,
  primaryCurrency: "INR" | "DKK",
  fxRate: DkkInrRate | undefined,
): string {
  const payload = payloadFor(action);
  const params = asRecord(payload.params);
  const kind = textValue(payload.kind, action.kind);
  if (isSellAction(action)) return englishCopy.home.sell_steps;
  if (isReviewExitAction(action)) return englishCopy.home.review_steps;
  const member = person
    ? members.find((candidate) => candidate.id === person.id)
    : undefined;
  if (kind === "PAY_EXTRA_ON_LOAN") {
    return textValue(
      asRecord(payload.params).advice,
      "Use this amount for an extra loan payment.",
    );
  }
  if (!isBuyAction(action)) return englishCopy.home.generic_steps;
  const accountType = textValue(params.account_type ?? payload.account_type);
  const rawAmount = rawActionAmount(action);
  const amount = rawAmount
    ? formatHouseholdAmount(rawAmount, primaryCurrency, fxRate, false)
    : englishCopy.household_ui.suggested_amount;
  if (person && accountType === "ASK") {
    return englishCopy.household_ui.howto_ask.replace("{amount}", amount);
  }
  if (person && accountType === "ORDINARY") {
    return englishCopy.household_ui.howto_ordinary.replace("{amount}", amount);
  }
  if (
    member?.residency.country === "IN" &&
    /mf|mutual|index/i.test(textValue(payload.instrument_category))
  ) {
    return `${englishCopy.household_ui.action_for.replace("{name}", person?.name ?? "")}: ${englishCopy.household_ui.howto_india_mf}`;
  }
  if (person) {
    return englishCopy.household_ui.howto_member.replace("{amount}", amount);
  }
  const route = textValue(
    payload.route,
    textValue(asRecord(plan?.route).route, "A"),
  );
  const base = routeHowTo(
    route,
    rawAmount
      ? formatHouseholdAmount(rawAmount, primaryCurrency, fxRate, false)
      : "",
  );
  return base || englishCopy.home.generic_steps;
}

function exitCardForAction(
  action: ActionResponse,
  plan: WeeklyPlan | undefined,
): Record<string, unknown> | undefined {
  const payload = payloadFor(action);
  const params = asRecord(payload.params);
  const securityId = textValue(
    payload.security_id ?? params.security_id ?? asRecord(action).security_id,
  );
  const symbol = textValue(payload.symbol ?? params.symbol);
  const cards = asArray(plan?.exit_cards).map(asRecord);
  if (securityId)
    return cards.find((card) => textValue(card.security_id) === securityId);
  if (symbol) return cards.find((card) => textValue(card.symbol) === symbol);
  return undefined;
}

function TriggeredExitRows({
  action,
  plan,
}: {
  action: ActionResponse;
  plan: WeeklyPlan | undefined;
}) {
  const card = exitCardForAction(action, plan);
  const triggers = asRecord(card?.triggers ?? card?.trigger_statuses);
  const rows = Object.entries(triggers).flatMap(([key, value]) => {
    const trigger = asRecord(value);
    if (textValue(trigger.status) !== "triggered") return [];
    return [
      {
        key,
        value: textValue(trigger.value),
        threshold: textValue(trigger.threshold),
      },
    ];
  });
  if (!rows.length) return null;
  return (
    <div>
      <h3 className="text-sm font-semibold">
        {englishCopy.home.triggered_rows}
      </h3>
      <dl className="mt-2 grid gap-2">
        {rows.map((row) => (
          <div
            key={row.key}
            className="flex justify-between gap-4 border-t border-line pt-2 text-sm"
          >
            <dt>{triggerKeyLabel(row.key)}</dt>
            <dd className="text-right">
              {labelForCode("triggered")} · {row.value}
              {row.threshold ? ` · ${row.threshold}` : ""}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function activeActions(
  actions: ActionResponse[] | undefined,
): ActionResponse[] {
  return (actions ?? [])
    .filter((action) => action.status === "proposed")
    .sort((left, right) => left.rank - right.rank)
    .slice(0, 3);
}

function upcomingMaturities(
  networth: Awaited<ReturnType<typeof getNetworth>> | undefined,
) {
  const today = new Date();
  const end = new Date(today);
  end.setDate(end.getDate() + 30);
  return (networth?.maturity_calendar ?? [])
    .filter((event) => {
      const date = new Date(`${event.date}T00:00:00`);
      return date >= new Date(today.toDateString()) && date <= end;
    })
    .sort((left, right) => left.date.localeCompare(right.date));
}

type InvestingAmountCurrency = "INR" | "DKK";

function amountEntryCurrency(
  isDanishMember: boolean,
  fxRate: DkkInrRate | undefined,
): InvestingAmountCurrency {
  return isDanishMember && fxRate && inrToDkk("0.01", fxRate) ? "DKK" : "INR";
}

function amountInEntryCurrency(
  amountInr: string,
  currency: InvestingAmountCurrency,
  fxRate: DkkInrRate | undefined,
): string {
  if (currency === "DKK" && fxRate) return inrToDkk(amountInr, fxRate) ?? "";
  return amountInr;
}

function amountInRupees(
  amount: string,
  currency: InvestingAmountCurrency,
  fxRate: DkkInrRate | undefined,
): string | null {
  if (currency === "DKK") return fxRate ? dkkToInr(amount, fxRate) : null;
  return amount;
}

function investingPrefsInput(
  current: InvestingPrefs | undefined,
  amountInr: string,
  mode: "monthly" | "weekly",
): InvestingPrefsInput {
  return {
    monthly_sip_capacity:
      mode === "monthly" ? amountInr : (current?.monthly_sip_capacity ?? "0"),
    existing_sips: current?.existing_sips ?? [],
    new_money_this_week:
      mode === "weekly" ? amountInr : (current?.new_money_this_week ?? "0"),
    has_lrs_broker: current?.has_lrs_broker ?? false,
    prefers_no_paperwork: current?.prefers_no_paperwork ?? false,
    satellite_pct: current?.satellite_pct ?? "20",
    birth_year: current?.birth_year ?? null,
    ask_used_this_fy_dkk: current?.ask_used_this_fy_dkk ?? "0",
    ask_value_start_of_year_dkk: current?.ask_value_start_of_year_dkk ?? null,
    lrs_used_this_fy: current?.lrs_used_this_fy ?? "0",
    broker: current?.broker ?? "nordnet",
    has_indian_nri_share_account:
      current?.has_indian_nri_share_account ?? false,
    has_danish_pension_account: current?.has_danish_pension_account ?? false,
    us_stock_picks_opt_in: current?.us_stock_picks_opt_in ?? false,
    us_stock_picks_pct: current?.us_stock_picks_pct ?? "10",
  };
}

function ActionCard({
  action,
  person,
  holdingName,
  readOnly,
  onOpen,
  onDone,
  saving,
  primary = false,
}: {
  action: ActionResponse;
  person?: ReturnType<typeof memberForAction>;
  holdingName?: string;
  readOnly: boolean;
  onOpen(action: ActionResponse): void;
  onDone(action: ActionResponse): void;
  saving: boolean;
  primary?: boolean;
}) {
  const amount = actionAmount(action);
  const actionOwner = person
    ? person.isYou
      ? englishCopy.household_ui.action_for_you
      : englishCopy.household_ui.action_for.replace("{name}", person.name)
    : "";
  const cardLabel = [
    actionTitle(action),
    holdingName,
    actionCategory(action),
    actionOwner,
    action.status === "done" ? englishCopy.household_ui.completed : "",
  ]
    .filter(Boolean)
    .join(", ");
  return (
    <article
      className="grid gap-3 border-t border-line py-3 first:border-0 first:pt-0"
      aria-label={cardLabel}
      data-testid={`action-${action.kind}`}
    >
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
          <MoveRight aria-hidden="true" className="size-4" />
        </span>
        <div className="flex min-w-0 flex-1 items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-base font-semibold">{actionTitle(action)}</h3>
            {holdingName ? (
              <p className="mt-1 break-words text-sm font-medium [overflow-wrap:anywhere]">
                {holdingName}
              </p>
            ) : null}
            {actionCategory(action) ? (
              <p className="mt-1 text-xs text-muted-foreground">
                {actionCategory(action)}
              </p>
            ) : null}
            {actionOwner ? (
              <p className="mt-1 text-sm text-muted-foreground">
                {actionOwner}
              </p>
            ) : null}
          </div>
          {amount ? (
            <span className="shrink-0 text-sm font-semibold tabular-nums">
              <MoneyValue amountInr={amount} />
            </span>
          ) : null}
        </div>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          variant={primary ? undefined : "secondary"}
          data-primary={primary ? "true" : undefined}
          onClick={() => onOpen(action)}
        >
          {englishCopy.home.show_steps}
        </Button>
        <Button
          variant="secondary"
          aria-label={
            action.status === "done"
              ? englishCopy.household_ui.completed
              : isBuyAction(action)
                ? englishCopy.home.bought_it
                : isSellAction(action)
                  ? englishCopy.home.sold_it
                  : "Done"
          }
          onClick={() => onDone(action)}
          loading={saving}
          disabled={readOnly || action.status !== "proposed"}
        >
          {readOnly
            ? englishCopy.household_ui.view_only
            : action.status === "done"
              ? englishCopy.household_ui.completed
              : isBuyAction(action)
                ? englishCopy.home.bought_it
                : isSellAction(action)
                  ? englishCopy.home.sold_it
                  : "Done"}
        </Button>
      </div>
    </article>
  );
}

export function HomePage() {
  const location = useLocation();
  const goalSavedNotice =
    (location.state as { goalSaved?: unknown } | null)?.goalSaved === true;
  const [sheetAction, setSheetAction] = useState<ActionResponse | null>(null);
  const [recordAction, setRecordAction] = useState<ActionResponse | null>(null);
  const [amountMode, setAmountMode] = useState<"monthly" | "weekly" | null>(
    null,
  );
  const [amountDraft, setAmountDraft] = useState("");
  const [amountMessage, setAmountMessage] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const queryClient = useQueryClient();
  const {
    activeMemberId,
    selectedHouseholdId,
    meMemberId,
    members,
    isViewer,
    primaryCurrency,
    fxRate,
  } = useHousehold();
  const isDanishSignedInMember = members.some(
    (member) => member.id === meMemberId && member.residency.country === "DK",
  );
  const preferredAmountCurrency = amountEntryCurrency(
    isDanishSignedInMember,
    fxRate,
  );
  const [amountCurrency, setAmountCurrency] = useState<InvestingAmountCurrency>(
    preferredAmountCurrency,
  );
  useEffect(() => {
    setAmountCurrency(preferredAmountCurrency);
  }, [preferredAmountCurrency]);
  const canUseDkk = Boolean(fxRate && inrToDkk("0.01", fxRate));

  function changeAmountCurrency(next: InvestingAmountCurrency) {
    if (next === amountCurrency || (next === "DKK" && !canUseDkk)) return;
    if (!fxRate) return;
    setAmountDraft((current) => {
      if (!current) return current;
      return next === "DKK"
        ? (inrToDkk(current, fxRate) ?? current)
        : (dkkToInr(current, fxRate) ?? current);
    });
    setAmountCurrency(next);
    setAmountMessage("");
  }
  const networth = useFeatureQuery(
    ["c27b", "networth", selectedHouseholdId, activeMemberId],
    "/v1/networth",
    () => getNetworth(activeMemberId),
  );
  const assets = useFeatureQuery(
    ["c27b", "assets", selectedHouseholdId, activeMemberId],
    "/v1/assets",
    () => listAssets(activeMemberId),
  );
  const loans = useFeatureQuery(
    ["c30", "loans", selectedHouseholdId],
    "/v1/loans",
    listLoans,
  );
  const plan = useFeatureQuery(
    ["c27b", "plan", selectedHouseholdId, activeMemberId],
    "/v1/plan",
    getPlan,
  );
  const actions = useFeatureQuery(
    ["c27b", "actions", selectedHouseholdId],
    "/v1/actions",
    getActions,
  );
  async function refreshAfterMissingAction(error: unknown): Promise<void> {
    if (!(error instanceof ApiError) || error.status !== 404) return;
    setRecordAction(null);
    setActionMessage(englishCopy.home.step_updated);
    await refreshActionQueries(queryClient);
  }
  const investing = useFeatureQuery(
    ["settings", "investing"],
    "/v1/me/investing",
    getInvestingPrefs,
  );
  const markDone = useMutation({
    mutationFn: (action: ActionResponse) =>
      updateActionStatus(action.id, "done"),
    onSuccess: (updated) => {
      queryClient.setQueryData<ActionResponse[]>(
        ["c27b", "actions", selectedHouseholdId],
        (current) =>
          (current ?? []).map((action) =>
            action.id === updated.id ? updated : action,
          ),
      );
    },
    onError: refreshAfterMissingAction,
  });
  const recordBuy = useMutation({
    mutationFn: ({
      action,
      payload,
    }: {
      action: ActionResponse;
      payload: RecordBuyRequest;
    }) => recordBought(action.id, payload),
    onSuccess: async (result) => {
      queryClient.setQueryData<ActionResponse[]>(
        ["c27b", "actions", selectedHouseholdId],
        (current) =>
          (current ?? []).map((action) =>
            action.id === result.action.id ? result.action : action,
          ),
      );
      setRecordAction(null);
      await queryClient.invalidateQueries({ queryKey: ["c27b"] });
    },
    onError: refreshAfterMissingAction,
  });
  const hideHighlight = useMutation({
    mutationFn: dismissHighlight,
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["c27b", "plan", selectedHouseholdId, activeMemberId],
      }),
  });
  const saveAmount = useMutation({
    mutationFn: ({
      amountInr,
      mode,
    }: {
      amountInr: string;
      mode: "monthly" | "weekly";
    }) =>
      saveInvestingPrefs(investingPrefsInput(investing.data, amountInr, mode)),
    onSuccess: async (saved) => {
      queryClient.setQueryData(["settings", "investing"], saved);
      setAmountMode(null);
      setAmountMessage(englishCopy.home.amount_saved);
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: ["c27b", "plan", selectedHouseholdId],
        }),
        queryClient.invalidateQueries({
          queryKey: ["c27b", "actions", selectedHouseholdId],
        }),
      ]);
    },
    onError: () => setAmountMessage(englishCopy.home.amount_error),
  });
  const planValue = plan.data as WeeklyPlan | undefined;
  const visibleActions = (actions.data ?? []).filter(
    (action) => !activeMemberId || actionMemberId(action) === activeMemberId,
  );
  const actionList = activeActions(visibleActions);
  const completedBuyActions = visibleActions
    .filter((action) => action.status === "done" && action.has_recorded_buys)
    .sort((left, right) => left.rank - right.rank)
    .slice(0, 1);
  const nextMaturity = upcomingMaturities(networth.data)[0];
  const netWorthPrimary = networth.data
    ? formatHouseholdAmount(networth.data.net_worth, primaryCurrency, fxRate)
    : null;
  const monthlyAmountSet = isPositiveAmount(
    investing.data?.monthly_sip_capacity,
  );
  const editorMode =
    amountMode ??
    (!monthlyAmountSet || (actionList.length > 0 && !isViewer)
      ? "monthly"
      : null);
  const showMonthlyDisclosure =
    actionList.length > 0 &&
    monthlyAmountSet &&
    amountMode !== "weekly" &&
    !isViewer;
  const noRecordedItems = Boolean(
    networth.supported &&
    networth.data &&
    assets.supported &&
    assets.data &&
    loans.supported &&
    loans.data &&
    networth.data.holdings.length === 0 &&
    assets.data.length === 0 &&
    loans.data.loans.length === 0,
  );

  function renderAmountForm(mode: "monthly" | "weekly") {
    return (
      <form
        className="grid gap-3 border-t border-line pt-3"
        onSubmit={(event) => {
          event.preventDefault();
          const amount = amountDraft.trim();
          setAmountMessage("");
          if (!/^\d{1,12}(?:\.\d{1,2})?$/.test(amount)) {
            setAmountMessage(englishCopy.home.amount_error);
            return;
          }
          if (!isPositiveAmount(amount)) {
            setAmountMessage(englishCopy.home.amount_required);
            return;
          }
          const amountInr = amountInRupees(amount, amountCurrency, fxRate);
          if (amountInr === null) {
            setAmountMessage(englishCopy.home.amount_error);
            return;
          }
          saveAmount.mutate({ amountInr, mode });
        }}
      >
        {mode === "monthly" ? (
          <p className="text-sm font-medium">
            {englishCopy.home.amount_question}
          </p>
        ) : null}
        <div className="grid gap-1">
          <label
            className="text-sm font-medium"
            htmlFor="home-investing-amount"
          >
            {mode === "weekly"
              ? englishCopy.home.weekly_amount
              : englishCopy.home.monthly_amount}
          </label>
          <div className="flex items-center gap-2">
            <input
              id="home-investing-amount"
              type="number"
              min="0.01"
              step="0.01"
              inputMode="decimal"
              value={amountDraft}
              onChange={(event) => setAmountDraft(event.currentTarget.value)}
              className="min-h-12 w-0 min-w-0 flex-1 rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
            />
            <div
              role="radiogroup"
              aria-label={englishCopy.home.amount_currency}
              className="flex shrink-0 rounded-xl border border-line bg-surface p-0.5"
            >
              <button
                type="button"
                role="radio"
                aria-checked={amountCurrency === "DKK"}
                disabled={!canUseDkk}
                onClick={() => changeAmountCurrency("DKK")}
                className={`min-h-11 min-w-11 rounded-lg px-2 text-sm font-medium disabled:cursor-not-allowed disabled:opacity-40 ${amountCurrency === "DKK" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
              >
                kr
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={amountCurrency === "INR"}
                onClick={() => changeAmountCurrency("INR")}
                className={`min-h-11 min-w-11 rounded-lg px-2 text-sm font-medium ${amountCurrency === "INR" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
              >
                ₹
              </button>
            </div>
          </div>
          {!canUseDkk ? (
            <p className="text-xs text-muted-foreground">
              {englishCopy.home.dkk_rate_missing}
            </p>
          ) : null}
        </div>
        <Button
          type="submit"
          loading={saveAmount.isPending}
          variant={
            actionList.length || noRecordedItems ? "secondary" : undefined
          }
          data-primary={
            actionList.length || noRecordedItems ? undefined : "true"
          }
        >
          {englishCopy.home.amount_save}
        </Button>
        {amountMessage ? (
          <p role="status" className="text-sm text-muted-foreground">
            {amountMessage}
          </p>
        ) : null}
      </form>
    );
  }

  if (
    (networth.supported && networth.isPending) ||
    (plan.supported && plan.isPending) ||
    (actions.supported && actions.isPending) ||
    (investing.supported && investing.isPending)
  ) {
    return <ScreenLoading label={englishCopy.home.loading_week} />;
  }
  if (
    (networth.error && !networth.data && !isMissingFeature(networth.error)) ||
    (plan.error && !plan.data && !isMissingFeature(plan.error)) ||
    (actions.error && !actions.data && !isMissingFeature(actions.error))
  ) {
    return (
      <ScreenError
        title="Your week couldn't load"
        onRetry={() => {
          void Promise.all([
            networth.refetch(),
            plan.refetch(),
            actions.refetch(),
            investing.refetch(),
          ]);
        }}
      />
    );
  }

  return (
    <div data-copy-screen="home" className="grid gap-4">
      {goalSavedNotice ? (
        <Card className="flex flex-wrap items-center justify-between gap-2 text-sm">
          <p>{englishCopy.household_ui.goal_saved}</p>
          <Link
            to="/onboarding/goals"
            state={{ returnTo: "/" }}
            className="inline-flex min-h-11 items-center font-medium text-brand"
          >
            {englishCopy.household_ui.add_another_goal}
          </Link>
        </Card>
      ) : null}
      <Card
        className="home-hero grid gap-3"
        aria-labelledby="this-week-heading"
      >
        <h2 id="this-week-heading" className="text-lg font-semibold">
          This week
        </h2>
        {actionList.length ? (
          <div className="grid gap-2">
            {actionList.map((action, index) => (
              <ActionCard
                key={action.id}
                action={action}
                holdingName={
                  isSellAction(action) || isReviewExitAction(action)
                    ? sellStepName(action, networth.data?.holdings ?? [])
                    : undefined
                }
                person={memberForAction(action, meMemberId, members, planValue)}
                readOnly={isViewer}
                saving={markDone.isPending || recordBuy.isPending}
                primary={index === 0 && !isViewer}
                onDone={(selected) => {
                  setActionMessage("");
                  if (isBuyAction(selected)) {
                    setRecordAction(selected);
                    recordBuy.reset();
                  } else {
                    markDone.mutate(selected);
                  }
                }}
                onOpen={setSheetAction}
              />
            ))}
            {isViewer ? (
              <Link
                to="/plan"
                data-primary="true"
                className={buttonClassName()}
              >
                View your plan
              </Link>
            ) : null}
          </div>
        ) : noRecordedItems && !isViewer ? (
          <Link
            to="/money?add=1"
            data-primary="true"
            className={buttonClassName()}
          >
            {englishCopy.money_ui.add_update}
          </Link>
        ) : monthlyAmountSet ? (
          <div className="grid justify-items-center gap-3 py-3 text-center">
            <span className="grid size-11 place-items-center rounded-full bg-success-soft text-success-foreground">
              <CheckCircle2 aria-hidden="true" className="size-5" />
            </span>
            <p className="text-base font-medium">
              {englishCopy.home.nothing_to_buy}
            </p>
          </div>
        ) : isViewer ? (
          <p className="text-sm text-muted-foreground">
            {englishCopy.home.amount_not_set}
          </p>
        ) : null}
        {completedBuyActions.map((action) => (
          <div
            key={action.id}
            role="status"
            className="grid gap-1 border-t border-line pt-3"
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-base font-semibold">{buyStepName(action)}</h3>
              <span className="text-sm font-medium">
                {englishCopy.home.buy_done}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              {englishCopy.home.buy_saved}
            </p>
          </div>
        ))}
        {showMonthlyDisclosure ? (
          <details
            onToggle={(event) =>
              setAmountMode(event.currentTarget.open ? "monthly" : null)
            }
          >
            <summary className="min-h-12 cursor-pointer py-2 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand">
              {englishCopy.home.monthly_investment}
            </summary>
            {amountMode === "monthly" ? renderAmountForm("monthly") : null}
          </details>
        ) : editorMode ? (
          renderAmountForm(editorMode)
        ) : null}
        {amountMessage && !editorMode ? (
          <p role="status" className="text-sm text-muted-foreground">
            {amountMessage}
          </p>
        ) : null}
        {actionMessage ? (
          <p role="status" className="text-sm text-muted-foreground">
            {actionMessage}
          </p>
        ) : null}
        {markDone.isError &&
        !(
          markDone.error instanceof ApiError && markDone.error.status === 404
        ) ? (
          <p role="alert" className="text-sm text-muted-foreground">
            We couldn't mark this done. Try again.
          </p>
        ) : null}
        {!isViewer ? (
          <Button
            type="button"
            variant="ghost"
            className="min-h-11 w-fit justify-start px-0 py-2 text-sm font-medium text-brand hover:bg-transparent hover:underline"
            onClick={() => {
              setAmountDraft(
                amountInEntryCurrency(
                  investing.data?.new_money_this_week ?? "",
                  amountCurrency,
                  fxRate,
                ),
              );
              setAmountMessage("");
              setAmountMode("weekly");
            }}
          >
            {englishCopy.home.extra_money_link}
          </Button>
        ) : null}
        <details className="border-t border-line pt-2">
          <summary className="min-h-11 cursor-pointer py-2 text-sm font-medium">
            Make it more accurate
          </summary>
          <div className="grid gap-2 pb-2">
            <p className="text-sm text-muted-foreground">
              {englishCopy.home.more_accurate_description}
            </p>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/risk"
              state={{ returnTo: "/" }}
            >
              Choose your comfort with ups and downs
            </Link>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/goals"
              state={{ returnTo: "/" }}
            >
              Add a goal
            </Link>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/consent"
              state={{ returnTo: "/" }}
            >
              Choose email updates
            </Link>
          </div>
        </details>
      </Card>

      {noRecordedItems ? (
        <Link
          to="/money"
          className="flex min-h-12 items-center justify-between gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-sm font-medium"
        >
          <span>{englishCopy.home.no_holdings}</span>
          <ChevronRight
            aria-hidden="true"
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Link>
      ) : netWorthPrimary ? (
        <Link
          to="/money"
          className="flex min-h-12 items-center justify-between gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-sm font-medium"
        >
          <span>
            {englishCopy.home.money_summary.replace(
              "{amount}",
              netWorthPrimary,
            )}
          </span>
          <ChevronRight
            aria-hidden="true"
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Link>
      ) : (
        <Link
          to="/money"
          className="flex min-h-12 items-center justify-between gap-3 rounded-xl border border-line bg-surface px-4 py-3 text-sm font-medium"
        >
          <span>{englishCopy.home.money_summary_empty}</span>
          <ChevronRight
            aria-hidden="true"
            className="size-4 shrink-0 text-muted-foreground"
          />
        </Link>
      )}

      <Highlights
        items={planValue?.highlights_by_screen?.home}
        onDismiss={(key) => hideHighlight.mutate(key)}
      />

      {nextMaturity ? (
        <Card className="grid gap-2" aria-labelledby="coming-up-heading">
          <div className="flex items-center justify-between gap-3">
            <h2 id="coming-up-heading" className="text-lg font-semibold">
              Coming up
            </h2>
            <Link to="/money" className="text-sm font-semibold text-brand">
              See all
            </Link>
          </div>
          <Link
            to={`/money/asset/${nextMaturity.asset_id}`}
            className="flex min-h-16 items-center justify-between gap-3 border-t border-line py-2"
          >
            <span className="flex min-w-0 items-center gap-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <CalendarDays aria-hidden="true" className="size-4" />
              </span>
              <span className="min-w-0">
                <strong className="block truncate text-sm">
                  {nextMaturity.label}
                </strong>
                <span className="text-xs text-muted-foreground">
                  {new Intl.DateTimeFormat("en-IN", {
                    day: "numeric",
                    month: "short",
                  }).format(new Date(`${nextMaturity.date}T00:00:00`))}
                </span>
              </span>
            </span>
            <span className="flex items-center gap-2 text-sm tabular-nums">
              {nextMaturity.amount ? (
                <MoneyValue amountInr={nextMaturity.amount} />
              ) : null}
              <ChevronRight
                aria-hidden="true"
                className="size-4 text-muted-foreground"
              />
            </span>
          </Link>
        </Card>
      ) : null}

      {sheetAction
        ? (() => {
            const payload = payloadFor(sheetAction);
            const reason = textValue(payload.reason_code);
            const params = asRecord(payload.params);
            const clubbingCheck =
              asRecord(sheetAction).clubbing_check ??
              payload.clubbing_check ??
              params.clubbing_check;
            const hasClubbingCheck =
              clubbingCheck === true ||
              (typeof clubbingCheck === "string" && clubbingCheck.length > 0);
            const sale = sellDetails(sheetAction);
            const person = memberForAction(
              sheetAction,
              meMemberId,
              members,
              planValue,
            );
            return (
              <Sheet
                open
                onOpenChange={(open) => {
                  if (!open) setSheetAction(null);
                }}
                title="How to take this step"
              >
                <div className="grid gap-3">
                  {person ? (
                    <p className="text-sm font-medium">
                      {person.isYou
                        ? englishCopy.household_ui.action_for_you
                        : englishCopy.household_ui.action_for.replace(
                            "{name}",
                            person.name,
                          )}
                    </p>
                  ) : null}
                  <p className="text-base">
                    {actionHowTo(
                      sheetAction,
                      person,
                      members,
                      planValue,
                      primaryCurrency,
                      fxRate,
                    )}
                  </p>
                  {sale.quantity ? (
                    <p className="text-sm">
                      {englishCopy.home.sell_quantity.replace(
                        "{quantity}",
                        sale.quantity,
                      )}
                    </p>
                  ) : null}
                  {sale.tax ? (
                    <p className="text-sm">
                      {englishCopy.home.estimated_tax.replace("{amount}", "")}
                      <MoneyValue amountInr={sale.tax} />
                    </p>
                  ) : null}
                  <details>
                    <summary className="min-h-12 cursor-pointer py-2 text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand">
                      Why?
                    </summary>
                    <div className="grid gap-3 pt-2">
                      <p className="text-base">
                        {reason
                          ? reasonCopy(reason)
                          : "This step follows your plan."}
                      </p>
                      {hasClubbingCheck ? (
                        <p className="text-base">
                          {englishCopy.household_ui.clubbing_check}
                        </p>
                      ) : null}
                      <TriggeredExitRows
                        action={sheetAction}
                        plan={planValue}
                      />
                    </div>
                  </details>
                </div>
              </Sheet>
            );
          })()
        : null}
      {recordAction ? (
        <BuyRecordSheet
          key={recordAction.id}
          action={recordAction}
          saving={recordBuy.isPending}
          error={recordBuy.isError}
          onClose={() => setRecordAction(null)}
          onSubmit={(payload) =>
            recordBuy.mutate({ action: recordAction, payload })
          }
        />
      ) : null}
    </div>
  );
}
export { actionHowTo, actionMemberId, actionTitle, activeActions, payloadFor };
