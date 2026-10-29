import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  CalendarDays,
  CheckCircle2,
  ChevronRight,
  MoveRight,
} from "lucide-react";
import { Link } from "react-router";
import { getNetworth } from "../money/api";
import { roundActionAmount } from "../lib/money";
import { actionLabel, labelForCode, triggerKeyLabel } from "../lib/labels";
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
  updateActionStatus,
} from "../planning/api";
import type { ActionResponse, WeeklyPlan } from "../planning/api";
import { Button, Card, Sheet, Term } from "../ui";
import { Highlights } from "../features/Highlights";
import englishCopy from "../copy/en.json";
import { useHousehold } from "../household/HouseholdContext";
import { MoneyValue } from "../household/currency-view";
import {
  formatHouseholdAmount,
  formatSecondaryAmount,
} from "../household/currency";
import { formatRateToTwoDecimals } from "../household/fx";
import type { DkkInrRate } from "../household/fx";
import type { MemberResponse } from "../money/api";

function payloadFor(action: ActionResponse): Record<string, unknown> {
  const payload = asRecord(action.payload);
  const nested = asRecord(payload.action);
  return Object.keys(nested).length ? nested : payload;
}

function actionTitle(action: ActionResponse): string {
  const payload = payloadFor(action);
  const kind = textValue(payload.kind, action.kind);
  const tranche = asRecord(payload.tranche);
  if (kind === "BUY_STOCK_TRANCHE") {
    const part = textValue(tranche.n, "1");
    const of = textValue(tranche.of, "3");
    return `${actionLabel(kind)} (part ${part} of ${of})`;
  }
  const title = actionLabel(kind);
  const category = textValue(payload.instrument_category);
  return category ? `${title} · ${instrumentLabel(category)}` : title;
}

function rawActionAmount(action: ActionResponse): string {
  return textValue(payloadFor(action).amount, "0.00");
}

function actionAmount(action: ActionResponse): string {
  return roundActionAmount(rawActionAmount(action));
}

function networthAsOf(date: string | undefined): string {
  if (!date) return englishCopy.home.net_worth_as_of_today;
  const shownDate = new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(new Date(`${date}T00:00:00`));
  return englishCopy.home.net_worth_as_of.replace("{date}", shownDate);
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
  return {
    quantity: textValue(payload.quantity ?? payload.quantity_to_sell),
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
  const member = person
    ? members.find((candidate) => candidate.id === person.id)
    : undefined;
  if (textValue(payload.kind, action.kind) === "PAY_EXTRA_ON_LOAN") {
    return textValue(
      asRecord(payload.params).advice,
      "Use this amount for an extra loan payment.",
    );
  }
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
  return base;
}

function exitCardForAction(
  action: ActionResponse,
  plan: WeeklyPlan | undefined,
): Record<string, unknown> | undefined {
  const payload = payloadFor(action);
  const params = asRecord(payload.params);
  const securityId = textValue(
    payload.security_id ?? payload.asset_id ?? params.security_id,
  );
  const symbol = textValue(payload.symbol ?? params.symbol);
  const targetAction: Record<string, string> = {
    SELL_EXIT: "EXIT",
    SELL_TRIM: "TRIM_TO_TARGET",
    REVIEW_EXIT: "REVIEW_THEN_EXIT",
    SWAP: "SWAP",
  };
  const cards = asArray(plan?.exit_cards).map(asRecord);
  return (
    cards.find(
      (card) =>
        (securityId && textValue(card.security_id) === securityId) ||
        (symbol && textValue(card.symbol) === symbol),
    ) ??
    cards.find(
      (card) =>
        textValue(card.action) ===
        targetAction[textValue(payload.kind, action.kind)],
    )
  );
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

function ActionCard({
  action,
  person,
  readOnly,
  onOpen,
  onDone,
  saving,
  primary = false,
}: {
  action: ActionResponse;
  person?: ReturnType<typeof memberForAction>;
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
    actionOwner,
    action.status === "done" ? englishCopy.household_ui.completed : "",
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <article
      className="grid gap-3 border-t border-line py-3 first:border-0 first:pt-0"
      aria-label={cardLabel}
    >
      <div className="flex items-start gap-3">
        <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
          <MoveRight aria-hidden="true" className="size-4" />
        </span>
        <div className="flex min-w-0 flex-1 items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold">{actionTitle(action)}</h3>
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
          variant="ghost"
          aria-label="How to"
          onClick={() => onOpen(action)}
        >
          How?
        </Button>
        <Button
          variant={primary ? undefined : "secondary"}
          data-primary={primary ? "true" : undefined}
          aria-label={
            action.status === "done"
              ? englishCopy.household_ui.completed
              : "Mark done"
          }
          onClick={() => onDone(action)}
          loading={saving}
          disabled={readOnly || action.status !== "proposed"}
        >
          {readOnly
            ? englishCopy.household_ui.view_only
            : action.status === "done"
              ? englishCopy.household_ui.completed
              : "Done"}
        </Button>
      </div>
    </article>
  );
}

export function HomePage() {
  const [sheetAction, setSheetAction] = useState<ActionResponse | null>(null);
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
  const networth = useFeatureQuery(
    ["c27b", "networth", selectedHouseholdId, activeMemberId],
    "/v1/networth",
    () => getNetworth(activeMemberId),
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
  });
  const hideHighlight = useMutation({
    mutationFn: dismissHighlight,
    onSuccess: () =>
      queryClient.invalidateQueries({
        queryKey: ["c27b", "plan", selectedHouseholdId, activeMemberId],
      }),
  });
  const planValue = plan.data as WeeklyPlan | undefined;
  const visibleActions = (actions.data ?? []).filter(
    (action) => !activeMemberId || actionMemberId(action) === activeMemberId,
  );
  const actionList = activeActions(visibleActions);
  const nextMaturity = upcomingMaturities(networth.data)[0];
  const netWorthPrimary = networth.data
    ? formatHouseholdAmount(networth.data.net_worth, primaryCurrency, fxRate)
    : null;
  const netWorthSecondary = networth.data
    ? formatSecondaryAmount(networth.data.net_worth, primaryCurrency, fxRate)
    : null;

  if (
    (networth.supported && networth.isPending) ||
    (plan.supported && plan.isPending) ||
    (actions.supported && actions.isPending)
  ) {
    return <ScreenLoading label="Loading your week" />;
  }
  if (
    [networth.error, plan.error, actions.error].some(
      (error) => error && !isMissingFeature(error),
    )
  ) {
    return <ScreenError title="Your week couldn't load" />;
  }

  return (
    <div data-copy-screen="home" className="grid gap-4">
      <Card className="home-hero">
        <p className="text-sm font-medium text-muted-foreground">
          <Term
            explanation="The value of everything you own, minus what you owe."
            prompt="What's this?"
            learnMoreHref="/help#term-net-worth"
          >
            Net worth
          </Term>
        </p>
        {networth.supported && netWorthPrimary ? (
          <p className="mt-1 text-hero font-bold tabular-nums">
            <span className="tracking-tight">{netWorthPrimary}</span>
            {netWorthSecondary ? (
              <span className="block text-xs font-normal tracking-normal text-muted-foreground">
                {netWorthSecondary}
              </span>
            ) : null}
          </p>
        ) : (
          <p className="mt-1 text-base">
            Add an item in Money to see your total.
          </p>
        )}
        <div className="mt-3 grid gap-1.5">
          {fxRate === undefined &&
          members.some((member) => member.residency.country === "DK") ? (
            <p className="text-xs text-muted-foreground">
              {englishCopy.household_ui.no_fx_rate}
            </p>
          ) : null}
          {networth.data || fxRate ? (
            <p className="text-xs text-muted-foreground">
              {networthAsOf(networth.data?.as_of ?? fxRate?.date)}
              {fxRate
                ? ` · 1 DKK = ₹${formatRateToTwoDecimals(fxRate.rate)}`
                : ""}
            </p>
          ) : null}
        </div>
      </Card>

      <Card className="grid gap-3" aria-labelledby="this-week-heading">
        <h2 id="this-week-heading" className="text-lg font-semibold">
          This week
        </h2>
        {actionList.length ? (
          <div className="grid gap-2">
            {actionList.map((action, index) => (
              <ActionCard
                key={action.id}
                action={action}
                person={memberForAction(action, meMemberId, members, planValue)}
                readOnly={isViewer}
                saving={markDone.isPending}
                primary={index === 0 && !isViewer}
                onDone={(selected) => markDone.mutate(selected)}
                onOpen={setSheetAction}
              />
            ))}
            {isViewer ? (
              <Link
                to="/plan"
                data-primary="true"
                className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
              >
                View your plan
              </Link>
            ) : null}
          </div>
        ) : (
          <div className="grid justify-items-center gap-3 py-3 text-center">
            <span className="grid size-11 place-items-center rounded-full bg-success-soft text-success-foreground">
              <CheckCircle2 aria-hidden="true" className="size-5" />
            </span>
            <p className="text-base font-medium">
              {englishCopy.home.nothing_to_do}
            </p>
            <Link
              to="/plan"
              data-primary="true"
              className="motion-safe:active:scale-[.98] inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground transition"
            >
              Review your plan
            </Link>
          </div>
        )}
        {markDone.isError ? (
          <p role="alert" className="text-sm text-muted-foreground">
            We couldn't mark this done. Try again.
          </p>
        ) : null}
        <details className="border-t border-line pt-2">
          <summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">
            Make it more accurate
          </summary>
          <div className="grid gap-2 pb-2">
            <p className="text-sm text-muted-foreground">
              Your plan starts balanced. Goals and email updates stay off until
              you choose them.
            </p>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/risk"
            >
              Choose your comfort with ups and downs
            </Link>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/goals"
            >
              Add a goal
            </Link>
            <Link
              className="min-h-10 text-sm font-medium text-brand"
              to="/onboarding/consent"
            >
              Choose email updates
            </Link>
          </div>
        </details>
      </Card>

      <Highlights
        items={planValue?.highlights_by_screen?.home}
        onDismiss={(key) => hideHighlight.mutate(key)}
      />

      <Card className="grid gap-2" aria-labelledby="coming-up-heading">
        <div className="flex items-center justify-between gap-3">
          <h2 id="coming-up-heading" className="text-lg font-semibold">
            Coming up
          </h2>
          <Link to="/money" className="text-sm font-semibold text-brand">
            See all
          </Link>
        </div>
        {nextMaturity ? (
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
        ) : (
          <p className="border-t border-line pt-2 text-sm text-muted-foreground">
            Nothing is due in the next 30 days.
          </p>
        )}
      </Card>

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
            const whyParams = Object.entries(params).filter(
              ([key]) => key !== "clubbing_check",
            );
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
                    <summary className="cursor-pointer text-sm font-semibold">
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
                      {whyParams.length ? (
                        <dl className="grid gap-2">
                          {whyParams.map(([key, value]) => (
                            <div
                              key={key}
                              className="flex justify-between gap-4 border-t border-line pt-2 text-sm"
                            >
                              <dt className="text-muted-foreground">
                                {key.replaceAll("_", " ")}
                              </dt>
                              <dd className="text-right">
                                {typeof value === "string"
                                  ? value
                                  : JSON.stringify(value)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      ) : null}
                    </div>
                  </details>
                </div>
              </Sheet>
            );
          })()
        : null}
    </div>
  );
}
export { actionHowTo, actionMemberId, actionTitle, activeActions, payloadFor };
