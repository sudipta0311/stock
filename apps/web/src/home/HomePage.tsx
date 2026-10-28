import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  CalendarDays,
  CircleCheck,
  CircleHelp,
} from "lucide-react";
import { Link } from "react-router";
import { getMe, getNetworth } from "../money/api";
import { formatInr } from "../lib/money";
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
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import { getActions, getPlan, updateActionStatus } from "../planning/api";
import type { ActionResponse, WeeklyPlan } from "../planning/api";
import { Badge, Button, Card, Sheet } from "../ui";
import englishCopy from "../copy/en.json";
import { useHousehold } from "../household/HouseholdContext";
import { ExchangeRateNote, MoneyValue } from "../household/currency-view";
import { formatHouseholdAmount } from "../household/currency";
import type { DkkInrRate } from "../household/fx";
import type { MemberResponse } from "../money/api";

type SelectedSheet = { action: ActionResponse; kind: "howto" | "why" } | null;

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

function actionAmount(action: ActionResponse): string {
  return textValue(payloadFor(action).amount);
}

const SELL_ACTIONS = new Set(["SELL_EXIT", "SELL_TRIM", "REVIEW_EXIT", "SWAP"]);

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
  const accountType = textValue(params.account_type ?? payload.account_type);
  const rawAmount = actionAmount(action);
  const amount = rawAmount
    ? formatHouseholdAmount(rawAmount, primaryCurrency, fxRate)
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
    rawAmount ? formatInr(rawAmount, "compact") : "",
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

function showDailyChangesEnabled(): boolean {
  try {
    return (
      window.localStorage.getItem("wealth.settings.show-daily-changes") ===
      "true"
    );
  } catch {
    return false;
  }
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
  doneBy,
  readOnly,
  onOpen,
  onDone,
  saving,
}: {
  action: ActionResponse;
  person?: ReturnType<typeof memberForAction>;
  doneBy?: string;
  readOnly: boolean;
  onOpen(kind: "howto" | "why", action: ActionResponse): void;
  onDone(action: ActionResponse): void;
  saving: boolean;
}) {
  const payload = payloadFor(action);
  const amount = actionAmount(action);
  const reason = textValue(payload.reason_code);
  const kind = textValue(payload.kind, action.kind);
  const details = sellDetails(action);
  const sellAction = SELL_ACTIONS.has(kind);
  const clubbingCheck =
    asRecord(action).clubbing_check ?? payload.clubbing_check;
  const clubbing =
    clubbingCheck !== null &&
    clubbingCheck !== undefined &&
    clubbingCheck !== false &&
    clubbingCheck !== "";
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
    <Card className="grid gap-3" aria-label={cardLabel}>
      {person ? (
        <div className="flex items-center gap-2 text-sm">
          <span
            aria-hidden="true"
            className="grid size-8 place-items-center rounded-full bg-brand/10 text-xs font-semibold text-brand"
          >
            {person.name.slice(0, 1).toUpperCase()}
          </span>
          <span>
            {person.isYou
              ? englishCopy.household_ui.action_for_you
              : englishCopy.household_ui.action_for.replace(
                  "{name}",
                  person.name,
                )}
          </span>
        </div>
      ) : null}
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-base font-semibold">{actionTitle(action)}</h3>
        {amount ? (
          <span className="shrink-0 text-sm font-semibold tabular-nums">
            <MoneyValue amountInr={amount} />
          </span>
        ) : null}
      </div>
      <p className="text-sm leading-6 text-muted-foreground">
        {reason ? reasonCopy(reason) : "This action follows your plan."}
      </p>
      {sellAction && details.quantity ? (
        <p className="text-sm">
          {englishCopy.home.sell_quantity.replace(
            "{quantity}",
            details.quantity,
          )}
        </p>
      ) : null}
      {sellAction && details.tax ? (
        <p className="text-sm">
          {englishCopy.home.estimated_tax.replace("{amount}", "")}
          <MoneyValue amountInr={details.tax} />
        </p>
      ) : null}
      {doneBy ? (
        <p className="text-xs text-muted-foreground">
          {englishCopy.household_ui.done_by.replace("{name}", doneBy)}
        </p>
      ) : null}
      {clubbing ? (
        <p className="rounded-xl bg-muted p-3 text-sm leading-6">
          {englishCopy.household_ui.clubbing_check}
        </p>
      ) : null}
      <div className="flex flex-wrap gap-2">
        <Button variant="secondary" onClick={() => onOpen("howto", action)}>
          How to
        </Button>
        <Button variant="secondary" onClick={() => onOpen("why", action)}>
          <CircleHelp aria-hidden="true" className="size-4" />
          Why
        </Button>
        <Button
          onClick={() => onDone(action)}
          loading={saving}
          disabled={readOnly || action.status !== "proposed"}
        >
          {readOnly
            ? englishCopy.household_ui.view_only
            : action.status === "done"
              ? englishCopy.household_ui.completed
              : "Mark done"}
        </Button>
      </div>
    </Card>
  );
}

export function HomePage() {
  const [sheet, setSheet] = useState<SelectedSheet>(null);
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
  const me = useFeatureQuery(
    ["c27b", "me", selectedHouseholdId],
    "/v1/me",
    getMe,
  );
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
  const planValue = plan.data as WeeklyPlan | undefined;
  const visibleActions = (actions.data ?? []).filter(
    (action) => !activeMemberId || actionMemberId(action) === activeMemberId,
  );
  const actionList = activeActions(visibleActions);
  const doneActions = visibleActions
    .filter((action) => action.status === "done")
    .sort((left, right) => left.rank - right.rank)
    .slice(0, 3);
  const nextMaturities = upcomingMaturities(networth.data);
  const foundationGaps = (planValue?.foundation ?? []).filter((item) =>
    isPositiveAmount(item.gap),
  );
  const personName = textValue(me.data?.display_name).trim().split(/\s+/)[0];
  const planRecord = asRecord(planValue);
  const score = textValue(
    planRecord.health_score ??
      planRecord.plan_health_score ??
      planRecord.health,
  );
  const weekChange = textValue(
    asRecord(networth.data).week_change_pct ??
      asRecord(networth.data).week_change,
  );

  if (
    (networth.supported && networth.isPending) ||
    (plan.supported && plan.isPending) ||
    (actions.supported && actions.isPending) ||
    (me.supported && me.isPending)
  ) {
    return <ScreenLoading label="Loading your week" />;
  }
  if (
    [networth.error, plan.error, actions.error].some(
      (error) => error && !isMissingFeature(error),
    )
  ) {
    return <ScreenError title="Your week couldn’t load" />;
  }

  return (
    <div className="grid gap-4">
      <section className="grid gap-3" aria-labelledby="home-greeting">
        <div>
          <h2 id="home-greeting" className="text-xl font-semibold">
            Hello{personName ? `, ${personName}` : ""}
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Here’s your household money at a glance.
          </p>
        </div>
        {networth.supported && networth.data ? (
          <Card className="grid gap-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Net worth</p>
                <p className="mt-1 text-xl font-semibold tabular-nums">
                  <MoneyValue amountInr={networth.data.net_worth} />
                </p>
                {weekChange && showDailyChangesEnabled() ? (
                  <p className="mt-1 text-xs text-muted-foreground">
                    This week {weekChange}
                  </p>
                ) : null}
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Plan health</p>
                <p className="mt-1 text-xl font-semibold">
                  {score ? score : "Your plan"}
                </p>
                <Link
                  className="mt-1 inline-flex min-h-10 items-center gap-1 text-sm font-medium text-brand"
                  to="/plan"
                >
                  See targets{" "}
                  <ArrowRight aria-hidden="true" className="size-4" />
                </Link>
              </div>
            </div>
            {!activeMemberId && (planValue?.members?.length ?? 0) > 0 ? (
              <details className="border-t border-line pt-3">
                <summary className="cursor-pointer text-sm font-semibold">
                  {englishCopy.household.money.member_breakdown}
                </summary>
                <div className="mt-3 grid gap-2">
                  {planValue?.members?.map((member) => (
                    <div
                      key={member.id}
                      className="flex items-center justify-between gap-3 text-sm"
                    >
                      <span>
                        {member.id === meMemberId
                          ? englishCopy.household.you
                          : member.name}
                      </span>
                      <MoneyValue amountInr={member.net_worth} />
                    </div>
                  ))}
                </div>
              </details>
            ) : null}
          </Card>
        ) : (
          <ComingSoon title="Net worth is coming soon" />
        )}
        <ExchangeRateNote />
        {fxRate === undefined &&
        members.some((member) => member.residency.country === "DK") ? (
          <p className="text-xs text-muted-foreground">
            {englishCopy.household_ui.no_fx_rate}
          </p>
        ) : null}
      </section>

      <section className="grid gap-3" aria-labelledby="this-week-heading">
        <div className="flex items-center justify-between gap-3">
          <h2 id="this-week-heading" className="text-lg font-semibold">
            This week
          </h2>
          {actions.supported && actions.data ? (
            <Badge>
              {actionList.length}{" "}
              {actionList.length === 1 ? "action" : "actions"}
            </Badge>
          ) : null}
        </div>
        {actions.supported && actions.data ? (
          actionList.length ? (
            actionList.map((action) => (
              <ActionCard
                key={action.id}
                action={action}
                person={memberForAction(action, meMemberId, members, planValue)}
                doneBy={textValue(action.done_by)}
                readOnly={isViewer}
                saving={markDone.isPending}
                onDone={(selected) => markDone.mutate(selected)}
                onOpen={(kind, selected) =>
                  setSheet({ kind, action: selected })
                }
              />
            ))
          ) : (
            <Card className="flex items-start gap-3">
              <CircleCheck
                aria-hidden="true"
                className="mt-0.5 size-5 text-brand"
              />
              <div>
                <p className="font-semibold">
                  You’re on plan. Nothing to do this week ✓
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  You can check back next week.
                </p>
              </div>
            </Card>
          )
        ) : (
          <ComingSoon title="This week’s actions are coming soon" />
        )}
        {doneActions.length ? (
          <details className="grid gap-3">
            <summary className="cursor-pointer text-sm font-semibold">
              {englishCopy.household_ui.done_this_week}
            </summary>
            {doneActions.map((action) => (
              <ActionCard
                key={action.id}
                action={action}
                person={memberForAction(action, meMemberId, members, planValue)}
                doneBy={textValue(action.done_by)}
                readOnly={isViewer}
                saving={false}
                onDone={() => undefined}
                onOpen={(kind, selected) =>
                  setSheet({ kind, action: selected })
                }
              />
            ))}
          </details>
        ) : null}
        {markDone.isError ? (
          <p role="alert" className="text-sm text-muted-foreground">
            We couldn’t mark this done. Try again.
          </p>
        ) : null}
      </section>

      <section className="grid gap-3" aria-labelledby="coming-up-heading">
        <h2 id="coming-up-heading" className="text-lg font-semibold">
          Coming up · next 30 days
        </h2>
        {!networth.supported || !networth.data ? (
          <ComingSoon title="Your calendar is coming soon" />
        ) : nextMaturities.length ? (
          <Card className="grid gap-2">
            {nextMaturities.map((event) => (
              <Link
                key={`${event.asset_id}-${event.date}-${event.kind}`}
                to={`/money/asset/${event.asset_id}`}
                className="flex min-h-12 items-center justify-between gap-3 border-t border-line pt-2 first:border-0 first:pt-0"
              >
                <span>
                  <strong className="block text-sm">{event.label}</strong>
                  <span className="text-xs text-muted-foreground">
                    {new Intl.DateTimeFormat("en-IN", {
                      day: "numeric",
                      month: "short",
                    }).format(new Date(`${event.date}T00:00:00`))}
                  </span>
                </span>
                <span className="flex items-center gap-2 text-sm tabular-nums">
                  {event.amount ? <MoneyValue amountInr={event.amount} /> : ""}
                  <CalendarDays
                    aria-hidden="true"
                    className="size-4 text-muted-foreground"
                  />
                </span>
              </Link>
            ))}
          </Card>
        ) : (
          <Card>
            <p className="text-sm text-muted-foreground">
              No maturities or premiums are due in the next 30 days.
            </p>
          </Card>
        )}
      </section>

      {plan.supported && foundationGaps.length > 0 ? (
        <Card className="flex items-center justify-between gap-3 border-warning/40">
          <div>
            <p className="font-semibold">A foundation item needs attention</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Your plan has {foundationGaps.length} gap
              {foundationGaps.length === 1 ? "" : "s"} to review.
            </p>
          </div>
          <Link
            to="/plan"
            className="inline-flex min-h-12 shrink-0 items-center gap-1 text-sm font-semibold text-brand"
          >
            Review <ArrowRight aria-hidden="true" className="size-4" />
          </Link>
        </Card>
      ) : null}

      {sheet
        ? (() => {
            const payload = payloadFor(sheet.action);
            const reason = textValue(payload.reason_code);
            const params = asRecord(payload.params);
            const person = memberForAction(
              sheet.action,
              meMemberId,
              members,
              planValue,
            );
            return (
              <Sheet
                open
                onOpenChange={(open) => {
                  if (!open) setSheet(null);
                }}
                title={
                  sheet.kind === "howto"
                    ? "How to take this step"
                    : "Why this is in your plan"
                }
              >
                {sheet.kind === "howto" ? (
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
                    <p className="text-sm leading-6">
                      {actionHowTo(
                        sheet.action,
                        person,
                        members,
                        planValue,
                        primaryCurrency,
                        fxRate,
                      )}
                    </p>
                  </div>
                ) : (
                  <div className="grid gap-3">
                    <p className="text-sm leading-6">
                      {reason
                        ? reasonCopy(reason)
                        : "This action follows your plan."}
                    </p>
                    <TriggeredExitRows action={sheet.action} plan={planValue} />
                    {Object.keys(params).length ? (
                      <details>
                        <summary className="cursor-pointer text-sm font-semibold">
                          Details
                        </summary>
                        <dl className="mt-2 grid gap-2">
                          {Object.entries(params).map(([key, value]) => (
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
                      </details>
                    ) : null}
                  </div>
                )}
              </Sheet>
            );
          })()
        : null}
      <p className="text-xs text-muted-foreground">
        {englishCopy.disclaimers.personal}
      </p>
    </div>
  );
}

export { actionHowTo, actionMemberId, actionTitle, activeActions, payloadFor };
