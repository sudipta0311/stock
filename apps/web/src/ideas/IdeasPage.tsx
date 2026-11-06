import { Link } from "react-router";
import { Lightbulb } from "lucide-react";
import { formatInr, formatUsd } from "../lib/money";
import { formatDkk } from "../household/fx";
import {
  asArray,
  asRecord,
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
import { getActions, getPlan, getResearchIdeas } from "../planning/api";
import { getInvestingPrefs } from "../settings/api";
import type {
  ActionResponse,
  ResearchIdeasResponse,
  WeeklyPlan,
} from "../planning/api";
import { Card, EmptyState, Term } from "../ui";
import * as englishCopy from "../copy/sections/index";

const ENTRY_HINT_REASON_COPY: Record<string, string> = {
  price_far_above_average:
    englishCopy.reasons.entry_hint_price_far_above_average,
  market_valuation_high: englishCopy.reasons.entry_hint_market_expensive,
  market_valuation_elevated: englishCopy.reasons.entry_hint_market_elevated,
  us_price_history_unavailable:
    englishCopy.reasons.entry_hint_us_price_history_unavailable,
  price_history_unavailable:
    englishCopy.reasons.entry_hint_price_history_unavailable,
  market_valuation_unavailable:
    englishCopy.reasons.entry_hint_market_history_unavailable,
  market_history_short: englishCopy.reasons.entry_hint_market_history_short,
  regular_entry_schedule: englishCopy.reasons.entry_hint_regular_schedule,
};

function dkkAmount(raw: unknown): string {
  const amount = String(raw ?? "").trim();
  if (!/^-?\d+$/.test(amount)) return "";
  return `DKK ${new Intl.NumberFormat("en-US").format(BigInt(amount))}`;
}

function usdAmount(raw: unknown): string {
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(String(raw ?? "").trim());
  if (!match) return "";
  let amount = BigInt(match[2] ?? "0");
  if ((match[3]?.[0] ?? "0") >= "5") amount += 1n;
  if (match[1]) amount = -amount;
  return `USD ${new Intl.NumberFormat("en-US").format(amount)}`;
}

function addInrAmounts(left: string, right: string): string {
  const toPaise = (value: string): bigint => {
    const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
    if (!match) return 0n;
    const whole = BigInt(match[2] ?? "0");
    const fraction = BigInt((match[3] ?? "").padEnd(2, "0").slice(0, 2));
    const amount = whole * 100n + fraction;
    return match[1] ? -amount : amount;
  };
  const total = toPaise(left) + toPaise(right);
  const sign = total < 0n ? "-" : "";
  const magnitude = total < 0n ? -total : total;
  return `${sign}${magnitude / 100n}.${String(magnitude % 100n).padStart(2, "0")}`;
}

function DenmarkBuyCard({ plan }: { plan?: WeeklyPlan }) {
  const copy = englishCopy.denmark;
  const residency = asRecord(plan?.residency);
  const members = asArray(plan?.members).map(asRecord);
  const lines = asArray(plan?.abroad_buy_list).slice(0, 3).map(asRecord);
  const hasDkResident = members.some(
    (member) => asRecord(member.residency).country === "DK",
  );
  if (residency.country !== "DK" && !hasDkResident && !lines.length)
    return null;
  if (!lines.length) return null;
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">{copy.buy_title}</h2>
        <p className="mt-1 text-sm text-muted-foreground">{copy.buy_intro}</p>
      </div>
      {lines.length ? (
        <ul className="grid gap-3">
          {lines.map((line, index) => {
            const instrument = asRecord(line.instrument);
            const details = asRecord(line.details);
            const member = members.find((item) => item.id === line.member_id);
            const broker = textValue(line.broker, "other");
            const howTo = asArray(asRecord(copy.how_to)[broker]);
            const listedName = textValue(
              instrument.listed_name,
              textValue(instrument.name, copy.fund_fallback),
            );
            const searchCode = textValue(
              instrument.isin,
              textValue(instrument.ticker),
            );
            const heldNotes = asArray(details.held_us_etf_notes).map(asRecord);
            const alternatives = asArray(details.alternatives_considered).map(
              asRecord,
            );
            const comparison = asRecord(details.us_comparison);
            const reasonCodes = asArray(line.reason_codes).map(String);
            const usStockTax = asRecord(details.us_stock_tax);
            const usTaxSources = asArray(usStockTax.sources).map(asRecord);
            const sources = asArray(details.sources).map(asRecord);
            return (
              <li
                key={`${textValue(instrument.isin)}-${index}`}
                className="grid gap-2 rounded-xl bg-muted p-3"
              >
                <div className="flex items-start justify-between gap-3">
                  <span className="flex min-w-0 items-center gap-3">
                    <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-surface text-brand">
                      <Lightbulb aria-hidden="true" className="size-4" />
                    </span>
                    <h3 className="min-w-0 font-semibold">
                      <span data-security>
                        {textValue(instrument.name, copy.fund_fallback)}
                      </span>
                    </h3>
                  </span>
                  <span className="shrink-0 text-sm font-semibold tabular-nums">
                    {dkkAmount(line.amount_dkk)}
                  </span>
                </div>
                {members.length > 1 && textValue(member?.name) ? (
                  <p className="text-xs text-muted-foreground">
                    {copy.buy_for_member.replace(
                      "{name}",
                      textValue(member?.name),
                    )}
                  </p>
                ) : null}
                {listedName || searchCode ? (
                  <details>
                    <summary className="min-h-11 cursor-pointer py-2 text-sm font-semibold text-brand">
                      {copy.find_it_at_broker}
                    </summary>
                    <div className="mt-2 grid gap-1 text-sm text-muted-foreground">
                      {listedName ? <p>{listedName}</p> : null}
                      {searchCode ? (
                        <p>
                          {copy.fund_code_label}: {searchCode}
                        </p>
                      ) : null}
                      {instrument.accumulating ? (
                        <p>{copy.reinvests_dividends}</p>
                      ) : null}
                    </div>
                  </details>
                ) : null}
                <p className="text-xs text-muted-foreground">
                  {textValue(line.account) === "ask"
                    ? copy.account_ask
                    : textValue(line.account) === "pension"
                      ? copy.account_pension
                      : copy.account_ordinary}
                </p>
                <div className="flex flex-wrap justify-between gap-x-3 text-xs text-muted-foreground">
                  <span>
                    {copy.broker}:{" "}
                    {textValue(asRecord(copy.brokers)[broker], broker)}
                  </span>
                  <span>{textValue(instrument.currency)}</span>
                </div>
                <details>
                  <summary className="min-h-11 cursor-pointer py-2 text-sm font-semibold text-brand">
                    {copy.how}
                  </summary>
                  <ol className="mt-2 grid gap-1 pl-5 text-base list-decimal">
                    {howTo.map((step, stepIndex) => (
                      <li key={`${broker}-${stepIndex}`}>{String(step)}</li>
                    ))}
                  </ol>
                </details>
                <details>
                  <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
                    {copy.why}
                  </summary>
                  <div className="mt-2 grid gap-2 text-base">
                    <p>{copy.why_ranked}</p>
                    {details.ask_unverified ? (
                      <p className="text-xs text-muted-foreground">
                        {copy.ask_unverified_note}
                      </p>
                    ) : null}
                    {reasonCodes.includes("broker_availability_unchecked") ? (
                      <p>{copy.broker_availability_unchecked}</p>
                    ) : null}
                    {alternatives.length ? (
                      <div>
                        <p>{copy.alternative_reason}</p>
                        <ul className="list-disc pl-5">
                          {alternatives.map((alternative, alternativeIndex) => (
                            <li
                              key={`${textValue(alternative.name)}-${alternativeIndex}`}
                            >
                              {textValue(alternative.name)}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {comparison.annual_difference_pct ? (
                      <div className="grid gap-1">
                        <p>
                          {(textValue(
                            comparison.annual_difference_direction,
                          ) === "less"
                            ? copy.comparison_summary_less
                            : copy.comparison_summary_more
                          ).replace(
                            "{difference}",
                            textValue(comparison.annual_difference_pct),
                          )}
                        </p>
                        <p>{copy.us_comparison_advantage_one}</p>
                        <p>{copy.us_comparison_advantage_two}</p>
                        <p className="text-xs text-muted-foreground">
                          {copy.sources}
                        </p>
                        <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                          <li>
                            <a
                              className="underline underline-offset-2"
                              href={textValue(comparison.us_ter_source_url)}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {copy.fee_source}
                            </a>
                          </li>
                          <li>
                            <a
                              className="underline underline-offset-2"
                              href={textValue(
                                comparison.dividend_yield_source_url,
                              )}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {copy.dividend_source}
                            </a>
                          </li>
                        </ul>
                      </div>
                    ) : null}
                    {heldNotes.slice(0, 1).map((note, noteIndex) => {
                      const [beforeSymbol = "", afterSymbol = ""] =
                        copy.held_us_etf.split("{symbol}");
                      const [between = "", afterName = ""] =
                        afterSymbol.split("{name}");
                      const symbol = textValue(note.symbol);
                      const name = textValue(note.eu_twin);
                      if (!symbol || !name) return null;
                      return (
                        <p key={`${symbol}-${noteIndex}`}>
                          {beforeSymbol}
                          <span data-security>{symbol}</span>
                          {between}
                          <span data-security>{name}</span>
                          {afterName}
                        </p>
                      );
                    })}
                    {Object.keys(usStockTax).length ? (
                      <div className="grid gap-1">
                        <p>
                          {copy.us_stock_dividend_before}
                          <Term
                            explanation={copy.us_stock_form_help}
                            prompt={copy.term_prompt}
                            learnMoreHref="/help#term-w-8ben"
                          >
                            {copy.us_stock_form}
                          </Term>
                          {copy.us_stock_dividend_after.replace(
                            "{rate}",
                            textValue(usStockTax.dividend_withholding_pct),
                          )}
                        </p>
                        {usStockTax.projected_us_situs_usd ? (
                          <p>
                            {copy.us_stock_situs_total
                              .replace(
                                "{current}",
                                usdAmount(usStockTax.current_us_situs_usd),
                              )
                              .replace(
                                "{projected}",
                                usdAmount(usStockTax.projected_us_situs_usd),
                              )
                              .replace(
                                "{threshold}",
                                usdAmount(
                                  usStockTax.estate_filing_threshold_usd,
                                ),
                              )}
                          </p>
                        ) : (
                          <p>
                            {copy.us_stock_situs_unavailable.replace(
                              "{threshold}",
                              usdAmount(usStockTax.estate_filing_threshold_usd),
                            )}
                          </p>
                        )}
                        <p>{copy.us_stock_estate_treaty}</p>
                        <p className="text-xs text-muted-foreground">
                          {copy.sources}
                        </p>
                        <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                          {usTaxSources.map((source, sourceIndex) => {
                            const label = textValue(source.label);
                            const labelText =
                              label === "w8ben"
                                ? copy.us_stock_form
                                : label === "dividend_treaty"
                                  ? copy.us_stock_dividend_source
                                  : label === "danish_credit"
                                    ? copy.us_stock_credit_source
                                    : label === "us_estate"
                                      ? copy.us_stock_estate_source
                                      : copy.us_stock_treaty_source;
                            return (
                              <li key={`${label}-${sourceIndex}`}>
                                <a
                                  className="underline underline-offset-2"
                                  href={textValue(source.url)}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  {labelText}
                                </a>
                              </li>
                            );
                          })}
                        </ul>
                      </div>
                    ) : null}
                    {details.pension_note ? <p>{copy.pension_note}</p> : null}
                    <p>{copy.unverified}</p>
                    <ul className="flex flex-wrap gap-x-3 gap-y-1 text-xs">
                      {sources.map((source, sourceIndex) => {
                        const label = textValue(source.label);
                        const labelText =
                          label === "factsheet"
                            ? copy.factsheet
                            : label === "tax_list"
                              ? copy.tax_list
                              : copy.share_savings_rules;
                        return (
                          <li key={`${label}-${sourceIndex}`}>
                            <a
                              className="underline underline-offset-2"
                              href={textValue(source.url)}
                              target="_blank"
                              rel="noreferrer"
                            >
                              {labelText}
                            </a>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </details>
              </li>
            );
          })}
        </ul>
      ) : null}
    </Card>
  );
}

function actionPayload(action: ActionResponse): Record<string, unknown> {
  const payload = asRecord(action.payload);
  const nested = asRecord(payload.action);
  return Object.keys(nested).length ? nested : payload;
}

function categoryName(category: string, sleeve: string): string {
  if (category === "in_core_index") return "Indian large-company fund";
  if (category === "in_international_index_fund")
    return "U.S. large-company fund in rupees";
  const label = instrumentLabel(category);
  return category ? label : `${instrumentLabel(sleeve)} category`;
}

export function corePicks(
  actions: ActionResponse[] | undefined,
  plan: WeeklyPlan | undefined,
): string[] {
  const values = new Set<string>();
  for (const action of actions ?? []) {
    if (action.status !== "proposed") continue;
    const payload = actionPayload(action);
    const category = textValue(payload.instrument_category);
    if (category) values.add(categoryName(category, textValue(payload.sleeve)));
  }
  const route = asRecord(plan?.route);
  const routeCategory = textValue(route.vehicle);
  if (routeCategory && routeCategory !== "no_open_route") {
    values.add(categoryName(routeCategory, "equity_global"));
  }
  return [...values];
}

function valuationWords(value: string): string | null {
  if (value === "cheap") return englishCopy.ideas.valuation_cheap;
  if (value === "fair") return englishCopy.ideas.valuation_fair;
  if (value === "expensive") return englishCopy.ideas.valuation_expensive;
  return null;
}

function confidenceWords(value: string): string | null {
  if (value.toUpperCase() === "HIGH") return englishCopy.ideas.confidence_high;
  if (value.toUpperCase() === "LOW") return englishCopy.ideas.confidence_low;
  return null;
}

function referencedTranche(
  idea: Record<string, unknown>,
  actions: ActionResponse[] | undefined,
): string | null {
  const ideaId = textValue(idea.security_id);
  const ideaSymbol = textValue(idea.symbol);
  for (const action of actions ?? []) {
    const payload = actionPayload(action);
    if (
      textValue(payload.security_id) !== ideaId &&
      textValue(payload.symbol) !== ideaSymbol
    )
      continue;
    const tranche = asRecord(payload.tranche);
    if (tranche.n && tranche.of)
      return `Part ${textValue(tranche.n)} of ${textValue(tranche.of)}`;
  }
  return null;
}

function SelectorFundName({ name }: { name: string }) {
  if (/\b(?:ETF|iNAV|tracking difference)\b/i.test(name)) {
    return (
      <Term
        explanation={englishCopy.ideas.fund_term_help}
        prompt={englishCopy.ideas.term_prompt}
      >
        {name}
      </Term>
    );
  }
  return name;
}

function SelectorFunds({ research }: { research?: ResearchIdeasResponse }) {
  const categories = Object.entries(asRecord(research?.core_picks)).flatMap(
    ([category, rawPicks]) => {
      const picks = asArray(rawPicks).slice(0, 2).map(asRecord);
      return picks.length ? [{ category, picks }] : [];
    },
  );
  if (!categories.length) return null;
  return (
    <details className="rounded-xl border border-line px-3">
      <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
        {englishCopy.ideas.funds_title}
      </summary>
      <div className="grid gap-3 pb-3">
        {categories.map(({ category, picks }) => (
          <section key={category} className="grid gap-1">
            <h3 className="text-sm font-medium">
              {textValue(picks[0]?.category_label, category)}
            </h3>
            <ul className="grid gap-1 text-sm text-muted-foreground">
              {picks.map((pick, index) => (
                <li key={`${category}-${index}`}>
                  <SelectorFundName
                    name={textValue(pick.name, "Fund name unavailable")}
                  />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </details>
  );
}

function groupIdeasByMarket(ideas: unknown[]): Map<string, unknown[]> {
  const groups = new Map<string, unknown[]>();
  for (const raw of ideas) {
    const market = textValue(asRecord(raw).market, "Other");
    groups.set(market, [...(groups.get(market) ?? []), raw]);
  }
  return groups;
}

function splitIdeasByMarket(ideas: unknown[]): {
  top: Map<string, unknown[]>;
  rest: Map<string, unknown[]>;
  alreadyOwned: unknown[];
} {
  const top = new Map<string, unknown[]>();
  const rest = new Map<string, unknown[]>();
  const counts = new Map<string, number>();
  const alreadyOwned: unknown[] = [];
  for (const raw of ideas) {
    if (textValue(asRecord(asRecord(raw).owned).status) === "enough") {
      alreadyOwned.push(raw);
      continue;
    }
    const market = textValue(asRecord(raw).market, "Other");
    const count = counts.get(market) ?? 0;
    const target = count < 3 ? top : rest;
    target.set(market, [...(target.get(market) ?? []), raw]);
    counts.set(market, count + 1);
  }
  return { top, rest, alreadyOwned };
}

function ideaMarketLabel(market: string): string {
  if (market === "IN") return "India";
  if (market === "US") return "United States";
  return market;
}

function ResearchIdea({
  raw,
  actions,
  index,
}: {
  raw: unknown;
  actions?: ActionResponse[];
  index: number;
}) {
  const idea = asRecord(raw);
  const symbol = textValue(idea.symbol, "Research idea");
  const name = textValue(idea.name);
  const displayName = name || symbol;
  const confidence = confidenceWords(textValue(idea.confidence));
  const valuation = valuationWords(textValue(idea.valuation_label));
  const sector = textValue(idea.sector);
  const tranche = referencedTranche(idea, actions);
  const why = asRecord(idea.why);
  const whyReasons = asArray(why.reasons)
    .slice(0, 2)
    .map((reason) => textValue(reason))
    .filter(Boolean);
  const whyRisk = textValue(why.risk);
  const ownership = asRecord(idea.owned);
  const ownershipStatus = textValue(ownership.status);
  const ownershipCurrency = textValue(ownership.currency, "INR");
  const directOwned = textValue(ownership.direct ?? ownership.direct_inr, "0");
  const throughOwned = textValue(
    ownership.through_funds ?? ownership.through_funds_inr,
    "0",
  );
  const ownedFunds = asArray(ownership.funds).slice(0, 3);
  const hasDirectOwned = Number(directOwned) > 0;
  const hasThroughOwned = Number(throughOwned) > 0;
  const formatOwnedAmount = (amount: string) =>
    ownershipCurrency === "DKK"
      ? formatDkk(amount)
      : formatInr(amount, "compact");
  const ownershipLine =
    ownershipStatus !== "some" && ownershipStatus !== "enough"
      ? null
      : hasThroughOwned && hasDirectOwned
        ? englishCopy.ideas.owned_direct_and_funds
            .replace(
              "{total}",
              formatOwnedAmount(addInrAmounts(directOwned, throughOwned)),
            )
            .replace("{through}", formatOwnedAmount(throughOwned))
        : hasThroughOwned
          ? englishCopy.ideas.owned_funds_only.replace(
              "{amount}",
              formatOwnedAmount(throughOwned),
            )
          : englishCopy.ideas.owned_direct_only.replace(
              "{amount}",
              formatOwnedAmount(directOwned),
            );
  const hint = asRecord(idea.entry_hint);
  const hintCode = textValue(hint.code);
  const hintParts = Number(hint.parts);
  const hintLine =
    hintCode === "wait"
      ? englishCopy.ideas.entry_hint_wait
      : hintCode === "buy_in_parts" && hintParts === 6
        ? englishCopy.ideas.entry_hint_buy_expensive
        : hintCode === "buy_in_parts" &&
            Number.isInteger(hintParts) &&
            hintParts > 0
          ? englishCopy.ideas.entry_hint_buy_parts
              .replace("{parts}", String(hintParts))
              .replace("{months}", String(hintParts))
          : null;
  const hintReasons = asArray(hint.reasons)
    .map((reason) => ENTRY_HINT_REASON_COPY[textValue(reason)])
    .filter((reason): reason is string => Boolean(reason))
    .slice(0, 2);
  return (
    <article
      key={`${symbol}-${index}`}
      className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
    >
      <header className="min-w-0">
        <h3
          className="break-words font-semibold [overflow-wrap:anywhere]"
          data-security
        >
          {displayName}
        </h3>
        {name ? (
          <p className="mt-1 text-xs text-muted-foreground" data-security>
            {symbol}
          </p>
        ) : null}
      </header>
      {ownershipLine ? (
        <p className="min-w-0 break-words text-sm text-muted-foreground [overflow-wrap:anywhere]">
          {ownershipLine}
        </p>
      ) : null}
      {hintLine ? <p className="text-sm font-medium">{hintLine}</p> : null}
      {idea.price ? (
        <p className="text-xs text-muted-foreground tabular-nums">
          Reference price{" "}
          {textValue(idea.market) === "US"
            ? formatUsd(textValue(idea.price), 0)
            : formatInr(textValue(idea.price), "compact")}
        </p>
      ) : null}
      {whyReasons.length ||
      whyRisk ||
      hintReasons.length ||
      ownedFunds.length ||
      confidence ||
      sector ||
      valuation ? (
        <details>
          <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
            {englishCopy.ideas.why_this_idea}
          </summary>
          <div className="grid gap-2 pb-2 text-sm text-muted-foreground">
            {confidence || sector || valuation ? (
              <div className="flex flex-wrap gap-2">
                {confidence ? (
                  <span className="rounded-full bg-muted px-2 py-1 text-xs">
                    {confidence}
                  </span>
                ) : null}
                {sector ? (
                  <span className="rounded-full bg-muted px-2 py-1 text-xs">
                    {sector}
                  </span>
                ) : null}
                {valuation ? (
                  <span className="rounded-full bg-muted px-2 py-1 text-xs">
                    {valuation}
                  </span>
                ) : null}
              </div>
            ) : null}
            {whyReasons.length ? (
              <ul className="grid gap-1">
                {whyReasons.map((reason, reasonIndex) => (
                  <li key={`${symbol}-reason-${reasonIndex}`}>{reason}</li>
                ))}
              </ul>
            ) : null}
            {whyRisk ? (
              <p>
                <strong className="font-medium text-foreground">
                  {englishCopy.ideas.main_risk}:{" "}
                </strong>
                {whyRisk}
              </p>
            ) : null}
            {hintReasons.length ? (
              <p>
                <strong className="font-medium text-foreground">
                  {englishCopy.ideas.about_timing}:{" "}
                </strong>
                {hintReasons.join(" ")}
              </p>
            ) : null}
            {ownedFunds.length ? (
              <div>
                <p className="font-medium text-foreground">
                  {englishCopy.ideas.owned_details_title}
                </p>
                <ul className="grid gap-1">
                  {ownedFunds.map((rawFund, fundIndex) => {
                    const fund = asRecord(rawFund);
                    const estimate =
                      fund.estimated === true
                        ? englishCopy.ideas.owned_estimate
                        : "";
                    return (
                      <li
                        className="break-words [overflow-wrap:anywhere]"
                        key={`${symbol}-fund-${fundIndex}`}
                      >
                        {englishCopy.ideas.owned_fund_detail
                          .replace("{name}", textValue(fund.name))
                          .replace(
                            "{amount}",
                            formatOwnedAmount(
                              textValue(fund.amount ?? fund.amount_inr, "0"),
                            ),
                          )
                          .replace("{estimate}", estimate)}
                      </li>
                    );
                  })}
                </ul>
              </div>
            ) : null}
          </div>
        </details>
      ) : null}
      {tranche ? <p className="text-sm font-medium">{tranche}</p> : null}
    </article>
  );
}

function IdeaGroups({
  groups,
  actions,
}: {
  groups: Map<string, unknown[]>;
  actions?: ActionResponse[];
}) {
  return (
    <div className="grid gap-3">
      {[...groups].map(([market, ideas]) => (
        <div key={market}>
          <h3 className="mb-2 text-xs font-semibold text-muted-foreground">
            {ideaMarketLabel(market)}
          </h3>
          <div className="grid gap-3">
            {ideas.map((raw, index) => (
              <ResearchIdea
                key={`${market}-${index}`}
                raw={raw}
                actions={actions}
                index={index}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function ResearchCard({
  ideas,
  actions,
}: {
  ideas: unknown[];
  actions?: ActionResponse[];
}) {
  const groups = splitIdeasByMarket(ideas);
  const hasIdeas = ideas.length > 0;
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">
        {englishCopy.ideas.weekly_title}
      </h2>
      <details className="border-b border-line pb-2">
        <summary className="min-h-11 cursor-pointer py-1 text-sm font-medium text-muted-foreground">
          {englishCopy.ideas.term_prompt}
        </summary>
        <p className="pb-1 text-sm text-muted-foreground">
          {englishCopy.ideas.confidence_help}
        </p>
      </details>
      {hasIdeas ? (
        <>
          <IdeaGroups groups={groups.top} actions={actions} />
          {groups.rest.size ? (
            <details>
              <summary className="min-h-12 cursor-pointer border-t border-line pt-3 text-sm font-semibold text-brand">
                {englishCopy.ideas.see_all}
              </summary>
              <div className="pt-3">
                <IdeaGroups groups={groups.rest} actions={actions} />
              </div>
            </details>
          ) : null}
          {groups.alreadyOwned.length ? (
            <details className="border-t border-line pt-3">
              <summary className="min-h-12 cursor-pointer text-sm font-semibold text-brand">
                {englishCopy.ideas.already_in_funds_title.replace(
                  "{count}",
                  String(groups.alreadyOwned.length),
                )}
              </summary>
              <div className="grid gap-3 pt-2">
                <p className="text-sm text-muted-foreground">
                  {englishCopy.ideas.already_enough_intro}
                </p>
                <IdeaGroups
                  groups={groupIdeasByMarket(groups.alreadyOwned)}
                  actions={actions}
                />
              </div>
            </details>
          ) : null}
        </>
      ) : (
        <EmptyState
          title="No new ideas this week"
          description="Open Plan to review your targets and saved money."
          icon={<Lightbulb aria-hidden="true" className="size-5" />}
        />
      )}
    </Card>
  );
}

function IndexFundsCard({
  research,
  showSetMonthlyAmount,
  showMonthlyBuying,
}: {
  research?: ResearchIdeasResponse;
  showSetMonthlyAmount: boolean;
  showMonthlyBuying: boolean;
}) {
  const funds = asArray(research?.index_funds).map(asRecord);
  if (!funds.length) return null;

  const copy = englishCopy.ideas;
  const market = asRecord(research?.us_market);
  const marketLabel = textValue(market.label);
  const marketLine =
    marketLabel === "cheap"
      ? copy.index_market_cheap
      : marketLabel === "fair"
        ? copy.index_market_fair
        : marketLabel === "expensive"
          ? copy.index_market_expensive
          : copy.index_market_unavailable;
  const extraMoneyParts = Number(market.extra_money_parts);
  const brokerNames = asRecord(englishCopy.denmark.brokers);
  const rowLabels: Record<string, string> = {
    global_all_world: copy.index_world_fund,
    global_sp500: copy.index_sp500_fund,
    global_nasdaq100: copy.index_nasdaq100_fund,
  };
  const mainFund =
    funds.find((fund) => textValue(fund.role) === "plan_main") ?? funds[0];
  const otherFunds = funds.filter((fund) => fund !== mainFund);
  const renderFund = (fund: Record<string, unknown>, index: number) => {
    const exposure = textValue(fund.exposure);
    const instrument = asRecord(fund.instrument);
    const listedName = textValue(instrument.listed_name);
    const searchCode = textValue(instrument.isin);
    const brokers = asArray(instrument.brokers)
      .map((broker) => textValue(broker))
      .map((broker) => textValue(brokerNames[broker], broker));
    const secondLine = [
      instrument.ask_eligible ? englishCopy.denmark.account_ask : null,
      brokers.length ? brokers.join(", ") : null,
    ]
      .filter(Boolean)
      .join(", ");
    return (
      <li
        key={`${exposure}-${index}`}
        className="border-t border-line pt-3 first:border-0 first:pt-0"
      >
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-sm font-semibold">
            {textValue(rowLabels[exposure], copy.index_world_fund)}
          </h3>
          {textValue(fund.role) === "plan_main" ? (
            <span className="rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground">
              {copy.index_plan_main}
            </span>
          ) : null}
        </div>
        <p className="mt-1 text-sm">{textValue(instrument.name)}</p>
        {secondLine ? (
          <p className="mt-1 text-xs text-muted-foreground">{secondLine}</p>
        ) : null}
        {listedName || searchCode ? (
          <details>
            <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
              {copy.index_find_it_at_broker}
            </summary>
            <div className="grid gap-1 pb-1 text-sm text-muted-foreground">
              {listedName ? <p>{listedName}</p> : null}
              {searchCode ? (
                <p>
                  {copy.index_search_code_label}: {searchCode}
                </p>
              ) : null}
              {instrument.accumulating ? (
                <p>{copy.index_reinvests_dividends}</p>
              ) : null}
            </div>
          </details>
        ) : null}
      </li>
    );
  };

  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">{copy.index_funds_title}</h2>
      <div className="grid gap-1 text-sm">
        <p>{marketLine}</p>
        {Number.isInteger(extraMoneyParts) && extraMoneyParts > 0 ? (
          <p>
            {copy.index_extra_parts.replace("{parts}", String(extraMoneyParts))}
          </p>
        ) : null}
        {showMonthlyBuying ? <p>{copy.index_monthly_buying}</p> : null}
      </div>
      <ul className="grid gap-3">
        {mainFund ? renderFund(mainFund, 0) : null}
      </ul>
      {otherFunds.length ? (
        <details>
          <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
            {copy.index_other_funds}
          </summary>
          <ul className="grid gap-3 pb-1">
            {otherFunds.map((fund, index) => renderFund(fund, index + 1))}
          </ul>
        </details>
      ) : null}
      <details>
        <summary className="min-h-11 cursor-pointer py-2 text-sm text-muted-foreground">
          {copy.index_why}
        </summary>
        <div className="grid gap-2 pb-1 text-sm text-muted-foreground">
          <p>{copy.index_world_reason_one}</p>
          <p>{copy.index_world_reason_two}</p>
          <p>{copy.index_us_reason_one}</p>
          <p>{copy.index_us_reason_two}</p>
        </div>
      </details>
      {showSetMonthlyAmount ? (
        <Link
          to="/"
          className="inline-flex min-h-11 w-fit items-center text-sm font-medium text-muted-foreground"
        >
          {englishCopy.denmark.set_monthly_amount}
        </Link>
      ) : null}
    </Card>
  );
}

export function IdeasPage() {
  const research = useFeatureQuery(
    ["c22", "research-ideas"],
    "/v1/research/ideas",
    getResearchIdeas,
  );
  const actions = useFeatureQuery(
    ["c22", "actions"],
    "/v1/actions",
    getActions,
  );
  const plan = useFeatureQuery(["c22", "plan"], "/v1/plan", getPlan);
  const investing = useFeatureQuery(
    ["c22", "investing"],
    "/v1/me/investing",
    getInvestingPrefs,
  );
  if (!research.supported || isMissingFeature(research.error))
    return <ComingSoon title="Research ideas are coming soon" />;
  if (research.isPending) return <ScreenLoading label="Loading ideas" />;
  if (research.error && !research.data && !isMissingFeature(research.error))
    return <ScreenError title="Ideas couldn’t load" />;
  const planValue = plan.data as WeeklyPlan | undefined;
  const planMembers = asArray(planValue?.members).map(asRecord);
  const denmarkResident =
    asRecord(planValue?.residency).country === "DK" ||
    planMembers.some((member) => asRecord(member.residency).country === "DK");
  const denmarkBuyCount = asArray(planValue?.abroad_buy_list).length;
  const monthlyAmountSet = isPositiveAmount(
    investing.data?.monthly_sip_capacity,
  );
  const showSetMonthlyAmount =
    denmarkResident &&
    !investing.isPending &&
    Boolean(investing.data) &&
    !monthlyAmountSet;
  return (
    <div data-copy-screen="ideas" className="grid gap-4">
      <header>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.ideas.intro}
        </p>
      </header>
      {denmarkResident && denmarkBuyCount ? (
        <DenmarkBuyCard plan={planValue} />
      ) : null}
      <ResearchCard
        ideas={asArray(research.data?.ideas)}
        actions={actions.data}
      />
      <IndexFundsCard
        research={research.data}
        showSetMonthlyAmount={showSetMonthlyAmount}
        showMonthlyBuying={monthlyAmountSet}
      />
      {denmarkResident && !denmarkBuyCount ? (
        <div className="grid gap-1 text-sm text-muted-foreground">
          <p>{englishCopy.denmark.no_buy_short}</p>
        </div>
      ) : null}
      <SelectorFunds research={research.data} />
      <Link
        to="/ideas/world"
        className="min-h-12 py-3 text-sm font-semibold text-brand"
      >
        {englishCopy.ideas.wider_link} <span aria-hidden="true">→</span>
      </Link>
    </div>
  );
}

export { valuationWords, referencedTranche, categoryName };
