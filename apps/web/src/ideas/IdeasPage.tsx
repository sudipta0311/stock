import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router";
import { Lightbulb } from "lucide-react";
import { formatInr, formatUsd } from "../lib/money";
import { labelForCode } from "../lib/labels";
import {
  asArray,
  asRecord,
  instrumentLabel,
  textValue,
} from "../features/data";
import {
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import {
  dismissHighlight,
  getActions,
  getPlan,
  getResearchIdeas,
} from "../planning/api";
import type {
  ActionResponse,
  ResearchIdeasResponse,
  WeeklyPlan,
} from "../planning/api";
import { Badge, Card, EmptyState, Term } from "../ui";
import englishCopy from "../copy/en.json";
import { Highlights } from "../features/Highlights";

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
                <div className="text-xs">
                  {copy.fund_code_label}:{" "}
                  <Term
                    code
                    explanation={copy.fund_code_help}
                    prompt={copy.term_prompt}
                    learnMoreHref="/help#term-isin"
                  >
                    {textValue(instrument.isin)}
                  </Term>
                </div>
                <p className="text-xs text-muted-foreground">
                  {textValue(line.account) === "ask"
                    ? copy.account_ask
                    : textValue(line.account) === "pension"
                      ? copy.account_pension
                      : copy.account_ordinary}
                  {" · "}
                  {copy.broker}:{" "}
                  {textValue(asRecord(copy.brokers)[broker], broker)}
                  {" · "}
                  {textValue(instrument.currency)}
                </p>
                <details>
                  <summary className="cursor-pointer text-sm font-semibold text-brand">
                    {copy.how}
                  </summary>
                  <ol className="mt-2 grid gap-1 pl-5 text-base list-decimal">
                    {howTo.map((step, stepIndex) => (
                      <li key={`${broker}-${stepIndex}`}>{String(step)}</li>
                    ))}
                  </ol>
                </details>
                <details>
                  <summary className="cursor-pointer text-sm text-muted-foreground">
                    {copy.why}
                  </summary>
                  <div className="mt-2 grid gap-2 text-base">
                    <p>{copy.why_ranked}</p>
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
      ) : (
        <p className="text-sm text-muted-foreground">{copy.no_buy}</p>
      )}
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

function valuationWords(value: string): string {
  const percentile = Number(value);
  if (!Number.isFinite(percentile)) return "Valuation range unavailable";
  if (percentile <= 20) return "Lower valuation range";
  if (percentile <= 50) return "Below the middle";
  if (percentile <= 80) return "Above the middle";
  return "Upper valuation range";
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

function CorePicksCard({
  actions,
  plan,
}: {
  actions?: ActionResponse[];
  plan?: WeeklyPlan;
}) {
  const picks = corePicks(actions, plan);
  const copy = englishCopy.ideas;
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">
          {copy.plan_categories_title}
        </h2>
      </div>
      {picks.length ? (
        <ul className="grid gap-2">
          {picks.map((pick) => (
            <li
              key={pick}
              className="flex min-h-14 items-center gap-3 border-t border-line pt-2 first:border-0 first:pt-0"
            >
              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <Lightbulb aria-hidden="true" className="size-4" />
              </span>
              <span className="text-sm">{pick}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          {copy.plan_categories_empty}
        </p>
      )}
    </Card>
  );
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

function TrackRecord({ research }: { research?: ResearchIdeasResponse }) {
  const track = asRecord(research?.track_record);
  const markets = asRecord(track.markets);
  const summaries = Object.values(markets)
    .map(asRecord)
    .filter((market) => market.status === "available")
    .map((market) => textValue(market.summary))
    .filter(Boolean);
  if (!summaries.length) return null;
  return (
    <section aria-labelledby="track-record-title" className="grid gap-1">
      <h3 id="track-record-title" className="text-sm font-semibold">
        {englishCopy.ideas.track_record_title}
      </h3>
      {summaries.slice(0, 2).map((summary) => (
        <p key={summary} className="text-sm text-muted-foreground">
          {summary}
        </p>
      ))}
    </section>
  );
}

function ResearchCard({
  ideas,
  actions,
}: {
  ideas: unknown[];
  actions?: ActionResponse[];
}) {
  return (
    <Card className="grid gap-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold">Research ideas</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            For learning and review. These are not product picks.
          </p>
        </div>
        <Badge>Research, not personal advice</Badge>
      </div>
      {ideas.length ? (
        ideas.map((raw, index) => {
          const idea = asRecord(raw);
          const symbol = textValue(idea.symbol, "Research idea");
          const confidence = textValue(idea.confidence);
          const valuation = textValue(idea.valuation_percentile);
          const tranche = referencedTranche(idea, actions);
          return (
            <article
              key={`${symbol}-${index}`}
              className="flex gap-3 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-muted text-muted-foreground">
                <Lightbulb aria-hidden="true" className="size-4" />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex items-start justify-between gap-3">
                  <h3 className="font-semibold" data-security>
                    {symbol}
                  </h3>
                  {confidence ? (
                    <span className="text-right text-xs text-muted-foreground">
                      {labelForCode(confidence)}
                    </span>
                  ) : null}
                </div>
                <p className="text-sm text-muted-foreground">
                  {textValue(idea.sector, "Sector unavailable")} ·{" "}
                  {valuationWords(valuation)}
                </p>
                {tranche ? (
                  <p className="mt-1 text-sm font-medium">{tranche}</p>
                ) : null}
                {idea.price ? (
                  <p className="mt-1 text-xs text-muted-foreground tabular-nums">
                    Reference price{" "}
                    {textValue(idea.market) === "US"
                      ? formatUsd(textValue(idea.price), 0)
                      : formatInr(textValue(idea.price), "compact")}
                  </p>
                ) : null}
              </div>
            </article>
          );
        })
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

export function IdeasPage() {
  const queryClient = useQueryClient();
  const hideHighlight = useMutation({
    mutationFn: dismissHighlight,
    onSuccess: () =>
      queryClient.invalidateQueries({ queryKey: ["c22", "plan"] }),
  });
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
  if (!research.supported || isMissingFeature(research.error))
    return <ComingSoon title="Research ideas are coming soon" />;
  if (research.isPending) return <ScreenLoading label="Loading ideas" />;
  if (research.error && !isMissingFeature(research.error))
    return <ScreenError title="Ideas couldn’t load" />;
  const planValue = plan.data as WeeklyPlan | undefined;
  const planMembers = asArray(planValue?.members).map(asRecord);
  const denmarkResident =
    asRecord(planValue?.residency).country === "DK" ||
    planMembers.some((member) => asRecord(member.residency).country === "DK");
  return (
    <div data-copy-screen="ideas" className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">Ideas</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Ideas are research to review, not a reason to rush.
        </p>
      </header>
      {denmarkResident ? (
        <DenmarkBuyCard plan={planValue} />
      ) : (
        <CorePicksCard actions={actions.data} plan={planValue} />
      )}
      <Highlights
        items={planValue?.highlights_by_screen?.ideas}
        onDismiss={(key) => hideHighlight.mutate(key)}
      />
      <Link
        to="/plan"
        data-primary="true"
        className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
      >
        Review your plan
      </Link>
      <details className="grid gap-4">
        <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
          {englishCopy.ideas.details_title}
        </summary>
        <TrackRecord research={research.data} />
        <SelectorFunds research={research.data} />
        <ResearchCard
          ideas={asArray(research.data?.ideas)}
          actions={actions.data}
        />
      </details>
      <Card className="grid gap-2">
        <h2 className="text-base font-semibold">Explore the wider picture</h2>
        <p className="text-sm text-muted-foreground">
          See how common market shocks could affect your current mix and your
          target plan.
        </p>
        <Link
          to="/ideas/world"
          className="inline-flex min-h-12 items-center gap-2 text-sm font-semibold text-brand"
        >
          World & your money <span aria-hidden="true">→</span>
        </Link>
      </Card>
    </div>
  );
}

export { valuationWords, referencedTranche, categoryName };
