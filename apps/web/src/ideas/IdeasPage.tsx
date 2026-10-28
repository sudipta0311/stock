import { Link } from "react-router";
import { formatInr } from "../lib/money";
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
import { getActions, getPlan, getResearchIdeas } from "../planning/api";
import type { ActionResponse, WeeklyPlan } from "../planning/api";
import { Badge, Card } from "../ui";

function actionPayload(action: ActionResponse): Record<string, unknown> {
  const payload = asRecord(action.payload);
  const nested = asRecord(payload.action);
  return Object.keys(nested).length ? nested : payload;
}

function categoryName(category: string, sleeve: string): string {
  if (category === "in_core_index") return "Nifty 50 index fund";
  if (category === "in_international_index_fund")
    return "S&P 500 index fund (in INR)";
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
  return (
    <Card className="grid gap-3">
      <div>
        <h2 className="text-base font-semibold">Core picks for you</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Categories from this week’s plan, with no product promotion.
        </p>
      </div>
      {picks.length ? (
        <ul className="grid gap-2">
          {picks.map((pick) => (
            <li key={pick} className="rounded-xl bg-muted p-3 text-sm">
              {pick}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-muted-foreground">
          Your core categories will appear when there is a planned action.
        </p>
      )}
    </Card>
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
              className="grid gap-2 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <div className="flex items-start justify-between gap-3">
                <h3 className="font-semibold">{symbol}</h3>
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
                <p className="text-sm font-medium">{tranche}</p>
              ) : null}
              {idea.price ? (
                <p className="text-xs text-muted-foreground">
                  Reference price {formatInr(textValue(idea.price), "compact")}
                </p>
              ) : null}
            </article>
          );
        })
      ) : (
        <p className="text-sm text-muted-foreground">
          There are no research ideas this week.
        </p>
      )}
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
  if (!research.supported || isMissingFeature(research.error))
    return <ComingSoon title="Research ideas are coming soon" />;
  if (research.isPending) return <ScreenLoading label="Loading ideas" />;
  if (research.error && !isMissingFeature(research.error))
    return <ScreenError title="Ideas couldn’t load" />;
  return (
    <div className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">Ideas</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Ideas are research to review, not a reason to rush.
        </p>
      </header>
      <CorePicksCard
        actions={actions.data}
        plan={plan.data as WeeklyPlan | undefined}
      />
      <ResearchCard
        ideas={asArray(research.data?.ideas)}
        actions={actions.data}
      />
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
