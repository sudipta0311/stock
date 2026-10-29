import { ExternalLink } from "lucide-react";
import { formatInr } from "../lib/money";
import {
  asArray,
  asRecord,
  formatScenarioLabel,
  textValue,
} from "../features/data";
import {
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
  useFeatureQuery,
} from "../features/api";
import { getPlan, getStress } from "../planning/api";
import type { WeeklyPlan } from "../planning/api";
import { Card } from "../ui";

/** "down ₹23.69 L" / "up ₹83,238": a bare "₹83,238" next to "-₹23.69 L" reads as a loss. */
function changeText(total: string): string {
  if (!total) return "—";
  const negative = total.trim().startsWith("-");
  const magnitude = total.trim().replace(/^[-+]/, "");
  if (/^0+(?:\.0+)?$/.test(magnitude)) return "no change";
  return `${negative ? "down" : "up"} ${formatInr(magnitude, "compact")}`;
}

function StressList({ stress }: { stress: Record<string, unknown> }) {
  const scenarios = Object.entries(stress);
  if (!scenarios.length)
    return (
      <p className="text-sm text-muted-foreground">
        Stress scenarios will appear when data is available.
      </p>
    );
  return (
    <div className="grid gap-3">
      {scenarios.map(([key, raw]) => {
        const scenario = asRecord(raw);
        const current = asRecord(scenario.current);
        const target = asRecord(scenario.target);
        const currentTotal = textValue(current.total);
        const targetTotal = textValue(target.total);
        if (!currentTotal && !targetTotal) return null;
        return (
          <article
            key={key}
            className="grid gap-1 border-t border-line pt-3 first:border-0 first:pt-0"
          >
            <h3 className="text-sm font-semibold">
              {formatScenarioLabel(key)}
            </h3>
            <p className="text-sm">
              Your money:{" "}
              <strong className="tabular-nums">
                {changeText(currentTotal)}
              </strong>
            </p>
            <p className="text-sm text-muted-foreground">
              With your target plan: {changeText(targetTotal)}
            </p>
          </article>
        );
      })}
    </div>
  );
}

function TiltNotes({ plan }: { plan: WeeklyPlan | undefined }) {
  const planRecord = asRecord(plan);
  const tilt = asRecord(
    planRecord.geo_tilts ??
      planRecord.geopolitical_tilts ??
      planRecord.world_view,
  );
  const changed = textValue(tilt.what_changed ?? tilt.change);
  const why = textValue(tilt.why ?? tilt.reason);
  const sources = asArray(tilt.sources ?? tilt.evidence_urls).filter(
    (source): source is string => typeof source === "string",
  );
  if (!changed && !why && sources.length === 0) return null;
  return (
    <Card className="grid gap-2">
      <h2 className="text-base font-semibold">What we changed and why</h2>
      {changed ? <p className="text-sm">{changed}</p> : null}
      {why ? <p className="text-sm text-muted-foreground">{why}</p> : null}
      {sources.map((source) => {
        try {
          const url = new URL(source);
          if (url.protocol !== "https:" && url.protocol !== "http:")
            return null;
          return (
            <a
              key={source}
              className="inline-flex min-h-10 items-center gap-1 text-sm font-medium text-brand"
              href={url.href}
              target="_blank"
              rel="noreferrer"
            >
              Source <ExternalLink aria-hidden="true" className="size-3.5" />
            </a>
          );
        } catch {
          return null;
        }
      })}
    </Card>
  );
}

export function WorldPage() {
  const stress = useFeatureQuery(["c22", "stress"], "/v1/stress", getStress);
  const plan = useFeatureQuery(["c22", "plan"], "/v1/plan", getPlan);
  if (!stress.supported || isMissingFeature(stress.error))
    return <ComingSoon title="World & your money is coming soon" />;
  if (stress.isPending)
    return <ScreenLoading label="Loading stress scenarios" />;
  if (stress.error && !isMissingFeature(stress.error))
    return <ScreenError title="Stress scenarios couldn’t load" />;
  const stressData = asRecord(stress.data?.stress);
  return (
    <div className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">World & your money</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Illustrations of what selected market events could mean for your
          holdings.
        </p>
      </header>
      <Card className="grid gap-3">
        <h2 className="text-base font-semibold">If this happens…</h2>
        <StressList stress={stressData} />
      </Card>
      <TiltNotes plan={plan.data as WeeklyPlan | undefined} />
    </div>
  );
}

export { StressList };
