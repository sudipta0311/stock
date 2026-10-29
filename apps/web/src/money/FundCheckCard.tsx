import { useEffect, useState } from "react";
import { Link } from "react-router";
import { apiFetch } from "../api/client";
import englishCopy from "../copy/en.json";
import { useHousehold } from "../household/HouseholdContext";
import { formatInr } from "../lib/money";
import { Button, Card } from "../ui";

interface FundCheckResponse {
  status: "fine" | "attention";
  summary_key: string;
  checks: Array<{
    reason_code: string;
    details: Record<string, unknown>;
    member_id?: string | null;
  }>;
  early_warning: Record<string, unknown> | null;
  details: {
    funds?: Array<Record<string, unknown>>;
    notes?: string[];
    member_id?: string | null;
  };
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function stringOrNumberValue(value: unknown): string | undefined {
  if (typeof value === "string") return value;
  return typeof value === "number" && Number.isFinite(value)
    ? String(value)
    : undefined;
}

function bucketLabel(bucket: string): string {
  const value = bucket.toLowerCase();
  const labels: Record<string, string> = {
    large: englishCopy.fund_check.large_companies,
    mid: englishCopy.fund_check.mid_sized_companies,
    small: englishCopy.fund_check.small_companies,
    flexi_multi: englishCopy.fund_check.mixed_companies,
    sector_thematic: englishCopy.fund_check.sector_funds,
    international: englishCopy.fund_check.international_funds,
    equity: englishCopy.fund_check.shares,
    hybrid: englishCopy.fund_check.mixed,
    debt: englishCopy.fund_check.bonds,
  };
  if (value.includes("large")) return englishCopy.fund_check.large_companies;
  if (value.includes("mid")) return englishCopy.fund_check.mid_sized_companies;
  if (value.includes("small")) return englishCopy.fund_check.small_companies;
  if (value.includes("sector") || value.includes("thematic")) {
    return englishCopy.fund_check.sector_funds;
  }
  if (value.includes("international") || value.includes("overseas")) {
    return englishCopy.fund_check.international_funds;
  }
  return labels[value] ?? englishCopy.fund_check.fund;
}

export function FundCheckCard() {
  const { selectedHouseholdId, activeMemberId } = useHousehold();
  const [response, setResponse] = useState<FundCheckResponse | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let active = true;
    const member = activeMemberId
      ? `?member=${encodeURIComponent(activeMemberId)}`
      : "";
    void apiFetch<FundCheckResponse>(`/v1/fund-check${member}`)
      .then((result) => {
        if (active) setResponse(result);
      })
      .catch(() => {
        if (active) setFailed(true);
      });
    return () => {
      active = false;
    };
  }, [activeMemberId, selectedHouseholdId]);

  const hasFunds = Boolean(response?.details.funds?.length);
  const familyWarning = response?.early_warning?.family;
  const warning =
    familyWarning && typeof familyWarning === "object"
      ? (familyWarning as Record<string, unknown>)
      : response?.early_warning;
  const warningBucket = stringValue(warning?.bucket);
  const warningMonths = stringOrNumberValue(warning?.months_until);
  const highlightLimit = Math.max(0, 3 - (warningBucket ? 1 : 0));
  const visibleChecks = response?.checks.slice(0, highlightLimit) ?? [];
  const visibleNotes =
    response?.details.notes?.slice(0, highlightLimit - visibleChecks.length) ??
    [];

  return (
    <Card className="grid gap-3" aria-labelledby="fund-check-heading">
      <div>
        <h2 id="fund-check-heading" className="text-base font-semibold">
          {englishCopy.fund_check.title}
        </h2>
        <p className="mt-1 text-base" aria-live="polite">
          {failed
            ? englishCopy.fund_check.unavailable
            : !response
              ? englishCopy.fund_check.loading
              : response.summary_key === "fund_check.upload"
                ? englishCopy.fund_check.upload_summary
                : response.status === "attention"
                  ? englishCopy.fund_check.attention_summary
                  : englishCopy.fund_check.good_summary}
        </p>
      </div>
      {warningBucket && warningMonths ? (
        <p className="text-base text-muted-foreground">
          {englishCopy.fund_check.early_warning
            .replace("{bucket}", bucketLabel(warningBucket).toLowerCase())
            .replace("{months}", warningMonths)}
        </p>
      ) : null}
      {hasFunds ? (
        <Button
          variant="secondary"
          aria-expanded={expanded}
          aria-controls="fund-check-details"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded
            ? englishCopy.fund_check.hide_details
            : englishCopy.fund_check.see_why}
        </Button>
      ) : (
        <Link
          className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground hover:bg-brand-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand"
          to="/import"
        >
          {englishCopy.fund_check.upload_statement}
        </Link>
      )}
      {hasFunds && response ? (
        <div id="fund-check-details" className="grid gap-3" hidden={!expanded}>
          {visibleChecks.length ? (
            <ul className="grid gap-2 text-sm">
              {visibleChecks.map((check, index) => (
                <li
                  key={`${check.reason_code}-${check.member_id ?? "family"}-${index}`}
                >
                  <p>
                    {englishCopy.fund_check.reasons[
                      check.reason_code as keyof typeof englishCopy.fund_check.reasons
                    ] ?? englishCopy.fund_check.reason_default}
                  </p>
                  {stringValue(check.details.yearly_saving_inr) ? (
                    <p className="mt-1 text-muted-foreground">
                      {englishCopy.fund_check.yearly_saving.replace(
                        "{amount}",
                        formatInr(
                          stringValue(check.details.yearly_saving_inr) ?? "0",
                          "compact",
                          0,
                        ),
                      )}
                    </p>
                  ) : null}
                  {stringValue(check.details.tax_inr) ? (
                    <p className="mt-1 text-muted-foreground">
                      {englishCopy.fund_check.estimated_tax.replace(
                        "{amount}",
                        formatInr(
                          stringValue(check.details.tax_inr) ?? "0",
                          "compact",
                          0,
                        ),
                      )}
                    </p>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm">{englishCopy.fund_check.no_changes}</p>
          )}
          {(response.details.funds ?? []).map((fund, index) => {
            const name = stringValue(fund.name) ?? englishCopy.fund_check.fund;
            const subCategory = stringValue(fund.sub_category);
            const planType = stringValue(fund.plan_type);
            return (
              <details
                className="rounded-lg border border-line px-3 py-2"
                key={`${stringValue(fund.security_id) ?? name}-${index}`}
              >
                <summary className="cursor-pointer text-sm font-medium">
                  {name}
                </summary>
                <dl className="mt-2 grid gap-1 text-sm text-muted-foreground">
                  {subCategory ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.category}</dt>
                      <dd>{bucketLabel(subCategory.toLowerCase())}</dd>
                    </div>
                  ) : null}
                  {planType ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.plan}</dt>
                      <dd>
                        {planType === "direct"
                          ? englishCopy.fund_check.direct_plan
                          : englishCopy.fund_check.regular_plan}
                      </dd>
                    </div>
                  ) : null}
                  {stringValue(fund.option) ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.option}</dt>
                      <dd>
                        {fund.option === "idcw"
                          ? englishCopy.fund_check.payout_option
                          : englishCopy.fund_check.growth_option}
                      </dd>
                    </div>
                  ) : null}
                  {stringValue(fund.return_3y_pct) ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.three_year_comparison}</dt>
                      <dd>
                        {stringValue(fund.return_3y_pct)}%
                        {stringValue(fund.median_3y_pct)
                          ? ` / ${stringValue(fund.median_3y_pct)}%`
                          : ""}
                      </dd>
                    </div>
                  ) : null}
                  {stringValue(fund.return_5y_pct) ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.five_year_comparison}</dt>
                      <dd>
                        {stringValue(fund.return_5y_pct)}%
                        {stringValue(fund.median_5y_pct)
                          ? ` / ${stringValue(fund.median_5y_pct)}%`
                          : ""}
                      </dd>
                    </div>
                  ) : null}
                  {stringValue(fund.ter_pct) ? (
                    <div className="flex justify-between gap-3">
                      <dt>{englishCopy.fund_check.yearly_cost}</dt>
                      <dd>{stringValue(fund.ter_pct)}%</dd>
                    </div>
                  ) : null}
                </dl>
              </details>
            );
          })}
          {visibleNotes.map((note) => (
            <p className="text-xs text-muted-foreground" key={note}>
              {englishCopy.fund_check.notes[
                note as keyof typeof englishCopy.fund_check.notes
              ] ?? englishCopy.fund_check.note_default}
            </p>
          ))}
        </div>
      ) : null}
    </Card>
  );
}
