import { useEffect, useMemo, useState } from "react";
import { apiFetch } from "../api/client";
import englishCopy from "../copy/en.json";
import { useHousehold } from "../household/HouseholdContext";
import { Button, Card } from "../ui";

interface FundCheckData {
  details?: {
    funds?: Array<Record<string, unknown>>;
    sips?: Array<Record<string, unknown>>;
  };
}

interface FundRow {
  securityId: string;
  memberId: string;
  name: string;
  current: string;
}

function readText(value: unknown): string {
  return typeof value === "string" ? value : "";
}

export function FundSipFineTune() {
  const { selectedHouseholdId, isViewer } = useHousehold();
  const [data, setData] = useState<FundCheckData | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState<string | null>(null);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let active = true;
    void apiFetch<FundCheckData>("/v1/fund-check")
      .then((result) => {
        if (active) setData(result);
      })
      .catch(() => {
        if (active) setMessage(englishCopy.fund_check.settings_unavailable);
      });
    return () => {
      active = false;
    };
  }, [selectedHouseholdId]);

  const rows = useMemo<FundRow[]>(() => {
    const sips = data?.details?.sips ?? [];
    const sipByKey = new Map<string, string>();
    for (const sip of sips) {
      const key = `${readText(sip.member_id)}:${readText(sip.security_id)}`;
      sipByKey.set(key, readText(sip.monthly_amount_inr));
    }
    return (data?.details?.funds ?? []).flatMap((fund) => {
      const securityId = readText(fund.security_id);
      const memberId = readText(fund.member_id);
      if (!securityId || !memberId) return [];
      return [
        {
          securityId,
          memberId,
          name: readText(fund.name) || englishCopy.fund_check.fund,
          current: sipByKey.get(`${memberId}:${securityId}`) ?? "0.00",
        },
      ];
    });
  }, [data]);

  async function save(row: FundRow) {
    const amount = drafts[row.securityId] ?? row.current;
    if (!/^\d{1,12}(?:\.\d{1,2})?$/.test(amount)) {
      setMessage(englishCopy.fund_check.settings_amount_error);
      return;
    }
    setSaving(row.securityId);
    setMessage("");
    try {
      await apiFetch(
        `/v1/fund-check/sips/${encodeURIComponent(row.securityId)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            member_id: row.memberId,
            monthly_amount_inr: amount,
          }),
        },
      );
      setMessage(englishCopy.fund_check.settings_saved);
    } catch {
      setMessage(englishCopy.fund_check.settings_unavailable);
    } finally {
      setSaving(null);
    }
  }

  return (
    <Card className="grid gap-3">
      <div>
        <h2 id="fund-sip-settings-heading" className="text-base font-semibold">
          {englishCopy.fund_check.settings_title}
        </h2>
        <p className="mt-1 text-base text-muted-foreground">
          {englishCopy.fund_check.settings_help}
        </p>
      </div>
      {rows.length ? (
        // One row per fund is long with a real statement, so it starts folded.
        <details className="group">
          <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold text-brand">
            Show {rows.length} {rows.length === 1 ? "fund" : "funds"}
          </summary>
          <ul className="grid gap-3">
            {rows.map((row) => (
              <li
                className="grid gap-2 rounded-xl border border-line p-3"
                key={`${row.memberId}:${row.securityId}`}
              >
                <label
                  className="text-sm font-medium"
                  htmlFor={`fund-sip-${row.securityId}`}
                >
                  {row.name}
                </label>
                <div className="flex flex-wrap items-end gap-2">
                  <div className="grid gap-1">
                    <span className="text-xs text-muted-foreground">
                      {englishCopy.fund_check.monthly_amount}
                    </span>
                    <div className="relative">
                      <span
                        aria-hidden="true"
                        className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-muted-foreground"
                      >
                        ₹
                      </span>
                      <input
                        className="min-h-12 w-40 rounded-lg border border-line bg-background pl-9 pr-3 text-sm tabular-nums"
                        id={`fund-sip-${row.securityId}`}
                        inputMode="decimal"
                        min="0"
                        onChange={(event) =>
                          setDrafts((current) => ({
                            ...current,
                            [row.securityId]: event.target.value,
                          }))
                        }
                        step="0.01"
                        type="number"
                        value={drafts[row.securityId] ?? row.current}
                        disabled={isViewer}
                      />
                    </div>
                  </div>
                  <Button
                    disabled={isViewer}
                    loading={saving === row.securityId}
                    onClick={() => void save(row)}
                    type="button"
                    variant="secondary"
                  >
                    {englishCopy.fund_check.settings_save}
                  </Button>
                </div>
              </li>
            ))}
          </ul>
        </details>
      ) : (
        <p className="text-sm text-muted-foreground">
          {englishCopy.fund_check.settings_empty}
        </p>
      )}
      {message ? (
        <p className="text-sm" role="status">
          {message}
        </p>
      ) : null}
    </Card>
  );
}
