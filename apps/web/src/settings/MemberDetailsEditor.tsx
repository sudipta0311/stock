import { useState, type FormEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError } from "../api/client";
import type { MemberPatch, SettingsMember } from "./api";
import { updateSettingsMember } from "./api";
import { useHousehold } from "../household/HouseholdContext";
import { Button, Field, Select, Sheet } from "../ui";
import englishCopy from "../copy/en.json";

type ResidenceStatus = "RI" | "NRI" | "RNOR";
type Relationship = MemberPatch["relationship"];

function residenceStatus(member: SettingsMember): ResidenceStatus {
  if (member.residency.status === "NRI") return "NRI";
  if (member.residency.status === "RNOR") return "RNOR";
  return "RI";
}

function residenceLabel(status: ResidenceStatus): string {
  if (status === "NRI") return englishCopy.household.residency_nri_dk;
  if (status === "RNOR") return englishCopy.household.residency_rnor;
  return englishCopy.household.residency_ri;
}

function nextResidency(status: ResidenceStatus, year: string) {
  return {
    status,
    country: status === "NRI" ? "DK" : "IN",
    settle_plan: status === "NRI" ? "undecided" : "india",
    planned_return_year:
      status === "NRI" && /^\d{4}$/.test(year.trim())
        ? Number(year.trim())
        : null,
  } as const;
}

export function MemberDetailsEditor({
  member,
  disabled,
}: {
  member: SettingsMember;
  disabled: boolean;
}) {
  const queryClient = useQueryClient();
  const { meMemberId } = useHousehold();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(member.display_name);
  const [relationship, setRelationship] = useState<Relationship>(
    member.relationship as Relationship,
  );
  const [status, setStatus] = useState<ResidenceStatus>(
    residenceStatus(member),
  );
  const [returnYear, setReturnYear] = useState(
    member.residency.planned_return_year
      ? String(member.residency.planned_return_year)
      : "",
  );
  const [taxRegime, setTaxRegime] = useState<"old" | "new">(
    member.tax.tax_regime ?? "new",
  );
  const [taxSlab, setTaxSlab] = useState(member.tax.tax_slab_pct ?? "");
  const [birthYear, setBirthYear] = useState(
    member.tax.birth_year ? String(member.tax.birth_year) : "",
  );
  const [annualIncome, setAnnualIncome] = useState(
    member.tax.annual_income ?? "",
  );
  const [incomeCurrency, setIncomeCurrency] = useState<"INR" | "DKK">(
    member.tax.annual_income_currency === "DKK" ? "DKK" : "INR",
  );
  const [monthlySip, setMonthlySip] = useState(
    member.investing.monthly_sip_capacity || "0",
  );
  const [newMoney, setNewMoney] = useState(
    member.investing.new_money_this_week || "0",
  );
  const [broker, setBroker] = useState(member.investing.broker ?? "nordnet");
  const [askStartValue, setAskStartValue] = useState(
    member.investing.ask_value_start_of_year_dkk ?? "",
  );
  const [hasNriShareAccount, setHasNriShareAccount] = useState(
    member.investing.has_indian_nri_share_account ?? false,
  );
  const [hasDanishPension, setHasDanishPension] = useState(
    member.investing.has_danish_pension_account ?? false,
  );
  const [usStockPicksOptIn, setUsStockPicksOptIn] = useState(
    member.investing.us_stock_picks_opt_in ?? false,
  );
  const [usStockPicksPct, setUsStockPicksPct] = useState(
    member.investing.us_stock_picks_pct ?? "10",
  );
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");
  const residenceChanged =
    status !== residenceStatus(member) ||
    (status === "NRI" &&
      (Number(returnYear) || null) !==
        (member.residency.planned_return_year ?? null));

  async function persistMember(): Promise<void> {
    setSaving(true);
    setMessage("");
    try {
      const tax: MemberPatch["tax"] = {
        annual_income: annualIncome.trim() || null,
        annual_income_currency: incomeCurrency,
        birth_year: birthYear.trim() ? Number(birthYear) : null,
        employment_type: member.tax.employment_type,
        fy_realised_equity_ltcg: member.tax.fy_realised_equity_ltcg,
        fy_realised_equity_stcg: member.tax.fy_realised_equity_stcg,
        surcharge_pct: member.tax.surcharge_pct,
        tax_regime: taxRegime,
        tax_slab_pct: taxSlab || null,
      };
      const investing: MemberPatch["investing"] = {
        monthly_sip_capacity: monthlySip.trim() || "0",
        new_money_this_week: newMoney.trim() || "0",
        existing_sips: member.investing.existing_sips ?? [],
        has_lrs_broker: member.investing.has_lrs_broker,
        prefers_no_paperwork: member.investing.prefers_no_paperwork,
        satellite_pct: member.investing.satellite_pct,
        ask_used_this_fy_dkk: member.investing.ask_used_this_fy_dkk,
        ask_value_start_of_year_dkk: askStartValue.trim() || null,
        lrs_used_this_fy: member.investing.lrs_used_this_fy,
        broker,
        has_indian_nri_share_account: hasNriShareAccount,
        has_danish_pension_account: hasDanishPension,
        us_stock_picks_opt_in: usStockPicksOptIn,
        us_stock_picks_pct: usStockPicksPct,
        birth_year: birthYear.trim()
          ? Number(birthYear)
          : (member.investing.birth_year ?? null),
      };
      const patch: MemberPatch = {
        display_name: name.trim(),
        relationship,
        residency: nextResidency(status, returnYear),
        tax,
        investing,
      };
      await updateSettingsMember(member.id, patch);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["settings", "members"] }),
        queryClient.invalidateQueries({
          queryKey: ["household-context", "members"],
        }),
        queryClient.invalidateQueries({ queryKey: ["c27b"] }),
        queryClient.invalidateQueries({ queryKey: ["c22"] }),
      ]);
      setOpen(false);
      setConfirmOpen(false);
      setMessage(englishCopy.household.settings.details_saved);
    } catch (error) {
      setMessage(
        error instanceof ApiError && error.status === 409
          ? englishCopy.household.residence_error
          : englishCopy.household.settings.details_error,
      );
    } finally {
      setSaving(false);
    }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    if (!name.trim()) {
      setMessage(englishCopy.settings.family.name_error);
      return;
    }
    if (residenceChanged) setConfirmOpen(true);
    else void persistMember();
  }

  return (
    <div className="grid gap-2 border-t border-line pt-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-muted-foreground">
          {residenceLabel(residenceStatus(member))}
        </p>
        <Button
          type="button"
          variant="ghost"
          className="min-h-10 px-3 py-2"
          disabled={disabled}
          onClick={() => {
            setName(member.display_name);
            setRelationship(member.relationship as Relationship);
            setStatus(residenceStatus(member));
            setReturnYear(
              member.residency.planned_return_year
                ? String(member.residency.planned_return_year)
                : "",
            );
            setTaxRegime(member.tax.tax_regime ?? "new");
            setTaxSlab(member.tax.tax_slab_pct ?? "");
            setBirthYear(
              member.tax.birth_year ? String(member.tax.birth_year) : "",
            );
            setAnnualIncome(member.tax.annual_income ?? "");
            setIncomeCurrency(
              member.tax.annual_income_currency === "DKK" ? "DKK" : "INR",
            );
            setMonthlySip(member.investing.monthly_sip_capacity || "0");
            setNewMoney(member.investing.new_money_this_week || "0");
            setBroker(member.investing.broker ?? "nordnet");
            setAskStartValue(
              member.investing.ask_value_start_of_year_dkk ?? "",
            );
            setHasNriShareAccount(
              member.investing.has_indian_nri_share_account ?? false,
            );
            setHasDanishPension(
              member.investing.has_danish_pension_account ?? false,
            );
            setUsStockPicksOptIn(
              member.investing.us_stock_picks_opt_in ?? false,
            );
            setUsStockPicksPct(member.investing.us_stock_picks_pct ?? "10");
            setOpen((current) => !current);
            setMessage("");
          }}
        >
          {englishCopy.household.settings.edit_member}
        </Button>
      </div>
      {open ? (
        <form
          onSubmit={submit}
          className="grid gap-3 rounded-xl bg-muted/50 p-3"
        >
          <Field
            id={`member-name-${member.id}`}
            label={englishCopy.settings.family.name}
          >
            <input
              id={`member-name-${member.id}`}
              value={name}
              onChange={(event) => setName(event.currentTarget.value)}
              maxLength={40}
              required
              className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
            />
          </Field>
          <Select
            id={`member-relationship-${member.id}`}
            label={englishCopy.settings.family.relationship}
            value={relationship ?? "other"}
            onChange={(event) =>
              setRelationship(event.currentTarget.value as Relationship)
            }
          >
            {(member.id === meMemberId
              ? ["self", "spouse", "parent", "other"]
              : ["spouse", "child", "parent", "other"]
            ).map((value) => (
              <option key={value} value={value}>
                {
                  englishCopy.household.member_relationship[
                    value as keyof typeof englishCopy.household.member_relationship
                  ]
                }
              </option>
            ))}
          </Select>
          <Select
            id={`member-residence-${member.id}`}
            label={englishCopy.household.residency_title}
            value={status}
            onChange={(event) =>
              setStatus(event.currentTarget.value as ResidenceStatus)
            }
          >
            <option value="RI">{englishCopy.household.residency_ri}</option>
            <option value="NRI">
              {englishCopy.household.residency_nri_dk}
            </option>
            <option value="RNOR">{englishCopy.household.residency_rnor}</option>
          </Select>
          <details>
            <summary className="cursor-pointer text-sm font-medium">
              {englishCopy.household.add_details}
            </summary>
            <div className="mt-3 grid gap-3">
              {status === "NRI" ? (
                <>
                  <Select
                    id={`member-broker-${member.id}`}
                    label={englishCopy.denmark.broker}
                    value={broker}
                    onChange={(event) =>
                      setBroker(
                        event.currentTarget.value as
                          "nordnet" | "saxo" | "danske_bank" | "other",
                      )
                    }
                  >
                    <option value="nordnet">
                      {englishCopy.denmark.brokers.nordnet}
                    </option>
                    <option value="saxo">
                      {englishCopy.denmark.brokers.saxo}
                    </option>
                    <option value="danske_bank">
                      {englishCopy.denmark.brokers.danske_bank}
                    </option>
                    <option value="other">
                      {englishCopy.denmark.other_broker}
                    </option>
                  </Select>
                  <Field
                    id={`member-ask-start-value-${member.id}`}
                    label={englishCopy.denmark.ask_start_value_label}
                    help={englishCopy.denmark.ask_start_value_hint}
                  >
                    <input
                      id={`member-ask-start-value-${member.id}`}
                      type="number"
                      min="0"
                      step="any"
                      value={askStartValue}
                      onChange={(event) =>
                        setAskStartValue(event.currentTarget.value)
                      }
                      className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                    />
                  </Field>
                  <label className="flex min-h-10 items-center gap-3 text-sm">
                    <input
                      type="checkbox"
                      checked={hasNriShareAccount}
                      onChange={(event) =>
                        setHasNriShareAccount(event.currentTarget.checked)
                      }
                      className="size-5 accent-brand"
                    />
                    {englishCopy.denmark.nri_account}
                  </label>
                  <label className="flex min-h-10 items-center gap-3 text-sm">
                    <input
                      type="checkbox"
                      checked={hasDanishPension}
                      onChange={(event) =>
                        setHasDanishPension(event.currentTarget.checked)
                      }
                      className="size-5 accent-brand"
                    />
                    {englishCopy.denmark.pension_account}
                  </label>
                  <label className="flex min-h-10 items-center gap-3 text-sm">
                    <input
                      type="checkbox"
                      checked={usStockPicksOptIn}
                      onChange={(event) =>
                        setUsStockPicksOptIn(event.currentTarget.checked)
                      }
                      className="size-5 accent-brand"
                    />
                    {englishCopy.denmark.stock_picks_opt_in}
                  </label>
                  {usStockPicksOptIn ? (
                    <Field
                      id={`member-us-stock-picks-pct-${member.id}`}
                      label={englishCopy.denmark.stock_picks_pct}
                    >
                      <input
                        id={`member-us-stock-picks-pct-${member.id}`}
                        type="number"
                        min="0"
                        max="25"
                        step="0.1"
                        value={usStockPicksPct}
                        onChange={(event) =>
                          setUsStockPicksPct(event.currentTarget.value)
                        }
                        className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                      />
                    </Field>
                  ) : null}
                </>
              ) : null}
              {status === "NRI" ? (
                <Field
                  id={`member-return-year-${member.id}`}
                  label={englishCopy.household.return_year}
                >
                  <input
                    id={`member-return-year-${member.id}`}
                    type="number"
                    min="1900"
                    max="9999"
                    value={returnYear}
                    onChange={(event) =>
                      setReturnYear(event.currentTarget.value)
                    }
                    className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                  />
                </Field>
              ) : null}
              <Select
                id={`member-tax-regime-${member.id}`}
                label={englishCopy.settings.family.tax_regime}
                value={taxRegime}
                onChange={(event) =>
                  setTaxRegime(event.currentTarget.value as "old" | "new")
                }
              >
                <option value="new">
                  {englishCopy.settings.family.new_regime}
                </option>
                <option value="old">
                  {englishCopy.settings.family.old_regime}
                </option>
              </Select>
              <Select
                id={`member-tax-slab-${member.id}`}
                label={englishCopy.settings.family.income_bracket}
                value={taxSlab}
                onChange={(event) => setTaxSlab(event.currentTarget.value)}
              >
                <option value="">
                  {englishCopy.settings.family.no_bracket}
                </option>
                {[0, 5, 10, 15, 20, 25, 30].map((value) => (
                  <option key={value} value={String(value)}>
                    {value}%
                  </option>
                ))}
              </Select>
              <Field
                id={`member-birth-year-${member.id}`}
                label={englishCopy.settings.family.birth_year}
              >
                <input
                  id={`member-birth-year-${member.id}`}
                  type="number"
                  min="1900"
                  max="9999"
                  value={birthYear}
                  onChange={(event) => setBirthYear(event.currentTarget.value)}
                  className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                />
              </Field>
              <div className="grid grid-cols-[1fr_auto] gap-2">
                <Field
                  id={`member-annual-income-${member.id}`}
                  label={englishCopy.household.onboarding.annual_income}
                >
                  <input
                    id={`member-annual-income-${member.id}`}
                    inputMode="decimal"
                    value={annualIncome}
                    onChange={(event) =>
                      setAnnualIncome(event.currentTarget.value)
                    }
                    className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                  />
                </Field>
                <Select
                  id={`member-income-currency-${member.id}`}
                  label={englishCopy.household.onboarding.currency}
                  value={incomeCurrency}
                  onChange={(event) =>
                    setIncomeCurrency(
                      event.currentTarget.value as "INR" | "DKK",
                    )
                  }
                >
                  <option value="INR">INR</option>
                  <option value="DKK">DKK</option>
                </Select>
              </div>
              <Field
                id={`member-monthly-sip-${member.id}`}
                label={englishCopy.settings.family.monthly_sip}
              >
                <input
                  id={`member-monthly-sip-${member.id}`}
                  inputMode="decimal"
                  value={monthlySip}
                  onChange={(event) => setMonthlySip(event.currentTarget.value)}
                  className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                />
              </Field>
              <Field
                id={`member-new-money-${member.id}`}
                label={englishCopy.settings.family.this_week}
              >
                <input
                  id={`member-new-money-${member.id}`}
                  inputMode="decimal"
                  value={newMoney}
                  onChange={(event) => setNewMoney(event.currentTarget.value)}
                  className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                />
              </Field>
            </div>
          </details>
          {message ? (
            <p role="status" className="text-sm text-danger-foreground">
              {message}
            </p>
          ) : null}
          <Button type="submit" loading={saving}>
            {englishCopy.household.settings.save_member}
          </Button>
        </form>
      ) : null}
      {message && !open ? (
        <p role="status" className="text-sm text-muted-foreground">
          {message}
        </p>
      ) : null}
      <Sheet
        open={confirmOpen}
        onOpenChange={setConfirmOpen}
        title={englishCopy.household.residence_confirm_title}
      >
        <div className="grid gap-4">
          <p className="text-sm leading-6">
            {englishCopy.household.residence_confirm
              .replace("{name}", name.trim() || member.display_name)
              .replace("{residence}", residenceLabel(status).toLowerCase())}
          </p>
          <Button onClick={() => void persistMember()} loading={saving}>
            {englishCopy.household.confirm}
          </Button>
          <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
            {englishCopy.household.cancel}
          </Button>
        </div>
      </Sheet>
    </div>
  );
}
