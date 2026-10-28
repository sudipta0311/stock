import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useForm, useWatch } from "react-hook-form";
import type { Path } from "react-hook-form";
import { ArrowLeft, ArrowRight, Check, ChevronDown } from "lucide-react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { ApiError, type ApiFieldError } from "../api/client";
import { formatInr } from "../lib/money";
import { Button, Card, Field, MoneyInput, MonthYearInput, Select } from "../ui";
import { institutionNames } from "./institutions.generated";
import { closeAsset, createAsset, getAsset, patchAsset } from "./api";
import type { AssetDetailResponse, AssetResponse } from "./api";
import { addPathForCategory } from "./config";
import type { ChecklistWorkflow } from "./config";
import { assetForms } from "./forms.generated";
import type {
  AssetDraft,
  AssetType,
  GeneratedAssetForm,
} from "./forms.generated";
import type { components } from "../api/schema";
import {
  type AssetFormValues,
  initialAssetValues,
  serializeAssetValues,
  validateAssetValues,
} from "./form-utils";
import { fieldCopy } from "./labels.en";
import { queueAssetWrite } from "./offline-queue";
import { useHousehold } from "../household/HouseholdContext";
import { getPlan } from "../planning/api";
import { apiPathAvailable } from "../api/feature-support";
import englishCopy from "../copy/en.json";

type PlanMember = components["schemas"]["PlanMember"];

const INSURANCE_TYPES = [
  "traditional_policy",
  "term_insurance",
  "health_insurance",
  "ulip",
] as const;
const EMPTY_DETAILS: Record<string, unknown> = {};

const assetTitles: Record<string, string> = {
  bond: "Bond",
  epf: "EPF / VPF",
  fd: "Fixed deposit",
  gold: "Gold",
  health_insurance: "Health insurance",
  loan: "Loan",
  nps: "NPS",
  po_scheme: "Post Office investment",
  ppf: "PPF",
  property: "Property",
  rd: "Recurring deposit",
  savings: "Savings account",
  term_insurance: "Term insurance",
  traditional_policy: "LIC / Insurance policy",
  ulip: "ULIP",
};

function optionLabel(value: string | number): string {
  const labels: Record<string, string> = {
    bond_54ec: "Section 54EC bond",
    corporate: "Corporate bond",
    gsec: "Government security",
    rbi_frsb: "RBI floating rate bond",
    tbill: "Treasury bill",
    tax_free: "Tax-free bond",
    half_yearly: "Half-yearly",
    yearly: "Yearly",
    cumulative: "Cumulative",
    monthly: "Monthly",
    quarterly: "Quarterly",
    simple: "Simple",
    sgb: "Sovereign Gold Bond",
    etf: "Gold ETF",
    mf: "Gold mutual fund",
    physical: "Physical gold",
    digital: "Digital gold",
    home: "Home loan",
    car: "Car loan",
    personal: "Personal loan",
    education: "Education loan",
    credit_card: "Credit card",
    other: "Other",
    nsc: "National Savings Certificate (NSC)",
    kvp: "Kisan Vikas Patra (KVP)",
    scss: "Senior Citizens Savings Scheme (SCSS)",
    mis: "Monthly Income Scheme (MIS)",
    td: "Time deposit",
    rd: "Recurring deposit",
    ssy: "Sukanya Samriddhi Yojana (SSY)",
    endowment: "Endowment",
    money_back: "Money-back",
    whole_life: "Whole life",
  };
  const text = String(value);
  return (
    labels[text] ??
    text.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase())
  );
}

function fieldHelp(type: AssetType, name: string, required: boolean): string {
  const copy = fieldCopy(type, name);
  if (copy.help) return copy.help;
  if (name === "member_id") return englishCopy.household_ui.member_help;
  if (!required)
    return "Leave blank if you are not sure. Rough numbers are fine.";
  if (name.endsWith("_year")) return "The year is enough.";
  return "Rough numbers are fine. You can edit this anytime.";
}

function isInstitutionField(name: string): boolean {
  return (
    name === "bank" ||
    name === "insurer" ||
    name === "institution" ||
    name === "held_at"
  );
}

function errorForField(
  name: string,
  fieldErrors: ApiFieldError[],
  clientErrors: Record<string, string>,
): string | undefined {
  return (
    clientErrors[name] ??
    fieldErrors.find((error) => error.field === name)?.message
  );
}

function useWorkflowState(): {
  workflow?: ChecklistWorkflow;
  memberId?: string | null;
} {
  const location = useLocation();
  const state = location.state as {
    workflow?: ChecklistWorkflow;
    memberId?: string | null;
  } | null;
  return { workflow: state?.workflow, memberId: state?.memberId };
}

function eligibilityInstrument(
  assetType: AssetType,
  values: AssetFormValues,
): string | null {
  if (assetType === "ppf") return "ppf";
  if (assetType === "nps") return "nps";
  if (assetType === "epf") return "epf";
  if (assetType === "gold") {
    return values.kind === "sgb" ? "sgb" : "gold_etf";
  }
  if (assetType === "po_scheme") {
    const scheme = String(values.scheme ?? "");
    if (scheme === "td" || scheme === "rd" || scheme === "mis") return "pomis";
    return scheme || null;
  }
  if (assetType === "fd") return "bank_fd";
  return null;
}

function eligibilityCondition(condition: string): string {
  const conditions: Record<string, string> = {
    ...englishCopy.household_ui.conditions,
  };
  return conditions[condition] ?? englishCopy.household_ui.eligibility_unknown;
}

function eligibilityNote(
  member: PlanMember | undefined,
  instrument: string | null,
): string | null {
  if (!member || !instrument) return null;
  const status = member.eligibility_status?.[instrument];
  if (status === "allowed") return englishCopy.household_ui.eligibility_allowed;
  const condition = member.eligibility_conditions?.[instrument];
  const conditionText = condition
    ? eligibilityCondition(condition)
    : englishCopy.household_ui.eligibility_unknown;
  if (status === "not_allowed") {
    return englishCopy.household_ui.eligibility_not_allowed.replace(
      "{condition}",
      conditionText,
    );
  }
  if (status === "conditional") {
    return englishCopy.household_ui.eligibility_conditional.replace(
      "{condition}",
      conditionText,
    );
  }
  return null;
}

export function USInvestmentStep() {
  const navigate = useNavigate();
  const { workflow } = useWorkflowState();
  function next() {
    const nextIndex = (workflow?.index ?? -1) + 1;
    const category = workflow?.categories[nextIndex];
    if (workflow && category) {
      navigate(addPathForCategory(category), {
        state: { workflow: { ...workflow, index: nextIndex } },
      });
    } else {
      navigate("/money");
    }
  }
  return (
    <Card className="grid gap-4">
      <Link
        to="/money"
        className="inline-flex min-h-12 items-center gap-2 text-sm text-muted-foreground"
      >
        <ArrowLeft aria-hidden="true" className="size-4" />
        Your Money Map
      </Link>
      <div>
        <h2 className="text-xl font-semibold">US investments</h2>
        <p className="mt-2 text-sm leading-6 text-muted-foreground">
          Upload a broker CSV to add global holdings. The account import screen
          is the available entry point for these assets.
        </p>
      </div>
      <Link
        to="/import"
        className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-white"
      >
        Upload a broker file{" "}
        <ArrowRight aria-hidden="true" className="size-4" />
      </Link>
      <p className="text-sm text-muted-foreground">
        You can also continue and add other items first.
      </p>
      <Button variant="secondary" onClick={next}>
        Continue to next item
      </Button>
    </Card>
  );
}

function AssetForm({
  assetType,
  draft,
  missingFields = [],
  initialAsset,
  selectedMemberId,
  workflow,
}: {
  assetType: AssetType;
  draft?: Record<string, unknown>;
  missingFields?: string[];
  initialAsset?: AssetDetailResponse;
  selectedMemberId?: string | null;
  workflow?: ChecklistWorkflow;
}) {
  const navigate = useNavigate();
  const household = useHousehold();
  const readOnly = household.isViewer;
  const activeMemberId =
    selectedMemberId ?? household.activeMemberId ?? household.meMemberId;
  const planQuery = useQuery({
    queryKey: ["c27b", "eligibility", household.selectedHouseholdId],
    queryFn: getPlan,
    enabled: apiPathAvailable("/v1/plan"),
  });
  const formDefinition: GeneratedAssetForm = assetForms[assetType];
  const fields = formDefinition.fields;
  const baseDetails = initialAsset?.details ?? EMPTY_DETAILS;
  const defaults = useMemo(
    () =>
      initialAssetValues(
        assetType,
        fields,
        { ...baseDetails, ...(draft ?? {}) },
        initialAsset?.member_id ?? activeMemberId,
      ),
    [
      assetType,
      baseDetails,
      draft,
      fields,
      initialAsset?.member_id,
      activeMemberId,
    ],
  );
  const form = useForm<AssetFormValues>({
    defaultValues: defaults,
    mode: "onChange",
  });
  const watched = useWatch({ control: form.control }) as AssetFormValues;
  const [assetId, setAssetId] = useState(initialAsset?.id ?? null);
  const assetIdRef = useRef<string | null>(initialAsset?.id ?? null);
  const [savedAsset, setSavedAsset] = useState<AssetResponse | null>(
    initialAsset ?? null,
  );
  const members = household.members;
  const [fieldErrors, setFieldErrors] = useState<ApiFieldError[]>([]);
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});
  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "saved" | "queued" | "error"
  >(initialAsset ? "saved" : "idle");
  const [saveError, setSaveError] = useState("");
  const [revision, setRevision] = useState(0);
  const changed = useRef(false);
  const saving = useRef(false);
  const pendingBody = useRef<Record<string, unknown> | null>(null);
  const lastSignature = useRef("");
  const localDraftKey = useRef(`${assetType}:${crypto.randomUUID()}`);

  useEffect(() => {
    if (!initialAsset) return;
    form.reset(defaults);
    lastSignature.current = JSON.stringify(
      serializeAssetValues(assetType, fields, defaults, baseDetails),
    );
  }, [assetType, baseDetails, defaults, fields, form, initialAsset]);

  function markEdited() {
    changed.current = true;
    setRevision((current) => current + 1);
    setFieldErrors([]);
    setSaveError("");
  }

  async function saveBody(body: Record<string, unknown>): Promise<void> {
    const signature = JSON.stringify(body);
    if (signature === lastSignature.current) {
      setSaveState("saved");
      return;
    }
    if (saving.current) {
      pendingBody.current = body;
      return;
    }
    saving.current = true;
    let nextBody: Record<string, unknown> | null = body;
    while (nextBody) {
      const bodyToSave = nextBody;
      try {
        if (!navigator.onLine) {
          await queueAssetWrite(
            localDraftKey.current,
            assetIdRef.current,
            bodyToSave,
          );
          setSaveState("queued");
          nextBody = pendingBody.current;
          pendingBody.current = null;
          continue;
        }
        setSaveState("saving");
        const currentAssetId = assetIdRef.current;
        const saved = currentAssetId
          ? await patchAsset(currentAssetId, bodyToSave)
          : await createAsset(bodyToSave as AssetDraft);
        assetIdRef.current = saved.id;
        setAssetId(saved.id);
        setSavedAsset(saved);
        lastSignature.current = JSON.stringify(bodyToSave);
        setSaveState("saved");
        setSaveError("");
        nextBody = pendingBody.current;
        pendingBody.current = null;
      } catch (error) {
        if (typeof navigator !== "undefined" && !navigator.onLine) {
          try {
            await queueAssetWrite(
              localDraftKey.current,
              assetIdRef.current,
              bodyToSave,
            );
            setSaveState("queued");
            setSaveError("");
          } catch {
            setSaveState("error");
            setSaveError(
              "Offline storage is unavailable. Keep this screen open and try again when connected.",
            );
          }
        } else if (error instanceof ApiError) {
          const errors =
            error.code === "rate_needed"
              ? [
                  {
                    field: "rate_pct",
                    message:
                      "Enter the interest rate printed on your certificate",
                    code: error.code,
                  },
                ]
              : error.errors;
          setFieldErrors(errors);
          setSaveError(
            errors.length
              ? "Please check the highlighted details."
              : error.title,
          );
          setSaveState("error");
        } else {
          setSaveError("We couldn't save this item. Please try again.");
          setSaveState("error");
        }
        nextBody = pendingBody.current;
        pendingBody.current = null;
      }
    }
    saving.current = false;
  }

  useEffect(() => {
    if (!changed.current) return;
    const values = form.getValues();
    const errors = validateAssetValues(fields, values);
    setClientErrors(errors);
    if (Object.keys(errors).length) return;
    let body: Record<string, unknown>;
    try {
      body = serializeAssetValues(assetType, fields, values, baseDetails);
    } catch {
      return;
    }
    const timer = window.setTimeout(() => void saveBody(body), 600);
    return () => window.clearTimeout(timer);
  }, [assetType, baseDetails, fields, form, revision]);

  useEffect(() => {
    function handleReplay(event: Event) {
      const savedEvent = event as CustomEvent<{
        draftKey: string;
        asset: AssetResponse;
      }>;
      if (savedEvent.detail.draftKey !== localDraftKey.current) return;
      assetIdRef.current = savedEvent.detail.asset.id;
      setAssetId(savedEvent.detail.asset.id);
      setSavedAsset(savedEvent.detail.asset);
      setSaveState("saved");
      lastSignature.current = JSON.stringify(savedEvent.detail.asset.details);
    }
    window.addEventListener("wealth:offline-asset-saved", handleReplay);
    return () =>
      window.removeEventListener("wealth:offline-asset-saved", handleReplay);
  }, []);

  function saveNow() {
    const values = form.getValues();
    const errors = validateAssetValues(fields, values);
    setClientErrors(errors);
    if (Object.keys(errors).length) return;
    try {
      void saveBody(
        serializeAssetValues(assetType, fields, values, baseDetails),
      );
    } catch {
      setSaveError("Enter a valid amount before saving.");
      setSaveState("error");
    }
  }

  function goNext() {
    const nextIndex = (workflow?.index ?? -1) + 1;
    const category = workflow?.categories[nextIndex];
    if (workflow && category) {
      navigate(addPathForCategory(category), {
        state: { workflow: { ...workflow, index: nextIndex } },
      });
    } else {
      navigate("/money");
    }
  }

  function renderField(field: (typeof fields)[number], required: boolean) {
    if (field.kind === "hidden") return null;
    const copy = fieldCopy(assetType, field.name);
    const id = `${assetType}-${field.name}`;
    const help = fieldHelp(assetType, field.name, required);
    const error = errorForField(field.name, fieldErrors, clientErrors);
    const name = field.name as Path<AssetFormValues>;
    const highlighted = missingFields.includes(field.name);
    const baseInput =
      "min-h-12 w-full rounded-xl border bg-surface px-3 text-base outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30";
    const highlightClass = highlighted
      ? "border-warning ring-1 ring-warning/40"
      : "border-line";

    if (field.kind === "rupees") {
      return (
        <div
          key={field.name}
          className={highlighted ? "rounded-xl ring-1 ring-warning/40" : ""}
        >
          <MoneyInput
            control={form.control}
            name={name}
            id={id}
            errorName={field.name}
            label={copy.label}
            help={help}
            required={required}
            problemErrors={fieldErrors.filter(
              (apiError) => apiError.field === field.name || !apiError.field,
            )}
            validationError={clientErrors[field.name]}
            placeholder="5L or 1,25,000"
            onValueChange={markEdited}
          />
          {highlighted ? (
            <p className="text-sm text-warning-foreground">
              Quick add could not fill this field.
            </p>
          ) : null}
        </div>
      );
    }
    if (field.kind === "month_year") {
      return (
        <div
          key={field.name}
          className={highlighted ? "rounded-xl ring-1 ring-warning/40" : ""}
        >
          <MonthYearInput
            id={id}
            label={copy.label}
            help={help}
            required={required}
            value={String(watched[field.name] ?? "")}
            onChange={(event) => {
              form.setValue(name, event.currentTarget.value, {
                shouldDirty: true,
              });
              markEdited();
            }}
            error={error}
            aria-describedby={error ? `${id}-error` : undefined}
            aria-invalid={Boolean(error) || undefined}
          />
          {highlighted ? (
            <p className="text-sm text-warning-foreground">
              Quick add could not fill this field.
            </p>
          ) : null}
        </div>
      );
    }
    if (field.name === "member_id") {
      return (
        <div
          key={field.name}
          className={highlighted ? "rounded-xl ring-1 ring-warning/40" : ""}
        >
          <Select
            id={id}
            label={englishCopy.household_ui.whose_is_this}
            help={help}
            value={String(watched.member_id ?? "")}
            onChange={(event) => {
              form.setValue(name, event.currentTarget.value || null, {
                shouldDirty: true,
              });
              markEdited();
            }}
          >
            <option value={household.meMemberId ?? ""}>
              {englishCopy.household_ui.you}
            </option>
            {members
              .filter((member) => member.id !== household.meMemberId)
              .map((member) => (
                <option key={member.id} value={member.id}>
                  {household.memberLabel(member)}
                </option>
              ))}
          </Select>
        </div>
      );
    }
    if (field.kind === "enum" || field.kind === "boolean") {
      const options =
        field.kind === "boolean"
          ? ["true", "false"]
          : (field.options ?? []).map(String);
      const current =
        field.kind === "boolean"
          ? String(watched[field.name] ?? "")
          : String(watched[field.name] ?? "");
      return (
        <div
          key={field.name}
          className={highlighted ? "rounded-xl ring-1 ring-warning/40" : ""}
        >
          <Select
            id={id}
            label={copy.label}
            help={help}
            required={required}
            value={current}
            error={error}
            onChange={(event) => {
              form.setValue(
                name,
                field.kind === "boolean"
                  ? event.currentTarget.value === "true"
                  : event.currentTarget.value,
                { shouldDirty: true },
              );
              markEdited();
            }}
          >
            <option value="">Choose one</option>
            {options.map((option) => (
              <option key={option} value={option}>
                {field.kind === "boolean"
                  ? option === "true"
                    ? "Yes"
                    : "No"
                  : optionLabel(option)}
              </option>
            ))}
          </Select>
          {highlighted ? (
            <p className="text-sm text-warning-foreground">
              Quick add could not fill this field.
            </p>
          ) : null}
        </div>
      );
    }
    const listValue = watched[field.name];
    const shownValue = Array.isArray(listValue)
      ? listValue.join(", ")
      : String(listValue ?? "");
    const inputType = field.kind === "number" ? "number" : "text";
    return (
      <div
        key={field.name}
        className={highlighted ? "rounded-xl ring-1 ring-warning/40" : ""}
      >
        <Field
          id={id}
          label={copy.label}
          help={help}
          error={error}
          required={required}
        >
          <input
            {...form.register(name)}
            id={id}
            type={inputType}
            inputMode={
              field.kind === "rate_pct"
                ? "decimal"
                : field.kind === "number"
                  ? "numeric"
                  : undefined
            }
            step={field.kind === "rate_pct" ? "0.01" : "any"}
            list={
              isInstitutionField(field.name) ? "wealth-institutions" : undefined
            }
            value={shownValue}
            aria-invalid={Boolean(error) || undefined}
            className={`${baseInput} ${highlightClass}`}
            onChange={(event) => {
              const value: string | number | null =
                field.kind === "number" && event.currentTarget.value
                  ? Number(event.currentTarget.value)
                  : event.currentTarget.value;
              form.setValue(name, value, {
                shouldDirty: true,
                shouldValidate: true,
              });
              markEdited();
            }}
          />
        </Field>
        {isInstitutionField(field.name) ? (
          <datalist id="wealth-institutions">
            {institutionNames.map((institution) => (
              <option key={institution} value={institution} />
            ))}
          </datalist>
        ) : null}
        {highlighted ? (
          <p className="text-sm text-warning-foreground">
            Quick add could not fill this field.
          </p>
        ) : null}
      </div>
    );
  }

  const requiredFields = fields.filter(
    (field) => field.required && field.kind !== "hidden",
  );
  const optionalFields = fields.filter(
    (field) => !field.required && field.kind !== "hidden",
  );
  const valuation = savedAsset?.valuation as Record<string, unknown> | null;
  const valueToday =
    typeof valuation?.value_today === "string"
      ? valuation.value_today
      : savedAsset?.value_cached;
  const maturityValue =
    typeof valuation?.maturity_value === "string"
      ? valuation.maturity_value
      : undefined;
  const maturityDate =
    typeof valuation?.maturity_date === "string"
      ? valuation.maturity_date
      : typeof savedAsset?.details.maturity === "string"
        ? savedAsset.details.maturity
        : undefined;
  const isPriceNeeded = savedAsset?.valuation_status === "price_needed";
  const selectedMember = members.find(
    (member) => member.id === String(watched.member_id ?? activeMemberId ?? ""),
  );
  const planMember = planQuery.data?.members?.find(
    (member) => member.id === selectedMember?.id,
  );
  const eligibilityKey = eligibilityInstrument(assetType, watched);
  const eligibility = eligibilityNote(planMember, eligibilityKey);
  const statusText =
    saveState === "saving"
      ? "Saving…"
      : saveState === "saved"
        ? "Saved"
        : saveState === "queued"
          ? "Saved on this device · waiting for connection"
          : saveState === "error"
            ? "Couldn't save"
            : "Changes save automatically";

  return (
    <div className="grid gap-4">
      <Card className="grid gap-4">
        <Link
          to="/money"
          className="inline-flex min-h-12 items-center gap-2 text-sm text-muted-foreground"
        >
          <ArrowLeft aria-hidden="true" className="size-4" />
          Your Money Map
        </Link>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold">{assetTitles[assetType]}</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Rough numbers are fine. You can edit this anytime.
            </p>
          </div>
          {workflow ? (
            <span className="rounded-full bg-muted px-3 py-2 text-sm tabular-nums">
              {workflow.index + 1} of {workflow.categories.length}
            </span>
          ) : null}
        </div>
        <div className="grid gap-4">
          <fieldset
            disabled={readOnly}
            className="grid gap-4 disabled:opacity-70"
          >
            <div className="grid gap-4">
              {requiredFields.map((field) => renderField(field, true))}
            </div>
            {eligibility ? (
              <section
                className="grid gap-1 rounded-xl border border-line bg-muted/40 p-3 text-sm"
                aria-label={englishCopy.household_ui.eligibility_title.replace(
                  "{name}",
                  selectedMember
                    ? household.memberLabel(selectedMember)
                    : englishCopy.household_ui.you,
                )}
              >
                <p className="font-medium">
                  {englishCopy.household_ui.eligibility_title.replace(
                    "{name}",
                    selectedMember
                      ? household.memberLabel(selectedMember)
                      : englishCopy.household_ui.you,
                  )}
                </p>
                <p>{eligibility}</p>
                {planMember?.eligibility_status?.[eligibilityKey ?? ""] ===
                "not_allowed" ? (
                  <p className="text-muted-foreground">
                    {englishCopy.household_ui.existing_record_note}
                  </p>
                ) : null}
              </section>
            ) : null}
            {optionalFields.length ? (
              <details className="rounded-xl border border-line px-3">
                <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between gap-2 text-sm font-medium">
                  <span>More details (for a closer estimate)</span>
                  <ChevronDown aria-hidden="true" className="size-4" />
                </summary>
                <div className="grid gap-4 border-t border-line py-4">
                  {optionalFields.map((field) => renderField(field, false))}
                </div>
              </details>
            ) : null}
            <div className="grid gap-2 sm:grid-cols-2">
              <Button
                type="button"
                onClick={saveNow}
                disabled={
                  Object.keys(validateAssetValues(fields, watched)).length > 0
                }
              >
                {saveState === "saved" ? (
                  <>
                    <Check aria-hidden="true" className="size-4" /> Save now
                  </>
                ) : (
                  "Save now"
                )}
              </Button>
              {workflow ? (
                <Button
                  type="button"
                  variant="secondary"
                  disabled={!savedAsset || saveState === "saving"}
                  onClick={goNext}
                >
                  {workflow.index + 1 < workflow.categories.length
                    ? "Continue to next item"
                    : "Finish Money Map setup"}
                </Button>
              ) : assetId ? (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => navigate(`/money/asset/${assetId}`)}
                >
                  View asset
                </Button>
              ) : null}
            </div>
          </fieldset>
        </div>
        {readOnly ? (
          <p className="text-sm text-muted-foreground">
            {englishCopy.household_ui.view_only}
          </p>
        ) : null}
        <p
          className="min-h-5 text-sm text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {statusText}
        </p>
        {saveError ? (
          <p className="min-h-5 text-sm text-danger-foreground" role="alert">
            {saveError}
          </p>
        ) : null}
        {savedAsset ? (
          <div className="grid gap-2 rounded-xl bg-muted/60 p-3 text-sm">
            {isPriceNeeded ? (
              <p className="font-medium">
                We'll value this once today's price is in.
              </p>
            ) : valueToday ? (
              <p>
                Value today ≈ <strong>{formatInr(valueToday, "full")}</strong>
              </p>
            ) : null}
            {maturityValue ? (
              <p>
                At maturity ≈{" "}
                <strong>{formatInr(maturityValue, "full")}</strong>
              </p>
            ) : null}
            {maturityDate ? (
              <p className="text-muted-foreground">
                Matures {maturityDate}. We'll remind you 30 days before.
              </p>
            ) : null}
            {savedAsset.valuation_note ? (
              <p className="text-muted-foreground">
                {savedAsset.valuation_note}
              </p>
            ) : null}
          </div>
        ) : null}
        {missingFields.length ? (
          <p className="text-sm text-warning-foreground">
            Quick add left {missingFields.length}{" "}
            {missingFields.length === 1 ? "field" : "fields"} to complete.
          </p>
        ) : null}
      </Card>
    </div>
  );
}

export function AddAssetPage() {
  const { type = "" } = useParams();
  const location = useLocation();
  const { workflow, memberId } = useWorkflowState();
  const state = location.state as {
    draft?: Record<string, unknown>;
    missing?: string[];
  } | null;
  const [insuranceType, setInsuranceType] =
    useState<AssetType>("traditional_policy");
  const assetType: AssetType =
    type === "insurance" ? insuranceType : (type as AssetType);
  if (type === "us_investments") return <USInvestmentStep />;
  if (!(assetType in assetForms)) {
    return (
      <Card>
        <h2 className="text-lg font-semibold">Choose an asset type</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          This item isn't available as a manual form yet.
        </p>
        <Link to="/import" className="mt-4 block">
          <Button className="w-full">Upload what you have</Button>
        </Link>
      </Card>
    );
  }
  return (
    <div className="grid gap-4">
      {type === "insurance" ? (
        <Card className="grid gap-3">
          <label htmlFor="insurance-type" className="text-sm font-medium">
            Insurance type
          </label>
          <select
            id="insurance-type"
            value={assetType}
            onChange={(event) =>
              setInsuranceType(event.currentTarget.value as AssetType)
            }
            className="min-h-12 rounded-xl border border-line bg-surface px-3"
          >
            {INSURANCE_TYPES.map((choice) => (
              <option key={choice} value={choice}>
                {assetTitles[choice]}
              </option>
            ))}
          </select>
        </Card>
      ) : null}
      <AssetForm
        key={`${assetType}:${workflow?.index ?? "single"}:${location.key}`}
        assetType={assetType}
        draft={state?.draft}
        missingFields={state?.missing}
        selectedMemberId={memberId}
        workflow={workflow}
      />
    </div>
  );
}

export function AssetDetailPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const [asset, setAsset] = useState<AssetDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [confirmClose, setConfirmClose] = useState(false);
  const [closing, setClosing] = useState(false);

  useEffect(() => {
    let active = true;
    void getAsset(id)
      .then((loaded) => {
        if (active) setAsset(loaded);
      })
      .catch(() => {
        if (active) setError("We couldn't load this asset.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [id]);

  async function closeCurrentAsset() {
    setClosing(true);
    setError("");
    try {
      await closeAsset(id);
      navigate("/money");
    } catch {
      setError("This item couldn't be closed. Please try again.");
    } finally {
      setClosing(false);
    }
  }

  if (loading)
    return (
      <Card aria-label="Loading asset">
        <div className="h-6 w-36 animate-pulse rounded bg-muted" />
        <div className="mt-4 h-20 animate-pulse rounded bg-muted" />
      </Card>
    );
  if (!asset)
    return (
      <Card role="alert">
        <p>{error || "This asset wasn't found."}</p>
        <Link to="/money" className="mt-4 block">
          <Button className="w-full">Back to Money Map</Button>
        </Link>
      </Card>
    );

  return (
    <div className="grid gap-4">
      <AssetForm
        key={asset.id}
        assetType={asset.asset_type as AssetType}
        initialAsset={asset}
      />
      <Card className="grid gap-3">
        <h2 className="text-base font-semibold">Edit history</h2>
        {asset.history.length ? (
          asset.history.map((entry) => (
            <div key={entry.id} className="border-t border-line pt-3 text-sm">
              <p className="font-medium">
                {new Intl.DateTimeFormat("en-IN", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                }).format(new Date(entry.changed_at))}
              </p>
              <p className="mt-1 text-muted-foreground">
                Changed{" "}
                {Object.keys(entry.changed_fields)
                  .map((field) => field.replaceAll("_", " "))
                  .join(", ") || "details"}
              </p>
            </div>
          ))
        ) : (
          <p className="text-sm text-muted-foreground">
            No edits have been recorded yet.
          </p>
        )}
        {error ? (
          <p className="text-sm text-danger-foreground" role="alert">
            {error}
          </p>
        ) : null}
        {confirmClose ? (
          <div className="grid gap-2 rounded-xl border border-danger/40 bg-danger-soft p-3">
            <p className="text-sm">
              Close this asset? It will leave your active Money Map but remain
              in its history.
            </p>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="danger"
                loading={closing}
                onClick={() => void closeCurrentAsset()}
              >
                Close asset
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() => setConfirmClose(false)}
              >
                Keep it
              </Button>
            </div>
          </div>
        ) : (
          <Button
            type="button"
            variant="danger"
            onClick={() => setConfirmClose(true)}
          >
            Close this asset
          </Button>
        )}
      </Card>
    </div>
  );
}
