import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useForm, useWatch } from "react-hook-form";
import type { Path } from "react-hook-form";
import { ArrowLeft, ArrowRight, ChevronDown } from "lucide-react";
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
  formatAmountInput,
  type AssetFormValues,
  initialAssetValues,
  serializeAssetValues,
  validateAssetValues,
} from "./form-utils";
import { applyFieldRules, withoutHiddenValues } from "./field-rules";
import { assetTitles } from "./titles";
import { fieldCopy } from "./labels.en";
import { queueAssetWrite } from "./offline-queue";
import { useHousehold } from "../household/HouseholdContext";
import { getPlan } from "../planning/api";
import { apiPathAvailable } from "../api/feature-support";
import * as englishCopy from "../copy/sections/index";
import { helpArticleForAssetType } from "../help/context";

type PlanMember = components["schemas"]["PlanMember"];

const INSURANCE_TYPES = [
  "traditional_policy",
  "term_insurance",
  "health_insurance",
  "ulip",
] as const;
const EMPTY_DETAILS: Record<string, unknown> = {};
const EURO_COUNTRIES = new Set([
  "AD",
  "AT",
  "BE",
  "BG",
  "CY",
  "DE",
  "EE",
  "ES",
  "FI",
  "FR",
  "GR",
  "HR",
  "IE",
  "IT",
  "LT",
  "LU",
  "LV",
  "MC",
  "ME",
  "MT",
  "NL",
  "PT",
  "SI",
  "SK",
  "SM",
  "VA",
  "XK",
]);

function savingsCurrencyForCountry(
  country: string | undefined,
): "INR" | "DKK" | "EUR" {
  if (country === "DK") return "DKK";
  if (country && EURO_COUNTRIES.has(country.toUpperCase())) return "EUR";
  return "INR";
}

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
    gold: "Gold loan",
    other: "Other",
    in_home_self_occupied: "India: home you live in",
    in_home_let_out: "India: rented-out home",
    in_education: "India: education loan",
    dk_interest_deduction: "Denmark: interest deduction",
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

function fieldHelp(
  type: AssetType,
  name: string,
  required: boolean,
): string | undefined {
  const copy = fieldCopy(type, name);
  if (copy.help) return copy.help;
  if (name === "member_id") return englishCopy.household_ui.member_help;
  if (!required) return undefined;
  if (name.endsWith("_year")) return englishCopy.household_ui.year_only_help;
  return undefined;
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
    if (values.kind === "sgb") return "sgb";
    // Physical and digital gold have no residence rule.
    return values.kind === "etf" || values.kind === "mf" ? "gold_etf" : null;
  }
  if (assetType === "po_scheme") {
    const scheme = String(values.scheme ?? "");
    if (scheme === "td" || scheme === "rd" || scheme === "mis") return "pomis";
    return scheme || null;
  }
  if (assetType === "fd") return "bank_fd";
  return null;
}

function eligibilityCondition(condition: string | undefined): string | null {
  const conditions: Record<string, string> = {
    ...englishCopy.household_ui.conditions,
  };
  return condition ? (conditions[condition] ?? null) : null;
}

function eligibilityNote(
  member: PlanMember | undefined,
  instrument: string | null,
): string | null {
  if (!member || !instrument) return null;
  const status = member.eligibility_status?.[instrument];
  if (status === "allowed") return englishCopy.household_ui.eligibility_allowed;
  // The rules give a reason only for "conditional", and only some reasons have
  // copy, so each status also needs a sentence that stands on its own.
  if (status === "not_allowed")
    return englishCopy.household_ui.eligibility_not_allowed;
  if (status === "conditional") {
    const conditionText = eligibilityCondition(
      member.eligibility_conditions?.[instrument],
    );
    return conditionText
      ? englishCopy.household_ui.eligibility_conditional.replace(
          "{condition}",
          conditionText,
        )
      : englishCopy.household_ui.eligibility_unknown;
  }
  return null;
}

function readableMaturityDate(value: string | undefined): string | undefined {
  if (!value) return undefined;
  const match = /^(\d{4})-(\d{2})(?:-\d{2})?$/.exec(value);
  if (!match) return value;
  const year = Number(match[1]);
  const month = Number(match[2]);
  if (month < 1 || month > 12) return value;
  return new Intl.DateTimeFormat("en-IN", {
    month: "long",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(Date.UTC(year, month - 1, 1)));
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
        <p className="mt-2 text-base text-muted-foreground">
          Upload a broker CSV to add global holdings. The account import screen
          is the available entry point for these assets.
        </p>
      </div>
      <Link
        to="/import"
        className="inline-flex min-h-12 items-center justify-center gap-2 rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
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
  saveQueued = false,
  onReturn,
}: {
  assetType: AssetType;
  draft?: Record<string, unknown>;
  missingFields?: string[];
  initialAsset?: AssetDetailResponse;
  selectedMemberId?: string | null;
  workflow?: ChecklistWorkflow;
  saveQueued?: boolean;
  onReturn?: (queued: boolean, saved?: AssetResponse) => void;
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
  const baseFormFields = useMemo(() => {
    const shaped = fields.map((field) => ({
      ...field,
      required:
        field.required ||
        (assetType === "savings" && field.name === "currency") ||
        (assetType === "loan" && !initialAsset && field.name === "emi") ||
        // Loans can be in kroner, and the amounts below follow this choice.
        (assetType === "loan" && field.name === "currency"),
    }));
    if (assetType === "savings") {
      const bank = shaped.find((field) => field.name === "bank");
      const currency = shaped.find((field) => field.name === "currency");
      const balance = shaped.find((field) => field.name === "balance");
      const rest = shaped.filter(
        (field) => !["bank", "currency", "balance"].includes(field.name),
      );
      return [
        ...[bank, currency, balance].filter(
          (field): field is (typeof shaped)[number] => field !== undefined,
        ),
        ...rest,
      ];
    }
    if (assetType === "fd") {
      const firstFields = ["bank", "principal", "rate_pct", "maturity"];
      return [
        ...firstFields.flatMap((name) =>
          shaped.filter((field) => field.name === name),
        ),
        ...shaped.filter((field) => !firstFields.includes(field.name)),
      ];
    }
    if (assetType !== "loan") return shaped;
    const currency = shaped.filter((field) => field.name === "currency");
    const rest = shaped.filter((field) => field.name !== "currency");
    const kindIndex = rest.findIndex((field) => field.name === "loan_kind");
    return [
      ...rest.slice(0, kindIndex + 1),
      ...currency,
      ...rest.slice(kindIndex + 1),
    ];
  }, [assetType, fields, initialAsset]);
  const baseDetails = initialAsset?.details ?? EMPTY_DETAILS;
  const defaults = useMemo(() => {
    const detailValues = { ...baseDetails, ...(draft ?? {}) };
    if (assetType === "savings" && detailValues.currency === undefined) {
      const member = household.members.find(
        (item) => item.id === (initialAsset?.member_id ?? activeMemberId),
      );
      detailValues.currency = initialAsset
        ? "INR"
        : savingsCurrencyForCountry(member?.residency.country);
    }
    const values = initialAssetValues(
      assetType,
      fields,
      detailValues,
      initialAsset?.member_id ?? activeMemberId,
    );
    if (draft) {
      for (const field of fields) {
        const raw = draft[field.name];
        if (
          (field.kind === "amount" || field.kind === "rupees") &&
          (typeof raw === "string" || typeof raw === "number")
        ) {
          try {
            values[field.name] = formatAmountInput(
              String(raw),
              field.kind === "rupees" ? "INR" : values.currency,
            );
          } catch {
            values[field.name] = String(raw);
          }
        }
      }
    }
    return values;
  }, [
    assetType,
    baseDetails,
    draft,
    fields,
    household.members,
    activeMemberId,
    initialAsset,
  ]);
  const form = useForm<AssetFormValues>({
    defaultValues: defaults,
    mode: "onChange",
  });
  useEffect(() => {
    if (assetType !== "savings" || initialAsset) return;
    const member = household.members.find((item) => item.id === activeMemberId);
    if (!member || form.getFieldState("currency").isDirty) return;
    const currency = savingsCurrencyForCountry(member.residency.country);
    if (form.getValues("currency") !== currency) {
      form.setValue("currency", currency);
    }
  }, [activeMemberId, assetType, form, household.members, initialAsset]);
  const watched = useWatch({ control: form.control }) as AssetFormValues;
  const formFields = useMemo(
    () => applyFieldRules(assetType, baseFormFields, watched),
    [assetType, baseFormFields, watched],
  );
  const assetIdRef = useRef<string | null>(initialAsset?.id ?? null);
  const [savedAsset, setSavedAsset] = useState<AssetResponse | null>(
    initialAsset ?? null,
  );
  const savedAssetRef = useRef<AssetResponse | null>(initialAsset ?? null);
  const members = household.members;
  const [fieldErrors, setFieldErrors] = useState<ApiFieldError[]>([]);
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});
  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "saved" | "queued" | "error"
  >(saveQueued ? "queued" : initialAsset ? "saved" : "idle");
  const [saveError, setSaveError] = useState("");
  const [revision, setRevision] = useState(0);
  const [changed, setChanged] = useState(false);
  const moreDetailsRef = useRef<HTMLDetailsElement>(null);
  const saving = useRef(false);
  const savePromise = useRef<Promise<boolean> | null>(null);
  const activeBody = useRef("");
  const pendingBody = useRef<Record<string, unknown> | null>(null);
  const queuedWrite = useRef(saveQueued);
  const lastSignature = useRef("");
  const localDraftKey = useRef(`${assetType}:${crypto.randomUUID()}`);

  useEffect(() => {
    if (!initialAsset) return;
    form.reset(defaults);
    lastSignature.current = JSON.stringify(
      serializeAssetValues(
        assetType,
        fields,
        withoutHiddenValues(assetType, defaults),
        baseDetails,
      ),
    );
  }, [assetType, baseDetails, defaults, fields, form, initialAsset]);

  function markEdited() {
    setChanged(true);
    setRevision((current) => current + 1);
    setFieldErrors([]);
    setSaveError("");
  }

  async function saveBody(body: Record<string, unknown>): Promise<boolean> {
    const signature = JSON.stringify(body);
    if (signature === lastSignature.current) {
      setSaveState("saved");
      return true;
    }
    if (savePromise.current) {
      if (signature !== activeBody.current) pendingBody.current = body;
      return savePromise.current;
    }
    const task = (async () => {
      saving.current = true;
      activeBody.current = signature;
      let nextBody: Record<string, unknown> | null = body;
      let latestSucceeded = true;
      while (nextBody) {
        const bodyToSave = nextBody;
        activeBody.current = JSON.stringify(bodyToSave);
        let succeeded = false;
        try {
          if (typeof navigator !== "undefined" && !navigator.onLine) {
            await queueAssetWrite(
              localDraftKey.current,
              assetIdRef.current,
              bodyToSave,
            );
            setSaveState("queued");
            queuedWrite.current = true;
            succeeded = true;
          } else {
            setSaveState("saving");
            const currentAssetId = assetIdRef.current;
            const saved = currentAssetId
              ? await patchAsset(currentAssetId, bodyToSave)
              : await createAsset(bodyToSave as AssetDraft);
            assetIdRef.current = saved.id;
            setSavedAsset(saved);
            savedAssetRef.current = saved;
            lastSignature.current = JSON.stringify(bodyToSave);
            setSaveState("saved");
            setSaveError("");
            succeeded = true;
          }
        } catch (error) {
          if (typeof navigator !== "undefined" && !navigator.onLine) {
            try {
              await queueAssetWrite(
                localDraftKey.current,
                assetIdRef.current,
                bodyToSave,
              );
              setSaveState("queued");
              queuedWrite.current = true;
              setSaveError("");
              succeeded = true;
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
        }
        latestSucceeded = succeeded;
        nextBody = pendingBody.current;
        pendingBody.current = null;
      }
      saving.current = false;
      activeBody.current = "";
      return latestSucceeded;
    })();
    savePromise.current = task;
    try {
      return await task;
    } finally {
      if (savePromise.current === task) savePromise.current = null;
    }
  }

  useEffect(() => {
    if (!changed) return;
    const values = form.getValues();
    const errors = validateAssetValues(formFields, values);
    setClientErrors(errors);
    if (Object.keys(errors).length) return;
    let body: Record<string, unknown>;
    try {
      body = serializeAssetValues(
        assetType,
        fields,
        withoutHiddenValues(assetType, values),
        baseDetails,
      );
    } catch {
      return;
    }
    const timer = window.setTimeout(() => void saveBody(body), 600);
    return () => window.clearTimeout(timer);
  }, [assetType, baseDetails, changed, fields, form, formFields, revision]);

  useEffect(() => {
    function handleReplay(event: Event) {
      const savedEvent = event as CustomEvent<{
        draftKey: string;
        asset: AssetResponse;
      }>;
      if (savedEvent.detail.draftKey !== localDraftKey.current) return;
      assetIdRef.current = savedEvent.detail.asset.id;
      setSavedAsset(savedEvent.detail.asset);
      savedAssetRef.current = savedEvent.detail.asset;
      setSaveState("saved");
      queuedWrite.current = false;
      lastSignature.current = JSON.stringify(savedEvent.detail.asset.details);
    }
    window.addEventListener("wealth:offline-asset-saved", handleReplay);
    return () =>
      window.removeEventListener("wealth:offline-asset-saved", handleReplay);
  }, []);

  async function persistCurrent(): Promise<boolean> {
    const values = form.getValues();
    const errors = validateAssetValues(formFields, values);
    setClientErrors(errors);
    if (Object.keys(errors).length) {
      if (
        moreDetailsRef.current &&
        optionalFields.some((field) => errors[field.name])
      )
        moreDetailsRef.current.open = true;
      setSaveError("Please check the highlighted details.");
      setSaveState("error");
      return false;
    }
    try {
      const saved = await saveBody(
        serializeAssetValues(
          assetType,
          fields,
          withoutHiddenValues(assetType, values),
          baseDetails,
        ),
      );
      return saved;
    } catch {
      setSaveError("Enter a valid amount before saving.");
      setSaveState("error");
      return false;
    }
  }

  function goNext(queued = false) {
    const nextIndex = (workflow?.index ?? -1) + 1;
    const category = workflow?.categories[nextIndex];
    if (workflow && category) {
      navigate(addPathForCategory(category), {
        state: {
          workflow: { ...workflow, index: nextIndex },
          saveQueued: queued,
        },
      });
    } else {
      navigate("/money", { state: { saveQueued: queued } });
    }
  }

  async function saveAndLeave() {
    const saved = await persistCurrent();
    if (!saved) return;
    const queued = queuedWrite.current;
    if (onReturn) {
      onReturn(
        queued,
        queued ? undefined : (savedAssetRef.current ?? undefined),
      );
    } else if (workflow) {
      goNext(queued);
    } else {
      navigate("/money", { state: { saveQueued: queued } });
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
    if (field.kind === "amount") {
      const currency = String(watched.currency ?? "INR");
      const prefix = currency === "DKK" ? "kr" : currency === "EUR" ? "€" : "₹";
      return (
        <div key={field.name}>
          <Field
            id={id}
            label={copy.label}
            help={help}
            error={error}
            required={required}
            prefix={prefix}
          >
            <input
              {...form.register(name)}
              id={id}
              type="text"
              inputMode="decimal"
              value={String(watched[field.name] ?? "")}
              placeholder={currency === "INR" ? "5L or 1,25,000" : "0.00"}
              aria-invalid={Boolean(error) || undefined}
              className={`${baseInput.replace("px-3", "pr-3")} ${prefix.length > 1 ? "pl-14" : "pl-9"} ${highlightClass}`}
              onChange={(event) => {
                form.setValue(name, event.currentTarget.value, {
                  shouldDirty: true,
                  shouldValidate: true,
                });
                markEdited();
              }}
            />
          </Field>
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

  const requiredFields = formFields.filter(
    (field) => field.required && field.kind !== "hidden",
  );
  const memberField = formFields.find((field) => field.name === "member_id");
  const multipleAdults =
    members.filter((member) => member.relationship !== "child").length > 1;
  const showMemberField = assetType === "fd" && multipleAdults && memberField;
  const optionalFields = formFields.filter(
    (field) =>
      !field.required &&
      field.kind !== "hidden" &&
      !(showMemberField && field.name === "member_id"),
  );
  const optionalFieldHasError = optionalFields.some(
    (field) =>
      clientErrors[field.name] ??
      fieldErrors.some((error) => error.field === field.name),
  );
  useEffect(() => {
    // An error inside the collapsed section would otherwise be invisible.
    if (optionalFieldHasError && moreDetailsRef.current)
      moreDetailsRef.current.open = true;
  }, [optionalFieldHasError]);
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
  const readableDate = readableMaturityDate(maturityDate);
  const isPriceNeeded = savedAsset?.valuation_status === "price_needed";
  const selectedMember = members.find(
    (member) => member.id === String(watched.member_id ?? activeMemberId ?? ""),
  );
  const planMember = planQuery.data?.members?.find(
    (member) => member.id === selectedMember?.id,
  );
  const eligibilityKey = eligibilityInstrument(assetType, watched);
  const eligibility = eligibilityNote(planMember, eligibilityKey);
  const eligibilityLabel = selectedMember
    ? household.memberLabel(selectedMember)
    : englishCopy.household_ui.you;
  // "You" is a standalone label; mid-sentence it reads "What can fit for you".
  const eligibilityTitle = englishCopy.household_ui.eligibility_title.replace(
    "{name}",
    eligibilityLabel === englishCopy.household.you ||
      eligibilityLabel === englishCopy.household_ui.you
      ? eligibilityLabel.toLowerCase()
      : eligibilityLabel,
  );
  const statusText =
    saveState === "saving"
      ? "Saving…"
      : saveState === "saved"
        ? "Saved"
        : saveState === "queued"
          ? englishCopy.money_ui.saved_offline
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
              {englishCopy.household_ui.form_rough_numbers}
            </p>
            <Link
              to={`/help#${helpArticleForAssetType(assetType)}`}
              className="mt-2 inline-flex min-h-10 items-center text-sm font-medium text-brand underline underline-offset-2"
            >
              Where do I find this?
            </Link>
          </div>
          {workflow && workflow.categories.length > 1 ? (
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
            <div className="grid gap-5">
              {showMemberField ? renderField(memberField, false) : null}
            </div>
            {assetType === "fd" && (showMemberField || eligibility) ? (
              <section
                className="grid gap-1 rounded-xl border border-line bg-muted/40 p-3 text-sm"
                aria-label={englishCopy.household_ui.adding_new_money}
              >
                <p className="font-medium">
                  {englishCopy.household_ui.adding_new_money}
                </p>
                {planMember?.eligibility_status?.[eligibilityKey ?? ""] ===
                "not_allowed" ? (
                  <>
                    <p>{englishCopy.household_ui.existing_record_note}</p>
                    {eligibility ? <p>{eligibility}</p> : null}
                  </>
                ) : eligibility ? (
                  <p>{eligibility}</p>
                ) : null}
              </section>
            ) : eligibility ? (
              <section
                className="grid gap-1 rounded-xl border border-line bg-muted/40 p-3 text-sm"
                aria-label={eligibilityTitle}
              >
                <p className="font-medium">{eligibilityTitle}</p>
                <p>{eligibility}</p>
                {planMember?.eligibility_status?.[eligibilityKey ?? ""] ===
                "not_allowed" ? (
                  <p className="text-muted-foreground">
                    {englishCopy.household_ui.existing_record_note}
                  </p>
                ) : null}
              </section>
            ) : null}
            <div className="grid gap-5">
              {requiredFields.map((field) => renderField(field, true))}
            </div>
            {optionalFields.length ? (
              <details
                ref={moreDetailsRef}
                className="rounded-xl border border-line px-3"
              >
                <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between gap-2 text-sm font-medium">
                  <span>More details (for a closer estimate)</span>
                  <ChevronDown aria-hidden="true" className="size-4" />
                </summary>
                <div className="grid gap-5 border-t border-line py-4">
                  {optionalFields.map((field) => renderField(field, false))}
                </div>
              </details>
            ) : null}
            <Button type="button" onClick={() => void saveAndLeave()}>
              {workflow && workflow.index + 1 < workflow.categories.length
                ? englishCopy.money_ui.save_continue
                : englishCopy.money_ui.save_return}
            </Button>
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
                {englishCopy.money_ui.asset_maturity_date}{" "}
                <strong>{readableDate}</strong>
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
  if (type === "us_investments") {
    return (
      <div className="grid gap-3">
        <Link
          to={`/help#${helpArticleForAssetType(type)}`}
          className="inline-flex min-h-10 w-fit items-center text-sm font-medium text-brand underline underline-offset-2"
        >
          Where do I find this?
        </Link>
        <USInvestmentStep />
      </div>
    );
  }
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
        saveQueued={Boolean(
          (location.state as { saveQueued?: boolean } | null)?.saveQueued,
        )}
      />
    </div>
  );
}

export function AssetDetailPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const household = useHousehold();
  const [asset, setAsset] = useState<AssetDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [confirmClose, setConfirmClose] = useState(false);
  const [closing, setClosing] = useState(false);
  const [editing, setEditing] = useState(false);
  const [queuedStatus, setQueuedStatus] = useState(false);

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

  const details = asset.details;
  const valuation = asset.valuation as Record<string, unknown> | null;
  const valueToday =
    typeof valuation?.value_today === "string"
      ? valuation.value_today
      : asset.value_cached;
  const maturityValue =
    typeof valuation?.maturity_value === "string"
      ? valuation.maturity_value
      : undefined;
  const maturityDate = readableMaturityDate(
    typeof valuation?.maturity_date === "string"
      ? valuation.maturity_date
      : typeof details.maturity === "string"
        ? details.maturity
        : undefined,
  );
  const bankOrInstitution = [
    details.bank,
    details.institution,
    details.insurer,
    details.held_at,
    details.issuer,
  ].find(
    (value): value is string => typeof value === "string" && value.length > 0,
  );
  const owner = household.members.find(
    (member) => member.id === asset.member_id,
  );
  const ownerName = owner ? household.memberLabel(owner) : "";

  return (
    <div className="grid gap-4">
      {!editing ? (
        <Card className="grid gap-4">
          <h2 className="text-xl font-semibold">
            {assetTitles[asset.asset_type as AssetType]}
          </h2>
          <dl className="grid grid-cols-2 gap-4 text-sm">
            {valueToday ? (
              <div>
                <dt className="text-muted-foreground">
                  {englishCopy.money_ui.asset_value_today}
                </dt>
                <dd className="mt-1 font-semibold">
                  {formatInr(valueToday, "full")}
                </dd>
              </div>
            ) : null}
            {maturityValue ? (
              <div>
                <dt className="text-muted-foreground">
                  {englishCopy.money_ui.asset_maturity_value}
                </dt>
                <dd className="mt-1 font-semibold">
                  {formatInr(maturityValue, "full")}
                </dd>
              </div>
            ) : null}
            {maturityDate ? (
              <div>
                <dt className="text-muted-foreground">
                  {englishCopy.money_ui.asset_maturity_date}
                </dt>
                <dd className="mt-1">{maturityDate}</dd>
              </div>
            ) : null}
            {bankOrInstitution ? (
              <div>
                <dt className="text-muted-foreground">
                  {englishCopy.money_ui.asset_bank}
                </dt>
                <dd className="mt-1">{bankOrInstitution}</dd>
              </div>
            ) : null}
            {ownerName ? (
              <div>
                <dt className="text-muted-foreground">
                  {englishCopy.money_ui.asset_owner}
                </dt>
                <dd className="mt-1">{ownerName}</dd>
              </div>
            ) : null}
          </dl>
          {queuedStatus ? (
            <p role="status" className="text-sm text-muted-foreground">
              {englishCopy.money_ui.saved_offline}
            </p>
          ) : null}
          <Button type="button" onClick={() => setEditing(true)}>
            {englishCopy.money_ui.edit_details}
          </Button>
        </Card>
      ) : (
        <AssetForm
          key={asset.id}
          assetType={asset.asset_type as AssetType}
          initialAsset={asset}
          saveQueued={queuedStatus}
          onReturn={(queued, saved) => {
            if (saved) {
              setAsset((current) =>
                current
                  ? ({
                      ...current,
                      ...saved,
                      history: current.history,
                    } as AssetDetailResponse)
                  : current,
              );
            }
            setQueuedStatus(queued);
            setEditing(false);
          }}
        />
      )}
      <Card className="grid gap-3">
        <details>
          <summary className="flex min-h-12 cursor-pointer list-none items-center justify-between gap-2 text-sm font-medium">
            <span>{englishCopy.money_ui.edit_history}</span>
            <ChevronDown aria-hidden="true" className="size-4" />
          </summary>
          <div className="grid gap-3 border-t border-line pt-3">
            {asset.history.length ? (
              asset.history.map((entry) => (
                <div
                  key={entry.id}
                  className="border-t border-line pt-3 text-sm"
                >
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
                  {englishCopy.money_ui.close_item_message}
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    type="button"
                    variant="danger"
                    loading={closing}
                    onClick={() => void closeCurrentAsset()}
                  >
                    {englishCopy.money_ui.mark_closed}
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    onClick={() => setConfirmClose(false)}
                  >
                    {englishCopy.money_ui.keep_item}
                  </Button>
                </div>
              </div>
            ) : (
              <Button
                type="button"
                variant="ghost"
                onClick={() => setConfirmClose(true)}
              >
                {englishCopy.money_ui.mark_closed}
              </Button>
            )}
          </div>
        </details>
      </Card>
    </div>
  );
}
