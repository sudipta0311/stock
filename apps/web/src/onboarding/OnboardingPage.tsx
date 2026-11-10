import { useEffect, useRef, useState } from "react";
import type { FormEvent, RefObject } from "react";
import { useForm } from "react-hook-form";
import { ArrowLeft, ArrowRight, Check, HeartHandshake } from "lucide-react";
import { Link, useLocation, useNavigate } from "react-router";
import type { components } from "../api/schema";
import { formatInr, parseInr } from "../lib/money";
import {
  Button,
  Card,
  Field,
  MoneyInput,
  Progress,
  Select,
  TileToggle,
} from "../ui";
import { useAuth } from "../auth/AuthContext";
import {
  createGoal,
  getMe,
  getRiskQuestions,
  listGoals,
  putRiskAnswers,
  updateProfile,
} from "../money/api";
import type {
  GoalResponse,
  ProfileUpsert,
  RiskQuestion,
  RiskResult,
} from "../money/api";
import { ONBOARDING_STORAGE_KEY } from "../money/config";
import {
  createSettingsMember,
  listSettingsMembers,
  updateSettingsMember,
} from "../settings/api";
import type { MemberWrite, SettingsMember } from "../settings/api";
import * as englishCopy from "../copy/sections/index";

const STEPS = [
  "welcome",
  "household",
  "goals",
  "risk",
  "consent",
  "done",
] as const;
const STEP_LABELS: Record<OnboardingStep, string> = {
  welcome: "Welcome",
  household: "Household",
  goals: "Goals",
  risk: "Risk",
  consent: "Updates",
  done: "Done",
};
type OnboardingStep = (typeof STEPS)[number];
type TaxRegime = ProfileUpsert["tax_regime"];
type GoalType = components["schemas"]["GoalType"];
type AmountForm = { amount: string };
type ResidenceStatus = "RI" | "NRI" | "RNOR";
type MemberRelationship = MemberWrite["relationship"];

interface AdultDraft {
  id?: string;
  displayName: string;
  relationship: MemberRelationship;
  residence: ResidenceStatus;
  plannedReturnYear: string;
  taxRegime: TaxRegime;
  taxSlab: string;
  birthYear: string;
  annualIncome: string;
  annualIncomeCurrency: "INR" | "DKK";
}

interface OnboardingDraft {
  step: OnboardingStep;
  householdMode: "me" | "spouse" | "family";
  adults: AdultDraft[];
  taxRegime: TaxRegime;
  cityTier: string;
  timezone: string;
  monthlyExpenses: string;
  annualIncome: string;
  dependants: string;
  employmentType: string;
  answers: Record<string, string>;
  consents: { weeklyEmail: boolean; reminders: boolean; push: boolean };
}

const DEFAULT_DRAFT: OnboardingDraft = {
  step: "welcome",
  householdMode: "me",
  adults: [
    {
      displayName: "You",
      relationship: "self",
      residence: "RI",
      plannedReturnYear: "",
      taxRegime: "new",
      taxSlab: "",
      birthYear: "",
      annualIncome: "",
      annualIncomeCurrency: "INR",
    },
  ],
  taxRegime: "new",
  cityTier: "",
  timezone: "Asia/Kolkata",
  monthlyExpenses: "",
  annualIncome: "",
  dependants: "",
  employmentType: "",
  answers: {},
  consents: { weeklyEmail: false, reminders: false, push: false },
};

function residenceFor(member: SettingsMember | undefined): ResidenceStatus {
  if (member?.residency.status === "RNOR") return "RNOR";
  if (member?.residency.status === "NRI") return "NRI";
  return "RI";
}

function residenceValue(
  status: ResidenceStatus,
  plannedReturnYear: string,
): MemberWrite["residency"] {
  const year = plannedReturnYear.trim();
  return {
    status,
    country: status === "NRI" ? "DK" : "IN",
    settle_plan: status === "NRI" ? "undecided" : "india",
    planned_return_year:
      status === "NRI" && /^\d{4}$/.test(year) ? Number(year) : null,
  };
}

const GOAL_OPTIONS: Array<{
  value: GoalType;
  label: string;
  description: string;
}> = [
  {
    value: "emergency",
    label: "Emergency fund",
    description: "A cash buffer for surprises",
  },
  {
    value: "education_india",
    label: "Child education in India",
    description: "Plan for school or college",
  },
  {
    value: "education_abroad",
    label: "Education abroad",
    description: "Plan for study outside India",
  },
  { value: "home", label: "Home", description: "Buy or improve a home" },
  {
    value: "retirement",
    label: "Retirement",
    description: "Build long-term independence",
  },
  { value: "wealth", label: "Wealth", description: "Grow money over time" },
];

const GOAL_NAMES: Record<GoalType, string> = {
  emergency: "Emergency fund",
  education_india: "Child education in India",
  education_abroad: "Education abroad",
  home: "Home",
  retirement: "Retirement",
  wealth: "Wealth building",
  other: "Other goal",
};

function readDraft(): OnboardingDraft {
  try {
    const parsed: Partial<OnboardingDraft> = JSON.parse(
      localStorage.getItem(ONBOARDING_STORAGE_KEY) ?? "{}",
    );
    return {
      ...DEFAULT_DRAFT,
      ...parsed,
      consents: { ...DEFAULT_DRAFT.consents, ...parsed.consents },
      answers: parsed.answers ?? {},
    };
  } catch {
    return DEFAULT_DRAFT;
  }
}

function stepFromPath(pathname: string): OnboardingStep | null {
  const leaf = pathname.split("/").filter(Boolean).at(-1);
  return STEPS.includes(leaf as OnboardingStep)
    ? (leaf as OnboardingStep)
    : null;
}

function cleanOptionalAmount(value: string): string | null {
  if (!value.trim()) return null;
  return parseInr(value);
}

function AmountEntry({
  id,
  label,
  value,
  onChange,
  help,
  required = false,
}: {
  id: string;
  label: string;
  value: string;
  onChange(value: string): void;
  help?: string;
  required?: boolean;
}) {
  const form = useForm<AmountForm>({ defaultValues: { amount: value } });
  useEffect(() => form.setValue("amount", value), [form, value]);
  return (
    <MoneyInput
      control={form.control}
      name="amount"
      id={`${id}-amount`}
      label={label}
      help={help ?? "Rough numbers are fine."}
      required={required}
      onValueChange={onChange}
    />
  );
}

function StepHeading({
  step,
  title,
  description,
  showProgress = true,
}: {
  step: OnboardingStep;
  title: string;
  description: string;
  showProgress?: boolean;
}) {
  const index = STEPS.indexOf(step);
  return (
    <div className="grid gap-3">
      {showProgress ? (
        <Progress
          quiet
          value={((index + 1) / STEPS.length) * 100}
          label="Onboarding progress"
        />
      ) : null}
      <div>
        <h1 className="text-page-title font-semibold tracking-tight">
          {title}
        </h1>
        <p className="mt-2 text-base text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

function QuestionCard({
  question,
  answer,
  onAnswer,
  headingRef,
}: {
  question: RiskQuestion;
  answer?: string;
  onAnswer(value: string): void;
  headingRef: RefObject<HTMLLegendElement | null>;
}) {
  return (
    <fieldset className="grid gap-3">
      <legend
        ref={headingRef}
        tabIndex={-1}
        className="text-base font-semibold focus:outline-none"
      >
        {question.text}
      </legend>
      <div className="grid gap-2">
        {question.options.map((option) => (
          <button
            key={option.id}
            type="button"
            aria-pressed={answer === option.id}
            onClick={() => onAnswer(option.id)}
            className={`min-h-14 rounded-xl border px-4 py-3 text-left text-base ${answer === option.id ? "border-brand bg-brand/10 font-semibold" : "border-line bg-background hover:bg-muted"}`}
          >
            {option.text}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function OnboardingPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { refreshProfile } = useAuth();
  const [draft, setDraft] = useState<OnboardingDraft>(readDraft);
  const [profileLoaded, setProfileLoaded] = useState(false);
  const [profileExists, setProfileExists] = useState<boolean | null>(null);
  const [questions, setQuestions] = useState<RiskQuestion[]>([]);
  const [questionIndex, setQuestionIndex] = useState(0);
  const riskQuestionHeadingRef = useRef<HTMLLegendElement>(null);
  const shouldFocusRiskQuestion = useRef(false);
  const [goals, setGoals] = useState<GoalResponse[]>([]);
  const [riskResult, setRiskResult] = useState<RiskResult | null>(null);
  const [riskSaved, setRiskSaved] = useState(false);
  const [goalType, setGoalType] = useState<GoalType>("emergency");
  const [goalAmount, setGoalAmount] = useState("");
  const [goalYear, setGoalYear] = useState(
    String(new Date().getFullYear() + 5),
  );
  const [goalCurrency, setGoalCurrency] = useState("INR");
  const [goalError, setGoalError] = useState("");
  const [goalSaving, setGoalSaving] = useState(false);
  const [pageError, setPageError] = useState("");
  const [saving, setSaving] = useState(false);

  const pathStep = stepFromPath(location.pathname);
  const savedStep = pathStep ?? draft.step;
  const mustRestartProfileSetup =
    profileExists === false && savedStep === "done";
  const step = mustRestartProfileSetup ? "household" : savedStep;
  const goalSavedNotice =
    (location.state as { goalSaved?: unknown } | null)?.goalSaved === true;
  const answersComplete =
    questions.length > 0 &&
    questions.every((question) => Boolean(draft.answers[question.id]));
  const selectedGoalLabel = GOAL_NAMES[goalType] ?? "Goal";

  useEffect(() => {
    try {
      localStorage.setItem(
        ONBOARDING_STORAGE_KEY,
        JSON.stringify({ ...draft, step }),
      );
    } catch {
      // The API remains the source of saved profile, goal, and risk data.
    }
  }, [draft, step]);

  useEffect(() => {
    if (!profileLoaded) return;
    if (mustRestartProfileSetup) {
      setDraft((current) => ({ ...current, step: "household" }));
      if (pathStep !== "household")
        navigate("/onboarding/household", { replace: true });
      return;
    }
    if (pathStep) {
      if (pathStep !== draft.step)
        setDraft((current) => ({ ...current, step: pathStep }));
      return;
    }
    navigate(`/onboarding/${draft.step}`, { replace: true });
  }, [draft.step, mustRestartProfileSetup, navigate, pathStep, profileLoaded]);

  useEffect(() => {
    let active = true;
    void (async () => {
      try {
        const me = await getMe();
        const members = me.members?.length
          ? me.members
          : await listSettingsMembers();
        if (!active) return;
        const adults = members
          .filter((member) => member.relationship !== "child")
          .map((member): AdultDraft => ({
            id: member.id,
            displayName: member.display_name,
            relationship:
              member.id === me.me_member_id
                ? "self"
                : (member.relationship as MemberRelationship),
            residence: residenceFor(member),
            plannedReturnYear: member.residency.planned_return_year
              ? String(member.residency.planned_return_year)
              : "",
            taxRegime: member.tax.tax_regime ?? "new",
            taxSlab: member.tax.tax_slab_pct ?? "",
            birthYear: member.tax.birth_year
              ? String(member.tax.birth_year)
              : "",
            annualIncome: member.tax.annual_income ?? "",
            annualIncomeCurrency:
              member.tax.annual_income_currency === "DKK" ? "DKK" : "INR",
          }));
        const currentMember = members.find(
          (member) => member.id === me.me_member_id,
        );
        const profile = me.profile;
        setProfileExists(profile !== null);
        const context = profile?.financial_context;
        setDraft((current) => ({
          ...current,
          adults: adults.length
            ? adults
            : current.adults.map((adult, index) =>
                index === 0 && !adult.id
                  ? {
                      ...adult,
                      displayName:
                        me.display_name?.trim() ||
                        adult.displayName.trim() ||
                        "You",
                    }
                  : adult,
              ),
          householdMode: adults.some(
            (member) => member.relationship === "spouse",
          )
            ? "spouse"
            : adults.length > 1
              ? "family"
              : "me",
          taxRegime:
            profile?.tax_regime ??
            currentMember?.tax.tax_regime ??
            current.taxRegime,
          cityTier: profile?.city_tier
            ? String(profile.city_tier)
            : current.cityTier,
          timezone: profile?.timezone || current.timezone,
          monthlyExpenses: context?.monthly_expenses ?? current.monthlyExpenses,
          annualIncome: context?.annual_income ?? current.annualIncome,
          dependants:
            context?.dependants === null || context?.dependants === undefined
              ? current.dependants
              : String(context.dependants),
          employmentType: context?.employment_type ?? current.employmentType,
          answers: profile?.risk_answers ?? current.answers,
        }));
      } catch {
        // A new account can continue with the sensible default member details.
      } finally {
        if (active) setProfileLoaded(true);
      }
    })();
    void listGoals()
      .then((items) => {
        if (active) setGoals(items);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (step !== "risk") return;
    let active = true;
    void getRiskQuestions()
      .then((items) => {
        if (active) setQuestions(items);
      })
      .catch(() =>
        setPageError(
          "Risk questions couldn't be loaded. You can skip this step and return later.",
        ),
      );
    return () => {
      active = false;
    };
  }, [step]);

  useEffect(() => {
    if (step !== "risk") {
      setQuestionIndex(0);
      return;
    }
    if (shouldFocusRiskQuestion.current && questions[questionIndex]) {
      riskQuestionHeadingRef.current?.focus();
      shouldFocusRiskQuestion.current = false;
    }
  }, [questionIndex, questions, step]);

  function updateDraft(patch: Partial<OnboardingDraft>) {
    setDraft((current) => ({ ...current, ...patch }));
  }

  // Home opens single steps ("Add a goal") with a return path; finishing one goes back
  // there instead of walking on through the rest of setup.
  const returnTo = (location.state as { returnTo?: unknown } | null)?.returnTo;
  const returnPath =
    typeof returnTo === "string" && returnTo.startsWith("/") ? returnTo : null;

  function moveTo(nextStep: OnboardingStep) {
    setPageError("");
    if (returnPath) {
      navigate(returnPath);
      return;
    }
    updateDraft({ step: nextStep });
    navigate(`/onboarding/${nextStep}`);
  }

  function profilePayload(): ProfileUpsert {
    return {
      tax_regime: draft.taxRegime,
      city_tier: draft.cityTier ? (Number(draft.cityTier) as 1 | 2 | 3) : null,
      timezone: draft.timezone || "Asia/Kolkata",
      monthly_expenses: cleanOptionalAmount(draft.monthlyExpenses),
      annual_income: cleanOptionalAmount(draft.annualIncome),
      dependants: draft.dependants.trim() ? Number(draft.dependants) : null,
      employment_type: (draft.employmentType ||
        null) as ProfileUpsert["employment_type"],
    };
  }

  function updateAdult(index: number, patch: Partial<AdultDraft>) {
    setDraft((current) => ({
      ...current,
      adults: current.adults.map((adult, adultIndex) =>
        adultIndex === index ? { ...adult, ...patch } : adult,
      ),
    }));
  }

  function setHouseholdMode(mode: OnboardingDraft["householdMode"]) {
    setDraft((current) => {
      const self =
        current.adults.find(
          (adult) => adult.id && adult.relationship === "self",
        ) ??
        current.adults[0] ??
        DEFAULT_DRAFT.adults[0]!;
      const spouse = current.adults.find(
        (adult) => adult.relationship === "spouse",
      );
      const otherAdults = current.adults.filter(
        (adult) =>
          adult.relationship !== "self" && adult.relationship !== "spouse",
      );
      const adults =
        mode === "me"
          ? [{ ...self, relationship: "self" as const }]
          : [
              { ...self, relationship: "self" as const },
              spouse ?? {
                ...DEFAULT_DRAFT.adults[0]!,
                displayName: "Spouse",
                relationship: "spouse" as const,
              },
              ...(mode === "family" ? otherAdults : []),
            ];
      return { ...current, householdMode: mode, adults };
    });
  }

  function addAdult() {
    setDraft((current) => ({
      ...current,
      adults: [
        ...current.adults,
        { ...DEFAULT_DRAFT.adults[0]!, relationship: "parent" },
      ],
    }));
  }

  async function saveHousehold() {
    setSaving(true);
    setPageError("");
    try {
      if (
        draft.adults.some(
          (adult) => !adult.displayName.trim() || !adult.relationship,
        )
      ) {
        setPageError(englishCopy.household.onboarding.name_error);
        return;
      }
      const me = await getMe();
      const existingMembers = me.members?.length
        ? me.members
        : await listSettingsMembers();
      const savedMembers: SettingsMember[] = [];
      for (const [index, adult] of draft.adults.entries()) {
        const existing =
          existingMembers.find((member) => member.id === adult.id) ??
          existingMembers.find(
            (member) => member.id === me.me_member_id && index === 0,
          );
        const residency = residenceValue(
          adult.residence,
          adult.plannedReturnYear,
        );
        const tax = {
          annual_income: adult.annualIncome.trim() || null,
          annual_income_currency: adult.annualIncomeCurrency,
          birth_year: adult.birthYear.trim() ? Number(adult.birthYear) : null,
          employment_type: existing?.tax.employment_type ?? "other",
          fy_realised_equity_ltcg: existing?.tax.fy_realised_equity_ltcg ?? "0",
          fy_realised_equity_stcg: existing?.tax.fy_realised_equity_stcg ?? "0",
          surcharge_pct: existing?.tax.surcharge_pct ?? "0",
          tax_regime: adult.taxRegime,
          tax_slab_pct: adult.taxSlab || null,
        } as const;
        const saved = existing
          ? await updateSettingsMember(existing.id, {
              display_name: adult.displayName.trim(),
              relationship: adult.relationship,
              residency,
              tax,
            })
          : await createSettingsMember({
              display_name: adult.displayName.trim(),
              relationship: adult.relationship,
              is_self: index === 0 && !me.me_member_id,
              residency: residency ?? undefined,
              tax,
            });
        savedMembers.push(saved);
      }
      if (!savedMembers.length) throw new Error("No adult details to save.");
      await updateProfile(profilePayload());
      setProfileExists(true);
      await refreshProfile();
      moveTo("done");
    } catch (error) {
      setPageError(
        error instanceof Error && error.message === "No adult details to save."
          ? englishCopy.household.onboarding.save_error
          : englishCopy.household.onboarding.save_error,
      );
    } finally {
      setSaving(false);
    }
  }

  async function saveGoal(): Promise<boolean> {
    setGoalSaving(true);
    setGoalError("");
    try {
      const targetAmount =
        goalCurrency === "INR" ? parseInr(goalAmount) : goalAmount.trim();
      if (goalCurrency === "USD" && !/^\d+(?:\.\d{1,2})?$/.test(targetAmount))
        throw new Error("Enter an amount in dollars.");
      const goal = await createGoal({
        name: selectedGoalLabel,
        goal_type: goalType,
        target_amount: targetAmount,
        target_year: Number(goalYear),
        currency: goalCurrency as "INR" | "USD",
        priority: Math.min(5, goals.length + 1),
      });
      setGoals((current) => [...current, goal]);
      setGoalAmount("");
      return true;
    } catch {
      setGoalError("Enter a target amount and year, then try again.");
      return false;
    } finally {
      setGoalSaving(false);
    }
  }

  async function continueAfterGoal(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!goalAmount.trim()) {
      if (returnPath) {
        setGoalError(englishCopy.household_ui.goal_amount_required);
        return;
      }
      moveTo("risk");
      return;
    }
    if (!(await saveGoal())) return;
    if (returnPath) {
      navigate(returnPath, { state: { goalSaved: true } });
      return;
    }
    updateDraft({ step: "risk" });
    navigate("/onboarding/risk", { state: { goalSaved: true } });
  }

  async function saveRiskAnswers() {
    if (!answersComplete) return;
    setSaving(true);
    setPageError("");
    try {
      const result = await putRiskAnswers(draft.answers);
      setRiskResult(result);
      setRiskSaved(true);
    } catch {
      setPageError(
        "Your answers couldn't be saved. Please try again or skip for now.",
      );
    } finally {
      setSaving(false);
    }
  }

  function moveRiskQuestion(index: number) {
    if (index < 0 || index >= questions.length) return;
    shouldFocusRiskQuestion.current = true;
    setQuestionIndex(index);
  }

  function skipRisk() {
    setPageError("");
    if (returnPath) {
      navigate(returnPath);
      return;
    }
    moveTo("consent");
  }

  function updateConsent(
    key: keyof OnboardingDraft["consents"],
    checked: boolean,
  ) {
    updateDraft({ consents: { ...draft.consents, [key]: checked } });
  }

  function finish(destination: string) {
    try {
      localStorage.setItem(
        ONBOARDING_STORAGE_KEY,
        JSON.stringify({ ...draft, step: "done" }),
      );
    } catch {
      /* the account still opens */
    }
    navigate(destination);
  }

  if (!profileLoaded && step !== "welcome") {
    return (
      <main className="mx-auto grid min-h-screen w-full max-w-xl content-center gap-4 p-4">
        <Card>
          <p className="text-sm text-muted-foreground">Loading your setup…</p>
        </Card>
      </main>
    );
  }

  return (
    <main
      data-copy-screen="onboarding"
      className="mx-auto grid min-h-screen w-full max-w-xl content-start gap-4 px-4 py-6 sm:content-center sm:py-10"
    >
      <Card className="grid gap-5">
        {returnPath ? (
          <Link
            to={returnPath}
            className="inline-flex min-h-11 items-center gap-2 text-sm font-medium text-muted-foreground"
          >
            <ArrowLeft aria-hidden="true" className="size-4" />
            {englishCopy.household_ui.back_to_home}
          </Link>
        ) : step !== "welcome" && step !== "done" ? (
          <Link
            to={`/onboarding/${STEPS[STEPS.indexOf(step) - 1]}`}
            className="inline-flex min-h-11 items-center gap-2 text-sm font-medium text-muted-foreground"
          >
            <ArrowLeft aria-hidden="true" className="size-4" />
            {STEP_LABELS[STEPS[STEPS.indexOf(step) - 1] ?? "welcome"]}
          </Link>
        ) : null}
        {step === "welcome" ? (
          <>
            <div className="grid size-14 place-items-center rounded-2xl bg-brand/10 text-brand">
              <HeartHandshake aria-hidden="true" className="size-7" />
            </div>
            <StepHeading
              step="welcome"
              title="Your money, in one clear place"
              description="Start with what you have. Approximate amounts are welcome, and you can add or change details anytime."
            />
            <Button data-primary="true" onClick={() => moveTo("household")}>
              {englishCopy.household.onboarding.get_started}{" "}
              <ArrowRight aria-hidden="true" className="size-4" />
            </Button>
          </>
        ) : null}

        {step === "household" ? (
          <>
            <StepHeading
              step="household"
              title={englishCopy.household.onboarding.title}
              description={englishCopy.household.onboarding.description}
            />
            <TileToggle
              label={englishCopy.household.onboarding.title}
              options={[
                {
                  value: "me",
                  label: englishCopy.household.onboarding.just_me,
                  description: englishCopy.household.onboarding.just_me_help,
                },
                {
                  value: "spouse",
                  label: englishCopy.household.onboarding.me_spouse,
                  description: englishCopy.household.onboarding.me_spouse_help,
                },
                {
                  value: "family",
                  label: englishCopy.household.onboarding.my_family,
                  description: englishCopy.household.onboarding.my_family_help,
                },
              ]}
              value={draft.householdMode}
              onChange={(value) =>
                setHouseholdMode(value as OnboardingDraft["householdMode"])
              }
            />
            {draft.adults.map((adult, index) => (
              <fieldset
                key={adult.id ?? `adult-${index}`}
                className="grid gap-5 rounded-xl border border-line p-3"
              >
                <legend className="px-1 text-sm font-semibold">
                  {index === 0
                    ? englishCopy.household.onboarding.you
                    : draft.householdMode === "spouse"
                      ? englishCopy.household.onboarding.your_spouse
                      : englishCopy.household.onboarding.adult_number.replace(
                          "{number}",
                          String(index + 1),
                        )}
                </legend>
                <Field
                  id={`adult-${index}-name`}
                  label={englishCopy.settings.family.name}
                >
                  <input
                    id={`adult-${index}-name`}
                    required
                    maxLength={40}
                    value={adult.displayName}
                    onChange={(event) =>
                      updateAdult(index, {
                        displayName: event.currentTarget.value,
                      })
                    }
                    className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                  />
                </Field>
                {index === 0 || draft.householdMode === "spouse" ? null : (
                  <Select
                    id={`adult-${index}-relationship`}
                    label={englishCopy.household.onboarding.relationship}
                    value={adult.relationship}
                    onChange={(event) =>
                      updateAdult(index, {
                        relationship: event.currentTarget
                          .value as MemberRelationship,
                      })
                    }
                  >
                    {(["spouse", "parent", "other"] as const).map((value) => (
                      <option key={value} value={value}>
                        {englishCopy.household.member_relationship[value]}
                      </option>
                    ))}
                  </Select>
                )}
                <Select
                  id={`adult-${index}-residence`}
                  label={englishCopy.household.residency_title}
                  value={adult.residence}
                  onChange={(event) =>
                    updateAdult(index, {
                      residence: event.currentTarget.value as ResidenceStatus,
                      annualIncomeCurrency:
                        event.currentTarget.value === "NRI" ? "DKK" : "INR",
                    })
                  }
                >
                  <option value="RI">
                    {englishCopy.household.residency_ri}
                  </option>
                  <option value="NRI">
                    {englishCopy.household.residency_nri_dk}
                  </option>
                  <option value="RNOR">
                    {englishCopy.household.residency_rnor}
                  </option>
                </Select>
                {index === 0 ? (
                  <p className="-mt-4 text-sm text-muted-foreground">
                    {englishCopy.household.onboarding.tax_residence_help}
                  </p>
                ) : null}
                <details className="border-t border-line pt-2">
                  <summary className="min-h-11 cursor-pointer py-2 text-sm font-medium">
                    {englishCopy.household.add_details}
                  </summary>
                  <div className="mt-3 grid gap-5">
                    {adult.residence === "NRI" ? (
                      <Field
                        id={`adult-${index}-return-year`}
                        label={englishCopy.household.return_year}
                      >
                        <input
                          id={`adult-${index}-return-year`}
                          type="number"
                          min="1900"
                          max="9999"
                          value={adult.plannedReturnYear}
                          onChange={(event) =>
                            updateAdult(index, {
                              plannedReturnYear: event.currentTarget.value,
                            })
                          }
                          className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                        />
                      </Field>
                    ) : null}
                    <Select
                      id={`adult-${index}-tax-regime`}
                      label={englishCopy.settings.family.tax_regime}
                      value={adult.taxRegime}
                      onChange={(event) =>
                        updateAdult(index, {
                          taxRegime: event.currentTarget.value as TaxRegime,
                        })
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
                      id={`adult-${index}-tax-slab`}
                      label={englishCopy.settings.family.income_bracket}
                      value={adult.taxSlab}
                      onChange={(event) =>
                        updateAdult(index, {
                          taxSlab: event.currentTarget.value,
                        })
                      }
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
                      id={`adult-${index}-birth-year`}
                      label={englishCopy.settings.family.birth_year}
                    >
                      <input
                        id={`adult-${index}-birth-year`}
                        type="number"
                        min="1900"
                        max="9999"
                        value={adult.birthYear}
                        onChange={(event) =>
                          updateAdult(index, {
                            birthYear: event.currentTarget.value,
                          })
                        }
                        className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                      />
                    </Field>
                    <div className="grid grid-cols-[1fr_auto] gap-2">
                      <Field
                        id={`adult-${index}-income`}
                        label={englishCopy.household.onboarding.annual_income}
                        prefix={
                          adult.annualIncomeCurrency === "DKK" ? "DKK" : "₹"
                        }
                      >
                        <input
                          id={`adult-${index}-income`}
                          inputMode="decimal"
                          value={adult.annualIncome}
                          onChange={(event) =>
                            updateAdult(index, {
                              annualIncome: event.currentTarget.value,
                            })
                          }
                          className={`min-h-12 w-full rounded-xl border border-line bg-background pr-3 ${adult.annualIncomeCurrency === "DKK" ? "pl-14" : "pl-9"}`}
                        />
                      </Field>
                      <Select
                        id={`adult-${index}-income-currency`}
                        label={englishCopy.household.onboarding.currency}
                        value={adult.annualIncomeCurrency}
                        onChange={(event) =>
                          updateAdult(index, {
                            annualIncomeCurrency: event.currentTarget
                              .value as AdultDraft["annualIncomeCurrency"],
                          })
                        }
                      >
                        <option value="INR">INR</option>
                        <option value="DKK">DKK</option>
                      </Select>
                    </div>
                  </div>
                </details>
                {index > 1 && draft.householdMode === "family" ? (
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() =>
                      setDraft((current) => ({
                        ...current,
                        adults: current.adults.filter(
                          (_item, itemIndex) => itemIndex !== index,
                        ),
                      }))
                    }
                  >
                    {englishCopy.household.onboarding.remove_adult}
                  </Button>
                ) : null}
              </fieldset>
            ))}
            {draft.householdMode === "family" ? (
              <Button type="button" variant="secondary" onClick={addAdult}>
                {englishCopy.household.onboarding.add_adult}
              </Button>
            ) : null}
            {pageError ? (
              <p role="alert" className="text-sm text-danger-foreground">
                {pageError}
              </p>
            ) : null}
            <Button
              type="button"
              data-primary="true"
              onClick={() => void saveHousehold()}
              loading={saving}
            >
              {englishCopy.household.onboarding.continue}
            </Button>
          </>
        ) : null}

        {step === "goals" ? (
          <>
            <StepHeading
              step="goals"
              title={
                returnPath
                  ? englishCopy.household_ui.add_goal_title
                  : "What are you saving for?"
              }
              description={englishCopy.household_ui.goal_intro}
              showProgress={!returnPath}
            />
            <TileToggle
              label="Goal type"
              options={GOAL_OPTIONS.map(({ value, label, description }) => ({
                value,
                label,
                description,
              }))}
              value={goalType}
              onChange={(value) => setGoalType(value as GoalType)}
            />
            <form
              className="grid gap-5"
              onSubmit={(event) => void continueAfterGoal(event)}
            >
              {goalCurrency === "INR" ? (
                <AmountEntry
                  id="goal-amount"
                  label="Target amount"
                  value={goalAmount}
                  onChange={(value) => {
                    setGoalAmount(value);
                    setGoalError("");
                  }}
                />
              ) : (
                <Field
                  id="goal-amount-usd"
                  label="Target amount (USD)"
                  help="Enter a dollar amount, without a currency conversion."
                  prefix="USD"
                >
                  <input
                    id="goal-amount-usd"
                    inputMode="decimal"
                    value={goalAmount}
                    onChange={(event) => {
                      setGoalAmount(event.currentTarget.value);
                      setGoalError("");
                    }}
                    className="min-h-12 w-full rounded-xl border border-line bg-background pl-14 pr-3"
                  />
                </Field>
              )}
              <div className="grid grid-cols-2 gap-3">
                <Field id="goal-year" label="Target year">
                  <input
                    id="goal-year"
                    type="number"
                    inputMode="numeric"
                    min="1900"
                    max="9999"
                    value={goalYear}
                    onChange={(event) => setGoalYear(event.currentTarget.value)}
                    className="min-h-12 w-full rounded-xl border border-line bg-background px-3"
                  />
                </Field>
                <Select
                  id="goal-currency"
                  label="Currency"
                  value={goalCurrency}
                  onChange={(event) =>
                    setGoalCurrency(event.currentTarget.value)
                  }
                >
                  <option value="INR">INR ₹</option>
                  <option value="USD">USD $</option>
                </Select>
              </div>
              {goalError ? (
                <p className="text-sm text-danger-foreground" role="alert">
                  {goalError}
                </p>
              ) : null}
              <Button type="submit" data-primary="true" loading={goalSaving}>
                {returnPath
                  ? englishCopy.household_ui.goal_save
                  : englishCopy.household.onboarding.continue}
              </Button>
            </form>
            {goals.length ? (
              <div className="grid gap-2">
                <h2 className="text-sm font-semibold">Added goals</h2>
                {goals.map((goal) => (
                  <div
                    key={goal.id}
                    className="flex min-h-12 items-center justify-between gap-3 rounded-xl bg-muted px-3 py-2 text-sm"
                  >
                    <span>
                      {goal.name} · {goal.target_year}
                    </span>
                    <strong>
                      {goal.currency === "USD" ? "$" : "₹"}
                      {goal.currency === "INR"
                        ? formatInr(goal.target_amount, "compact").replace(
                            "₹",
                            "",
                          )
                        : goal.target_amount}
                    </strong>
                  </div>
                ))}
              </div>
            ) : null}
          </>
        ) : null}

        {step === "risk" ? (
          <>
            {goalSavedNotice ? (
              <div className="grid gap-1 rounded-xl bg-muted/60 p-3 text-sm">
                <p>{englishCopy.household_ui.goal_saved}</p>
                <Link
                  to="/onboarding/goals"
                  className="inline-flex min-h-11 items-center text-sm font-medium text-brand"
                >
                  {englishCopy.household_ui.add_another_goal}
                </Link>
              </div>
            ) : null}
            <StepHeading
              step="risk"
              title={
                returnPath
                  ? englishCopy.household_ui.risk_home_title
                  : englishCopy.household_ui.risk_title
              }
              description={englishCopy.household_ui.risk_description}
              showProgress={!returnPath}
            />
            {questions[questionIndex] ? (
              <div className="grid gap-3">
                <p className="text-sm text-muted-foreground">
                  {englishCopy.household_ui.risk_question_count
                    .replace("{number}", String(questionIndex + 1))
                    .replace("{count}", String(questions.length))}
                </p>
                <QuestionCard
                  key={questions[questionIndex]!.id}
                  question={questions[questionIndex]!}
                  answer={draft.answers[questions[questionIndex]!.id]}
                  headingRef={riskQuestionHeadingRef}
                  onAnswer={(answer) =>
                    updateDraft({
                      answers: {
                        ...draft.answers,
                        [questions[questionIndex]!.id]: answer,
                      },
                    })
                  }
                />
              </div>
            ) : null}
            {riskResult ? (
              <Card className="border-brand/40 bg-brand/5">
                <p className="text-sm font-semibold">
                  Your plan: {riskResult.final}
                </p>
                <p className="mt-1 text-sm text-muted-foreground">
                  {riskResult.limited_by
                    ? `Your ${riskResult.limited_by} sets the current limit.`
                    : "Based on your answers and circumstances."}
                </p>
              </Card>
            ) : null}
            {pageError ? (
              <p role="alert" className="text-sm text-danger-foreground">
                {pageError}
              </p>
            ) : null}
            {riskSaved ? (
              <Button data-primary="true" onClick={() => moveTo("consent")}>
                {englishCopy.household.onboarding.continue}
              </Button>
            ) : questions.length ? (
              <div className="grid gap-2">
                <div className="flex items-center gap-2">
                  {questionIndex > 0 ? (
                    <Button
                      type="button"
                      variant="ghost"
                      onClick={() => moveRiskQuestion(questionIndex - 1)}
                    >
                      {englishCopy.household_ui.risk_back}
                    </Button>
                  ) : null}
                  {questionIndex < questions.length - 1 ? (
                    <Button
                      type="button"
                      data-primary="true"
                      className="ml-auto"
                      disabled={!draft.answers[questions[questionIndex]!.id]}
                      onClick={() => moveRiskQuestion(questionIndex + 1)}
                    >
                      {englishCopy.household_ui.risk_next}
                    </Button>
                  ) : (
                    <Button
                      type="button"
                      data-primary="true"
                      className="ml-auto"
                      onClick={() => void saveRiskAnswers()}
                      loading={saving}
                      disabled={!answersComplete}
                    >
                      {englishCopy.household_ui.risk_save}
                    </Button>
                  )}
                </div>
                <Button variant="ghost" onClick={skipRisk}>
                  {englishCopy.household_ui.risk_skip}
                </Button>
              </div>
            ) : null}
          </>
        ) : null}

        {step === "consent" ? (
          <>
            <StepHeading
              step="consent"
              title="How should we keep you updated?"
              description="Each option is optional. Leave everything unticked if you prefer."
            />
            <div className="grid gap-2">
              <div className="flex min-h-16 items-start gap-3 rounded-xl border border-line px-3 py-3">
                <input
                  id="consent-weekly"
                  type="checkbox"
                  className="mt-1 size-5 accent-brand"
                  checked={draft.consents.weeklyEmail}
                  onChange={(event) =>
                    updateConsent("weeklyEmail", event.currentTarget.checked)
                  }
                />
                <label htmlFor="consent-weekly" className="cursor-pointer">
                  <strong className="text-sm">Weekly email</strong>
                  <span className="mt-1 block text-sm text-muted-foreground">
                    A short summary of your Money Map and next steps.
                  </span>
                </label>
              </div>
              <div className="flex min-h-16 items-start gap-3 rounded-xl border border-line px-3 py-3">
                <input
                  id="consent-reminders"
                  type="checkbox"
                  className="mt-1 size-5 accent-brand"
                  checked={draft.consents.reminders}
                  onChange={(event) =>
                    updateConsent("reminders", event.currentTarget.checked)
                  }
                />
                <label htmlFor="consent-reminders" className="cursor-pointer">
                  <strong className="text-sm">Reminders</strong>
                  <span className="mt-1 block text-sm text-muted-foreground">
                    Maturities, premiums and renewal dates.
                  </span>
                </label>
              </div>
              <div className="flex min-h-16 items-start gap-3 rounded-xl border border-line px-3 py-3">
                <input
                  id="consent-push"
                  type="checkbox"
                  className="mt-1 size-5 accent-brand"
                  checked={draft.consents.push}
                  onChange={(event) =>
                    updateConsent("push", event.currentTarget.checked)
                  }
                />
                <label htmlFor="consent-push" className="cursor-pointer">
                  <strong className="text-sm">Push notifications</strong>
                  <span className="mt-1 block text-sm text-muted-foreground">
                    Time-sensitive reminders on this device.
                  </span>
                </label>
              </div>
            </div>
            <p className="text-sm text-muted-foreground">
              Consent settings are saved on this device for now. Account sync is
              coming soon.
            </p>
            <Button data-primary="true" onClick={() => moveTo("done")}>
              Continue
            </Button>
            <Button variant="ghost" onClick={() => moveTo("done")}>
              Skip for now
            </Button>
          </>
        ) : null}

        {step === "done" ? (
          <>
            <div className="grid size-14 place-items-center rounded-full bg-success-soft text-success-foreground">
              <Check aria-hidden="true" className="size-7" />
            </div>
            <StepHeading
              step="done"
              title={englishCopy.household_ui.ready_title}
              description={englishCopy.household_ui.ready_description}
            />
            {riskResult ? (
              <p className="text-sm text-muted-foreground">
                Your plan: {riskResult.final}
                {riskResult.limited_by
                  ? `, limited by ${riskResult.limited_by}`
                  : ""}
                .
              </p>
            ) : null}
            <Button data-primary="true" onClick={() => finish("/money?add=1")}>
              {englishCopy.money_ui.add_update}{" "}
              <ArrowRight aria-hidden="true" className="size-4" />
            </Button>
            <Button variant="ghost" onClick={() => finish("/")}>
              {englishCopy.household_ui.ready_explore}
            </Button>
          </>
        ) : null}
      </Card>
    </main>
  );
}
