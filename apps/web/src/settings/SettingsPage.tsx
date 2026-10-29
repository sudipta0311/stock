import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Card, Checkbox, Select, Sheet } from "../ui";
import englishCopy from "../copy/en.json";
import { ApiError, apiFetch } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import {
  clearAllDeviceData,
  clearSnapshotCache,
  offlineCacheEnabled,
  setOfflineCacheEnabled,
} from "../pwa/offline-cache";
import {
  appLockEnabled,
  appLockSupported,
  createDeviceLockCredential,
  disableDeviceLock,
} from "../pwa/app-lock";
import { disablePush, enablePush, PushSetupError } from "../pwa/push";
import { isIosDevice, isStandaloneMode } from "../pwa/install";
import {
  createSettingsMember,
  downloadSettingsExport,
  getInvestingPrefs,
  getRiskQuestions,
  getSettingsConsents,
  getSettingsHouseholdAccess,
  getSettingsHouseholds,
  getSettingsMe,
  listSettingsMembers,
  removeSettingsMember,
  saveInvestingPrefs,
  saveProfile,
  saveRiskAnswers,
  saveSettingsConsents,
  settingsPathAvailable,
  updateSettingsMember,
} from "./api";
import type {
  ConsentValues,
  InvestingPrefsInput,
  MemberWrite,
  ProfileUpsert,
  RiskQuestion,
  SettingsHouseholdAccess,
  SettingsMember,
} from "./api";
import { asRecord } from "../features/data";
import { useTheme } from "../ui/theme";
import { SUPPORTED_LANGUAGES } from "../i18n/config";
import packageMetadata from "../../package.json";
import { HouseholdSharingPanel } from "./HouseholdSharingPanel";
import { MemberDetailsEditor } from "./MemberDetailsEditor";
import { FundSipFineTune } from "./FundSipFineTune";
import { useHousehold } from "../household/HouseholdContext";

const LOCAL_CONSENTS_KEY = "wealth.settings.consents";
const DAILY_CHANGES_KEY = "wealth.settings.show-daily-changes";
const emptyConsents: ConsentValues = {
  weekly_email: false,
  reminder_email: false,
  push: false,
  urgent_alerts: false,
};

type SipDraft = {
  sleeve: "equity_india" | "equity_global" | "debt" | "gold" | "cash";
  instrument_category: string;
  monthly_amount: string;
};
type Relationship = MemberWrite["relationship"];
type ConsentKey = keyof ConsentValues;

function localConsentValues(): ConsentValues {
  try {
    const parsed: unknown = JSON.parse(
      window.localStorage.getItem(LOCAL_CONSENTS_KEY) ?? "{}",
    );
    const value = asRecord(parsed);
    return {
      weekly_email: value.weekly_email === true,
      reminder_email: value.reminder_email === true,
      push: value.push === true,
      urgent_alerts: value.urgent_alerts === true,
    };
  } catch {
    return emptyConsents;
  }
}

function formatError(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.status === 409) return error.title;
  return fallback;
}

function activeHouseholdAccess(
  access: SettingsHouseholdAccess[] | undefined,
): SettingsHouseholdAccess[] {
  return (access ?? []).filter((person) => person.revoked_at === null);
}

function deleteWarning(
  me: Awaited<ReturnType<typeof getSettingsMe>> | undefined,
  sharedPeople: SettingsHouseholdAccess[],
  households: Awaited<ReturnType<typeof getSettingsHouseholds>> | undefined,
): string {
  if (me?.role === "owner" && sharedPeople.length > 0) {
    const personName =
      sharedPeople.at(0)?.display_name ||
      englishCopy.settings.privacy.delete_other_person;
    return englishCopy.settings.privacy.delete_owner_shared.replace(
      "{name}",
      personName,
    );
  }
  if (me && me.role !== "owner") {
    const ownerName =
      households?.find(
        (household) => household.owner_id === me.household_owner_id,
      )?.owner_name || englishCopy.settings.privacy.delete_owner_name_fallback;
    return englishCopy.settings.privacy.delete_partner_shared.replace(
      "{name}",
      ownerName,
    );
  }
  return englishCopy.settings.privacy.delete_generic;
}

function isStandalone(): boolean {
  const displayMode =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(display-mode: standalone)").matches;
  const iosStandalone = Boolean(
    (navigator as Navigator & { standalone?: boolean }).standalone,
  );
  return isStandaloneMode(displayMode, iosStandalone);
}

function saveBlob(blob: Blob): void {
  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = `wealth-copilot-${new Date().toISOString().slice(0, 10)}.json`;
  link.style.display = "none";
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
}

export function SettingsPage() {
  const queryClient = useQueryClient();
  const { logout } = useAuth();
  const { meMemberId, primaryCurrency, setPrimaryCurrency } = useHousehold();
  const { resolved, toggleTheme } = useTheme();
  const meSupported = settingsPathAvailable("/v1/me");
  const membersSupported = settingsPathAvailable("/v1/members");
  const profileSupported = settingsPathAvailable("/v1/me/profile");
  const investingSupported = settingsPathAvailable("/v1/me/investing");
  const consentsSupported = settingsPathAvailable("/v1/me/consents");
  const householdsSupported = settingsPathAvailable("/v1/households");
  const accessSupported = settingsPathAvailable("/v1/households/access");
  const riskSupported =
    settingsPathAvailable("/v1/meta/risk-questions") &&
    settingsPathAvailable("/v1/me/risk-answers");
  const meQuery = useQuery({
    queryKey: ["settings", "me"],
    queryFn: getSettingsMe,
    enabled: meSupported,
  });
  const membersQuery = useQuery({
    queryKey: ["settings", "members"],
    queryFn: listSettingsMembers,
    enabled: membersSupported,
  });
  const householdQuery = useQuery({
    queryKey: ["settings", "households"],
    queryFn: getSettingsHouseholds,
    enabled: householdsSupported,
  });
  const accessQuery = useQuery({
    queryKey: ["settings", "household-access"],
    queryFn: getSettingsHouseholdAccess,
    enabled: accessSupported,
  });
  const consentQuery = useQuery({
    queryKey: ["settings", "consents"],
    queryFn: getSettingsConsents,
    enabled: consentsSupported,
  });
  const investingQuery = useQuery({
    queryKey: ["settings", "investing"],
    queryFn: getInvestingPrefs,
    enabled: investingSupported,
  });
  const riskQuestionQuery = useQuery({
    queryKey: ["settings", "risk-questions"],
    queryFn: getRiskQuestions,
    enabled: false,
  });

  const [consents, setConsents] = useState<ConsentValues>(emptyConsents);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [profileName, setProfileName] = useState("");
  const [taxRegime, setTaxRegime] = useState<"old" | "new">("new");
  const [taxSlab, setTaxSlab] = useState("");
  const [cityTier, setCityTier] = useState("");
  const [timezone, setTimezone] = useState("Asia/Kolkata");
  const [monthlySip, setMonthlySip] = useState("0");
  const [newMoney, setNewMoney] = useState("0");
  const [hasLrsBroker, setHasLrsBroker] = useState(false);
  const [lessPaperwork, setLessPaperwork] = useState(false);
  const [satellitePct, setSatellitePct] = useState("20");
  const [birthYear, setBirthYear] = useState("");
  const [existingSips, setExistingSips] = useState<SipDraft[]>([]);
  const [showDailyChanges, setShowDailyChanges] = useState(false);
  const [profileMessage, setProfileMessage] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [sharedDevice, setSharedDevice] = useState(
    () => !offlineCacheEnabled(),
  );
  const [sharedDeviceMessage, setSharedDeviceMessage] = useState("");
  const [lockEnabled, setLockEnabled] = useState(appLockEnabled);
  const [lockMessage, setLockMessage] = useState("");
  const [membersMessage, setMembersMessage] = useState("");
  const [newMemberName, setNewMemberName] = useState("");
  const [newRelationship, setNewRelationship] = useState<Relationship>("other");
  const [newResidence, setNewResidence] = useState<"RI" | "NRI" | "RNOR">("RI");
  const [newReturnYear, setNewReturnYear] = useState("");
  const [addPersonOpen, setAddPersonOpen] = useState(false);
  const [removingMember, setRemovingMember] = useState<SettingsMember | null>(
    null,
  );
  const [removeMemberMessage, setRemoveMemberMessage] = useState("");
  const [riskOpen, setRiskOpen] = useState(false);
  const [riskAnswers, setRiskAnswers] = useState<Record<string, string>>({});
  const [emergencyOk, setEmergencyOk] = useState(true);
  const [riskMessage, setRiskMessage] = useState("");
  const [riskSaving, setRiskSaving] = useState(false);
  const [exportMessage, setExportMessage] = useState("");
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteText, setDeleteText] = useState("");
  const [deleteMessage, setDeleteMessage] = useState("");
  const [deletingAccount, setDeletingAccount] = useState(false);
  const [consentSaving, setConsentSaving] = useState(false);

  const me = meQuery.data;
  const members = membersQuery.data ?? me?.members ?? [];
  const sharedPeople = useMemo(
    () => activeHouseholdAccess(accessQuery.data),
    [accessQuery.data],
  );
  const canEditHousehold = me?.role !== "viewer";
  const currentMember =
    members.find((member) => member.id === meMemberId) ?? members[0];
  const currentTax = currentMember?.tax;

  useEffect(() => {
    if (consentQuery.data) setConsents(consentQuery.data);
    else if (!consentsSupported) setConsents(localConsentValues());
  }, [consentQuery.data, consentsSupported]);

  useEffect(() => {
    if (me) {
      setProfileName(currentMember?.display_name ?? me.display_name ?? "");
      setTaxRegime(me.profile?.tax_regime ?? currentTax?.tax_regime ?? "new");
      setTaxSlab(me.profile?.tax_slab_pct ?? currentTax?.tax_slab_pct ?? "");
      setCityTier(me.profile?.city_tier ? String(me.profile.city_tier) : "");
      setTimezone(me.profile?.timezone || "Asia/Kolkata");
    }
  }, [me, currentMember, currentTax]);

  useEffect(() => {
    const preferences = investingQuery.data;
    if (!preferences) return;
    setMonthlySip(preferences.monthly_sip_capacity || "0");
    setNewMoney(preferences.new_money_this_week || "0");
    setHasLrsBroker(preferences.has_lrs_broker);
    setLessPaperwork(preferences.prefers_no_paperwork);
    setSatellitePct(preferences.satellite_pct || "20");
    setBirthYear(preferences.birth_year ? String(preferences.birth_year) : "");
    setExistingSips(
      (preferences.existing_sips ?? []).map((sip) => ({
        sleeve: sip.sleeve,
        instrument_category: sip.instrument_category,
        monthly_amount: sip.monthly_amount,
      })),
    );
  }, [investingQuery.data]);

  useEffect(() => {
    try {
      setShowDailyChanges(
        window.localStorage.getItem(DAILY_CHANGES_KEY) === "true",
      );
    } catch {
      setShowDailyChanges(false);
    }
  }, []);

  async function updateConsent(
    key: ConsentKey,
    enabled: boolean,
  ): Promise<void> {
    setNotificationMessage("");
    if (key === "push" && enabled) {
      if (
        isIosDevice(navigator.userAgent, navigator.platform) &&
        !isStandalone()
      ) {
        setNotificationMessage(
          englishCopy.settings.notifications.push_install_ios,
        );
        return;
      }
      try {
        await enablePush(import.meta.env.VITE_VAPID_PUBLIC_KEY ?? "");
      } catch (error) {
        const message =
          error instanceof PushSetupError && error.reason === "permission"
            ? englishCopy.settings.notifications.permission_denied
            : error instanceof PushSetupError && error.reason === "unavailable"
              ? import.meta.env.VITE_VAPID_PUBLIC_KEY
                ? englishCopy.settings.notifications.push_unavailable
                : englishCopy.settings.notifications.vapid_missing
              : englishCopy.settings.notifications.push_error;
        setNotificationMessage(message);
        return;
      }
    }
    if (key === "push" && !enabled) {
      try {
        await disablePush();
      } catch {
        setNotificationMessage(englishCopy.settings.notifications.push_error);
        return;
      }
    }
    const next = { ...consents, [key]: enabled };
    setConsentSaving(true);
    try {
      if (consentsSupported) {
        await saveSettingsConsents(next);
        await queryClient.invalidateQueries({
          queryKey: ["settings", "consents"],
        });
      } else {
        window.localStorage.setItem(LOCAL_CONSENTS_KEY, JSON.stringify(next));
      }
      setConsents(next);
      setNotificationMessage(
        key === "push"
          ? enabled
            ? englishCopy.settings.notifications.push_success
            : englishCopy.settings.notifications.push_off_success
          : englishCopy.settings.notifications.saved,
      );
    } catch {
      if (key === "push" && enabled) {
        try {
          await disablePush();
        } catch {
          /* The next settings visit can retry cleanup. */
        }
      }
      setNotificationMessage(englishCopy.settings.notifications.save_error);
    } finally {
      setConsentSaving(false);
    }
  }

  async function savePlanDetails(
    event: FormEvent<HTMLFormElement>,
  ): Promise<void> {
    event.preventDefault();
    setProfileMessage("");
    setSavingProfile(true);
    try {
      const memberId = me?.me_member_id ?? currentMember?.id;
      if (
        profileName.trim() &&
        memberId &&
        settingsPathAvailable("/v1/members/{member_id}")
      ) {
        await updateSettingsMember(memberId, {
          display_name: profileName.trim(),
        });
      }
      if (profileSupported) {
        const context = asRecord(me?.profile?.financial_context);
        const profile: ProfileUpsert = {
          tax_regime: taxRegime,
          tax_slab_pct: taxSlab || null,
          city_tier: cityTier ? (Number(cityTier) as 1 | 2 | 3) : null,
          timezone: timezone || "Asia/Kolkata",
          language: SUPPORTED_LANGUAGES[0].code,
          monthly_expenses:
            typeof context.monthly_expenses === "string"
              ? context.monthly_expenses
              : null,
          annual_income:
            typeof context.annual_income === "string"
              ? context.annual_income
              : null,
          dependants:
            typeof context.dependants === "number" ? context.dependants : null,
          employment_type:
            (context.employment_type as ProfileUpsert["employment_type"]) ??
            null,
          income_stability:
            (context.income_stability as ProfileUpsert["income_stability"]) ??
            null,
        };
        await saveProfile(profile);
      }
      if (investingSupported) {
        const current = investingQuery.data;
        const preferences: InvestingPrefsInput = {
          monthly_sip_capacity: monthlySip.trim() || "0",
          existing_sips: existingSips
            .filter((sip) => sip.monthly_amount.trim())
            .map((sip) => ({
              sleeve: sip.sleeve,
              instrument_category: sip.instrument_category,
              monthly_amount: sip.monthly_amount.trim(),
            })),
          new_money_this_week: newMoney.trim() || "0",
          has_lrs_broker: hasLrsBroker,
          prefers_no_paperwork: lessPaperwork,
          satellite_pct: satellitePct || "20",
          birth_year: birthYear ? Number(birthYear) : null,
          ask_used_this_fy_dkk: current?.ask_used_this_fy_dkk ?? "0",
          lrs_used_this_fy: current?.lrs_used_this_fy ?? "0",
          broker: current?.broker ?? "nordnet",
          has_indian_nri_share_account:
            current?.has_indian_nri_share_account ?? false,
          has_danish_pension_account:
            current?.has_danish_pension_account ?? false,
          us_stock_picks_opt_in: current?.us_stock_picks_opt_in ?? false,
          us_stock_picks_pct: current?.us_stock_picks_pct ?? "10",
        };
        await saveInvestingPrefs(preferences);
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["settings", "me"] }),
        queryClient.invalidateQueries({ queryKey: ["settings", "members"] }),
        queryClient.invalidateQueries({ queryKey: ["settings", "investing"] }),
      ]);
      setProfileMessage(englishCopy.settings.family.save_profile_success);
    } catch {
      setProfileMessage(englishCopy.settings.family.save_profile_error);
    } finally {
      setSavingProfile(false);
    }
  }

  async function addPerson(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setMembersMessage("");
    if (!newMemberName.trim()) {
      setMembersMessage(englishCopy.settings.family.name_error);
      return;
    }
    try {
      await createSettingsMember({
        display_name: newMemberName.trim(),
        relationship: newRelationship,
        is_self: false,
        residency: {
          status: newResidence,
          country: newResidence === "NRI" ? "DK" : "IN",
          settle_plan: newResidence === "NRI" ? "undecided" : "india",
          planned_return_year:
            newResidence === "NRI" && /^\d{4}$/.test(newReturnYear.trim())
              ? Number(newReturnYear.trim())
              : null,
        },
      });
      await queryClient.invalidateQueries({
        queryKey: ["settings", "members"],
      });
      await queryClient.invalidateQueries({ queryKey: ["settings", "me"] });
      setNewMemberName("");
      setNewRelationship("other");
      setNewResidence("RI");
      setNewReturnYear("");
      setAddPersonOpen(false);
    } catch (error) {
      setMembersMessage(
        formatError(error, englishCopy.settings.family.add_error),
      );
    }
  }

  async function removePerson(): Promise<void> {
    if (!removingMember) return;
    setRemoveMemberMessage("");
    try {
      await removeSettingsMember(removingMember.id);
      await queryClient.invalidateQueries({
        queryKey: ["settings", "members"],
      });
      await queryClient.invalidateQueries({ queryKey: ["settings", "me"] });
      setRemovingMember(null);
    } catch (error) {
      setRemoveMemberMessage(
        error instanceof ApiError && error.status === 409
          ? englishCopy.settings.family.remove_conflict
          : englishCopy.settings.family.remove_error,
      );
    }
  }

  async function saveRisk(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setRiskMessage("");
    setRiskSaving(true);
    try {
      await saveRiskAnswers({
        answers: riskAnswers,
        emergency_ok: emergencyOk,
      });
      setRiskMessage(englishCopy.settings.family.risk_saved);
      setRiskOpen(false);
      await queryClient.invalidateQueries({ queryKey: ["settings", "me"] });
    } catch {
      setRiskMessage(englishCopy.settings.family.risk_error);
    } finally {
      setRiskSaving(false);
    }
  }

  async function changeDeviceLock(enabled: boolean): Promise<void> {
    setLockMessage("");
    if (!enabled) {
      disableDeviceLock();
      setLockEnabled(false);
      setLockMessage(englishCopy.settings.privacy.lock_disabled);
      return;
    }
    if (!appLockSupported()) {
      setLockMessage(englishCopy.settings.privacy.lock_unsupported);
      return;
    }
    try {
      await createDeviceLockCredential();
      setLockEnabled(true);
      setLockMessage(englishCopy.settings.privacy.lock_enabled);
    } catch {
      setLockMessage(englishCopy.settings.privacy.lock_failed);
    }
  }

  async function exportData(): Promise<void> {
    setExportMessage("");
    try {
      const blob = await downloadSettingsExport();
      saveBlob(blob);
    } catch {
      setExportMessage(englishCopy.settings.privacy.export_error);
    }
  }

  async function deleteAccount(): Promise<void> {
    if (deleteText !== "DELETE") return;
    setDeletingAccount(true);
    setDeleteMessage("");
    try {
      await apiFetch<void>("/v1/me", { method: "DELETE" });
      await clearAllDeviceData();
      setDeleteOpen(false);
      await logout();
    } catch {
      setDeleteMessage(englishCopy.settings.privacy.delete_error);
    } finally {
      setDeletingAccount(false);
    }
  }

  async function toggleSharedDevice(enabled: boolean): Promise<void> {
    setSharedDevice(enabled);
    setOfflineCacheEnabled(!enabled);
    if (enabled) await clearSnapshotCache();
    setSharedDeviceMessage(
      enabled
        ? englishCopy.settings.privacy.shared_device_on
        : englishCopy.settings.privacy.shared_device_off,
    );
  }

  function updateSip(
    index: number,
    field: keyof SipDraft,
    value: string,
  ): void {
    setExistingSips((current) =>
      current.map((sip, position) =>
        position === index
          ? {
              ...sip,
              [field]:
                field === "sleeve" ? (value as SipDraft["sleeve"]) : value,
            }
          : sip,
      ),
    );
  }

  const riskQuestions: RiskQuestion[] = riskQuestionQuery.data ?? [];
  const canSaveRisk =
    riskQuestions.length > 0 &&
    riskQuestions.every((question) => Boolean(riskAnswers[question.id]));
  const warning = deleteWarning(me, sharedPeople, householdQuery.data);
  const viewOnly = me?.role === "viewer";

  return (
    <div className="grid gap-4">
      <section aria-labelledby="settings-family-heading" className="grid gap-3">
        <Card className="grid gap-4">
          <div>
            <h2
              id="settings-family-heading"
              className="text-base font-semibold"
            >
              {englishCopy.household.settings.household}
            </h2>
          </div>
          {viewOnly ? (
            <p className="text-sm text-muted-foreground">
              {englishCopy.settings.family.read_only}
            </p>
          ) : null}
          <div className="grid gap-2">
            {members.map((member) => (
              <div
                key={member.id}
                className="grid gap-2 rounded-xl border border-line px-3 py-2"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-medium">
                    {member.display_name}
                    {member.id === meMemberId
                      ? ` · ${englishCopy.household.you}`
                      : ""}
                  </p>
                  {member.id !== meMemberId &&
                  membersSupported &&
                  canEditHousehold ? (
                    <Button
                      variant="ghost"
                      className="min-h-10 px-3 py-2 text-danger-foreground"
                      onClick={() => {
                        setRemoveMemberMessage("");
                        setRemovingMember(member);
                      }}
                    >
                      {englishCopy.settings.family.remove_person}
                    </Button>
                  ) : null}
                </div>
                <MemberDetailsEditor
                  member={member}
                  disabled={!membersSupported || !canEditHousehold}
                />
              </div>
            ))}
            {members.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                {englishCopy.settings.family.no_members}
              </p>
            ) : null}
          </div>
          {membersSupported && canEditHousehold ? (
            addPersonOpen ? (
              <form
                onSubmit={(event) => void addPerson(event)}
                className="grid gap-3 rounded-xl border border-line p-3"
              >
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="new-member-name"
                >
                  {englishCopy.settings.family.name}
                  <input
                    id="new-member-name"
                    value={newMemberName}
                    onChange={(event) =>
                      setNewMemberName(event.currentTarget.value)
                    }
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                    required
                    maxLength={40}
                  />
                </label>
                <Select
                  id="new-member-relationship"
                  label={englishCopy.settings.family.relationship}
                  value={newRelationship}
                  onChange={(event) =>
                    setNewRelationship(
                      event.currentTarget.value as Relationship,
                    )
                  }
                >
                  {(["spouse", "child", "parent", "other"] as const).map(
                    (relationship) => (
                      <option key={relationship} value={relationship}>
                        {englishCopy.settings.family[relationship]}
                      </option>
                    ),
                  )}
                </Select>
                <Select
                  id="new-member-residence"
                  label={englishCopy.household.residency_title}
                  value={newResidence}
                  onChange={(event) =>
                    setNewResidence(
                      event.currentTarget.value as "RI" | "NRI" | "RNOR",
                    )
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
                {newResidence === "NRI" ? (
                  <details>
                    <summary className="cursor-pointer text-sm font-medium">
                      {englishCopy.household.add_details}
                    </summary>
                    <label
                      className="mt-2 grid gap-1 text-sm font-medium"
                      htmlFor="new-member-return-year"
                    >
                      {englishCopy.household.return_year}
                      <input
                        id="new-member-return-year"
                        type="number"
                        min="1900"
                        max="9999"
                        value={newReturnYear}
                        onChange={(event) =>
                          setNewReturnYear(event.currentTarget.value)
                        }
                        className="min-h-12 rounded-xl border border-line bg-surface px-3"
                      />
                    </label>
                  </details>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  <Button type="submit" variant="secondary">
                    {englishCopy.settings.family.add_person}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setAddPersonOpen(false)}
                  >
                    {englishCopy.settings.family.cancel}
                  </Button>
                </div>
              </form>
            ) : (
              <Button
                variant="secondary"
                className="w-full"
                onClick={() => setAddPersonOpen(true)}
              >
                {englishCopy.settings.family.add_person}
              </Button>
            )
          ) : null}
          {membersMessage ? (
            <p role="status" className="text-sm text-danger-foreground">
              {membersMessage}
            </p>
          ) : null}
          <details className="border-t border-line pt-3">
            <summary className="cursor-pointer text-sm font-semibold">
              {englishCopy.settings.family.fine_tune}
            </summary>
            <form
              onSubmit={(event) => void savePlanDetails(event)}
              className="mt-4 grid gap-4"
            >
              <fieldset disabled={viewOnly} className="grid gap-4">
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="profile-name"
                >
                  {englishCopy.settings.family.display_name}
                  <input
                    id="profile-name"
                    value={profileName}
                    onChange={(event) =>
                      setProfileName(event.currentTarget.value)
                    }
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                    maxLength={40}
                  />
                </label>
                <Select
                  id="tax-regime"
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
                  id="tax-slab"
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
                <Select
                  id="city-tier"
                  label={englishCopy.settings.family.city_tier}
                  value={cityTier}
                  onChange={(event) => setCityTier(event.currentTarget.value)}
                >
                  <option value="">
                    {englishCopy.settings.family.no_bracket}
                  </option>
                  <option value="1">
                    {englishCopy.settings.family.tier_1}
                  </option>
                  <option value="2">
                    {englishCopy.settings.family.tier_2}
                  </option>
                  <option value="3">
                    {englishCopy.settings.family.tier_3}
                  </option>
                </Select>
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="timezone"
                >
                  {englishCopy.settings.family.timezone}
                  <input
                    id="timezone"
                    value={timezone}
                    onChange={(event) => setTimezone(event.currentTarget.value)}
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                  />
                </label>
                <Select
                  id="language"
                  label={englishCopy.settings.family.language}
                  value={SUPPORTED_LANGUAGES[0].code}
                  disabled={SUPPORTED_LANGUAGES.length === 1}
                >
                  {SUPPORTED_LANGUAGES.map((language) => (
                    <option key={language.code} value={language.code}>
                      {language.label}
                    </option>
                  ))}
                </Select>
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="monthly-sip"
                >
                  {englishCopy.settings.family.monthly_sip}
                  <input
                    id="monthly-sip"
                    inputMode="decimal"
                    value={monthlySip}
                    onChange={(event) =>
                      setMonthlySip(event.currentTarget.value)
                    }
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
                  />
                </label>
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="new-money"
                >
                  {englishCopy.settings.family.this_week}
                  <input
                    id="new-money"
                    inputMode="decimal"
                    value={newMoney}
                    onChange={(event) => setNewMoney(event.currentTarget.value)}
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
                  />
                </label>
                <div className="grid gap-2">
                  <h3 className="text-sm font-semibold">
                    {englishCopy.settings.family.existing_sips}
                  </h3>
                  {existingSips.map((sip, index) => (
                    <div
                      key={`${sip.sleeve}-${index}`}
                      className="grid gap-3 rounded-xl border border-line p-3"
                    >
                      <Select
                        id={`sip-sleeve-${index}`}
                        label={
                          englishCopy.labels.sleeves[
                            sip.sleeve as keyof typeof englishCopy.labels.sleeves
                          ] ?? englishCopy.settings.family.existing_sips
                        }
                        value={sip.sleeve}
                        onChange={(event) =>
                          updateSip(index, "sleeve", event.currentTarget.value)
                        }
                      >
                        {Object.entries(englishCopy.labels.sleeves).map(
                          ([key, label]) => (
                            <option key={key} value={key}>
                              {label}
                            </option>
                          ),
                        )}
                      </Select>
                      <Select
                        id={`sip-instrument-${index}`}
                        label={englishCopy.settings.family.existing_sips}
                        value={sip.instrument_category}
                        onChange={(event) =>
                          updateSip(
                            index,
                            "instrument_category",
                            event.currentTarget.value,
                          )
                        }
                      >
                        {Object.entries(
                          englishCopy.labels.instrument_categories,
                        ).map(([key, label]) => (
                          <option key={key} value={key}>
                            {label}
                          </option>
                        ))}
                      </Select>
                      <label
                        className="grid gap-1 text-sm font-medium"
                        htmlFor={`sip-amount-${index}`}
                      >
                        {englishCopy.settings.family.monthly_sip}
                        <input
                          id={`sip-amount-${index}`}
                          inputMode="decimal"
                          value={sip.monthly_amount}
                          onChange={(event) =>
                            updateSip(
                              index,
                              "monthly_amount",
                              event.currentTarget.value,
                            )
                          }
                          className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base tabular-nums"
                        />
                      </label>
                      <Button
                        type="button"
                        variant="ghost"
                        className="justify-start"
                        onClick={() =>
                          setExistingSips((current) =>
                            current.filter((_, position) => position !== index),
                          )
                        }
                      >
                        {englishCopy.settings.family.remove_sip}
                      </Button>
                    </div>
                  ))}
                  <Button
                    type="button"
                    variant="secondary"
                    className="w-full"
                    onClick={() =>
                      setExistingSips((current) => [
                        ...current,
                        {
                          sleeve: "equity_india",
                          instrument_category: "in_core_index",
                          monthly_amount: "0",
                        },
                      ])
                    }
                  >
                    {englishCopy.settings.family.add_sip}
                  </Button>
                </div>
                <Checkbox
                  id="has-lrs-broker"
                  label={englishCopy.settings.family.lrs_broker}
                  checked={hasLrsBroker}
                  onChange={(event) =>
                    setHasLrsBroker(event.currentTarget.checked)
                  }
                />
                <Checkbox
                  id="less-paperwork"
                  label={englishCopy.settings.family.no_paperwork}
                  checked={lessPaperwork}
                  onChange={(event) =>
                    setLessPaperwork(event.currentTarget.checked)
                  }
                />
                <label
                  className="grid gap-2 text-sm font-medium"
                  htmlFor="satellite-pct"
                >
                  {englishCopy.settings.family.satellite_pct}
                  <span className="text-xs font-normal text-muted-foreground">
                    {englishCopy.settings.family.satellite_help}
                  </span>
                  <input
                    id="satellite-pct"
                    type="range"
                    min="0"
                    max="25"
                    step="1"
                    value={satellitePct}
                    onChange={(event) =>
                      setSatellitePct(event.currentTarget.value)
                    }
                  />
                  <span className="tabular-nums text-muted-foreground">
                    {satellitePct}%
                  </span>
                </label>
                <label
                  className="grid gap-1 text-sm font-medium"
                  htmlFor="birth-year"
                >
                  {englishCopy.settings.family.birth_year}
                  <input
                    id="birth-year"
                    type="number"
                    min="1900"
                    max="9999"
                    value={birthYear}
                    onChange={(event) =>
                      setBirthYear(event.currentTarget.value)
                    }
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                  />
                </label>
                {investingSupported ||
                profileSupported ||
                settingsPathAvailable("/v1/members/{member_id}") ? (
                  <Button
                    type="submit"
                    variant="secondary"
                    loading={savingProfile}
                    className="w-full"
                  >
                    {englishCopy.settings.family.save_plan}
                  </Button>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    {englishCopy.settings.family.unsupported}
                  </p>
                )}
                {profileMessage ? (
                  <p role="status" className="text-sm text-muted-foreground">
                    {profileMessage}
                  </p>
                ) : null}
              </fieldset>
            </form>
            {riskSupported ? (
              <div className="mt-4 border-t border-line pt-4">
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => {
                    setRiskMessage("");
                    setRiskOpen(true);
                    void riskQuestionQuery
                      .refetch()
                      .catch(() =>
                        setRiskMessage(
                          englishCopy.settings.family.risk_question_error,
                        ),
                      );
                  }}
                >
                  {englishCopy.settings.family.retake_risk}
                </Button>
                {riskMessage ? (
                  <p
                    role="status"
                    className="mt-2 text-sm text-muted-foreground"
                  >
                    {riskMessage}
                  </p>
                ) : null}
              </div>
            ) : null}
            <Checkbox
              id="show-daily-changes"
              className="mt-4"
              label={englishCopy.settings.family.daily_changes}
              checked={showDailyChanges}
              onChange={(event) => {
                const enabled = event.currentTarget.checked;
                setShowDailyChanges(enabled);
                try {
                  window.localStorage.setItem(
                    DAILY_CHANGES_KEY,
                    String(enabled),
                  );
                } catch {
                  /* The default remains off for this session. */
                }
              }}
            />
          </details>
        </Card>
        <HouseholdSharingPanel />
        <section
          aria-labelledby="fund-sip-settings-heading"
          className="grid gap-3"
        >
          <FundSipFineTune />
        </section>
        <Card className="grid gap-3">
          <h3 className="text-sm font-semibold">
            {englishCopy.household.primary_currency}
          </h3>
          <Select
            id="primary-currency"
            label={englishCopy.household.primary_currency}
            value={primaryCurrency}
            onChange={(event) =>
              setPrimaryCurrency(event.currentTarget.value as "INR" | "DKK")
            }
          >
            <option value="INR">{englishCopy.household.inr}</option>
            <option value="DKK">{englishCopy.household.dkk}</option>
          </Select>
        </Card>
      </section>

      <section aria-labelledby="settings-notifications-heading">
        <Card className="grid gap-3">
          <div>
            <h2
              id="settings-notifications-heading"
              className="text-base font-semibold"
            >
              {englishCopy.settings.notifications.title}
            </h2>
          </div>
          <Checkbox
            id="weekly-email"
            label={englishCopy.settings.notifications.weekly_email}
            checked={consents.weekly_email}
            disabled={
              consentSaving || (consentsSupported && consentQuery.isPending)
            }
            onChange={(event) =>
              void updateConsent("weekly_email", event.currentTarget.checked)
            }
          />
          <Checkbox
            id="reminder-email"
            label={englishCopy.settings.notifications.reminders}
            checked={consents.reminder_email}
            disabled={
              consentSaving || (consentsSupported && consentQuery.isPending)
            }
            onChange={(event) =>
              void updateConsent("reminder_email", event.currentTarget.checked)
            }
          />
          <Checkbox
            id="push-alerts"
            label={englishCopy.settings.notifications.push}
            checked={consents.push}
            disabled={
              consentSaving || (consentsSupported && consentQuery.isPending)
            }
            onChange={(event) =>
              void updateConsent("push", event.currentTarget.checked)
            }
          />
          <Checkbox
            id="urgent-alerts"
            label={englishCopy.settings.notifications.urgent}
            checked={consents.urgent_alerts}
            disabled={
              consentSaving || (consentsSupported && consentQuery.isPending)
            }
            onChange={(event) =>
              void updateConsent("urgent_alerts", event.currentTarget.checked)
            }
          />
          {!consentsSupported ? (
            <p className="text-sm text-muted-foreground">
              {englishCopy.settings.notifications.sync_later}
            </p>
          ) : null}
          {notificationMessage ? (
            <p role="status" className="text-sm text-muted-foreground">
              {notificationMessage}
            </p>
          ) : null}
        </Card>
      </section>

      <section aria-labelledby="settings-privacy-heading">
        <Card className="grid gap-3">
          <div>
            <h2
              id="settings-privacy-heading"
              className="text-base font-semibold"
            >
              {englishCopy.settings.privacy.title}
            </h2>
          </div>
          <Checkbox
            id="shared-device"
            label={englishCopy.settings.privacy.shared_device}
            help={englishCopy.settings.privacy.shared_device_help}
            checked={sharedDevice}
            onChange={(event) =>
              void toggleSharedDevice(event.currentTarget.checked)
            }
          />
          {sharedDeviceMessage ? (
            <p role="status" className="text-sm text-muted-foreground">
              {sharedDeviceMessage}
            </p>
          ) : null}
          <div className="border-t border-line pt-3">
            <Checkbox
              id="app-lock"
              label={englishCopy.settings.privacy.lock}
              help={englishCopy.settings.privacy.lock_help}
              checked={lockEnabled}
              onChange={(event) =>
                void changeDeviceLock(event.currentTarget.checked)
              }
            />
            {!lockEnabled && !appLockSupported() ? (
              <p className="pl-9 text-sm text-muted-foreground">
                {englishCopy.settings.privacy.lock_unsupported}
              </p>
            ) : null}
            {lockMessage ? (
              <p role="status" className="pl-9 text-sm text-muted-foreground">
                {lockMessage}
              </p>
            ) : null}
          </div>
          <div className="grid gap-2 border-t border-line pt-3">
            {meSupported ? (
              <Button
                variant="secondary"
                className="w-full"
                onClick={() => void exportData()}
              >
                {englishCopy.settings.privacy.export}
              </Button>
            ) : (
              <p className="text-sm text-muted-foreground">
                {englishCopy.settings.family.unsupported}
              </p>
            )}
            {exportMessage ? (
              <p role="status" className="text-sm text-danger-foreground">
                {exportMessage}
              </p>
            ) : null}
          </div>
          <div className="grid gap-2 border-t border-line pt-3">
            {meSupported ? (
              <Button
                variant="danger"
                className="w-full"
                onClick={() => {
                  setDeleteText("");
                  setDeleteMessage("");
                  setDeleteOpen(true);
                }}
              >
                {englishCopy.settings.privacy.delete}
              </Button>
            ) : (
              <p className="text-sm text-muted-foreground">
                {englishCopy.settings.family.unsupported}
              </p>
            )}
            <Button
              variant="ghost"
              className="w-full"
              onClick={() => void logout()}
            >
              {englishCopy.settings.family.sign_out}
            </Button>
          </div>
        </Card>
      </section>

      <section aria-labelledby="settings-about-heading">
        <Card className="grid gap-3">
          <h2 id="settings-about-heading" className="text-base font-semibold">
            {englishCopy.settings.about.title}
          </h2>
          <p className="text-sm text-muted-foreground">
            {englishCopy.settings.about.version}: {packageMetadata.version}
          </p>
          <p className="text-sm text-muted-foreground">
            {englishCopy.settings.about.disclaimer}
          </p>
          <div className="flex flex-wrap gap-4 border-t border-line pt-3 text-sm">
            <a
              className="min-h-10 text-brand underline"
              href={englishCopy.settings.about.investor_url}
              target="_blank"
              rel="noreferrer"
            >
              {englishCopy.settings.about.investor_link}
            </a>
            <a
              className="min-h-10 text-brand underline"
              href={englishCopy.settings.about.data_url}
              target="_blank"
              rel="noreferrer"
            >
              {englishCopy.settings.about.data_link}
            </a>
          </div>
          <Button variant="secondary" className="w-full" onClick={toggleTheme}>
            {resolved === "dark"
              ? englishCopy.settings.about.use_light_theme
              : englishCopy.settings.about.use_dark_theme}
          </Button>
        </Card>
      </section>

      <Sheet
        open={Boolean(removingMember)}
        onOpenChange={(open) => {
          if (!open) setRemovingMember(null);
        }}
        title={englishCopy.settings.family.remove_person}
      >
        <div className="grid gap-3">
          <p className="text-sm">
            {englishCopy.settings.family.remove_confirm}
          </p>
          {removeMemberMessage ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {removeMemberMessage}
            </p>
          ) : null}
          <Button
            variant="danger"
            className="w-full"
            onClick={() => void removePerson()}
          >
            {englishCopy.settings.family.remove_person}
          </Button>
          <Button
            variant="secondary"
            className="w-full"
            onClick={() => setRemovingMember(null)}
          >
            {englishCopy.settings.family.cancel}
          </Button>
        </div>
      </Sheet>

      <Sheet
        open={riskOpen}
        onOpenChange={(open) => setRiskOpen(open)}
        title={englishCopy.settings.family.risk_title}
        description={englishCopy.settings.family.risk_description}
      >
        <form onSubmit={(event) => void saveRisk(event)} className="grid gap-4">
          {riskQuestionQuery.isError ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {englishCopy.settings.family.risk_question_error}
            </p>
          ) : null}
          {riskQuestions.map((question) => (
            <fieldset
              key={question.id}
              className="grid gap-2 rounded-xl border border-line p-3"
            >
              <legend className="px-1 text-sm font-semibold">
                {question.text}
              </legend>
              {question.options.map((option) => (
                <label
                  key={option.id}
                  className="flex min-h-10 items-center gap-2 text-sm"
                >
                  <input
                    type="radio"
                    name={question.id}
                    value={option.id}
                    checked={riskAnswers[question.id] === option.id}
                    onChange={() =>
                      setRiskAnswers((current) => ({
                        ...current,
                        [question.id]: option.id,
                      }))
                    }
                  />
                  {option.text}
                </label>
              ))}
            </fieldset>
          ))}
          <Checkbox
            id="risk-emergency-ok"
            label={englishCopy.settings.family.risk_emergency}
            checked={emergencyOk}
            onChange={(event) => setEmergencyOk(event.currentTarget.checked)}
          />
          <Button
            type="submit"
            className="w-full"
            loading={riskSaving}
            disabled={!canSaveRisk}
          >
            {englishCopy.settings.family.risk_save}
          </Button>
        </form>
      </Sheet>

      <Sheet
        open={deleteOpen}
        onOpenChange={(open) => setDeleteOpen(open)}
        title={englishCopy.settings.privacy.delete}
      >
        <div className="grid gap-3">
          <p className="text-sm font-medium">{warning}</p>
          <p className="text-sm text-muted-foreground">
            {englishCopy.settings.privacy.delete_confirm}
          </p>
          <label
            className="grid gap-1 text-sm font-medium"
            htmlFor="delete-confirmation"
          >
            {englishCopy.settings.privacy.delete_confirm}
            <input
              id="delete-confirmation"
              autoComplete="off"
              value={deleteText}
              onChange={(event) => setDeleteText(event.currentTarget.value)}
              className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
            />
          </label>
          {deleteMessage ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {deleteMessage}
            </p>
          ) : null}
          <Button
            variant="danger"
            className="w-full"
            disabled={deleteText !== "DELETE"}
            loading={deletingAccount}
            onClick={() => void deleteAccount()}
          >
            {englishCopy.settings.privacy.delete_button}
          </Button>
          <Button
            variant="secondary"
            className="w-full"
            onClick={() => setDeleteOpen(false)}
          >
            {englishCopy.settings.family.cancel}
          </Button>
        </div>
      </Sheet>
    </div>
  );
}

export { localConsentValues, deleteWarning, saveBlob };
