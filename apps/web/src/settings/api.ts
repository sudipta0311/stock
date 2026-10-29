import { apiFetch } from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import type { components } from "../api/schema";

export type SettingsMe = components["schemas"]["MeResponse"];
export type SettingsMember = components["schemas"]["MemberResponse"];
export type StatementInboxKey =
  components["schemas"]["StatementInboxKeyResponse"];
export type StatementInboxKeyCreated =
  components["schemas"]["StatementInboxKeyCreated"];
export type SettingsHousehold = components["schemas"]["HouseholdSummary"];
export type SettingsHouseholdAccess =
  components["schemas"]["HouseholdAccessResponse"];
export type SettingsInvite = components["schemas"]["InviteResponse"];
export type SettingsInviteCreated =
  components["schemas"]["InviteCreatedResponse"];
export type SettingsInviteCreate = components["schemas"]["InviteCreate"];
export type SettingsAccessPatch = components["schemas"]["HouseholdAccessPatch"];
export type ConsentValues = components["schemas"]["ConsentValues"];
export type ProfileUpsert = components["schemas"]["ProfileUpsert"];
export type InvestingPrefs = components["schemas"]["InvestingPrefs-Output"];
export type InvestingPrefsInput = components["schemas"]["InvestingPrefs-Input"];
export type RiskQuestion = components["schemas"]["RiskQuestionResponse"];
export type RiskAnswers = components["schemas"]["RiskAnswersRequest"];
export type MemberWrite = components["schemas"]["MemberWrite"];
export type MemberPatch = components["schemas"]["MemberPatch"];

export interface SignedInDevice {
  id: string;
  device_label: string;
  created_at: string;
  last_used_at: string;
  current: boolean;
}

export function settingsPathAvailable(path: string): boolean {
  return apiPathAvailable(path);
}

export function listStatementInboxKeys(): Promise<StatementInboxKey[]> {
  return apiFetch("/v1/statement-inbox/keys");
}

export function createStatementInboxKey(
  label: string,
): Promise<StatementInboxKeyCreated> {
  return apiFetch("/v1/statement-inbox/keys", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
}

export function revokeStatementInboxKey(id: string): Promise<void> {
  return apiFetch(`/v1/statement-inbox/keys/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function forgetStatementPasswords(): Promise<void> {
  return apiFetch("/v1/statement-inbox/passwords", { method: "DELETE" });
}

export function getSettingsMe(): Promise<SettingsMe> {
  return apiFetch("/v1/me");
}

export function listSignedInDevices(): Promise<SignedInDevice[]> {
  return apiFetch("/v1/auth/sessions");
}

export function signOutEverywhere(): Promise<void> {
  return apiFetch("/v1/auth/logout-all", { method: "POST" });
}

export function listSettingsMembers(): Promise<SettingsMember[]> {
  return apiFetch("/v1/members");
}

export function createSettingsMember(
  value: MemberWrite,
): Promise<SettingsMember> {
  return apiFetch("/v1/members", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function updateSettingsMember(
  id: string,
  value: MemberPatch,
): Promise<SettingsMember> {
  return apiFetch(`/v1/members/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function removeSettingsMember(id: string): Promise<void> {
  return apiFetch(`/v1/members/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function getSettingsConsents(): Promise<ConsentValues> {
  return apiFetch("/v1/me/consents");
}

export function saveSettingsConsents(
  value: ConsentValues,
): Promise<ConsentValues> {
  return apiFetch("/v1/me/consents", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function getInvestingPrefs(): Promise<InvestingPrefs> {
  return apiFetch("/v1/me/investing");
}

export function saveInvestingPrefs(
  value: InvestingPrefsInput,
): Promise<InvestingPrefs> {
  return apiFetch("/v1/me/investing", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function saveProfile(
  value: ProfileUpsert,
): Promise<components["schemas"]["ProfileResponse"]> {
  return apiFetch("/v1/me/profile", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function getSettingsHouseholds(): Promise<SettingsHousehold[]> {
  return apiFetch("/v1/households");
}

export function getSettingsHouseholdAccess(): Promise<
  SettingsHouseholdAccess[]
> {
  return apiFetch("/v1/households/access");
}

export function getSettingsInvites(): Promise<SettingsInvite[]> {
  return apiFetch("/v1/households/invites");
}

export function createSettingsInvite(
  value: SettingsInviteCreate,
): Promise<SettingsInviteCreated> {
  return apiFetch("/v1/households/invites", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function revokeSettingsInvite(id: string): Promise<void> {
  return apiFetch(`/v1/households/invites/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function updateSettingsAccess(
  userId: string,
  value: SettingsAccessPatch,
): Promise<SettingsHouseholdAccess> {
  return apiFetch(`/v1/households/access/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function revokeSettingsAccess(userId: string): Promise<void> {
  return apiFetch(`/v1/households/access/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
}

export function joinSettingsHousehold(
  value: components["schemas"]["InviteJoin"],
): Promise<components["schemas"]["HouseholdJoinResponse"]> {
  return apiFetch("/v1/households/join", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export function leaveSettingsHousehold(): Promise<void> {
  return apiFetch("/v1/households/leave", { method: "DELETE" });
}

export function getRiskQuestions(): Promise<RiskQuestion[]> {
  return apiFetch("/v1/meta/risk-questions");
}

export function saveRiskAnswers(
  value: RiskAnswers,
): Promise<components["schemas"]["RiskResultResponse"]> {
  return apiFetch("/v1/me/risk-answers", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

export async function downloadSettingsExport(): Promise<Blob> {
  const me = await getSettingsMe();
  const paths: Array<[string, string]> = [
    ["members", "/v1/members"],
    ["accounts", "/v1/accounts"],
    ["assets", "/v1/assets"],
    ["goals", "/v1/goals"],
    ["imports", "/v1/imports"],
    ["actions", "/v1/actions"],
    ["inbox", "/v1/inbox"],
    ["highlight_dismissals", "/v1/highlights/dismissals"],
  ];
  const entries = await Promise.all(
    paths.map(async ([key, path]) => {
      if (!apiPathAvailable(path)) return [key, null] as const;
      try {
        return [key, await apiFetch<unknown>(path)] as const;
      } catch {
        return [key, null] as const;
      }
    }),
  );
  let householdName = me.display_name ?? "Household";
  if (apiPathAvailable("/v1/households")) {
    try {
      const households = await getSettingsHouseholds();
      householdName =
        households.find(
          (household) => household.owner_id === me.household_owner_id,
        )?.owner_name ?? householdName;
    } catch {
      // The export still completes with the household name returned by the account.
    }
  }
  const document = {
    exported_at: new Date().toISOString(),
    household_name: householdName,
    household_owner_id: me.household_owner_id,
    me,
    ...Object.fromEntries(entries),
  };
  return new Blob([JSON.stringify(document, null, 2)], {
    type: "application/json",
  });
}
