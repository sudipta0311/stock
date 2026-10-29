import type { components } from "../api/schema";
import { apiFetch } from "../api/client";
import type { AssetDraft } from "./forms.generated";

type Schema<Name extends keyof components["schemas"]> =
  components["schemas"][Name];

export type MeResponse = Schema<"MeResponse">;
export type ProfileUpsert = Schema<"ProfileUpsert">;
export type RiskQuestion = Schema<"RiskQuestionResponse">;
export type RiskResult = Schema<"RiskResultResponse">;
export type AssetResponse = Schema<"AssetResponse">;
export type AssetDetailResponse = Schema<"AssetDetailResponse">;
export type QuickAddResponse = Schema<"QuickAddResponse">;
export type MemberResponse = Schema<"MemberResponse">;
export type NetworthResponse = Schema<"NetworthResponse">;
export type GoalResponse = Schema<"GoalResponse">;
export type GoalWrite = Schema<"Goal">;
export type AccountResponse = Schema<"AccountResponse">;

function jsonRequest(method: string, payload: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

function memberQuery(memberId?: string | null): string {
  return memberId ? `?member_id=${encodeURIComponent(memberId)}` : "";
}

export function getMe(): Promise<MeResponse> {
  return apiFetch("/v1/me");
}

export function updateProfile(
  payload: ProfileUpsert,
): Promise<Schema<"ProfileResponse">> {
  return apiFetch("/v1/me/profile", jsonRequest("PUT", payload));
}

export function putRiskAnswers(
  answers: Record<string, string>,
): Promise<RiskResult> {
  return apiFetch("/v1/me/risk-answers", jsonRequest("PUT", { answers }));
}

export function getRiskQuestions(): Promise<RiskQuestion[]> {
  return apiFetch("/v1/meta/risk-questions");
}

export function updateChecklist(
  categories: string[],
): Promise<{ categories: string[] }> {
  return apiFetch("/v1/me/checklist", jsonRequest("PUT", { categories }));
}

export function listAssets(memberId?: string | null): Promise<AssetResponse[]> {
  return apiFetch(`/v1/assets${memberQuery(memberId)}`);
}

export function getAsset(assetId: string): Promise<AssetDetailResponse> {
  return apiFetch(`/v1/assets/${encodeURIComponent(assetId)}`);
}

export function createAsset(payload: AssetDraft): Promise<AssetResponse> {
  return apiFetch("/v1/assets", jsonRequest("POST", payload));
}

export function patchAsset(
  assetId: string,
  payload: Record<string, unknown>,
): Promise<AssetResponse> {
  return apiFetch(
    `/v1/assets/${encodeURIComponent(assetId)}`,
    jsonRequest("PATCH", payload),
  );
}

export function closeAsset(assetId: string): Promise<void> {
  return apiFetch(`/v1/assets/${encodeURIComponent(assetId)}`, {
    method: "DELETE",
  });
}

export function quickAdd(text: string): Promise<QuickAddResponse> {
  return apiFetch("/v1/assets/quick-add", jsonRequest("POST", { text }));
}

export function listMembers(): Promise<MemberResponse[]> {
  return apiFetch("/v1/members");
}

export function listAccounts(): Promise<AccountResponse[]> {
  return apiFetch("/v1/accounts");
}

export function createMember(
  payload: Schema<"MemberWrite">,
): Promise<MemberResponse> {
  return apiFetch("/v1/members", jsonRequest("POST", payload));
}

export function getNetworth(
  memberId?: string | null,
): Promise<NetworthResponse> {
  return apiFetch(`/v1/networth${memberQuery(memberId)}`);
}

export function listGoals(): Promise<GoalResponse[]> {
  return apiFetch("/v1/goals");
}

export function createGoal(payload: GoalWrite): Promise<GoalResponse> {
  return apiFetch("/v1/goals", jsonRequest("POST", payload));
}
