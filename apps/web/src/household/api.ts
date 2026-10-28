import { apiFetch } from "../api/client";
import type { components } from "../api/schema";

type Schema<Name extends keyof components["schemas"]> =
  components["schemas"][Name];

export type HouseholdSummary = Schema<"HouseholdSummary">;
export type HouseholdAccess = Schema<"HouseholdAccessResponse">;
export type HouseholdInvite = Schema<"InviteResponse">;
export type HouseholdInviteCreated = Schema<"InviteCreatedResponse">;
export type HouseholdJoin = Schema<"HouseholdJoinResponse">;
export type HouseholdRole = Schema<"HouseholdAccessPatch">["role"];

function jsonRequest(method: string, value: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  };
}

export function listHouseholds(): Promise<HouseholdSummary[]> {
  return apiFetch("/v1/households");
}

export function listHouseholdAccess(): Promise<HouseholdAccess[]> {
  return apiFetch("/v1/households/access");
}

export function listHouseholdInvites(): Promise<HouseholdInvite[]> {
  return apiFetch("/v1/households/invites");
}

export function createHouseholdInvite(value: {
  email: string;
  role: HouseholdRole;
  member_id?: string;
  send_email: boolean;
}): Promise<HouseholdInviteCreated> {
  return apiFetch("/v1/households/invites", jsonRequest("POST", value));
}

export function revokeHouseholdInvite(inviteId: string): Promise<void> {
  return apiFetch(`/v1/households/invites/${encodeURIComponent(inviteId)}`, {
    method: "DELETE",
  });
}

export function updateHouseholdAccess(
  userId: string,
  role: HouseholdRole,
): Promise<HouseholdAccess> {
  return apiFetch(
    `/v1/households/access/${encodeURIComponent(userId)}`,
    jsonRequest("PATCH", { role }),
  );
}

export function revokeHouseholdAccess(userId: string): Promise<void> {
  return apiFetch(`/v1/households/access/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
}

export function joinHousehold(value: {
  token: string;
  bring_my_data: boolean;
}): Promise<HouseholdJoin> {
  return apiFetch("/v1/households/join", jsonRequest("POST", value));
}

export function leaveHousehold(): Promise<void> {
  return apiFetch("/v1/households/leave", { method: "DELETE" });
}
