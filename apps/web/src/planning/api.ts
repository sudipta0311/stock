import { apiFetch } from "../api/client";
import type { components } from "../api/schema";

export type WeeklyPlan = components["schemas"]["WeeklyPlanResponse"];
export type ActionResponse = components["schemas"]["ActionResponse"];
export type StressResponse = components["schemas"]["StressResponse"];
export type ResearchIdeasResponse =
  components["schemas"]["ResearchIdeasResponse"];
export type RecordBuyRequest = components["schemas"]["RecordBuyRequest"];
export type ActionBoughtResponse =
  components["schemas"]["ActionBoughtResponse"];
export type RecordedBuyResponse = components["schemas"]["RecordedBuyResponse"];
export type RecordedBuyPatch = components["schemas"]["RecordedBuyPatch"];

export function getPlan(): Promise<WeeklyPlan> {
  return apiFetch("/v1/plan");
}

export function getActions(): Promise<ActionResponse[]> {
  return apiFetch("/v1/actions");
}

export function updateActionStatus(
  actionId: string,
  status: "done" | "dismissed",
): Promise<ActionResponse> {
  return apiFetch(`/v1/actions/${encodeURIComponent(actionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

export function recordBought(
  actionId: string,
  payload: RecordBuyRequest,
): Promise<ActionBoughtResponse> {
  return apiFetch(`/v1/actions/${encodeURIComponent(actionId)}/bought`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function updateRecordedBuy(
  recordedBuyId: string,
  payload: RecordedBuyPatch,
): Promise<RecordedBuyResponse> {
  return apiFetch(`/v1/recorded-buys/${encodeURIComponent(recordedBuyId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function removeRecordedBuy(
  recordedBuyId: string,
): Promise<RecordedBuyResponse> {
  return apiFetch(`/v1/recorded-buys/${encodeURIComponent(recordedBuyId)}`, {
    method: "DELETE",
  });
}

export function getStress(): Promise<StressResponse> {
  return apiFetch("/v1/stress");
}

export function getResearchIdeas(): Promise<ResearchIdeasResponse> {
  return apiFetch("/v1/research/ideas");
}

export function dismissHighlight(key: string): Promise<void> {
  return apiFetch(`/v1/highlights/${encodeURIComponent(key)}/dismiss`, {
    method: "POST",
  });
}
