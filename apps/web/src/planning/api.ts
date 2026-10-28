import { apiFetch } from "../api/client";
import type { components } from "../api/schema";

export type WeeklyPlan = components["schemas"]["WeeklyPlanResponse"];
export type ActionResponse = components["schemas"]["ActionResponse"];
export type StressResponse = components["schemas"]["StressResponse"];
export type ResearchIdeasResponse =
  components["schemas"]["ResearchIdeasResponse"];

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

export function getStress(): Promise<StressResponse> {
  return apiFetch("/v1/stress");
}

export function getResearchIdeas(): Promise<ResearchIdeasResponse> {
  return apiFetch("/v1/research/ideas");
}
