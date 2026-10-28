import { getAuthToken } from "../auth/token";
import {
  readOfflineSnapshot,
  storeOfflineSnapshot,
} from "../pwa/offline-cache";

export interface ApiFieldError {
  field?: string;
  message: string;
  code?: string;
}

export interface ApiProblem {
  code: string;
  title: string;
  errors: ApiFieldError[];
}

export class ApiError extends Error implements ApiProblem {
  readonly status: number;
  readonly code: string;
  readonly title: string;
  readonly errors: ApiFieldError[];

  constructor(status: number, problem: ApiProblem) {
    super(problem.title);
    this.name = "ApiError";
    this.status = status;
    this.code = problem.code;
    this.title = problem.title;
    this.errors = problem.errors;
  }
}

const RETRYABLE_STATUS = new Set([502, 503, 504]);
const MAX_RETRY_MS = 60_000;
const MAX_BACKOFF_MS = 15_000;
export const SELECTED_HOUSEHOLD_STORAGE_KEY = "wealth.household.selected";

export function getSelectedHouseholdId(): string | null {
  try {
    return window.sessionStorage.getItem(SELECTED_HOUSEHOLD_STORAGE_KEY);
  } catch {
    return null;
  }
}

export function persistSelectedHouseholdId(id: string | null): void {
  try {
    if (id) window.sessionStorage.setItem(SELECTED_HOUSEHOLD_STORAGE_KEY, id);
    else window.sessionStorage.removeItem(SELECTED_HOUSEHOLD_STORAGE_KEY);
  } catch {
    // The API still defaults to the signed-in user's household.
  }
}

function emitWaking(waking: boolean): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(
      new CustomEvent("wealth:api-waking", { detail: { waking } }),
    );
  }
}

function requestUrl(path: string): URL {
  const base = import.meta.env.VITE_API_URL || window.location.origin;
  return new URL(path, base);
}

async function asApiError(response: Response): Promise<ApiError> {
  let problem: Partial<ApiProblem> = {};
  try {
    const parsed: unknown = await response.json();
    if (typeof parsed === "object" && parsed !== null) {
      const body = parsed as Record<string, unknown>;
      const rawErrors = Array.isArray(body.errors)
        ? body.errors
        : Array.isArray(body.detail)
          ? body.detail
          : [];
      problem = {
        code:
          typeof body.code === "string"
            ? body.code
            : typeof body.type === "string"
              ? body.type
              : undefined,
        title:
          typeof body.title === "string"
            ? body.title
            : typeof body.detail === "string"
              ? body.detail
              : undefined,
        errors: rawErrors.flatMap((item): ApiFieldError[] => {
          if (typeof item === "string") return [{ message: item }];
          if (typeof item !== "object" || item === null) return [];
          const field =
            "field" in item && typeof item.field === "string"
              ? item.field
              : "loc" in item && Array.isArray(item.loc)
                ? item.loc
                    .filter(
                      (part: unknown): part is string =>
                        typeof part === "string",
                    )
                    .at(-1)
                : undefined;
          const message =
            "message" in item && typeof item.message === "string"
              ? item.message
              : "msg" in item && typeof item.msg === "string"
                ? item.msg
                : undefined;
          const code =
            "code" in item && typeof item.code === "string"
              ? item.code
              : undefined;
          return message ? [{ field, message, code }] : [];
        }),
      };
    }
  } catch {
    // Keep an HTTP status based message when a server returns an empty or invalid body.
  }
  return new ApiError(response.status, {
    code: problem.code ?? `http_${response.status}`,
    title: problem.title ?? response.statusText ?? "The request failed.",
    errors: problem.errors ?? [],
  });
}

function wait(milliseconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("The request was aborted.", "AbortError"));
      return;
    }
    const timeout = window.setTimeout(() => {
      signal?.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    const abort = () => {
      window.clearTimeout(timeout);
      signal?.removeEventListener("abort", abort);
      reject(new DOMException("The request was aborted.", "AbortError"));
    };
    signal?.addEventListener("abort", abort, { once: true });
  });
}

/** Fetch an API path, adding the current bearer token and recovering from free-tier cold starts. */
export async function apiFetch<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const url = requestUrl(path);
  const headers = new Headers(init.headers);
  const token = getAuthToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  const startedAt = Date.now();
  const snapshotRequest =
    (init.method ?? "GET").toUpperCase() === "GET" &&
    /^\/v1\/(?:networth|plan|inbox)(?:\/|$)/.test(url.pathname);
  let attempt = 0;
  let waking = false;
  let lastNetworkError: unknown;
  let lastRetryableResponse: Response | undefined;
  let householdFallback = false;
  let snapshotUrl = url.href;

  while (Date.now() - startedAt < MAX_RETRY_MS) {
    try {
      const requestHeaders = new Headers(headers);
      const selectedHousehold = householdFallback
        ? null
        : getSelectedHouseholdId();
      const snapshotKey = new URL(url.href);
      if (selectedHousehold) {
        snapshotKey.searchParams.set("_household_snapshot", selectedHousehold);
      } else {
        snapshotKey.searchParams.delete("_household_snapshot");
      }
      snapshotUrl = snapshotKey.href;
      if (selectedHousehold) {
        requestHeaders.set("X-Household", selectedHousehold);
      } else {
        requestHeaders.delete("X-Household");
      }
      const response = await fetch(url, { ...init, headers: requestHeaders });
      if (response.status === 404 && selectedHousehold && !householdFallback) {
        householdFallback = true;
        persistSelectedHouseholdId(null);
        if (typeof window !== "undefined") {
          window.dispatchEvent(new Event("wealth:household-removed"));
        }
        continue;
      }
      if (RETRYABLE_STATUS.has(response.status)) {
        lastRetryableResponse = response;
        if (!waking) {
          waking = true;
          emitWaking(true);
        }
      } else {
        if (waking) emitWaking(false);
        if (!response.ok) throw await asApiError(response);
        if (response.status === 204) return undefined as T;
        if (snapshotRequest)
          await storeOfflineSnapshot(snapshotUrl, response.clone());
        return (await response.json()) as T;
      }
    } catch (error) {
      if (error instanceof ApiError) {
        if (waking) emitWaking(false);
        throw error;
      }
      if (init.signal?.aborted) {
        if (waking) emitWaking(false);
        throw error;
      }
      lastNetworkError = error;
      if (!waking) {
        waking = true;
        emitWaking(true);
      }
      if (snapshotRequest) {
        const cached = await readOfflineSnapshot(snapshotUrl);
        if (cached) {
          emitWaking(false);
          return (await cached.json()) as T;
        }
      }
    }

    const remaining = MAX_RETRY_MS - (Date.now() - startedAt);
    if (remaining <= 0) break;
    const delay = Math.min(1_000 * 2 ** attempt, MAX_BACKOFF_MS, remaining);
    await wait(delay, init.signal ?? undefined);
    attempt += 1;
  }

  emitWaking(false);
  if (lastRetryableResponse) throw await asApiError(lastRetryableResponse);
  throw new ApiError(0, {
    code: "network_error",
    title:
      lastNetworkError instanceof Error
        ? lastNetworkError.message
        : "The server could not be reached.",
    errors: [],
  });
}
