import {
  clearAuthToken,
  getAuthTokenExpiresAt,
  getRefreshToken,
  setAuthSession,
} from "./token";
import type { AuthSessionTokens } from "./token";

const RETRYABLE_STATUS = new Set([502, 503, 504]);
const MAX_REFRESH_RETRY_MS = 60_000;
const MAX_BACKOFF_MS = 15_000;
const REFRESH_EARLY_MS = 2 * 60_000;

export interface AuthSessionResponse extends AuthSessionTokens {
  session_expires_at: string;
  token_type?: string;
}

let refreshInFlight: Promise<AuthSessionResponse | null> | null = null;
let proactiveTimer: number | undefined;
let authBootstrapStarted = false;
let authBootstrapComplete: Promise<void> | null = null;
let finishAuthBootstrap: (() => void) | null = null;

export function beginAuthBootstrap(): void {
  if (authBootstrapStarted) return;
  authBootstrapStarted = true;
  authBootstrapComplete = new Promise((resolve) => {
    finishAuthBootstrap = resolve;
  });
}

export function completeAuthBootstrap(): void {
  finishAuthBootstrap?.();
  finishAuthBootstrap = null;
}

export function waitForAuthBootstrap(): Promise<void> {
  return authBootstrapComplete ?? Promise.resolve();
}

function refreshUrl(): URL {
  const base = import.meta.env.VITE_API_URL || window.location.origin;
  return new URL("/v1/auth/refresh", base);
}

function dispatchAuthExpired(): void {
  window.dispatchEvent(new Event("wealth:auth-expired"));
}

function validSessionResponse(value: unknown): value is AuthSessionResponse {
  if (typeof value !== "object" || value === null) return false;
  const body = value as Record<string, unknown>;
  return (
    typeof body.access_token === "string" &&
    typeof body.refresh_token === "string" &&
    typeof body.expires_in === "number" &&
    Number.isFinite(body.expires_in) &&
    body.expires_in > 0 &&
    typeof body.session_expires_at === "string"
  );
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function requestRefresh(
  initialRefreshToken: string,
): Promise<AuthSessionResponse | null> {
  let refreshToken = initialRefreshToken;
  const startedAt = Date.now();
  let delay = 1_000;
  while (Date.now() - startedAt < MAX_REFRESH_RETRY_MS) {
    let response: Response;
    try {
      response = await fetch(refreshUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
        cache: "no-store",
      });
    } catch (error) {
      if (Date.now() - startedAt >= MAX_REFRESH_RETRY_MS) throw error;
      await wait(Math.min(delay, MAX_BACKOFF_MS));
      delay = Math.min(delay * 2, MAX_BACKOFF_MS);
      continue;
    }
    if (RETRYABLE_STATUS.has(response.status)) {
      await wait(Math.min(delay, MAX_BACKOFF_MS));
      delay = Math.min(delay * 2, MAX_BACKOFF_MS);
      continue;
    }
    if (response.status === 409) {
      // Renewed moments ago by another tab or a lost response: use a newer stored
      // token if one exists; otherwise this device must sign in again.
      const latest = getRefreshToken();
      if (latest && latest !== refreshToken) {
        refreshToken = latest;
        continue;
      }
      clearAuthToken();
      return null;
    }
    if (response.status === 401 || response.status === 403) {
      clearAuthToken();
      return null;
    }
    if (!response.ok) {
      throw new Error("Unable to refresh the sign-in session.");
    }
    const value: unknown = await response.json();
    if (!validSessionResponse(value)) {
      throw new Error("The sign-in response was incomplete.");
    }
    setAuthSession(value);
    scheduleProactiveRefresh();
    return value;
  }
  throw new Error("Unable to reach the sign-in service.");
}

/**
 * Serialise refreshes across tabs: the refresh token is shared through localStorage and
 * rotates on every use, so two tabs refreshing at once would look like token theft.
 */
function withCrossTabRefreshLock<T>(work: () => Promise<T>): Promise<T> {
  if (
    typeof navigator !== "undefined" &&
    "locks" in navigator &&
    navigator.locks
  ) {
    // The callback's promise is awaited by the lock; this TS lib types it one level deep.
    return navigator.locks.request(
      "wealth-auth-refresh",
      work,
    ) as unknown as Promise<T>;
  }
  return work();
}

export function refreshAuthSession(): Promise<AuthSessionResponse | null> {
  if (!getRefreshToken()) return Promise.resolve(null);
  if (refreshInFlight) return refreshInFlight;

  const pending = withCrossTabRefreshLock(async () => {
    // Re-read inside the lock: another tab may have rotated the token meanwhile.
    const latest = getRefreshToken();
    return latest ? requestRefresh(latest) : null;
  });
  refreshInFlight = pending;
  void pending.then(
    () => {
      if (refreshInFlight === pending) refreshInFlight = null;
    },
    () => {
      if (refreshInFlight === pending) refreshInFlight = null;
    },
  );
  return pending;
}

export function cancelProactiveRefresh(): void {
  if (proactiveTimer !== undefined) window.clearTimeout(proactiveTimer);
  proactiveTimer = undefined;
}

export function scheduleProactiveRefresh(retryAfterMs?: number): void {
  cancelProactiveRefresh();
  if (!getRefreshToken()) return;
  const expiresAt = getAuthTokenExpiresAt();
  if (expiresAt === null) return;
  const delay =
    retryAfterMs ?? Math.max(0, expiresAt - Date.now() - REFRESH_EARLY_MS);
  proactiveTimer = window.setTimeout(() => {
    void refreshAuthSession()
      .then((result) => {
        if (result === null) dispatchAuthExpired();
      })
      .catch(() => scheduleProactiveRefresh(30_000));
  }, delay);
}
