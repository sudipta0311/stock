const WAKE_LIMIT_MS = 150_000;
const ATTEMPT_LIMIT_MS = 30_000;
const MAX_BACKOFF_MS = 5_000;
const SHOW_WAKING_AFTER_MS = 1_500;
const STAYS_AWAKE_MS = 60_000;

let inFlight: Promise<boolean> | null = null;
let awakeUntil = 0;

function healthUrl(): URL {
  const base = import.meta.env.VITE_API_URL || window.location.origin;
  return new URL("/healthz", base);
}

function emitWaking(waking: boolean): void {
  window.dispatchEvent(
    new CustomEvent("wealth:api-waking", { detail: { waking } }),
  );
}

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function answers(): Promise<boolean> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), ATTEMPT_LIMIT_MS);
  try {
    const response = await fetch(healthUrl(), {
      cache: "no-store",
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    window.clearTimeout(timeout);
  }
}

/**
 * The free API host sleeps after 15 idle minutes and can take over a minute to
 * wake. Resolves true once the API answers, or false if it stays unreachable.
 */
export function pollUntilApiAnswers(): Promise<boolean> {
  if (Date.now() < awakeUntil) return Promise.resolve(true);
  inFlight ??= (async () => {
    const startedAt = Date.now();
    const showWaking = window.setTimeout(
      () => emitWaking(true),
      SHOW_WAKING_AFTER_MS,
    );
    let delay = 1_000;
    try {
      while (Date.now() - startedAt < WAKE_LIMIT_MS) {
        if (await answers()) {
          awakeUntil = Date.now() + STAYS_AWAKE_MS;
          return true;
        }
        const remaining = WAKE_LIMIT_MS - (Date.now() - startedAt);
        if (remaining <= 0) break;
        await wait(Math.min(delay, remaining));
        delay = Math.min(delay * 2, MAX_BACKOFF_MS);
      }
      return false;
    } finally {
      window.clearTimeout(showWaking);
      emitWaking(false);
      inFlight = null;
    }
  })();
  return inFlight;
}

/** Waits for the deployed API; local development and tests have no sleeping host. */
export function waitForApi(): Promise<boolean> {
  return import.meta.env.PROD ? pollUntilApiAnswers() : Promise.resolve(true);
}

/** Starts waking the API without waiting, e.g. while the person is at Google. */
export function nudgeApi(): void {
  if (!import.meta.env.PROD) return;
  try {
    void fetch(healthUrl(), { cache: "no-store", keepalive: true }).catch(
      () => undefined,
    );
  } catch {
    // Waking early is only a head start; sign-in waits for the API anyway.
  }
}

export function resetApiWarmupForTests(): void {
  inFlight = null;
  awakeUntil = 0;
}
