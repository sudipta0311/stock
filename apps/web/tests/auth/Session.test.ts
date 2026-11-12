import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function deferred<T>(): {
  promise: Promise<T>;
  reject: (reason?: unknown) => void;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

function refreshResponse(): Response {
  return new Response(
    JSON.stringify({
      access_token: "synthetic-access",
      refresh_token: "synthetic-next-refresh",
      expires_in: 1_800,
      session_expires_at: "2030-01-01T00:00:00Z",
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

let cancelProactiveRefresh: (() => void) | undefined;

async function loadSession(): Promise<{
  session: typeof import("../../src/auth/session");
  token: typeof import("../../src/auth/token");
}> {
  vi.resetModules();
  const token = await import("../../src/auth/token");
  const session = await import("../../src/auth/session");
  cancelProactiveRefresh = session.cancelProactiveRefresh;
  return { session, token };
}

beforeEach(() => {
  window.localStorage.clear();
  vi.useRealTimers();
});

afterEach(() => {
  cancelProactiveRefresh?.();
  cancelProactiveRefresh = undefined;
  window.localStorage.clear();
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("waitForSignInRenewal", () => {
  it("resolves when no sign-in renewal is in flight", async () => {
    const { session } = await loadSession();

    await expect(session.waitForSignInRenewal()).resolves.toBeUndefined();
  });

  it.each(["success", "conflict"] as const)(
    "waits for a pending %s renewal fetch",
    async (outcome) => {
      const { session, token } = await loadSession();
      token.setAuthSession({
        access_token: "synthetic-current-access",
        refresh_token: "synthetic-current-refresh",
        expires_in: 1_800,
      });
      const responsePending = deferred<Response>();
      const fetchStarted = deferred<void>();
      const fetcher = vi.fn<typeof fetch>(() => {
        fetchStarted.resolve();
        return responsePending.promise;
      });
      vi.stubGlobal("fetch", fetcher);

      const renewal = session.refreshAuthSession();
      const waiting = session.waitForSignInRenewal();
      await fetchStarted.promise;
      let finished = false;
      void waiting.then(() => {
        finished = true;
      });
      await Promise.resolve();
      expect(finished).toBe(false);

      responsePending.resolve(
        outcome === "success"
          ? refreshResponse()
          : new Response(null, { status: 409 }),
      );
      await expect(waiting).resolves.toBeUndefined();
      if (outcome === "success") {
        await expect(renewal).resolves.toMatchObject({
          access_token: "synthetic-access",
        });
      } else {
        await expect(renewal).resolves.toBeNull();
      }
    },
  );

  it("resolves after a network failure exhausts its retries", async () => {
    vi.useFakeTimers();
    const { session, token } = await loadSession();
    token.setAuthSession({
      access_token: "synthetic-current-access",
      refresh_token: "synthetic-current-refresh",
      expires_in: 1_800,
    });
    const firstFetch = deferred<Response>();
    const fetchStarted = deferred<void>();
    const fetcher = vi.fn<typeof fetch>(() => {
      fetchStarted.resolve();
      return firstFetch.promise;
    });
    vi.stubGlobal("fetch", fetcher);

    const renewal = session.refreshAuthSession();
    const waiting = session.waitForSignInRenewal();
    await fetchStarted.promise;
    let finished = false;
    void waiting.then(() => {
      finished = true;
    });
    firstFetch.reject(new TypeError("Synthetic network failure."));
    await vi.advanceTimersByTimeAsync(0);
    expect(finished).toBe(false);

    await vi.advanceTimersByTimeAsync(240_000);
    await expect(renewal).rejects.toThrow(
      "Unable to reach the sign-in service.",
    );
    await expect(waiting).resolves.toBeUndefined();
  });

  it("waits for a renewal that starts while startup sign-in is pending", async () => {
    const { session, token } = await loadSession();
    session.beginAuthBootstrap();
    token.setAuthSession({
      access_token: "synthetic-current-access",
      refresh_token: "synthetic-current-refresh",
      expires_in: 1_800,
    });
    const responsePending = deferred<Response>();
    const fetchStarted = deferred<void>();
    const fetcher = vi.fn<typeof fetch>(() => {
      fetchStarted.resolve();
      return responsePending.promise;
    });
    vi.stubGlobal("fetch", fetcher);

    const waiting = session.waitForSignInRenewal();
    const renewal = session.refreshAuthSession();
    await fetchStarted.promise;
    session.completeAuthBootstrap();
    let finished = false;
    void waiting.then(() => {
      finished = true;
    });
    await Promise.resolve();
    expect(finished).toBe(false);

    responsePending.resolve(refreshResponse());
    await expect(renewal).resolves.toMatchObject({
      access_token: "synthetic-access",
    });
    await expect(waiting).resolves.toBeUndefined();
  });
});
