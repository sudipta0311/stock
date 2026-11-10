import { afterEach, expect, it, vi } from "vitest";
import {
  ApiError,
  apiFetch,
  getSelectedHouseholdId,
  HOUSEHOLD_SELECTION_CHANGED_EVENT,
  persistSelectedHouseholdId,
  SELECTED_HOUSEHOLD_STORAGE_KEY,
} from "../../src/api/client";
import { cancelProactiveRefresh } from "../../src/auth/session";
import { clearAuthToken, setAuthSession } from "../../src/auth/token";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  cancelProactiveRefresh();
  clearAuthToken();
  window.sessionStorage.clear();
});

it("drops a selected household when the API says it is no longer available", async () => {
  const oldHouseholdId = "household-synthetic-old";
  persistSelectedHouseholdId(oldHouseholdId);
  const changed = vi.fn();
  window.addEventListener(HOUSEHOLD_SELECTION_CHANGED_EVENT, changed);
  const fetcher = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "The household was not found." }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ ready: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetcher);

  try {
    await expect(
      apiFetch<{ ready: boolean }>("/v1/me/investing"),
    ).resolves.toEqual({ ready: true });

    expect(
      new Headers(fetcher.mock.calls[0]?.[1]?.headers).get("X-Household"),
    ).toBe(oldHouseholdId);
    expect(
      new Headers(fetcher.mock.calls[1]?.[1]?.headers).get("X-Household"),
    ).toBeNull();
    expect(getSelectedHouseholdId()).toBeNull();
    expect(
      window.sessionStorage.getItem(SELECTED_HOUSEHOLD_STORAGE_KEY),
    ).toBeNull();
    expect(changed).toHaveBeenCalledOnce();
  } finally {
    window.removeEventListener(HOUSEHOLD_SELECTION_CHANGED_EVENT, changed);
  }
});

it("aborts old-household requests and omits its header after selection changes", async () => {
  persistSelectedHouseholdId("household-synthetic-old");
  let started!: () => void;
  let resolveOldResponse!: (response: Response) => void;
  let oldRequestSignal: AbortSignal | null | undefined;
  const requestStarted = new Promise<void>((resolve) => {
    started = resolve;
  });
  const fetcher = vi.fn<typeof fetch>(
    (_input, init) =>
      new Promise<Response>((resolve) => {
        oldRequestSignal = init?.signal;
        resolveOldResponse = resolve;
        started();
      }),
  );
  vi.stubGlobal("fetch", fetcher);

  const pending = apiFetch("/v1/plan");
  await requestStarted;
  persistSelectedHouseholdId(null);
  expect(oldRequestSignal?.aborted).toBe(true);
  resolveOldResponse(
    new Response(JSON.stringify({ stale: true }), { status: 200 }),
  );
  await expect(pending).rejects.toMatchObject({ name: "AbortError" });

  fetcher.mockResolvedValueOnce(
    new Response(JSON.stringify({ ready: true }), { status: 200 }),
  );
  await expect(apiFetch<{ ready: boolean }>("/v1/me")).resolves.toEqual({
    ready: true,
  });
  expect(
    new Headers(fetcher.mock.calls[1]?.[1]?.headers).get("X-Household"),
  ).toBeNull();
});

it("works without AbortSignal.any and still forwards the caller's abort", async () => {
  // Older iOS Safari (before 17.4) has no AbortSignal.any.
  const original = Object.getOwnPropertyDescriptor(AbortSignal, "any");
  Reflect.deleteProperty(AbortSignal, "any");
  persistSelectedHouseholdId("household-synthetic-shared");
  let requestSignal: AbortSignal | null | undefined;
  const fetcher = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ ready: true }), { status: 200 }),
    )
    .mockImplementationOnce(
      (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          requestSignal = init?.signal;
          init?.signal?.addEventListener("abort", () =>
            reject(new DOMException("Aborted", "AbortError")),
          );
        }),
    );
  vi.stubGlobal("fetch", fetcher);

  try {
    const caller = new AbortController();
    await expect(
      apiFetch<{ ready: boolean }>("/v1/me", { signal: caller.signal }),
    ).resolves.toEqual({ ready: true });

    const cancelled = new AbortController();
    const pending = apiFetch("/v1/plan", { signal: cancelled.signal });
    await vi.waitFor(() => expect(requestSignal).toBeDefined());
    cancelled.abort();
    expect(requestSignal?.aborted).toBe(true);
    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  } finally {
    if (original) Object.defineProperty(AbortSignal, "any", original);
  }
});

it("parses RFC 7807 validation problems", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          type: "validation_error",
          title: "Check the amount",
          errors: [
            {
              field: "amount",
              message: "Enter a valid amount",
              code: "invalid",
            },
          ],
        }),
        {
          status: 422,
          headers: { "content-type": "application/problem+json" },
        },
      ),
    ),
  );

  await expect(apiFetch("/v1/members")).rejects.toMatchObject({
    status: 422,
    code: "validation_error",
    title: "Check the amount",
    errors: [
      { field: "amount", message: "Enter a valid amount", code: "invalid" },
    ],
  } satisfies Partial<ApiError>);
});

it("retries a cold-start response and emits waking events", async () => {
  vi.useFakeTimers();
  const fetcher = vi
    .fn()
    .mockResolvedValueOnce(
      new Response("", { status: 503, statusText: "Unavailable" }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ ready: true }), { status: 200 }),
    );
  vi.stubGlobal("fetch", fetcher);
  const events: boolean[] = [];
  window.addEventListener("wealth:api-waking", (event) =>
    events.push((event as CustomEvent<{ waking: boolean }>).detail.waking),
  );

  const request = apiFetch<{ ready: boolean }>("/v1/networth");
  await vi.advanceTimersByTimeAsync(1_000);
  await expect(request).resolves.toEqual({ ready: true });
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect(events).toEqual([true, false]);
});

it("refreshes once after a 401 and retries the original request", async () => {
  setAuthSession({
    access_token: "synthetic-old-access",
    refresh_token: "synthetic-refresh-token",
    expires_in: 1_800,
  });
  const fetcher = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(new Response("", { status: 401 }))
    .mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          access_token: "synthetic-new-access",
          refresh_token: "synthetic-rotated-refresh",
          expires_in: 1_800,
          session_expires_at: "2030-01-01T00:00:00Z",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify({ ready: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
  vi.stubGlobal("fetch", fetcher);

  await expect(apiFetch<{ ready: boolean }>("/v1/me")).resolves.toEqual({
    ready: true,
  });
  expect(fetcher).toHaveBeenCalledTimes(3);
  expect(
    new Headers(fetcher.mock.calls[0]?.[1]?.headers).get("Authorization"),
  ).toBe("Bearer synthetic-old-access");
  expect(String(fetcher.mock.calls[1]?.[0])).toContain("/v1/auth/refresh");
  expect(
    new Headers(fetcher.mock.calls[2]?.[1]?.headers).get("Authorization"),
  ).toBe("Bearer synthetic-new-access");
});

it("shares one refresh across parallel 401 responses", async () => {
  setAuthSession({
    access_token: "synthetic-old-access",
    refresh_token: "synthetic-refresh-token",
    expires_in: 1_800,
  });
  let originalRequests = 0;
  let refreshRequests = 0;
  const fetcher = vi.fn<typeof fetch>(async (input) => {
    const url = new URL(String(input));
    if (url.pathname.endsWith("/v1/auth/refresh")) {
      refreshRequests += 1;
      return new Response(
        JSON.stringify({
          access_token: "synthetic-new-access",
          refresh_token: "synthetic-rotated-refresh",
          expires_in: 1_800,
          session_expires_at: "2030-01-01T00:00:00Z",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    originalRequests += 1;
    if (originalRequests <= 2) return new Response("", { status: 401 });
    return new Response(JSON.stringify({ ready: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetcher);

  const values = await Promise.all([
    apiFetch<{ ready: boolean }>("/v1/me"),
    apiFetch<{ ready: boolean }>("/v1/networth"),
  ]);

  expect(values).toEqual([{ ready: true }, { ready: true }]);
  expect(refreshRequests).toBe(1);
  expect(originalRequests).toBe(4);
});
