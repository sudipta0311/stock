import { afterEach, expect, it, vi } from "vitest";
import { ApiError, apiFetch } from "../../src/api/client";
import { cancelProactiveRefresh } from "../../src/auth/session";
import { clearAuthToken, setAuthSession } from "../../src/auth/token";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  cancelProactiveRefresh();
  clearAuthToken();
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
