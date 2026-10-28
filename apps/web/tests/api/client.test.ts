import { afterEach, expect, it, vi } from "vitest";
import { ApiError, apiFetch } from "../../src/api/client";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
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
