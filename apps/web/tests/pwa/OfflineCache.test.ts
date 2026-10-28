import { afterEach, describe, expect, it, vi } from "vitest";
import { apiFetch } from "../../src/api/client";
import {
  getOfflineSnapshotTime,
  storeOfflineSnapshot,
} from "../../src/pwa/offline-cache";

function memoryCacheStorage() {
  const values = new Map<string, Response>();
  const cache = {
    put: vi.fn(async (request: RequestInfo | URL, response: Response) => {
      const key =
        typeof request === "string"
          ? request
          : request instanceof URL
            ? request.href
            : request.url;
      values.set(key, response.clone());
    }),
    match: vi.fn(async (request: RequestInfo | URL) => {
      const key =
        typeof request === "string"
          ? request
          : request instanceof URL
            ? request.href
            : request.url;
      return values.get(key)?.clone();
    }),
    keys: vi.fn(async () => [...values.keys()].map((key) => new Request(key))),
  };
  return { cache, storage: { open: vi.fn().mockResolvedValue(cache) } };
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
});

describe("offline snapshots", () => {
  it("serves a cached read-only snapshot after a network failure", async () => {
    const { cache, storage } = memoryCacheStorage();
    vi.stubGlobal("caches", storage);
    const url = new URL("/v1/networth", window.location.origin).href;
    await storeOfflineSnapshot(
      url,
      new Response(
        JSON.stringify({ as_of: "2026-10-27", net_worth: "123.00" }),
        {
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("offline")));

    await expect(
      apiFetch<{ net_worth: string }>("/v1/networth"),
    ).resolves.toEqual({ as_of: "2026-10-27", net_worth: "123.00" });
    expect(cache.match).toHaveBeenCalledWith(url);
  });

  it("labels the snapshot with its data date instead of the response date", async () => {
    const { storage } = memoryCacheStorage();
    vi.stubGlobal("caches", storage);
    await storeOfflineSnapshot(
      new URL("/v1/plan", window.location.origin).href,
      new Response(JSON.stringify({ as_of: "2026-10-27" }), {
        headers: {
          "Content-Type": "application/json",
          Date: "Wed, 28 Oct 2026 08:00:00 GMT",
        },
      }),
    );

    await expect(getOfflineSnapshotTime()).resolves.toContain("27 Oct 2026");
  });
});
