import { render, waitFor } from "@testing-library/react";
import { act } from "react";
import { HttpResponse, http } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { OfflineWriteSync } from "../../src/money/OfflineWriteSync";
import {
  clearOfflineAssetWrites,
  listQueuedAssetWrites,
  queueAssetWrite,
} from "../../src/money/offline-queue";
import { server } from "../msw";

const onlineDescriptor = Object.getOwnPropertyDescriptor(navigator, "onLine");

afterEach(async () => {
  await clearOfflineAssetWrites();
  if (onlineDescriptor)
    Object.defineProperty(navigator, "onLine", onlineDescriptor);
});

describe("offline asset queue", () => {
  it("keeps the last write per draft and replays it on the online event", async () => {
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    const bodies: Array<Record<string, unknown>> = [];
    server.use(
      http.post(/\/v1\/assets$/, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        bodies.push(body);
        return HttpResponse.json(
          {
            id: "asset-offline",
            ...body,
            member_id: null,
            account_id: null,
            details: body,
            valuation: null,
            valuation_note: null,
            valuation_status: "valued",
            value_cached: "600000.00",
            value_as_of: "2026-10-28",
            status: "active",
            created_at: "2026-10-28T00:00:00Z",
            updated_at: "2026-10-28T00:00:00Z",
          },
          { status: 201 },
        );
      }),
    );
    await queueAssetWrite("draft-fd", null, {
      asset_type: "savings",
      bank: "Demo Bank",
      balance: "500000.00",
    });
    await queueAssetWrite("draft-fd", null, {
      asset_type: "savings",
      bank: "Demo Bank",
      balance: "600000.00",
    });
    expect(await listQueuedAssetWrites()).toHaveLength(1);

    const saved = vi.fn();
    window.addEventListener("wealth:offline-asset-saved", saved);
    render(<OfflineWriteSync />);
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: true,
    });
    await act(async () => {
      window.dispatchEvent(new Event("online"));
      await Promise.resolve();
    });

    await waitFor(async () =>
      expect(await listQueuedAssetWrites()).toHaveLength(0),
    );
    expect(bodies).toHaveLength(1);
    expect(bodies[0]).toMatchObject({ balance: "600000.00" });
    expect(saved).toHaveBeenCalledTimes(1);
    window.removeEventListener("wealth:offline-asset-saved", saved);
  });
});
