import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import englishCopy from "../../src/copy/en.json";
import { OfflineStatus } from "../../src/app/OfflineStatus";
import { WakeIndicator } from "../../src/app/WakeIndicator";

vi.mock("../../src/pwa/offline-cache", () => ({
  getOfflineSnapshotTime: vi.fn().mockResolvedValue("28 Oct 2026, 9:00 am"),
}));
vi.mock("../../src/money/offline-queue", () => ({
  listQueuedAssetWrites: vi
    .fn()
    .mockResolvedValue([
      { key: "asset:synthetic", draftKey: "draft-test", assetId: "asset-test" },
    ]),
}));

afterEach(() => {
  vi.useRealTimers();
  Object.defineProperty(navigator, "onLine", {
    configurable: true,
    value: true,
  });
});

describe("offline and cold-start status", () => {
  it("shows the offline snapshot time and pending-change count", async () => {
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    render(<OfflineStatus />);

    expect(
      await screen.findByText(
        new RegExp(
          englishCopy.offline.snapshot_time.replace(
            "{time}",
            "28 Oct 2026, 9:00 am",
          ),
        ),
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByRole("status")[0]).toHaveTextContent(
      englishCopy.offline.banner,
    );
    expect(
      screen.getByText(englishCopy.offline.pending.replace("{count}", "1")),
    ).toBeInTheDocument();
  });

  it("shows the longer cold-start message after twenty seconds", async () => {
    vi.useFakeTimers();
    render(<WakeIndicator />);
    act(() =>
      window.dispatchEvent(
        new CustomEvent("wealth:api-waking", { detail: { waking: true } }),
      ),
    );
    expect(
      screen.getByText(englishCopy.offline.short_wake),
    ).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });
    expect(screen.getByText(englishCopy.offline.long_wake)).toBeInTheDocument();
  });
});
