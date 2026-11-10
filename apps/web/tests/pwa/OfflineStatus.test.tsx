import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import englishCopy from "../../src/copy/en.json";
import { OfflineStatus } from "../../src/app/OfflineStatus";
import { WakeIndicator } from "../../src/app/WakeIndicator";
import { listQueuedAssetWrites } from "../../src/money/offline-queue";

vi.mock("../../src/pwa/offline-cache", () => ({
  getOfflineSnapshotTime: vi.fn().mockResolvedValue("28 Oct 2026, 9:00 am"),
}));
vi.mock("../../src/money/offline-queue", () => ({
  listQueuedAssetWrites: vi.fn().mockResolvedValue([
    {
      key: "asset:synthetic",
      draftKey: "draft-test",
      assetId: "asset-test",
      body: {},
      updatedAt: 1,
    },
  ]),
}));

afterEach(() => {
  vi.useRealTimers();
  vi.mocked(listQueuedAssetWrites)
    .mockReset()
    .mockResolvedValue([
      {
        key: "asset:synthetic",
        draftKey: "draft-test",
        assetId: "asset-test",
        body: {},
        updatedAt: 1,
      },
    ]);
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
    render(
      <MemoryRouter>
        <OfflineStatus />
      </MemoryRouter>,
    );

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
    expect(screen.getAllByRole("status")).toHaveLength(1);
    expect(screen.getByRole("status")).toHaveTextContent(
      englishCopy.offline.banner,
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      englishCopy.offline.pending.replace("{count}", "1"),
    );
  });

  it("refreshes the offline banner after navigation", async () => {
    const user = userEvent.setup();
    let reads = 0;
    vi.mocked(listQueuedAssetWrites)
      .mockReset()
      .mockImplementation(async () => {
        reads += 1;
        return reads === 1
          ? []
          : [
              {
                key: "asset:synthetic",
                draftKey: "draft-test",
                assetId: "asset-test",
                body: {},
                updatedAt: 1,
              },
            ];
      });
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });

    render(
      <MemoryRouter initialEntries={["/edit"]}>
        <Routes>
          <Route
            path="*"
            element={
              <>
                <OfflineStatus />
                <Link to="/money">Money</Link>
              </>
            }
          />
        </Routes>
      </MemoryRouter>,
    );
    await user.click(screen.getByRole("link", { name: "Money" }));

    await expect
      .poll(() => screen.getByRole("status").textContent)
      .toContain(englishCopy.offline.pending.replace("{count}", "1"));
  });

  it("keeps the wake state to a small updating chip", async () => {
    vi.useFakeTimers();
    render(<WakeIndicator />);
    act(() =>
      window.dispatchEvent(
        new CustomEvent("wealth:api-waking", { detail: { waking: true } }),
      ),
    );
    expect(screen.getByText(englishCopy.offline.updating)).toBeInTheDocument();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });
    expect(screen.getByText(englishCopy.offline.updating)).toBeInTheDocument();
    expect(
      screen.queryByText(englishCopy.offline.long_wake),
    ).not.toBeInTheDocument();
  });
});
