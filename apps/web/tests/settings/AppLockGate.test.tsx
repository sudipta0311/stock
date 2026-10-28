import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import englishCopy from "../../src/copy/en.json";
import { AppLockGate } from "../../src/app/AppLockGate";

const { unlockMock, initialLockMock } = vi.hoisted(() => ({
  unlockMock: vi.fn().mockResolvedValue(true),
  initialLockMock: vi.fn().mockReturnValue(false),
}));

vi.mock("../../src/auth/AuthContext", () => ({
  useAuth: () => ({ mode: "oidc", loginWithOidc: vi.fn() }),
}));
vi.mock("../../src/pwa/app-lock", () => ({
  APP_LOCK_BACKGROUND_MS: 300_000,
  appLockEnabled: () => true,
  appLockSupported: () => true,
  clearBackgroundTime: vi.fn(),
  rememberBackgroundTime: vi.fn(),
  shouldLockAfterBackground: initialLockMock,
  unlockDeviceLock: unlockMock,
}));

beforeEach(() => {
  unlockMock.mockReset();
  unlockMock.mockResolvedValue(true);
  initialLockMock.mockReset();
  initialLockMock.mockReturnValue(false);
});

afterEach(() => {
  vi.useRealTimers();
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: "visible",
  });
});

describe("device lock", () => {
  it("locks after five minutes in the background and hides page content until unlocked", async () => {
    vi.useFakeTimers();
    render(
      <AppLockGate>
        <p>Saved balance</p>
      </AppLockGate>,
    );
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    fireEvent(document, new Event("visibilitychange"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(300_000);
    });

    const dialog = screen.getByRole("dialog", {
      name: englishCopy.settings.privacy.lock_locked,
    });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText("Saved balance").parentElement).toHaveAttribute(
      "aria-hidden",
      "true",
    );
    expect(screen.getByText("Saved balance").parentElement).toHaveClass(
      "blur-lg",
    );
    fireEvent.click(
      screen.getByRole("button", { name: englishCopy.settings.privacy.unlock }),
    );
    await act(async () => {});
    expect(unlockMock).toHaveBeenCalledOnce();
    expect(
      screen.queryByRole("dialog", {
        name: englishCopy.settings.privacy.lock_locked,
      }),
    ).not.toBeInTheDocument();
  });

  it("offers the device unlock action after a background interval", async () => {
    initialLockMock.mockReturnValueOnce(true);
    const user = userEvent.setup();
    render(
      <AppLockGate>
        <p>Saved balance</p>
      </AppLockGate>,
    );
    expect(
      screen.getByRole("dialog", {
        name: englishCopy.settings.privacy.lock_locked,
      }),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: englishCopy.settings.privacy.unlock }),
    );
    expect(await screen.findByText("Saved balance")).toBeInTheDocument();
    expect(unlockMock).toHaveBeenCalledOnce();
  });
});
