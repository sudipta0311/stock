import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToastProvider } from "../../src/ui";

const register = vi.hoisted(() => ({
  needRefresh: false,
  updateServiceWorker: vi.fn(async () => undefined),
  options: undefined as
    | {
        onRegisteredSW?: (
          url: string,
          registration: ServiceWorkerRegistration | undefined,
        ) => void;
      }
    | undefined,
}));

vi.mock("virtual:pwa-register/react", () => ({
  useRegisterSW: (options: typeof register.options) => {
    register.options = options;
    return {
      needRefresh: [register.needRefresh, vi.fn()],
      offlineReady: [false, vi.fn()],
      updateServiceWorker: register.updateServiceWorker,
    };
  },
}));
vi.mock("../../src/pwa/offline-cache", () => ({
  syncOfflineCacheSetting: vi.fn(),
}));

const { PwaLifecycle } = await import("../../src/app/PwaLifecycle");

function setVisibility(state: DocumentVisibilityState) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

afterEach(() => {
  register.needRefresh = false;
  register.options = undefined;
  Reflect.deleteProperty(document, "visibilityState");
});

describe("app updates", () => {
  it("keeps the update notice open until the person taps refresh", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    register.needRefresh = true;
    render(
      <ToastProvider>
        <PwaLifecycle />
      </ToastProvider>,
    );

    const refresh = await screen.findByRole("button", {
      name: "Tap to refresh",
    });
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(screen.getByText("New version is ready")).toBeInTheDocument();
    vi.useRealTimers();

    fireEvent.click(refresh);
    expect(register.updateServiceWorker).toHaveBeenCalledWith(true);
  });

  it("looks for a new version when the person comes back to the app", () => {
    render(
      <ToastProvider>
        <PwaLifecycle />
      </ToastProvider>,
    );
    const update = vi.fn(async () => undefined);
    register.options?.onRegisteredSW?.("/sw.js", {
      update,
    } as unknown as ServiceWorkerRegistration);

    setVisibility("hidden");
    document.dispatchEvent(new Event("visibilitychange"));
    expect(update).not.toHaveBeenCalled();

    setVisibility("visible");
    document.dispatchEvent(new Event("visibilitychange"));
    expect(update).toHaveBeenCalledTimes(1);
  });
});
