import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
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

const offlineQueue = vi.hoisted(() => ({
  writes: [] as unknown[],
  listQueuedAssetWrites: vi.fn(async () => offlineQueue.writes),
}));

const authSession = vi.hoisted(() => ({
  waitForSignInRenewal: vi.fn<() => Promise<void>>(async () => undefined),
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
vi.mock("../../src/money/offline-queue", () => ({
  listQueuedAssetWrites: offlineQueue.listQueuedAssetWrites,
}));
vi.mock("../../src/auth/session", () => ({
  waitForSignInRenewal: authSession.waitForSignInRenewal,
}));

const { PwaLifecycle } = await import("../../src/app/PwaLifecycle");

function setVisibility(state: DocumentVisibilityState): void {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => state,
  });
}

function renderLifecycle() {
  return render(
    <ToastProvider>
      <PwaLifecycle />
    </ToastProvider>,
  );
}

async function expectUpdateToast(): Promise<void> {
  await screen.findByRole("button", { name: "Tap to refresh" });
  expect(register.updateServiceWorker).not.toHaveBeenCalled();
}

afterEach(() => {
  register.needRefresh = false;
  register.options = undefined;
  offlineQueue.writes = [];
  register.updateServiceWorker.mockClear();
  offlineQueue.listQueuedAssetWrites.mockClear();
  authSession.waitForSignInRenewal.mockReset().mockResolvedValue(undefined);
  vi.useRealTimers();
  vi.restoreAllMocks();
  Reflect.deleteProperty(document, "visibilityState");
});

describe("app updates", () => {
  it("applies a waiting version after load without showing a toast", async () => {
    register.needRefresh = true;
    renderLifecycle();

    await waitFor(() =>
      expect(register.updateServiceWorker).toHaveBeenCalledWith(true),
    );
    expect(screen.queryByText("New version is ready")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Tap to refresh" }),
    ).not.toBeInTheDocument();
  });

  it("applies a waiting version shortly after returning to the app", async () => {
    const view = renderLifecycle();
    setVisibility("hidden");
    document.dispatchEvent(new Event("visibilitychange"));
    setVisibility("visible");
    document.dispatchEvent(new Event("visibilitychange"));

    register.needRefresh = true;
    view.rerender(
      <ToastProvider>
        <PwaLifecycle />
      </ToastProvider>,
    );

    await waitFor(() =>
      expect(register.updateServiceWorker).toHaveBeenCalledWith(true),
    );
    expect(screen.queryByText("New version is ready")).not.toBeInTheDocument();
  });

  it("waits for sign-in renewal before applying a waiting version", async () => {
    let finishRenewal!: () => void;
    authSession.waitForSignInRenewal.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishRenewal = resolve;
        }),
    );
    register.needRefresh = true;
    renderLifecycle();

    await waitFor(() =>
      expect(authSession.waitForSignInRenewal).toHaveBeenCalledOnce(),
    );
    expect(offlineQueue.listQueuedAssetWrites).not.toHaveBeenCalled();
    expect(register.updateServiceWorker).not.toHaveBeenCalled();

    await act(async () => finishRenewal());
    await waitFor(() =>
      expect(register.updateServiceWorker).toHaveBeenCalledWith(true),
    );
    expect(screen.queryByText("New version is ready")).not.toBeInTheDocument();
  });

  it("shows the toast when sign-in renewal finishes outside the safe window", async () => {
    let now = Date.now();
    vi.spyOn(Date, "now").mockImplementation(() => now);
    let finishRenewal!: () => void;
    authSession.waitForSignInRenewal.mockImplementationOnce(
      () =>
        new Promise<void>((resolve) => {
          finishRenewal = resolve;
        }),
    );
    register.needRefresh = true;
    renderLifecycle();

    await waitFor(() =>
      expect(authSession.waitForSignInRenewal).toHaveBeenCalledOnce(),
    );
    now += 15_001;
    await act(async () => finishRenewal());
    await expectUpdateToast();
  });

  it("waits for sign-in renewal after the person taps refresh", async () => {
    offlineQueue.writes = [{ key: "queued-save" }];
    let finishRenewal!: () => void;
    authSession.waitForSignInRenewal
      .mockResolvedValueOnce(undefined)
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishRenewal = resolve;
          }),
      );
    register.needRefresh = true;
    renderLifecycle();

    const refresh = await screen.findByRole("button", {
      name: "Tap to refresh",
    });
    fireEvent.click(refresh);
    await waitFor(() =>
      expect(authSession.waitForSignInRenewal).toHaveBeenCalledTimes(2),
    );
    expect(register.updateServiceWorker).not.toHaveBeenCalled();

    await act(async () => finishRenewal());
    await waitFor(() =>
      expect(register.updateServiceWorker).toHaveBeenCalledWith(true),
    );
  });

  it("shows the toast when a field changes before the queue check completes", async () => {
    let finishQueueCheck!: (writes: unknown[]) => void;
    offlineQueue.listQueuedAssetWrites.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finishQueueCheck = resolve;
        }),
    );
    register.needRefresh = true;
    const { container } = render(
      <>
        <input aria-label="Example field" />
        <ToastProvider>
          <PwaLifecycle />
        </ToastProvider>
      </>,
    );

    fireEvent.input(container.querySelector("input")!, {
      target: { value: "changed" },
    });
    await waitFor(() =>
      expect(offlineQueue.listQueuedAssetWrites).toHaveBeenCalledOnce(),
    );
    await act(async () => finishQueueCheck([]));
    await expectUpdateToast();
  });

  it("shows the toast while a dialog is open", async () => {
    register.needRefresh = true;
    render(
      <>
        <div role="dialog" aria-modal="true">
          Dialog content
        </div>
        <ToastProvider>
          <PwaLifecycle />
        </ToastProvider>
      </>,
    );

    await expectUpdateToast();
  });

  it("shows the toast when an offline save is queued", async () => {
    offlineQueue.writes = [{ key: "queued-save" }];
    register.needRefresh = true;
    renderLifecycle();

    await expectUpdateToast();
  });

  it("shows the toast when the update arrives more than 15 seconds later", async () => {
    let now = Date.now();
    vi.spyOn(Date, "now").mockImplementation(() => now);
    const view = renderLifecycle();
    now += 15_001;
    register.needRefresh = true;
    view.rerender(
      <ToastProvider>
        <PwaLifecycle />
      </ToastProvider>,
    );

    await expectUpdateToast();
  });

  it("keeps the toast until the person taps refresh", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    offlineQueue.writes = [{ key: "queued-save" }];
    register.needRefresh = true;
    renderLifecycle();

    const refresh = await screen.findByRole("button", {
      name: "Tap to refresh",
    });
    await act(async () => {
      vi.advanceTimersByTime(60_000);
    });
    expect(screen.getByText("New version is ready")).toBeInTheDocument();

    fireEvent.click(refresh);
    await waitFor(() =>
      expect(register.updateServiceWorker).toHaveBeenCalledWith(true),
    );
  });

  it("looks for a new version when the person comes back to the app", () => {
    renderLifecycle();
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
