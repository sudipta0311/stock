import { act, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AuthProvider } from "../../src/auth/AuthContext";
import { AuthCallbackPage, LoginPage } from "../../src/app/AuthPages";
import { cancelProactiveRefresh } from "../../src/auth/session";
import {
  clearAuthToken,
  getRefreshToken,
  REFRESH_TOKEN_STORAGE_KEY,
} from "../../src/auth/token";
import {
  pollUntilApiAnswers,
  resetApiWarmupForTests,
} from "../../src/auth/warmup";

const warmup = vi.hoisted(() => ({ reachable: true }));
const oidcMocks = vi.hoisted(() => ({
  signinRedirectCallback: vi.fn(),
}));

vi.mock("../../src/auth/warmup", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/auth/warmup")>();
  return {
    ...actual,
    nudgeApi: vi.fn(),
    waitForApi: vi.fn(async () => warmup.reachable),
  };
});

vi.mock("../../src/auth/oidc", () => ({
  getOidcUserManager: async () => ({
    getUser: vi.fn().mockResolvedValue(null),
    removeUser: vi.fn().mockResolvedValue(undefined),
    signinRedirectCallback: oidcMocks.signinRedirectCallback,
  }),
}));

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  clearAuthToken();
  resetApiWarmupForTests();
  warmup.reachable = true;
  oidcMocks.signinRedirectCallback.mockReset();
  vi.stubEnv("VITE_AUTH_MODE", "oidc");
  vi.stubEnv("VITE_OIDC_AUTHORITY", "https://accounts.example.test");
  vi.stubEnv("VITE_OIDC_CLIENT_ID", "synthetic-client");
});

afterEach(() => {
  cancelProactiveRefresh();
  clearAuthToken();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("waits for a sleeping server and reports when it answers", async () => {
  vi.useFakeTimers();
  const waking: boolean[] = [];
  const onWaking = (event: Event) =>
    waking.push(
      (event as CustomEvent<{ waking: boolean }>).detail.waking === true,
    );
  window.addEventListener("wealth:api-waking", onWaking);
  let calls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async () => {
      calls += 1;
      if (calls < 3) throw new TypeError("Failed to fetch");
      return new Response("{}", { status: 200 });
    }),
  );

  const result = pollUntilApiAnswers();
  await vi.advanceTimersByTimeAsync(10_000);

  expect(await result).toBe(true);
  expect(calls).toBe(3);
  expect(waking).toEqual([true, false]);
  window.removeEventListener("wealth:api-waking", onWaking);
});

it("gives up on a server that never answers", async () => {
  vi.useFakeTimers();
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async () => new Response("", { status: 503 })),
  );

  const result = pollUntilApiAnswers();
  await vi.advanceTimersByTimeAsync(241_000);

  expect(await result).toBe(false);
});

it("offers Try again and keeps the saved sign-in when the server can't be reached", async () => {
  warmup.reachable = false;
  window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, "synthetic-refresh");
  const fetcher = vi.fn<typeof fetch>();
  vi.stubGlobal("fetch", fetcher);

  render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <LoginPage />
      </AuthProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByText(/We couldn't reach the server/),
  ).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Try again" })).toBeVisible();
  expect(fetcher).not.toHaveBeenCalled();
  expect(getRefreshToken()).toBe("synthetic-refresh");
});

it("quietly restores the saved sign-in after the API starts answering", async () => {
  warmup.reachable = false;
  window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, "synthetic-refresh");
  const paths: string[] = [];
  let retryCallback: (() => void) | undefined;
  const nativeSetTimeout = window.setTimeout.bind(window);
  vi.spyOn(window, "setTimeout").mockImplementation(((
    handler: TimerHandler,
    timeout?: number,
    ...args: unknown[]
  ) => {
    if (timeout === 20_000 && typeof handler === "function") {
      retryCallback = handler as () => void;
      return 42;
    }
    return nativeSetTimeout(handler, timeout, ...args);
  }) as typeof window.setTimeout);
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input) => {
      const path = new URL(String(input)).pathname;
      paths.push(path);
      if (path === "/v1/auth/refresh") {
        return new Response(
          JSON.stringify({
            access_token: "synthetic-access",
            refresh_token: "synthetic-rotated-refresh",
            expires_in: 1_800,
            session_expires_at: "2026-11-01T00:00:00Z",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        );
      }
      return new Response(JSON.stringify({ profile: {} }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );

  render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<p>Home restored</p>} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByText(/We couldn't reach the server/),
  ).toBeInTheDocument();
  expect(getRefreshToken()).toBe("synthetic-refresh");
  await waitFor(() => expect(retryCallback).toBeDefined());
  await act(async () => {
    retryCallback?.();
  });

  expect(await screen.findByText("Home restored")).toBeInTheDocument();
  expect(paths).toContain("/healthz");
  expect(paths).toContain("/v1/auth/refresh");
  expect(getRefreshToken()).toBe("synthetic-rotated-refresh");
});

it("doesn't spend the one-time Google code while the server is asleep", async () => {
  warmup.reachable = false;

  render(
    <MemoryRouter initialEntries={["/auth/callback"]}>
      <AuthProvider>
        <AuthCallbackPage />
      </AuthProvider>
    </MemoryRouter>,
  );

  expect(
    await screen.findByText(
      "We couldn't reach the server. Please sign in again in a minute.",
    ),
  ).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Back to sign in" })).toBeVisible();
  expect(oidcMocks.signinRedirectCallback).not.toHaveBeenCalled();
});
