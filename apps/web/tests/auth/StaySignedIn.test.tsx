import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "../../src/auth/AuthContext";
import { cancelProactiveRefresh } from "../../src/auth/session";
import {
  clearAuthToken,
  getAuthToken,
  getRefreshToken,
  REFRESH_TOKEN_STORAGE_KEY,
  setAuthSession,
} from "../../src/auth/token";

const oidcMocks = vi.hoisted(() => ({
  getUser: vi.fn().mockResolvedValue(null),
  removeUser: vi.fn().mockResolvedValue(undefined),
}));

vi.mock("../../src/auth/oidc", () => ({
  getOidcUserManager: async () => ({
    getUser: oidcMocks.getUser,
    removeUser: oidcMocks.removeUser,
  }),
}));

function response(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function AuthStatus() {
  const { state, hasProfile } = useAuth();
  return <p>{`${state}:${hasProfile}`}</p>;
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  clearAuthToken();
  oidcMocks.getUser.mockReset().mockResolvedValue(null);
  oidcMocks.removeUser.mockReset().mockResolvedValue(undefined);
  vi.stubEnv("VITE_AUTH_MODE", "oidc");
  vi.stubEnv("VITE_OIDC_AUTHORITY", "https://accounts.example.test");
  vi.stubEnv("VITE_OIDC_CLIENT_ID", "synthetic-client");
});

afterEach(() => {
  cancelProactiveRefresh();
  clearAuthToken();
  vi.unstubAllGlobals();
});

it("keeps only the refresh token in local storage", () => {
  setAuthSession({
    access_token: "synthetic-app-access",
    refresh_token: "synthetic-app-refresh",
    expires_in: 1_800,
  });

  expect(getAuthToken()).toBe("synthetic-app-access");
  expect(getRefreshToken()).toBe("synthetic-app-refresh");
  expect(window.localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY)).toBe(
    "synthetic-app-refresh",
  );
  expect(window.localStorage.getItem("wealth.auth.access-token")).toBeNull();
  expect(window.sessionStorage.getItem("wealth.auth.access-token")).toBeNull();
});

it("restores the app session before the first protected API call", async () => {
  window.localStorage.setItem(
    REFRESH_TOKEN_STORAGE_KEY,
    "synthetic-old-refresh",
  );
  const fetcher = vi.fn<typeof fetch>(async (input, init) => {
    const url = new URL(String(input));
    if (url.pathname === "/v1/auth/refresh") {
      return response({
        access_token: "synthetic-restored-access",
        refresh_token: "synthetic-new-refresh",
        expires_in: 1_800,
        session_expires_at: "2030-01-01T00:00:00Z",
      });
    }
    expect(url.pathname).toBe("/v1/me");
    expect(new Headers(init?.headers).get("Authorization")).toBe(
      "Bearer synthetic-restored-access",
    );
    return response({ profile: null });
  });
  vi.stubGlobal("fetch", fetcher);

  render(
    <AuthProvider>
      <AuthStatus />
    </AuthProvider>,
  );

  expect(await screen.findByText("authenticated:false")).toBeInTheDocument();
  expect(fetcher).toHaveBeenCalledTimes(2);
  expect(new URL(String(fetcher.mock.calls[0]?.[0])).pathname).toBe(
    "/v1/auth/refresh",
  );
  expect(getAuthToken()).toBe("synthetic-restored-access");
  expect(getRefreshToken()).toBe("synthetic-new-refresh");
  expect(window.sessionStorage.getItem("wealth.auth.access-token")).toBeNull();
});

it("falls back to memory when browser storage is blocked", () => {
  const originalDescriptor = Object.getOwnPropertyDescriptor(
    window,
    "localStorage",
  );
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: {
      getItem: () => {
        throw new DOMException("Storage is blocked.", "SecurityError");
      },
      setItem: () => {
        throw new DOMException("Storage is blocked.", "SecurityError");
      },
      removeItem: () => {
        throw new DOMException("Storage is blocked.", "SecurityError");
      },
    },
  });

  setAuthSession({
    access_token: "synthetic-memory-access",
    refresh_token: "synthetic-memory-refresh",
    expires_in: 1_800,
  });
  expect(getAuthToken()).toBe("synthetic-memory-access");
  expect(getRefreshToken()).toBe("synthetic-memory-refresh");

  if (originalDescriptor) {
    Object.defineProperty(window, "localStorage", originalDescriptor);
  }
});

it("uses the newer token another tab saved when a refresh was just renewed", async () => {
  const { refreshAuthSession } = await import("../../src/auth/session");
  window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, "synthetic-stale");
  const presented: string[] = [];
  const fetcher = vi.fn<typeof fetch>(async (_input, init) => {
    const token = (JSON.parse(String(init?.body)) as { refresh_token: string })
      .refresh_token;
    presented.push(token);
    if (token === "synthetic-stale") {
      // Another tab rotated the token a moment ago and stored the new one.
      window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, "synthetic-fresh");
      return new Response(
        JSON.stringify({ detail: "The sign-in session was just renewed." }),
        { status: 409, headers: { "Content-Type": "application/json" } },
      );
    }
    return response({
      access_token: "synthetic-access",
      refresh_token: "synthetic-next",
      expires_in: 1_800,
      session_expires_at: "2030-01-01T00:00:00Z",
    });
  });
  vi.stubGlobal("fetch", fetcher);

  const result = await refreshAuthSession();

  expect(presented).toEqual(["synthetic-stale", "synthetic-fresh"]);
  expect(result?.access_token).toBe("synthetic-access");
  expect(getRefreshToken()).toBe("synthetic-next");
});
