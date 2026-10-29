import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import type { components } from "../api/schema";
import { apiFetch } from "../api/client";
import { queryClient } from "../api/query-client";
import {
  clearAuthToken,
  getAuthToken,
  getRefreshToken,
  setAuthSession,
  setAuthToken,
} from "./token";
import { claimDeviceFor } from "./device-account";
import { goToLogin } from "./navigation";
import { clearAllDeviceData } from "../pwa/offline-cache";
import {
  beginAuthBootstrap,
  cancelProactiveRefresh,
  completeAuthBootstrap,
  refreshAuthSession,
  scheduleProactiveRefresh,
} from "./session";

type AuthMode = "oidc" | "dev";
type AuthState = "loading" | "anonymous" | "authenticated";
type MeResponse = components["schemas"]["MeResponse"];

interface AuthContextValue {
  mode: AuthMode;
  state: AuthState;
  hasProfile: boolean | null;
  configurationError: string | null;
  refreshProfile(): Promise<void>;
  loginWithDevToken(token: string): void;
  loginWithOidc(): Promise<void>;
  completeOidcLogin(): Promise<void>;
  logout(options?: {
    disablePush?: boolean;
    notice?: boolean;
    preserveTourHistory?: boolean;
    serverRevoked?: boolean;
  }): Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const PROFILE_STATUS_KEY = "wealth.auth.has-profile";
export const AUTH_RETURN_PATH_KEY = "wealth.auth.return-path";
export const AUTH_NOTICE_STORAGE_KEY = "wealth.auth.notice";

function configuredAuthMode(): AuthMode {
  return import.meta.env.VITE_AUTH_MODE === "dev" ? "dev" : "oidc";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  beginAuthBootstrap();
  const mode = configuredAuthMode();
  const [state, setState] = useState<AuthState>("loading");
  const [hasProfile, setHasProfile] = useState<boolean | null>(null);
  const [configurationError, setConfigurationError] = useState<string | null>(
    null,
  );

  const verifyProfile = useCallback(async () => {
    try {
      const me = await apiFetch<MeResponse>("/v1/me");
      const ready = me.profile !== null;
      setHasProfile(ready);
      try {
        window.sessionStorage.setItem(PROFILE_STATUS_KEY, String(ready));
      } catch {
        // Profile state is fetched again next time when session storage is unavailable.
      }
    } catch (error) {
      if (error instanceof Error && "status" in error && error.status === 401) {
        cancelProactiveRefresh();
        clearAuthToken();
        setHasProfile(null);
        setState("anonymous");
        return;
      }
      setHasProfile(true);
    }
  }, []);

  const acceptToken = useCallback(
    (token: string) => {
      setAuthToken(token);
      setState("authenticated");
      setHasProfile(null);
      void verifyProfile();
    },
    [verifyProfile],
  );

  useEffect(() => {
    if (mode === "dev" && import.meta.env.PROD) {
      setConfigurationError(
        "Development token sign-in is disabled in production builds.",
      );
      setState("anonymous");
      completeAuthBootstrap();
      return;
    }

    let active = true;
    void (async () => {
      let token = mode === "dev" ? getAuthToken() : null;
      if (mode === "oidc") {
        if (getRefreshToken()) {
          try {
            token = (await refreshAuthSession())?.access_token ?? null;
          } catch {
            // Keep the refresh token for the next attempt when the API is unavailable.
          }
        }
        if (
          import.meta.env.VITE_OIDC_AUTHORITY &&
          import.meta.env.VITE_OIDC_CLIENT_ID
        ) {
          try {
            const { getOidcUserManager } = await import("./oidc");
            const manager = await getOidcUserManager();
            const user = await manager.getUser();
            if (!token) {
              token = user?.id_token ?? user?.access_token ?? null;
              if (token) setAuthToken(token);
            }
            if (user) await manager.removeUser();
          } catch (error) {
            if (active)
              setConfigurationError(
                error instanceof Error
                  ? error.message
                  : "Unable to configure sign-in.",
              );
          }
        }
      }
      if (token) await claimDeviceFor(token);
      if (!active) return;
      completeAuthBootstrap();
      if (!token) {
        setState("anonymous");
        return;
      }
      setAuthToken(token);
      scheduleProactiveRefresh();
      setState("authenticated");
      try {
        const cachedProfile = window.sessionStorage.getItem(PROFILE_STATUS_KEY);
        setHasProfile(cachedProfile === null ? null : cachedProfile === "true");
      } catch {
        setHasProfile(null);
      }
      await verifyProfile();
    })().finally(() => completeAuthBootstrap());
    return () => {
      active = false;
    };
  }, [mode, verifyProfile]);

  useEffect(() => {
    function expireSession() {
      queryClient.clear();
      cancelProactiveRefresh();
      clearAuthToken();
      setHasProfile(null);
      setState("anonymous");
      try {
        const path = `${window.location.pathname}${window.location.search}${window.location.hash}`;
        window.sessionStorage.clear();
        if (path.startsWith("/") && !path.startsWith("//")) {
          window.sessionStorage.setItem(AUTH_RETURN_PATH_KEY, path);
        }
        window.sessionStorage.setItem(AUTH_NOTICE_STORAGE_KEY, "expired");
      } catch {
        // Sign-in still continues when browser storage is blocked.
      }
    }
    window.addEventListener("wealth:auth-expired", expireSession);
    return () =>
      window.removeEventListener("wealth:auth-expired", expireSession);
  }, []);

  const loginWithDevToken = useCallback(
    (token: string) => {
      if (mode !== "dev" || import.meta.env.PROD) {
        setConfigurationError(
          "Development token sign-in is only available in local development.",
        );
        return;
      }
      const trimmed = token.trim();
      if (!trimmed) {
        setConfigurationError(
          "Paste a development token generated by the API's local instructions.",
        );
        return;
      }
      setConfigurationError(null);
      void claimDeviceFor(trimmed).then(() => acceptToken(trimmed));
    },
    [acceptToken, mode],
  );

  const loginWithOidc = useCallback(async () => {
    const { getOidcUserManager } = await import("./oidc");
    await (await getOidcUserManager()).signinRedirect();
  }, []);

  const completeOidcLogin = useCallback(async () => {
    const { getOidcUserManager } = await import("./oidc");
    const manager = await getOidcUserManager();
    const user = await manager.signinRedirectCallback();
    const token = user.access_token || user.id_token;
    if (!token) throw new Error("Google sign-in returned no app token.");
    await claimDeviceFor(token);
    if (user.refresh_token) {
      setAuthSession({
        access_token: token,
        refresh_token: user.refresh_token,
        expires_in:
          user.expires_in ??
          (user.expires_at
            ? Math.max(1, user.expires_at - Math.floor(Date.now() / 1_000))
            : 30 * 60),
      });
      scheduleProactiveRefresh();
    } else {
      setAuthToken(token);
    }
    await manager.removeUser();
    setState("authenticated");
    setHasProfile(null);
    await verifyProfile();
  }, [verifyProfile]);

  const logout = useCallback(
    async (
      options: {
        disablePush?: boolean;
        notice?: boolean;
        preserveTourHistory?: boolean;
        serverRevoked?: boolean;
      } = {},
    ) => {
      if (options.disablePush) {
        try {
          const { disablePush } = await import("../pwa/push");
          await disablePush();
        } catch {
          // Signing out still works if this phone cannot reach the push service.
        }
      }
      const currentToken = getAuthToken();
      if (!options.serverRevoked && currentToken) {
        try {
          const apiBase =
            import.meta.env.VITE_API_URL || window.location.origin;
          await fetch(new URL("/v1/auth/logout", apiBase), {
            method: "POST",
            headers: { Authorization: `Bearer ${currentToken}` },
            cache: "no-store",
          });
        } catch {
          // Local sign-out still completes when the phone cannot reach the API.
        }
      }
      if (mode === "oidc") {
        try {
          const { getOidcUserManager } = await import("./oidc");
          await (await getOidcUserManager()).removeUser();
        } catch {
          // The app session still clears when local identity storage is unavailable.
        }
      }
      queryClient.clear();
      cancelProactiveRefresh();
      clearAuthToken();
      setState("anonymous");
      setHasProfile(null);
      try {
        await clearAllDeviceData({
          preservePushSubscription: options.disablePush !== true,
          preserveTourHistory: options.preserveTourHistory !== false,
        });
      } catch {
        // Sign-out continues when offline data cannot be cleared.
      }
      try {
        window.sessionStorage.clear();
        if (options.notice !== false) {
          window.sessionStorage.setItem(AUTH_NOTICE_STORAGE_KEY, "signed-out");
        }
      } catch {
        // Logout continues even when browser storage is blocked.
      }
      goToLogin();
    },
    [mode],
  );

  const value = useMemo(
    () => ({
      mode,
      state,
      hasProfile,
      configurationError,
      refreshProfile: verifyProfile,
      loginWithDevToken,
      loginWithOidc,
      completeOidcLogin,
      logout,
    }),
    [
      mode,
      state,
      hasProfile,
      configurationError,
      verifyProfile,
      loginWithDevToken,
      loginWithOidc,
      completeOidcLogin,
      logout,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider.");
  return context;
}
