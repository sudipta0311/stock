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
import { clearAuthToken, getAuthToken, setAuthToken } from "./token";
import { goToLogin } from "./navigation";
import { clearAllDeviceData } from "../pwa/offline-cache";

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
  completeSilentRenew(): Promise<void>;
  logout(options?: {
    disablePush?: boolean;
    notice?: boolean;
    preserveTourHistory?: boolean;
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
      return;
    }

    let active = true;
    void (async () => {
      let token = getAuthToken();
      if (
        mode === "oidc" &&
        import.meta.env.VITE_OIDC_AUTHORITY &&
        import.meta.env.VITE_OIDC_CLIENT_ID
      ) {
        try {
          const { getOidcUserManager } = await import("./oidc");
          const user = await (await getOidcUserManager()).getUser();
          token = user?.id_token ?? user?.access_token ?? token;
          if (token) setAuthToken(token);
        } catch (error) {
          if (active)
            setConfigurationError(
              error instanceof Error
                ? error.message
                : "Unable to configure sign-in.",
            );
        }
      }
      if (!active) return;
      if (!token) {
        setState("anonymous");
        return;
      }
      setState("authenticated");
      try {
        const cachedProfile = window.sessionStorage.getItem(PROFILE_STATUS_KEY);
        setHasProfile(cachedProfile === null ? null : cachedProfile === "true");
      } catch {
        setHasProfile(null);
      }
      await verifyProfile();
    })();
    return () => {
      active = false;
    };
  }, [mode, verifyProfile]);

  useEffect(() => {
    function expireSession() {
      if (!getAuthToken()) return;
      queryClient.clear();
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
      acceptToken(trimmed);
    },
    [acceptToken, mode],
  );

  const loginWithOidc = useCallback(async () => {
    const { getOidcUserManager } = await import("./oidc");
    await (await getOidcUserManager()).signinRedirect();
  }, []);

  const completeOidcLogin = useCallback(async () => {
    const { getOidcUserManager } = await import("./oidc");
    const user = await (await getOidcUserManager()).signinRedirectCallback();
    acceptToken(user.id_token ?? user.access_token);
  }, [acceptToken]);

  const completeSilentRenew = useCallback(async () => {
    const { getOidcUserManager } = await import("./oidc");
    await (await getOidcUserManager()).signinSilentCallback();
  }, []);

  const logout = useCallback(
    async (
      options: {
        disablePush?: boolean;
        notice?: boolean;
        preserveTourHistory?: boolean;
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
      if (mode === "oidc") {
        try {
          const { getOidcUserManager } = await import("./oidc");
          await (await getOidcUserManager()).removeUser();
        } catch {
          // The app session still clears when local identity storage is unavailable.
        }
      }
      queryClient.clear();
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
      completeSilentRenew,
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
      completeSilentRenew,
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
