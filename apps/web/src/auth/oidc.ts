import type { UserManager } from "oidc-client-ts";

let userManager: UserManager | undefined;

export async function getOidcUserManager(): Promise<UserManager> {
  if (userManager) return userManager;
  const { UserManager, WebStorageStateStore } = await import("oidc-client-ts");
  const authority = import.meta.env.VITE_OIDC_AUTHORITY;
  const clientId = import.meta.env.VITE_OIDC_CLIENT_ID;
  if (!authority || !clientId) {
    throw new Error(
      "Set VITE_OIDC_AUTHORITY and VITE_OIDC_CLIENT_ID to enable sign-in.",
    );
  }
  const audience = import.meta.env.VITE_OIDC_AUDIENCE;
  if (audience && audience !== clientId) {
    throw new Error(
      "VITE_OIDC_AUDIENCE must match VITE_OIDC_CLIENT_ID for Google sign-in.",
    );
  }
  const apiUrl = import.meta.env.VITE_API_URL || window.location.origin;
  userManager = new UserManager({
    authority,
    client_id: clientId,
    redirect_uri: `${window.location.origin}/auth/callback`,
    silent_redirect_uri: `${window.location.origin}/auth/callback/silent`,
    post_logout_redirect_uri: `${window.location.origin}/login`,
    response_type: "code",
    scope: import.meta.env.VITE_OIDC_SCOPE || "openid email profile",
    automaticSilentRenew: true,
    userStore: new WebStorageStateStore({ store: window.sessionStorage }),
    metadataSeed: {
      token_endpoint: new URL("/v1/auth/google/exchange", apiUrl).toString(),
    },
  });
  return userManager;
}
