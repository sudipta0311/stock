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
    ...(audience ? { extraQueryParams: { audience } } : {}),
  });
  return userManager;
}
