const TOKEN_KEY = "wealth.auth.access-token";
let accessToken: string | null = null;

export function getAuthToken(): string | null {
  if (accessToken) return accessToken;
  try {
    accessToken = window.sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
  return accessToken;
}

export function setAuthToken(token: string | null): void {
  accessToken = token;
  try {
    if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
    else window.sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // The in-memory token keeps this session usable when browser storage is unavailable.
  }
}

export function clearAuthToken(): void {
  setAuthToken(null);
}
