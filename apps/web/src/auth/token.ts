export const REFRESH_TOKEN_STORAGE_KEY = "wealth.auth.refresh-token";

let accessToken: string | null = null;
let refreshToken: string | null = null;
let accessTokenExpiresAt: number | null = null;

export interface AuthSessionTokens {
  access_token: string;
  refresh_token: string;
  expires_in: number;
}

export function getAuthToken(): string | null {
  return accessToken;
}

export function getRefreshToken(): string | null {
  try {
    const stored = window.localStorage.getItem(REFRESH_TOKEN_STORAGE_KEY);
    if (stored !== null) refreshToken = stored;
  } catch {
    // The in-memory token keeps this tab signed in when browser storage is blocked.
  }
  return refreshToken;
}

export function getAuthTokenExpiresAt(): number | null {
  return accessTokenExpiresAt;
}

export function setRefreshToken(token: string | null): void {
  refreshToken = token;
  try {
    if (token) window.localStorage.setItem(REFRESH_TOKEN_STORAGE_KEY, token);
    else window.localStorage.removeItem(REFRESH_TOKEN_STORAGE_KEY);
  } catch {
    // The in-memory token keeps this tab signed in when browser storage is blocked.
  }
}

export function setAuthToken(token: string | null): void {
  accessToken = token;
  if (token === null) accessTokenExpiresAt = null;
}

export function setAuthSession(tokens: AuthSessionTokens): void {
  accessToken = tokens.access_token;
  accessTokenExpiresAt = Date.now() + tokens.expires_in * 1_000;
  setRefreshToken(tokens.refresh_token);
}

export function clearAuthToken(): void {
  accessToken = null;
  accessTokenExpiresAt = null;
  setRefreshToken(null);
}
