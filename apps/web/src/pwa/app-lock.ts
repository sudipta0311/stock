const ENABLED_KEY = "wealth.security.app-lock.enabled";
const CREDENTIAL_KEY = "wealth.security.app-lock.credential-id";
const HIDDEN_AT_KEY = "wealth.security.app-lock.hidden-at";

export const APP_LOCK_BACKGROUND_MS = 5 * 60 * 1000;

function randomBytes(length: number): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(length));
  crypto.getRandomValues(bytes);
  return bytes;
}

function decodeCredentialId(value: string): Uint8Array<ArrayBuffer> {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  const base64 = (value + padding).replaceAll("-", "+").replaceAll("_", "/");
  const decoded = atob(base64);
  const bytes = new Uint8Array(new ArrayBuffer(decoded.length));
  for (let index = 0; index < decoded.length; index += 1) {
    bytes[index] = decoded.charCodeAt(index);
  }
  return bytes;
}

export function appLockEnabled(): boolean {
  try {
    return window.localStorage.getItem(ENABLED_KEY) === "true";
  } catch {
    return false;
  }
}

export function appLockSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "credentials" in navigator &&
    typeof PublicKeyCredential !== "undefined"
  );
}

export async function createDeviceLockCredential(): Promise<void> {
  if (!appLockSupported()) throw new Error("device_lock_unsupported");
  const credential = await navigator.credentials.create({
    publicKey: {
      challenge: randomBytes(32),
      rp: { name: "Wealth Copilot" },
      user: {
        id: randomBytes(16),
        name: "wealth-copilot-local",
        displayName: "Wealth Copilot",
      },
      pubKeyCredParams: [
        { type: "public-key", alg: -7 },
        { type: "public-key", alg: -257 },
      ],
      authenticatorSelection: {
        authenticatorAttachment: "platform",
        residentKey: "discouraged",
        userVerification: "required",
      },
      timeout: 60_000,
      attestation: "none",
    },
  });
  if (!(credential instanceof PublicKeyCredential))
    throw new Error("device_lock_failed");
  window.localStorage.setItem(CREDENTIAL_KEY, credential.id);
  window.localStorage.setItem(ENABLED_KEY, "true");
}

export function disableDeviceLock(): void {
  try {
    window.localStorage.removeItem(ENABLED_KEY);
    window.localStorage.removeItem(CREDENTIAL_KEY);
    window.localStorage.removeItem(HIDDEN_AT_KEY);
  } catch {
    // The app remains usable if browser storage is blocked.
  }
}

export async function unlockDeviceLock(): Promise<boolean> {
  if (!appLockSupported()) return false;
  const id = window.localStorage.getItem(CREDENTIAL_KEY);
  if (!id) return false;
  const credential = await navigator.credentials.get({
    publicKey: {
      challenge: randomBytes(32),
      allowCredentials: [{ id: decodeCredentialId(id), type: "public-key" }],
      userVerification: "required",
      timeout: 60_000,
    },
  });
  return credential instanceof PublicKeyCredential;
}

export function rememberBackgroundTime(timestamp: number): void {
  try {
    window.localStorage.setItem(HIDDEN_AT_KEY, String(timestamp));
  } catch {
    // Auto-lock still works for the current tab.
  }
}

export function clearBackgroundTime(): void {
  try {
    window.localStorage.removeItem(HIDDEN_AT_KEY);
  } catch {
    // The lock state still clears in memory.
  }
}

export function shouldLockAfterBackground(now: number): boolean {
  try {
    const raw = window.localStorage.getItem(HIDDEN_AT_KEY);
    const hiddenAt = raw ? Number(raw) : Number.NaN;
    return (
      appLockEnabled() &&
      Number.isFinite(hiddenAt) &&
      now - hiddenAt >= APP_LOCK_BACKGROUND_MS
    );
  } catch {
    return false;
  }
}
