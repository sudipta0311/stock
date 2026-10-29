import { clearAllDeviceData } from "../pwa/offline-cache";
import { DEVICE_ACCOUNT_KEY } from "../pwa/storage-keys";

async function unsubscribePushSubscription(): Promise<void> {
  try {
    const registration = await navigator.serviceWorker?.getRegistration?.();
    const subscription = await registration?.pushManager?.getSubscription?.();
    await subscription?.unsubscribe?.();
  } catch {
    // A missing or unavailable Push API must not block sign-in.
  }
}

function tokenSubject(token: string): string | null {
  try {
    const payload = token.split(".")[1] ?? "";
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    const subject: unknown = (JSON.parse(json) as { sub?: unknown }).sub;
    return typeof subject === "string" && subject ? subject : null;
  } catch {
    return null;
  }
}

/**
 * A session can end without sign-out (it expires), which leaves that person's
 * onboarding answers, offline copies and queued edits on the phone. When a
 * different account signs in next, wipe them before its session starts.
 */
export async function claimDeviceFor(token: string): Promise<void> {
  const subject = tokenSubject(token);
  if (!subject) return;
  let previous: string | null;
  try {
    previous = window.localStorage.getItem(DEVICE_ACCOUNT_KEY);
  } catch {
    return;
  }
  if (previous !== null && previous !== subject) {
    await unsubscribePushSubscription();
    try {
      await clearAllDeviceData({ preservePushSubscription: false });
    } catch {
      // Sign-in continues when browser storage cannot be cleared.
    }
  }
  try {
    window.localStorage.setItem(DEVICE_ACCOUNT_KEY, subject);
  } catch {
    // Without storage there is nothing left behind to protect.
  }
}
