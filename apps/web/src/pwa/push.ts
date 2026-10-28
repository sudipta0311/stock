import { apiFetch } from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import type { components } from "../api/schema";

const PUSH_SUBSCRIPTION_ID_KEY = "wealth.push.subscription-id";

export class PushSetupError extends Error {
  constructor(
    readonly reason:
      "unavailable" | "permission" | "configuration" | "subscription",
  ) {
    super(reason);
    this.name = "PushSetupError";
  }
}

export function applicationServerKey(value: string): Uint8Array {
  const padding = "=".repeat((4 - (value.length % 4)) % 4);
  const base64 = (value + padding).replaceAll("-", "+").replaceAll("_", "/");
  const decoded = atob(base64);
  return Uint8Array.from(decoded, (character) => character.charCodeAt(0));
}

export async function enablePush(publicKey: string): Promise<string> {
  if (
    !publicKey ||
    typeof Notification === "undefined" ||
    !navigator.serviceWorker ||
    !("PushManager" in window) ||
    !apiPathAvailable("/v1/push/subscriptions")
  ) {
    throw new PushSetupError("unavailable");
  }
  if (Notification.permission !== "granted") {
    const permission = await Notification.requestPermission();
    if (permission !== "granted") throw new PushSetupError("permission");
  }

  let subscription: PushSubscription | null = null;
  let createdHere = false;
  try {
    const registration = await navigator.serviceWorker.ready;
    subscription = await registration.pushManager.getSubscription();
    if (!subscription) {
      subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: applicationServerKey(publicKey) as BufferSource,
      });
      createdHere = true;
    }
    const payload = subscription.toJSON();
    if (!payload.endpoint || !payload.keys?.p256dh || !payload.keys.auth) {
      throw new PushSetupError("subscription");
    }
    const saved = await apiFetch<
      components["schemas"]["PushSubscriptionResponse"]
    >("/v1/push/subscriptions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ endpoint: payload.endpoint, keys: payload.keys }),
    });
    window.localStorage.setItem(PUSH_SUBSCRIPTION_ID_KEY, saved.id);
    return saved.id;
  } catch (error) {
    if (subscription && createdHere) await subscription.unsubscribe();
    if (error instanceof PushSetupError) throw error;
    throw new PushSetupError("subscription");
  }
}

export async function disablePush(): Promise<void> {
  const subscriptionId = window.localStorage.getItem(PUSH_SUBSCRIPTION_ID_KEY);
  let failure: unknown;
  try {
    if (
      subscriptionId &&
      apiPathAvailable("/v1/push/subscriptions/{subscription_id}")
    ) {
      await apiFetch<void>(
        `/v1/push/subscriptions/${encodeURIComponent(subscriptionId)}`,
        {
          method: "DELETE",
        },
      );
    }
  } catch (error) {
    failure = error;
  }
  try {
    const registration = await navigator.serviceWorker?.getRegistration();
    const subscription = await registration?.pushManager.getSubscription();
    if (subscription) await subscription.unsubscribe();
  } catch (error) {
    failure ??= error;
  } finally {
    window.localStorage.removeItem(PUSH_SUBSCRIPTION_ID_KEY);
  }
  if (failure) throw failure;
}
