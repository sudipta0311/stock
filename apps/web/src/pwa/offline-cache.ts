import {
  PUSH_SUBSCRIPTION_ID_KEY,
  TOUR_SEEN_STORAGE_PREFIX,
} from "./storage-keys";

const SNAPSHOT_CACHE = "api-snapshots";
const SETTINGS_KEY = "wealth.offline-cache.enabled";

function isSnapshotUrl(value: string): boolean {
  try {
    return /^\/v1\/(?:networth|plan|inbox)(?:\/|$)/.test(
      new URL(value, window.location.origin).pathname,
    );
  } catch {
    return false;
  }
}

export function offlineCacheEnabled(): boolean {
  try {
    return window.localStorage.getItem(SETTINGS_KEY) !== "false";
  } catch {
    return true;
  }
}

function postToServiceWorker(
  message:
    | { type: "SET_CACHE_ENABLED"; enabled: boolean }
    | { type: "CLEAR_SNAPSHOTS" },
): void {
  if (typeof navigator === "undefined" || !navigator.serviceWorker) return;
  const worker = navigator.serviceWorker.controller;
  worker?.postMessage(message);
  void navigator.serviceWorker.getRegistration().then((registration) => {
    registration?.active?.postMessage(message);
  });
}

export function setOfflineCacheEnabled(enabled: boolean): void {
  try {
    window.localStorage.setItem(SETTINGS_KEY, String(enabled));
  } catch {
    // The active service worker still receives the current preference.
  }
  postToServiceWorker({ type: "SET_CACHE_ENABLED", enabled });
  if (!enabled) void clearSnapshotCache();
}

export function syncOfflineCacheSetting(): void {
  if (typeof navigator === "undefined" || !navigator.serviceWorker) return;
  postToServiceWorker({
    type: "SET_CACHE_ENABLED",
    enabled: offlineCacheEnabled(),
  });
}

export async function clearSnapshotCache(): Promise<void> {
  postToServiceWorker({ type: "CLEAR_SNAPSHOTS" });
  if (typeof caches === "undefined") return;
  try {
    await caches.delete(SNAPSHOT_CACHE);
  } catch {
    // Private browsing modes can block Cache Storage access.
  }
}

export async function storeOfflineSnapshot(
  url: string,
  response: Response,
): Promise<void> {
  if (
    !offlineCacheEnabled() ||
    typeof caches === "undefined" ||
    !isSnapshotUrl(url)
  )
    return;
  try {
    const cache = await caches.open(SNAPSHOT_CACHE);
    await cache.put(url, response);
  } catch {
    // Snapshot caching is optional when the browser blocks Cache Storage.
  }
}

export async function readOfflineSnapshot(
  url: string,
): Promise<Response | null> {
  if (
    !offlineCacheEnabled() ||
    typeof caches === "undefined" ||
    !isSnapshotUrl(url)
  )
    return null;
  try {
    const cache = await caches.open(SNAPSHOT_CACHE);
    return (await cache.match(url)) ?? null;
  } catch {
    return null;
  }
}

async function clearAppCaches(): Promise<void> {
  if (typeof caches === "undefined") return;
  try {
    const names = await caches.keys();
    await Promise.all(names.map((name) => caches.delete(name)));
  } catch {
    // Cache Storage may be unavailable in private browsing modes.
  }
}

async function clearAppDatabases(): Promise<void> {
  if (typeof indexedDB === "undefined") return;
  try {
    const databases = await indexedDB.databases();
    await Promise.all(
      databases
        .filter((database) => database.name?.startsWith("wealth-"))
        .map(
          (database) =>
            new Promise<void>((resolve) => {
              const request = indexedDB.deleteDatabase(database.name ?? "");
              request.onsuccess =
                request.onerror =
                request.onblocked =
                  () => resolve();
            }),
        ),
    );
  } catch {
    // Browsers that cannot list IndexedDB databases still clear known app stores below.
    await Promise.all(
      ["wealth-money-map-offline"].map(
        (name) =>
          new Promise<void>((resolve) => {
            const request = indexedDB.deleteDatabase(name);
            request.onsuccess =
              request.onerror =
              request.onblocked =
                () => resolve();
          }),
      ),
    );
  }
}

export async function clearOfflineData(): Promise<void> {
  postToServiceWorker({ type: "CLEAR_SNAPSHOTS" });
  await Promise.all([clearAppCaches(), clearAppDatabases()]);
}

export async function clearAllDeviceData(
  options: {
    preservePushSubscription?: boolean;
    preserveTourHistory?: boolean;
  } = {},
): Promise<void> {
  await clearOfflineData();
  try {
    const pushSubscriptionId = options.preservePushSubscription
      ? window.localStorage.getItem(PUSH_SUBSCRIPTION_ID_KEY)
      : null;
    const tourHistory = options.preserveTourHistory
      ? Object.keys(window.localStorage)
          .filter((key) => key.startsWith(TOUR_SEEN_STORAGE_PREFIX))
          .map((key) => [key, window.localStorage.getItem(key)] as const)
      : [];
    window.localStorage.clear();
    if (pushSubscriptionId) {
      window.localStorage.setItem(PUSH_SUBSCRIPTION_ID_KEY, pushSubscriptionId);
    }
    for (const [key, value] of tourHistory) {
      if (value !== null) window.localStorage.setItem(key, value);
    }
    window.sessionStorage.clear();
  } catch {
    // Deletion and sign-out continue when browser storage is blocked.
  }
}

export async function getOfflineSnapshotTime(): Promise<string | null> {
  if (typeof caches === "undefined") return null;
  try {
    const cache = await caches.open(SNAPSHOT_CACHE);
    const requests = await cache.keys();
    const snapshots: number[] = [];
    for (const request of requests) {
      const path = new URL(request.url).pathname;
      if (!/^\/v1\/(?:networth|plan|inbox)(?:\/|$)/.test(path)) continue;
      const response = await cache.match(request);
      if (!response) continue;
      let timestamp = Number.NaN;
      try {
        const body = (await response.clone().json()) as { as_of?: unknown };
        if (typeof body.as_of === "string") timestamp = Date.parse(body.as_of);
      } catch {
        // A partial or non-JSON snapshot can still be shown without an age label.
      }
      if (Number.isNaN(timestamp)) {
        const dateHeader = response.headers.get("date");
        timestamp = dateHeader ? Date.parse(dateHeader) : Number.NaN;
      }
      if (!Number.isNaN(timestamp)) snapshots.push(timestamp);
    }
    if (!snapshots.length) return null;
    return new Intl.DateTimeFormat("en-IN", {
      day: "numeric",
      month: "short",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(Math.max(...snapshots));
  } catch {
    return null;
  }
}
