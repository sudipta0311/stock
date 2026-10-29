/// <reference lib="webworker" />

import { cleanupOutdatedCaches, precacheAndRoute } from "workbox-precaching";
import { NavigationRoute, registerRoute } from "workbox-routing";
import { createHandlerBoundToURL } from "workbox-precaching";
import { CacheableResponsePlugin } from "workbox-cacheable-response";
import { ExpirationPlugin } from "workbox-expiration";
import { NetworkFirst } from "workbox-strategies";
import { handleShareTargetPost, storeSharedFile } from "./imports/share-target";
import { notificationTarget } from "./pwa/notification-routing";
import englishCopy from "./copy/en.json";

declare const self: ServiceWorkerGlobalScope & { __WB_MANIFEST: Array<never> };

const SNAPSHOT_CACHE = "api-snapshots";
const CACHE_SETTING_BUCKET = "wealth-settings";
const CACHE_SETTING_URL = new URL(
  "/__wealth_offline_cache_setting__",
  self.location.origin,
).href;

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();
registerRoute(new NavigationRoute(createHandlerBoundToURL("/index.html")));

self.addEventListener("fetch", (event: FetchEvent) => {
  const requestUrl = new URL(event.request.url);
  if (
    event.request.method !== "POST" ||
    requestUrl.origin !== self.location.origin ||
    requestUrl.pathname !== "/import/share"
  )
    return;
  event.respondWith(
    (async () => {
      try {
        const form = await event.request.formData();
        const entry = form.get("file");
        const file =
          typeof File !== "undefined" && entry instanceof File ? entry : null;
        const response = await handleShareTargetPost(
          event.request.method,
          event.request.url,
          file,
          storeSharedFile,
        );
        return response ?? fetch(event.request);
      } catch {
        return new Response(null, {
          status: 303,
          headers: {
            Location: new URL(
              "/import?shareError=share-storage",
              self.location.origin,
            ).href,
          },
        });
      }
    })(),
  );
});

async function isSnapshotCacheEnabled(): Promise<boolean> {
  const settings = await caches.open(CACHE_SETTING_BUCKET);
  const setting = await settings.match(CACHE_SETTING_URL);
  return (await setting?.text()) !== "false";
}

self.addEventListener("message", (event: ExtendableMessageEvent) => {
  const message = event.data as { type?: string; enabled?: boolean } | null;
  if (message?.type === "SET_CACHE_ENABLED") {
    event.waitUntil(
      (async () => {
        const settings = await caches.open(CACHE_SETTING_BUCKET);
        await settings.put(
          CACHE_SETTING_URL,
          new Response(String(message.enabled !== false)),
        );
        if (message.enabled === false) await caches.delete(SNAPSHOT_CACHE);
      })(),
    );
  }
  if (message?.type === "CLEAR_SNAPSHOTS")
    event.waitUntil(caches.delete(SNAPSHOT_CACHE));
  // "Tap to refresh" (vite-plugin-pwa) asks the waiting version to take over; without
  // this the old version keeps running until every tab of the app is closed.
  if (message?.type === "SKIP_WAITING") event.waitUntil(self.skipWaiting());
});

const snapshotStrategy = new NetworkFirst({
  cacheName: SNAPSHOT_CACHE,
  networkTimeoutSeconds: 3,
  plugins: [
    new CacheableResponsePlugin({ statuses: [200] }),
    new ExpirationPlugin({ maxEntries: 20, maxAgeSeconds: 7 * 24 * 60 * 60 }),
  ],
});

registerRoute(
  ({ url }) =>
    url.pathname.startsWith("/v1/") &&
    /^\/v1\/(?:networth|plan|inbox)(?:\/.*)?$/.test(url.pathname) &&
    !/^\/v1\/(?:auth|imports)(?:\/|$)/.test(url.pathname),
  async ({ request, event }) => {
    if (!(await isSnapshotCacheEnabled())) return fetch(request);
    return snapshotStrategy.handle({ request, event });
  },
  "GET",
);

self.addEventListener("activate", (event: ExtendableEvent) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("push", (event: PushEvent) => {
  let payload: Record<string, unknown> = {};
  try {
    const parsed: unknown = event.data?.json();
    if (typeof parsed === "object" && parsed !== null)
      payload = parsed as Record<string, unknown>;
  } catch {
    // An empty notification still opens the app safely.
  }
  const title =
    typeof payload.title === "string" && payload.title.trim()
      ? payload.title
      : englishCopy.push.notification_title;
  const body =
    typeof payload.body === "string" && payload.body.trim()
      ? payload.body
      : englishCopy.push.notification_fallback;
  const target = notificationTarget(payload, self.location.origin);
  event.waitUntil(
    self.registration.showNotification(title, {
      body,
      data: { url: target },
      icon: "/icons/wealth-192.png",
      badge: "/icons/wealth-192.png",
    }),
  );
});

self.addEventListener("notificationclick", (event: NotificationEvent) => {
  event.notification.close();
  const target = notificationTarget(
    event.notification.data,
    self.location.origin,
  );
  event.waitUntil(
    (async () => {
      const windows = await self.clients.matchAll({
        type: "window",
        includeUncontrolled: true,
      });
      const app = windows.find(
        (client) => new URL(client.url).origin === self.location.origin,
      );
      if (app) {
        await app.navigate(new URL(target, self.location.origin).href);
        await app.focus();
        return;
      }
      await self.clients.openWindow(new URL(target, self.location.origin).href);
    })(),
  );
});
