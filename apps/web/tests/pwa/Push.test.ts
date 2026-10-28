import { afterEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";
import { server } from "../msw";
import { disablePush, enablePush } from "../../src/pwa/push";
import { notificationTarget } from "../../src/pwa/notification-routing";

const pushPayload = {
  endpoint: "https://push.example.test/subscription",
  keys: { p256dh: "synthetic-public-key", auth: "synthetic-auth-key" },
};

function setupPushBrowser(permission: "default" | "granted" = "default") {
  const requestPermission = vi.fn().mockResolvedValue("granted");
  vi.stubGlobal("Notification", { permission, requestPermission });
  Object.defineProperty(window, "PushManager", {
    configurable: true,
    value: class PushManager {},
  });
  const unsubscribe = vi.fn().mockResolvedValue(true);
  const subscription = { toJSON: () => pushPayload, unsubscribe };
  const subscribe = vi.fn().mockResolvedValue(subscription);
  const getSubscription = vi
    .fn()
    .mockResolvedValueOnce(null)
    .mockResolvedValue(subscription);
  const registration = { pushManager: { subscribe, getSubscription } };
  Object.defineProperty(navigator, "serviceWorker", {
    configurable: true,
    value: {
      ready: Promise.resolve(registration),
      getRegistration: vi.fn().mockResolvedValue(registration),
    },
  });
  return { requestPermission, subscribe, getSubscription, unsubscribe };
}

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  Reflect.deleteProperty(window, "PushManager");
  Reflect.deleteProperty(navigator, "serviceWorker");
});

describe("web push", () => {
  it("requests permission on enable, posts the subscription, and unsubscribes server-side and locally", async () => {
    const browser = setupPushBrowser();
    const posted: unknown[] = [];
    let deleted = false;
    server.use(
      http.post("*/v1/push/subscriptions", async ({ request }) => {
        posted.push(await request.json());
        return HttpResponse.json({ id: "subscription-test" });
      }),
      http.delete("*/v1/push/subscriptions/subscription-test", () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    expect(browser.requestPermission).not.toHaveBeenCalled();
    await expect(enablePush("AQIDBA")).resolves.toBe("subscription-test");
    expect(browser.requestPermission).toHaveBeenCalledOnce();
    expect(browser.subscribe).toHaveBeenCalledOnce();
    expect(posted).toEqual([pushPayload]);

    await disablePush();
    expect(deleted).toBe(true);
    expect(browser.unsubscribe).toHaveBeenCalledOnce();
    expect(
      window.localStorage.getItem("wealth.push.subscription-id"),
    ).toBeNull();
  });

  it("routes only same-origin home and inbox notification links", () => {
    expect(
      notificationTarget(
        { url: "/inbox/brief/2026-W44?open=1" },
        "https://wealth.example.test",
      ),
    ).toBe("/inbox/brief/2026-W44?open=1");
    expect(
      notificationTarget(
        { url: "https://elsewhere.example/path" },
        "https://wealth.example.test",
      ),
    ).toBe("/");
    expect(
      notificationTarget({ url: "/settings" }, "https://wealth.example.test"),
    ).toBe("/");
    expect(
      notificationTarget(
        { url: "//elsewhere.example/" },
        "https://wealth.example.test",
      ),
    ).toBe("/");
  });
});
