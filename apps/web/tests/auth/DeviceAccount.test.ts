import { afterEach, describe, expect, it, vi } from "vitest";
import { claimDeviceFor } from "../../src/auth/device-account";
import { ONBOARDING_STORAGE_KEY } from "../../src/money/config";
import {
  DEVICE_ACCOUNT_KEY,
  PUSH_SUBSCRIPTION_ID_KEY,
} from "../../src/pwa/storage-keys";

function token(subject: unknown): string {
  const encode = (value: object) =>
    btoa(JSON.stringify(value))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
  return `${encode({ alg: "HS256", typ: "JWT" })}.${encode({ sub: subject, email: "someone@example.com" })}.signature`;
}

function openDatabase(name: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(name);
    request.onsuccess = () => {
      request.result.close();
      resolve();
    };
    request.onerror = () => reject(request.error);
  });
}

const originalServiceWorker = Object.getOwnPropertyDescriptor(
  navigator,
  "serviceWorker",
);

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  if (originalServiceWorker) {
    Object.defineProperty(navigator, "serviceWorker", originalServiceWorker);
  } else {
    Reflect.deleteProperty(navigator, "serviceWorker");
  }
});

describe("signing in on a shared phone", () => {
  it("wipes what the previous account left behind when someone else signs in", async () => {
    const unsubscribe = vi.fn().mockResolvedValue(true);
    const getSubscription = vi.fn().mockResolvedValue({ unsubscribe });
    Object.defineProperty(navigator, "serviceWorker", {
      configurable: true,
      value: {
        getRegistration: vi.fn().mockResolvedValue({
          pushManager: { getSubscription },
        }),
      },
    });
    localStorage.setItem(DEVICE_ACCOUNT_KEY, "person-a");
    localStorage.setItem(
      ONBOARDING_STORAGE_KEY,
      JSON.stringify({ step: "risk", annualIncome: "1800000" }),
    );
    localStorage.setItem(PUSH_SUBSCRIPTION_ID_KEY, "subscription-1");
    await openDatabase("wealth-money-map-offline");

    await claimDeviceFor(token("person-b"));

    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBeNull();
    expect(localStorage.getItem(DEVICE_ACCOUNT_KEY)).toBe("person-b");
    expect(unsubscribe).toHaveBeenCalledOnce();
    expect(localStorage.getItem(PUSH_SUBSCRIPTION_ID_KEY)).toBeNull();
    const databases = await indexedDB.databases();
    expect(databases.map((database) => database.name)).not.toContain(
      "wealth-money-map-offline",
    );
  });

  it("keeps the same person's unfinished answers", async () => {
    localStorage.setItem(DEVICE_ACCOUNT_KEY, "person-a");
    localStorage.setItem(ONBOARDING_STORAGE_KEY, '{"step":"risk"}');
    localStorage.setItem(PUSH_SUBSCRIPTION_ID_KEY, "subscription-1");

    await claimDeviceFor(token("person-a"));

    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe(
      '{"step":"risk"}',
    );
    expect(localStorage.getItem(PUSH_SUBSCRIPTION_ID_KEY)).toBe(
      "subscription-1",
    );
  });

  it.each(["service worker", "Push API"])(
    "continues sign-in when the %s is unavailable",
    async (missingApi) => {
      localStorage.setItem(DEVICE_ACCOUNT_KEY, "person-a");
      localStorage.setItem(PUSH_SUBSCRIPTION_ID_KEY, "subscription-1");
      if (missingApi === "Push API") {
        Object.defineProperty(navigator, "serviceWorker", {
          configurable: true,
          value: {
            getRegistration: vi.fn().mockResolvedValue({}),
          },
        });
      }

      await expect(claimDeviceFor(token("person-b"))).resolves.toBeUndefined();

      expect(localStorage.getItem(DEVICE_ACCOUNT_KEY)).toBe("person-b");
      expect(localStorage.getItem(PUSH_SUBSCRIPTION_ID_KEY)).toBeNull();
    },
  );

  it("remembers the account on a phone that had no owner yet", async () => {
    localStorage.setItem(ONBOARDING_STORAGE_KEY, '{"step":"risk"}');

    await claimDeviceFor(token("person-a"));

    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe(
      '{"step":"risk"}',
    );
    expect(localStorage.getItem(DEVICE_ACCOUNT_KEY)).toBe("person-a");
  });

  it("leaves everything alone when the token can't be read", async () => {
    localStorage.setItem(DEVICE_ACCOUNT_KEY, "person-a");
    localStorage.setItem(ONBOARDING_STORAGE_KEY, "{}");

    await claimDeviceFor("not-a-token");
    await claimDeviceFor(token(42));

    expect(localStorage.getItem(ONBOARDING_STORAGE_KEY)).toBe("{}");
    expect(localStorage.getItem(DEVICE_ACCOUNT_KEY)).toBe("person-a");
  });
});
