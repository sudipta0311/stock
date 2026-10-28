import { afterEach, describe, expect, it, vi } from "vitest";

const captured = vi.hoisted(() => ({
  settings: [] as Array<Record<string, unknown>>,
}));

vi.mock("oidc-client-ts", () => ({
  UserManager: class {
    constructor(settings: Record<string, unknown>) {
      captured.settings.push(settings);
    }
  },
  WebStorageStateStore: class {},
}));

afterEach(() => {
  vi.unstubAllEnvs();
  captured.settings.length = 0;
});

describe("Google sign-in setup", () => {
  it("uses the API token exchange while keeping PKCE enabled", async () => {
    vi.stubEnv("VITE_OIDC_AUTHORITY", "https://accounts.google.com");
    vi.stubEnv("VITE_OIDC_CLIENT_ID", "synthetic-client-id");
    vi.stubEnv("VITE_OIDC_AUDIENCE", "synthetic-client-id");
    vi.stubEnv("VITE_API_URL", "https://api.example.test");

    const { getOidcUserManager } = await import("../../src/auth/oidc");
    await getOidcUserManager();

    expect(captured.settings).toHaveLength(1);
    expect(captured.settings[0]).toMatchObject({
      authority: "https://accounts.google.com",
      client_id: "synthetic-client-id",
      response_type: "code",
      automaticSilentRenew: true,
      metadataSeed: {
        token_endpoint: "https://api.example.test/v1/auth/google/exchange",
      },
    });
    expect(captured.settings[0]?.disablePKCE).not.toBe(true);
  });
});
