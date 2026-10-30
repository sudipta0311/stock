import { describe, expect, it } from "vitest";
import { developmentTokenSignInAllowed } from "../src/auth/dev-auth";

describe("development token sign-in build guard", () => {
  it("allows development builds and the explicit e2e build", () => {
    expect(developmentTokenSignInAllowed("dev", false, false)).toBe(true);
    expect(developmentTokenSignInAllowed("dev", true, true)).toBe(true);
  });

  it("blocks production dev tokens unless the e2e build switch is enabled", () => {
    expect(developmentTokenSignInAllowed("dev", true, false)).toBe(false);
    expect(developmentTokenSignInAllowed("oidc", true, true)).toBe(false);
  });
});
