import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

describe("built content security policy", () => {
  it("contains the policy generated for the exported PWA", async () => {
    const html = await readFile(
      resolve(process.cwd(), "dist/index.html"),
      "utf8",
    );
    const policy = html.match(
      /<meta http-equiv="Content-Security-Policy" content="([^"]+)"\s*\/>/,
    )?.[1];

    expect(policy).toBeDefined();
    expect(policy).toContain("default-src 'self'");
    expect(policy).toContain("script-src 'self'");
    expect(policy).toContain("style-src 'self'");
    expect(policy).toContain("connect-src 'self'");
    expect(policy).toContain("https://accounts.google.com");
    expect(policy).toContain("img-src 'self' data:");
    expect(policy).toContain("frame-src 'self' https://accounts.google.com");
    expect(policy).toContain("worker-src 'self'");
    expect(policy).toContain("manifest-src 'self'");
    expect(policy).toContain("base-uri 'self'");
    expect(policy).toContain("form-action 'self' https://accounts.google.com");
  });
});
