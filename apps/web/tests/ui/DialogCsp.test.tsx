import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Sheet } from "../../src/ui/sheet";

const scrollLockHash = "sha256-nzTgYzXYDNe6BAHiiI7NNlfK8n/auuOAhh2t92YvuXo=";

describe("dialog scroll lock styles", () => {
  it("matches the allowlisted library style and keeps the page locked", async () => {
    const innerWidth = Object.getOwnPropertyDescriptor(window, "innerWidth");
    const clientWidth = Object.getOwnPropertyDescriptor(
      document.documentElement,
      "clientWidth",
    );
    Object.defineProperty(window, "innerWidth", {
      configurable: true,
      value: 360,
    });
    Object.defineProperty(document.documentElement, "clientWidth", {
      configurable: true,
      value: 360,
    });

    const view = render(
      <Sheet open onOpenChange={() => undefined} title="Synthetic dialog">
        <p>Dialog content</p>
      </Sheet>,
    );
    try {
      expect(await screen.findByRole("dialog")).toBeVisible();
      await waitFor(() =>
        expect(document.body).toHaveAttribute("data-scroll-locked"),
      );
      const style = [...document.querySelectorAll("style")].find((element) =>
        element.textContent?.includes("body[data-scroll-locked]"),
      );
      expect(style).toBeDefined();
      const hash = `sha256-${createHash("sha256")
        .update(style?.textContent ?? "")
        .digest("base64")}`;
      expect(hash).toBe(scrollLockHash);
      // A CSP hash source only counts when it is single-quoted.
      const viteConfig = await readFile(
        resolve(process.cwd(), "vite.config.ts"),
        "utf8",
      );
      expect(viteConfig).toContain(`"'${scrollLockHash}'"`);
      const applicationStyles = await readFile(
        resolve(process.cwd(), "src/styles.css"),
        "utf8",
      );
      expect(applicationStyles).toMatch(
        /body\[data-scroll-locked\]\s*\{[^}]*overflow:\s*hidden/s,
      );
    } finally {
      view.unmount();
      if (innerWidth) {
        Object.defineProperty(window, "innerWidth", innerWidth);
      } else {
        Reflect.deleteProperty(window, "innerWidth");
      }
      if (clientWidth) {
        Object.defineProperty(
          document.documentElement,
          "clientWidth",
          clientWidth,
        );
      } else {
        Reflect.deleteProperty(document.documentElement, "clientWidth");
      }
    }
  });
});
