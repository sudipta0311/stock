import type { ConsoleMessage, Page } from "@playwright/test";
import { expect, test as appTest } from "./app";

const allowedConsoleErrors = new WeakMap<Page, RegExp[]>();

export function allowConsoleErrors(page: Page, patterns: RegExp[]): void {
  allowedConsoleErrors.set(page, [
    ...(allowedConsoleErrors.get(page) ?? []),
    ...patterns,
  ]);
}

export function watchBrowserErrors(page: Page): {
  stop(): string[];
  assertNoUnexpectedErrors(): void;
} {
  const errors: string[] = [];
  const onConsole = (message: ConsoleMessage) => {
    if (message.type() !== "error") return;
    const url = message.location().url;
    errors.push(
      "console: " + message.text().slice(0, 300) + (url ? " @ " + url : ""),
    );
  };
  const onPageError = (error: Error) => {
    errors.push("page error: " + error.message.slice(0, 300));
  };
  let stopped = false;
  const stop = () => {
    if (!stopped) {
      page.off("console", onConsole);
      page.off("pageerror", onPageError);
      stopped = true;
    }
    return errors;
  };
  page.on("console", onConsole);
  page.on("pageerror", onPageError);
  return {
    stop,
    assertNoUnexpectedErrors() {
      stop();
      const unexpected = [...errors];
      for (const allowed of allowedConsoleErrors.get(page) ?? []) {
        const index = unexpected.findIndex((message) => allowed.test(message));
        if (index >= 0) unexpected.splice(index, 1);
      }
      allowedConsoleErrors.delete(page);
      expect(unexpected, "Unexpected browser console or page errors").toEqual(
        [],
      );
    },
  };
}

export const test = appTest.extend<{ browserErrorGuard: void }>({
  browserErrorGuard: [
    async ({ page }, use) => {
      const watcher = watchBrowserErrors(page);
      let testFailed = false;
      try {
        await use();
      } catch (error) {
        testFailed = true;
        throw error;
      } finally {
        watcher.stop();
      }
      if (!testFailed) {
        watcher.assertNoUnexpectedErrors();
      }
    },
    { auto: true },
  ],
});

export { expect };
