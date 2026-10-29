import "@testing-library/jest-dom/vitest";
import "fake-indexeddb/auto";
import { cleanup, configure } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, expect } from "vitest";
import { server } from "./msw";
import { bannedVisibleTerms } from "./plain-language";

// findBy*/waitFor default to 1 s, which slow CI runners miss on first render.
configure({ asyncUtilTimeout: 5_000 });

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => {
  const violations = Array.from(
    document.querySelectorAll<HTMLElement>("[data-copy-screen]"),
  ).flatMap((root) =>
    bannedVisibleTerms(root).map(
      (term) => `${root.dataset.copyScreen ?? "screen"}: ${term}`,
    ),
  );
  expect(violations).toEqual([]);
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
