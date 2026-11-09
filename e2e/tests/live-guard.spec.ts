import { test, expect } from "@playwright/test";
import { isPathInsideRepository } from "../live/environment.mjs";
import { decideRequest, pathTemplate } from "../live/request-guard";

const pwaUrl = "https://wealth-pwa.onrender.com";
const apiUrl = "https://wealth-api-84le.onrender.com";

function decide(
  request: Parameters<typeof decideRequest>[0],
): "allow" | "block" {
  return decideRequest(request, { pwaUrl, apiUrl });
}

test("live request guard allows only reads and refresh", () => {
  const allowed = [
    {
      method: "GET",
      url: `${pwaUrl}/money`,
      isNavigation: true,
      frameIsMain: true,
    },
    {
      method: "GET",
      url: `${apiUrl}/v1/assets`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "HEAD",
      url: "https://fonts.example.test/font.woff2",
      isNavigation: false,
      frameIsMain: false,
    },
    {
      method: "OPTIONS",
      url: `${apiUrl}/v1/assets`,
      isNavigation: false,
      frameIsMain: false,
    },
    {
      method: "GET",
      url: "https://mail.example.test/frame",
      isNavigation: true,
      frameIsMain: false,
    },
    {
      method: "POST",
      url: `${apiUrl}/v1/auth/refresh`,
      isNavigation: false,
      frameIsMain: true,
    },
  ];
  for (const request of allowed) expect(decide(request)).toBe("allow");

  const blocked = [
    {
      method: "GET",
      url: "https://accounts.example.test/",
      isNavigation: true,
      frameIsMain: true,
    },
    {
      method: "POST",
      url: `${apiUrl}/v1/assets`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "PUT",
      url: `${apiUrl}/v1/me/profile`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "PATCH",
      url: `${apiUrl}/v1/actions/item`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "DELETE",
      url: `${apiUrl}/v1/auth/logout`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "POST",
      url: `${pwaUrl}/login`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "POST",
      url: "https://mail.example.test/connect",
      isNavigation: false,
      frameIsMain: false,
    },
    {
      method: "POST",
      url: `${apiUrl}/v1/loans/item/prepay-preview`,
      isNavigation: false,
      frameIsMain: true,
    },
    {
      method: "POST",
      url: `${apiUrl}/v1/auth/logout-all`,
      isNavigation: false,
      frameIsMain: true,
    },
  ];
  for (const request of blocked) expect(decide(request)).toBe("block");
});

test("live paths resolve outside the repository", () => {
  const repository = "C:/workspace/wealth";
  expect(
    isPathInsideRepository("C:/workspace/wealth/e2e-output", repository),
  ).toBe(true);
  expect(
    isPathInsideRepository("C:/workspace/wealth/../wealth-copy", repository),
  ).toBe(false);
  expect(
    isPathInsideRepository("C:/workspace/wealth-copy/output", repository),
  ).toBe(false);
});

test("blocked request paths replace record identifiers", () => {
  expect(pathTemplate("/v1/assets/12345678-1234-4234-8234-123456789abc")).toBe(
    "/v1/assets/:id",
  );
  expect(pathTemplate("/v1/imports/job-123/items/item-456")).toBe(
    "/v1/imports/:id/items/:id",
  );
  expect(pathTemplate("/v1/assets/quick-add")).toBe("/v1/assets/quick-add");
});
