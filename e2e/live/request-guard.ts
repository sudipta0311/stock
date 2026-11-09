import { DEFAULT_LIVE_API_URL, DEFAULT_LIVE_PWA_URL } from "./environment.mjs";

export interface RequestDecisionInput {
  method: string;
  url: string;
  isNavigation: boolean;
  frameIsMain: boolean;
}

export interface RequestOrigins {
  pwaUrl?: string;
  apiUrl?: string;
}

export function decideRequest(
  request: RequestDecisionInput,
  origins: RequestOrigins = {},
): "allow" | "block" {
  let url: URL;
  let pwaOrigin: string;
  let apiOrigin: string;
  try {
    url = new URL(request.url);
    pwaOrigin = new URL(origins.pwaUrl ?? DEFAULT_LIVE_PWA_URL).origin;
    apiOrigin = new URL(origins.apiUrl ?? DEFAULT_LIVE_API_URL).origin;
  } catch {
    return "block";
  }

  const method = request.method.toUpperCase();
  if (["GET", "HEAD", "OPTIONS"].includes(method)) {
    if (
      request.isNavigation &&
      request.frameIsMain &&
      url.origin !== pwaOrigin
    ) {
      return "block";
    }
    return "allow";
  }

  if (
    method === "POST" &&
    url.origin === apiOrigin &&
    url.pathname === "/v1/auth/refresh"
  ) {
    return "allow";
  }

  return "block";
}

const dynamicSegments: ReadonlyArray<readonly [string, number, string?]> = [
  ["/v1/assets/", 3, "quick-add"],
  ["/v1/loans/", 3],
  ["/v1/imports/", 3],
  ["/v1/brief/", 3],
  ["/v1/actions/", 3],
  ["/v1/recorded-buys/", 3],
  ["/v1/members/", 3],
  ["/v1/goals/", 3],
  ["/v1/highlights/", 3],
  ["/v1/households/access/", 4],
  ["/v1/households/invites/", 4],
  ["/v1/statement-inbox/keys/", 4],
  ["/v1/statement-inbox/items/", 4],
  ["/v1/fund-check/sips/", 4],
  ["/v1/push/subscriptions/", 4],
];

export function pathTemplate(pathname: string): string {
  const path = pathname.split("?")[0] ?? pathname;
  const segments = path.split("/");
  const pattern = dynamicSegments.find(([prefix]) => path.startsWith(prefix));
  if (pattern) {
    const [, segmentIndex, staticSegment] = pattern;
    if (segments[segmentIndex] && segments[segmentIndex] !== staticSegment) {
      segments[segmentIndex] = ":id";
    }
  }

  const importsIndex = segments.findIndex((segment) => segment === "imports");
  if (importsIndex >= 0 && segments[importsIndex + 2] === "items") {
    segments[importsIndex + 3] = ":id";
  }
  const actionsIndex = segments.findIndex((segment) => segment === "actions");
  if (actionsIndex >= 0 && segments[actionsIndex + 2] === "bought") {
    segments[actionsIndex + 1] = ":id";
  }
  const recordedBuyIndex = segments.findIndex(
    (segment) => segment === "recorded-buys",
  );
  if (recordedBuyIndex >= 0) segments[recordedBuyIndex + 1] = ":id";
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index] ?? "";
    if (
      /^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(segment) ||
      /^\d{6,}$/.test(segment) ||
      /@/.test(segment) ||
      /^[A-Za-z0-9_-]{24,}$/.test(segment)
    ) {
      segments[index] = ":id";
    }
  }
  return segments.join("/") || "/";
}

export function routeTemplate(
  url: string,
  origins: RequestOrigins = {},
): string {
  try {
    const parsed = new URL(url);
    const pwaOrigin = new URL(origins.pwaUrl ?? DEFAULT_LIVE_PWA_URL).origin;
    const apiOrigin = new URL(origins.apiUrl ?? DEFAULT_LIVE_API_URL).origin;
    if (parsed.origin !== pwaOrigin && parsed.origin !== apiOrigin) {
      return "/external";
    }
    return pathTemplate(parsed.pathname);
  } catch {
    return "/invalid-url";
  }
}
