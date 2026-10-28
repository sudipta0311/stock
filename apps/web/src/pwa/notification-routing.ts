export function notificationTarget(payload: unknown, origin: string): string {
  const raw =
    typeof payload === "object" &&
    payload !== null &&
    "url" in payload &&
    typeof payload.url === "string"
      ? payload.url
      : "/";
  try {
    const base = new URL(origin);
    const target = new URL(raw, base);
    const allowedPath =
      target.pathname === "/" ||
      target.pathname === "/inbox" ||
      target.pathname.startsWith("/inbox/");
    if (target.origin !== base.origin || !allowedPath) return "/";
    return `${target.pathname}${target.search}${target.hash}`;
  } catch {
    return "/";
  }
}
