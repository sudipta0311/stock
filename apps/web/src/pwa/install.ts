const VALUE_MOMENT_KEY = "wealth.pwa.value-moment";
const INSTALL_SHOWN_KEY = "wealth.pwa.install-shown-at";
const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

export function markValueMoment(): void {
  try {
    window.localStorage.setItem(VALUE_MOMENT_KEY, "true");
  } catch {
    // The current session can still show the install prompt.
  }
  window.dispatchEvent(new Event("wealth:value-moment"));
}

export function hasValueMoment(): boolean {
  try {
    return window.localStorage.getItem(VALUE_MOMENT_KEY) === "true";
  } catch {
    return false;
  }
}

export function isStandaloneMode(
  displayModeMatches: boolean,
  iosStandalone: boolean,
): boolean {
  return displayModeMatches || iosStandalone;
}

export function isIosDevice(
  userAgent: string,
  platform: string,
  touchPoints = typeof navigator === "undefined" ? 0 : navigator.maxTouchPoints,
): boolean {
  return (
    /iPad|iPhone|iPod/i.test(userAgent) ||
    (platform === "MacIntel" && touchPoints > 1)
  );
}

export function shouldShowIosInstallGuide(
  lastShownAt: number | null,
  now = Date.now(),
): boolean {
  return lastShownAt === null || now - lastShownAt >= THIRTY_DAYS_MS;
}

export function markInstallGuideShown(now = Date.now()): void {
  try {
    window.localStorage.setItem(INSTALL_SHOWN_KEY, String(now));
  } catch {
    // The guide can still be dismissed for the current session.
  }
}

export function installGuideLastShownAt(): number | null {
  try {
    const value = window.localStorage.getItem(INSTALL_SHOWN_KEY);
    if (!value) return null;
    const timestamp = Number(value);
    return Number.isFinite(timestamp) ? timestamp : null;
  } catch {
    return null;
  }
}

export function iosInstallGuideNeeded(now = Date.now()): boolean {
  return shouldShowIosInstallGuide(installGuideLastShownAt(), now);
}
