import { useEffect, useState } from "react";
import { useLocation } from "react-router";
import englishCopy from "../copy/en.json";
import { Button, Card, Sheet } from "../ui";
import {
  hasValueMoment,
  installGuideLastShownAt,
  isIosDevice,
  isStandaloneMode,
  markInstallGuideShown,
  shouldShowIosInstallGuide,
} from "../pwa/install";

interface DeferredInstallPrompt extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed"; platform?: string }>;
}

function standalone(): boolean {
  return isStandaloneMode(
    typeof window.matchMedia === "function" &&
      window.matchMedia("(display-mode: standalone)").matches,
    Boolean((navigator as Navigator & { standalone?: boolean }).standalone),
  );
}

export function InstallExperience() {
  const location = useLocation();
  const [valueMoment, setValueMoment] = useState(hasValueMoment);
  const [deferredPrompt, setDeferredPrompt] =
    useState<DeferredInstallPrompt | null>(null);
  const [androidCard, setAndroidCard] = useState(false);
  const [iosSheet, setIosSheet] = useState(false);

  useEffect(() => {
    const rememberValue = () => setValueMoment(true);
    const capturePrompt = (event: Event) => {
      event.preventDefault();
      setDeferredPrompt(event as DeferredInstallPrompt);
    };
    const installed = () => {
      setAndroidCard(false);
      setIosSheet(false);
      setDeferredPrompt(null);
    };
    window.addEventListener("wealth:value-moment", rememberValue);
    window.addEventListener("beforeinstallprompt", capturePrompt);
    window.addEventListener("appinstalled", installed);
    return () => {
      window.removeEventListener("wealth:value-moment", rememberValue);
      window.removeEventListener("beforeinstallprompt", capturePrompt);
      window.removeEventListener("appinstalled", installed);
    };
  }, []);

  useEffect(() => {
    setAndroidCard(false);
    setIosSheet(false);
    if (
      !valueMoment ||
      standalone() ||
      location.pathname.startsWith("/settings")
    )
      return;
    if (isIosDevice(navigator.userAgent, navigator.platform)) {
      const lastShown = installGuideLastShownAt();
      if (shouldShowIosInstallGuide(lastShown)) {
        markInstallGuideShown();
        setIosSheet(true);
      }
      return;
    }
    if (
      deferredPrompt &&
      shouldShowIosInstallGuide(installGuideLastShownAt())
    ) {
      markInstallGuideShown();
      setAndroidCard(true);
    }
  }, [deferredPrompt, location.pathname, valueMoment]);

  async function install(): Promise<void> {
    if (!deferredPrompt) return;
    await deferredPrompt.prompt();
    await deferredPrompt.userChoice;
    setAndroidCard(false);
    setDeferredPrompt(null);
  }

  if (androidCard) {
    return (
      <div className="mx-auto w-full max-w-3xl px-4 pt-3 sm:px-6">
        <Card
          className="flex items-center justify-between gap-3"
          aria-label={englishCopy.install.button}
        >
          <p className="text-sm">{englishCopy.install.card_sentence}</p>
          <div className="flex shrink-0 items-center gap-1">
            <Button onClick={() => void install()}>
              {englishCopy.install.button}
            </Button>
            <button
              type="button"
              aria-label={englishCopy.install.dismiss}
              className="grid size-10 place-items-center rounded-full text-muted-foreground hover:bg-muted"
              onClick={() => {
                setAndroidCard(false);
                markInstallGuideShown();
              }}
            >
              <span aria-hidden="true">×</span>
            </button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <Sheet
      open={iosSheet}
      onOpenChange={(open) => setIosSheet(open)}
      title={englishCopy.install.ios_title}
    >
      <div className="grid justify-items-center gap-4 text-center">
        <svg
          aria-hidden="true"
          viewBox="0 0 96 72"
          className="h-16 w-24 text-brand"
        >
          <rect
            x="10"
            y="6"
            width="76"
            height="60"
            rx="12"
            fill="currentColor"
            opacity="0.12"
          />
          <path
            d="M48 48V18m0 0-11 11m11-11 11 11M28 39v12a6 6 0 0 0 6 6h28a6 6 0 0 0 6-6V39"
            fill="none"
            stroke="currentColor"
            strokeWidth="4"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <p className="text-sm">{englishCopy.install.ios_sentence}</p>
        <Button className="w-full" onClick={() => setIosSheet(false)}>
          {englishCopy.install.ios_close}
        </Button>
      </div>
    </Sheet>
  );
}
