import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";
import englishCopy from "../copy/en.json";
import {
  APP_LOCK_BACKGROUND_MS,
  appLockEnabled,
  appLockSupported,
  clearBackgroundTime,
  rememberBackgroundTime,
  shouldLockAfterBackground,
  unlockDeviceLock,
} from "../pwa/app-lock";
import { useAuth } from "../auth/AuthContext";
import { Button, Card } from "../ui";

export function AppLockGate({ children }: { children: ReactNode }) {
  const [locked, setLocked] = useState(false);
  const [fallback, setFallback] = useState(false);
  const [unlocking, setUnlocking] = useState(false);
  const [unlockFailed, setUnlockFailed] = useState(false);
  const { mode, loginWithOidc } = useAuth();

  useEffect(() => {
    if (shouldLockAfterBackground(Date.now())) {
      setLocked(true);
      setFallback(!appLockSupported());
    }
    let timeout: number | undefined;
    const visibilityChanged = () => {
      if (document.visibilityState === "hidden") {
        if (!appLockEnabled()) return;
        rememberBackgroundTime(Date.now());
        if (timeout !== undefined) window.clearTimeout(timeout);
        timeout = window.setTimeout(() => {
          setLocked(true);
          setFallback(!appLockSupported());
        }, APP_LOCK_BACKGROUND_MS);
        return;
      }
      if (timeout !== undefined) window.clearTimeout(timeout);
      if (shouldLockAfterBackground(Date.now())) {
        setLocked(true);
        setFallback(!appLockSupported());
      } else {
        clearBackgroundTime();
      }
    };
    document.addEventListener("visibilitychange", visibilityChanged);
    return () => {
      if (timeout !== undefined) window.clearTimeout(timeout);
      document.removeEventListener("visibilitychange", visibilityChanged);
    };
  }, []);

  const signInAgain = useCallback(() => {
    if (mode === "oidc") void loginWithOidc();
    else window.location.assign("/login");
  }, [loginWithOidc, mode]);

  async function unlock(): Promise<void> {
    setUnlocking(true);
    setUnlockFailed(false);
    try {
      const success = await unlockDeviceLock();
      if (!success) {
        setFallback(true);
        setUnlockFailed(true);
        return;
      }
      clearBackgroundTime();
      setLocked(false);
    } catch {
      setFallback(true);
      setUnlockFailed(true);
    } finally {
      setUnlocking(false);
    }
  }

  return (
    <>
      <div
        aria-hidden={locked}
        inert={locked}
        className={locked ? "pointer-events-none select-none blur-lg" : ""}
      >
        {children}
      </div>
      {locked ? (
        <div
          className="fixed inset-0 z-[100] grid place-items-center bg-background/95 p-5 backdrop-blur"
          role="dialog"
          aria-modal="true"
          aria-labelledby="app-lock-title"
        >
          <Card className="w-full max-w-sm text-center">
            <h1 id="app-lock-title" className="text-xl font-semibold">
              {englishCopy.settings.privacy.lock_locked}
            </h1>
            <p className="mt-2 text-sm text-muted-foreground">
              {englishCopy.settings.privacy.lock_sentence}
            </p>
            {unlockFailed ? (
              <p role="alert" className="mt-3 text-sm text-danger-foreground">
                {englishCopy.settings.privacy.unlock_failed}
              </p>
            ) : null}
            <Button
              className="mt-5 w-full"
              loading={unlocking}
              onClick={fallback ? signInAgain : () => void unlock()}
            >
              {fallback
                ? englishCopy.settings.privacy.sign_in
                : englishCopy.settings.privacy.unlock}
            </Button>
          </Card>
        </div>
      ) : null}
    </>
  );
}
