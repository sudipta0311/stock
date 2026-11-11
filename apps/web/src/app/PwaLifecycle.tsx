import { useEffect, useRef } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";
import { listQueuedAssetWrites } from "../money/offline-queue";
import { syncOfflineCacheSetting } from "../pwa/offline-cache";
import { useToast } from "../ui";

const AUTO_UPDATE_WINDOW_MS = 15_000;

export function isSafeMomentForUpdate({
  now,
  visibleAt,
  hasChanged,
  dialogOpen,
  queuedWriteCount,
}: {
  now: number;
  visibleAt: number;
  hasChanged: boolean;
  dialogOpen: boolean;
  queuedWriteCount: number;
}): boolean {
  const visibleFor = now - visibleAt;
  return (
    visibleFor >= 0 &&
    visibleFor <= AUTO_UPDATE_WINDOW_MS &&
    !hasChanged &&
    !dialogOpen &&
    queuedWriteCount === 0
  );
}

export function PwaLifecycle() {
  const { toast } = useToast();
  const pageState = useRef<{
    visibleAt: number | null;
    hasChanged: boolean;
  }>({ visibleAt: null, hasChanged: false });
  const {
    needRefresh: [needRefresh],
    updateServiceWorker,
  } = useRegisterSW({
    immediate: true,
    onRegisteredSW(_url, registration) {
      if (!registration) return;
      // A phone tab can stay open for days without navigating, so look for a new
      // version whenever the person comes back to it.
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible")
          registration.update().catch(() => undefined);
      });
    },
  });

  useEffect(() => {
    if (pageState.current.visibleAt === null) {
      pageState.current.visibleAt = Date.now();
    }
    const markChanged = () => {
      if (pageState.current) pageState.current.hasChanged = true;
    };
    const markVisible = () => {
      if (document.visibilityState === "visible" && pageState.current) {
        pageState.current.visibleAt = Date.now();
      }
    };
    document.addEventListener("input", markChanged, true);
    document.addEventListener("change", markChanged, true);
    document.addEventListener("visibilitychange", markVisible);
    return () => {
      document.removeEventListener("input", markChanged, true);
      document.removeEventListener("change", markChanged, true);
      document.removeEventListener("visibilitychange", markVisible);
    };
  }, []);

  useEffect(() => {
    syncOfflineCacheSetting();
    if (!needRefresh) return;

    let active = true;
    const showUpdateToast = () => {
      if (!active) return;
      toast({
        title: "New version is ready",
        description: "Tap to refresh and continue with the latest app.",
        duration: Infinity,
        action: {
          label: "Tap to refresh",
          altText: "Refresh to install the new version",
          onClick: () => void updateServiceWorker(true),
        },
      });
    };

    void listQueuedAssetWrites()
      .then((writes) => {
        if (!active) return;
        const state = pageState.current;
        const safe =
          state.visibleAt !== null &&
          isSafeMomentForUpdate({
            now: Date.now(),
            visibleAt: state.visibleAt,
            hasChanged: state.hasChanged,
            dialogOpen:
              document.querySelector(
                '[role="dialog"]:not([aria-hidden="true"]), dialog[open]',
              ) !== null,
            queuedWriteCount: writes.length,
          });
        if (safe) void updateServiceWorker(true);
        else showUpdateToast();
      })
      .catch(showUpdateToast);

    return () => {
      active = false;
    };
  }, [needRefresh, toast, updateServiceWorker]);

  return null;
}
