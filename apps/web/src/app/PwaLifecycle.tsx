import { useEffect } from "react";
import { useRegisterSW } from "virtual:pwa-register/react";
import { syncOfflineCacheSetting } from "../pwa/offline-cache";
import { useToast } from "../ui";

export function PwaLifecycle() {
  const { toast } = useToast();
  const {
    needRefresh: [needRefresh],
    updateServiceWorker,
  } = useRegisterSW({ immediate: true });

  useEffect(() => {
    syncOfflineCacheSetting();
    if (needRefresh) {
      toast({
        title: "New version is ready",
        description: "Tap to refresh and continue with the latest app.",
        action: {
          label: "Tap to refresh",
          altText: "Refresh to install the new version",
          onClick: () => void updateServiceWorker(true),
        },
      });
    }
  }, [needRefresh, toast, updateServiceWorker]);

  return null;
}
