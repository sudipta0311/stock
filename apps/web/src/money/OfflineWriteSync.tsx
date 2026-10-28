import { useEffect } from "react";
import { replayOfflineWrites } from "./offline-queue";

export function OfflineWriteSync() {
  useEffect(() => {
    const replay = () => void replayOfflineWrites();
    window.addEventListener("online", replay);
    if (navigator.onLine) replay();
    return () => window.removeEventListener("online", replay);
  }, []);

  return null;
}
