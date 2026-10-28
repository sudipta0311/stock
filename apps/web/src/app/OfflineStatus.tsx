import { useEffect, useState } from "react";
import englishCopy from "../copy/en.json";
import { getOfflineSnapshotTime } from "../pwa/offline-cache";
import { listQueuedAssetWrites } from "../money/offline-queue";
import { CHECKLIST_PENDING_KEY } from "../money/config";

function currentOnlineStatus(): boolean {
  return typeof navigator === "undefined" || navigator.onLine;
}

export function OfflineStatus() {
  const [online, setOnline] = useState(currentOnlineStatus);
  const [pending, setPending] = useState(0);
  const [snapshotTime, setSnapshotTime] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const updatePending = () => {
      void listQueuedAssetWrites()
        .then((writes) => {
          let checklistPending = 0;
          try {
            checklistPending = window.localStorage.getItem(
              CHECKLIST_PENDING_KEY,
            )
              ? 1
              : 0;
          } catch {
            checklistPending = 0;
          }
          if (active) setPending(writes.length + checklistPending);
        })
        .catch(() => {
          if (active) setPending(0);
        });
    };
    const updateNetwork = () => setOnline(currentOnlineStatus());
    window.addEventListener("online", updateNetwork);
    window.addEventListener("offline", updateNetwork);
    window.addEventListener("wealth:pending-changes", updatePending);
    updatePending();
    return () => {
      active = false;
      window.removeEventListener("online", updateNetwork);
      window.removeEventListener("offline", updateNetwork);
      window.removeEventListener("wealth:pending-changes", updatePending);
    };
  }, []);

  useEffect(() => {
    if (online) {
      setSnapshotTime(null);
      return;
    }
    let active = true;
    void getOfflineSnapshotTime().then((time) => {
      if (active) setSnapshotTime(time);
    });
    return () => {
      active = false;
    };
  }, [online]);

  if (online && pending === 0) return null;
  return (
    <div
      className="mx-auto grid w-full max-w-3xl gap-2 px-4 pt-3 sm:px-6"
      aria-live="polite"
    >
      {!online ? (
        <p
          role="status"
          className="rounded-xl border border-warning/40 bg-warning-soft px-3 py-2 text-sm text-warning-foreground"
        >
          {englishCopy.offline.banner}
          {snapshotTime
            ? ` ${englishCopy.offline.snapshot_time.replace("{time}", snapshotTime)}`
            : ""}
        </p>
      ) : null}
      {pending > 0 ? (
        <p
          role="status"
          className="inline-flex w-fit min-h-8 items-center rounded-full bg-muted px-3 text-xs font-medium text-foreground"
        >
          {(pending === 1
            ? englishCopy.offline.pending
            : englishCopy.offline.pending_many
          ).replace("{count}", String(pending))}
        </p>
      ) : null}
    </div>
  );
}
