import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import englishCopy from "../copy/en.json";

export function WakeIndicator() {
  const [waking, setWaking] = useState(false);
  const [longWait, setLongWait] = useState(false);
  useEffect(() => {
    const update = (event: Event) => {
      const detail = (event as CustomEvent<{ waking?: boolean }>).detail;
      setWaking(detail?.waking === true);
    };
    window.addEventListener("wealth:api-waking", update);
    return () => window.removeEventListener("wealth:api-waking", update);
  }, []);

  useEffect(() => {
    if (!waking) {
      setLongWait(false);
      return;
    }
    const timeout = window.setTimeout(() => setLongWait(true), 20_000);
    return () => window.clearTimeout(timeout);
  }, [waking]);

  if (!waking) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed right-3 top-20 z-40 inline-flex min-h-10 items-center gap-2 rounded-full border border-line bg-surface px-3 text-xs font-medium text-foreground shadow-md sm:right-5"
    >
      <LoaderCircle
        aria-hidden="true"
        className="size-4 animate-spin text-brand"
      />
      <span>
        {longWait
          ? englishCopy.offline.long_wake
          : englishCopy.offline.short_wake}
      </span>
    </div>
  );
}
