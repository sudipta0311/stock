import { useEffect, useState } from "react";
import { LoaderCircle } from "lucide-react";
import * as englishCopy from "../copy/sections/index";

export function WakeIndicator() {
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    const update = (event: Event) => {
      const detail = (event as CustomEvent<{ waking?: boolean }>).detail;
      setWaking(detail?.waking === true);
    };
    window.addEventListener("wealth:api-waking", update);
    return () => window.removeEventListener("wealth:api-waking", update);
  }, []);

  if (!waking) return null;
  return (
    <div
      role="status"
      aria-live="polite"
      className="fixed right-3 top-20 z-40 inline-flex min-h-8 items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 text-xs font-medium text-muted-foreground shadow-sm sm:right-5"
    >
      <LoaderCircle
        aria-hidden="true"
        className="size-4 animate-spin text-brand"
      />
      <span>{englishCopy.offline.updating}</span>
    </div>
  );
}
