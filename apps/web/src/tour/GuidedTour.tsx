import { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { Button } from "../ui";
import { TOUR_SEEN_STORAGE_PREFIX } from "../pwa/storage-keys";

const steps = [
  {
    path: "/",
    label: "Home",
    sentence: "Home keeps your next steps together.",
  },
  {
    path: "/money",
    label: "Money",
    sentence: "Add what you have, even if the numbers are rough.",
  },
  {
    path: "/plan",
    label: "Plan",
    sentence: "Your plan compares your money with your targets.",
  },
  {
    path: "/ideas",
    label: "Ideas",
    sentence: "Ideas help you review choices at your own pace.",
  },
] as const;

export function tourSeenKey(userId: string): string {
  return `${TOUR_SEEN_STORAGE_PREFIX}${userId}`;
}

function hasSeenTour(userId: string): boolean {
  try {
    return window.localStorage.getItem(tourSeenKey(userId)) === "true";
  } catch {
    return false;
  }
}

function markTourSeen(userId: string): void {
  try {
    window.localStorage.setItem(tourSeenKey(userId), "true");
  } catch {
    // The tour remains dismissible for this visit when storage is unavailable.
  }
}

export function GuidedTour({ userId }: { userId: string }) {
  const navigate = useNavigate();
  const location = useLocation();
  const dialogRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [dismissedThisVisit, setDismissedThisVisit] = useState(false);
  const step = steps[stepIndex] ?? steps[0];
  const closeTour = useCallback(() => {
    markTourSeen(userId);
    setDismissedThisVisit(true);
    setOpen(false);
  }, [userId]);

  useEffect(() => {
    if (
      !dismissedThisVisit &&
      location.pathname === "/" &&
      !hasSeenTour(userId)
    ) {
      setStepIndex(0);
      setOpen(true);
    }
  }, [dismissedThisVisit, location.pathname, userId]);

  useEffect(() => {
    function replay() {
      setStepIndex(0);
      setDismissedThisVisit(false);
      setOpen(true);
      navigate("/");
    }
    window.addEventListener("wealth:show-tour", replay);
    return () => window.removeEventListener("wealth:show-tour", replay);
  }, [navigate]);

  useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement as HTMLElement | null;
    dialogRef.current
      ?.querySelector<HTMLElement>("button:not(:disabled)")
      ?.focus();
    function trapFocus(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        event.preventDefault();
        closeTour();
        return;
      }
      if (event.key !== "Tab") return;
      const controls = dialogRef.current?.querySelectorAll<HTMLElement>(
        "button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex='-1'])",
      );
      if (!controls?.length) {
        event.preventDefault();
        return;
      }
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    }
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      previousFocus?.focus();
    };
  }, [closeTour, open]);

  function next(): void {
    const nextIndex = stepIndex + 1;
    if (nextIndex >= steps.length) {
      closeTour();
      return;
    }
    setStepIndex(nextIndex);
    navigate(steps[nextIndex]?.path ?? "/");
  }

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[70] grid items-end bg-black/45 p-3 pb-[calc(1rem+env(safe-area-inset-bottom))] motion-safe:transition-opacity sm:items-center sm:justify-center">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tour-title"
        aria-describedby="tour-sentence"
        className="grid w-full max-w-md gap-4 rounded-2xl border border-line bg-surface p-5 shadow-xl motion-reduce:transition-none"
      >
        <div className="flex items-center justify-between gap-3">
          <h2 id="tour-title" className="text-lg font-semibold">
            {step.label}
          </h2>
          <span className="text-sm text-muted-foreground">
            {stepIndex + 1} of {steps.length}
          </span>
        </div>
        <p id="tour-sentence" className="text-base text-muted-foreground">
          {step.sentence}
        </p>
        <ol
          className="flex items-center justify-center gap-2"
          aria-label="Tour progress"
        >
          {steps.map((item, index) => (
            <li key={item.label}>
              <span
                aria-current={index === stepIndex ? "step" : undefined}
                aria-label={`Step ${index + 1}${index === stepIndex ? ", current" : ""}`}
                className={`block size-2.5 rounded-full ${index === stepIndex ? "bg-brand" : "bg-muted-foreground/40"}`}
              />
            </li>
          ))}
        </ol>
        <div className="grid grid-cols-2 gap-2">
          <Button type="button" variant="secondary" onClick={closeTour}>
            Skip
          </Button>
          <Button type="button" onClick={next}>
            {stepIndex === steps.length - 1 ? "Finish" : "Next"}
          </Button>
        </div>
      </div>
    </div>
  );
}
