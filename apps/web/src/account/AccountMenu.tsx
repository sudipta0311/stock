import { lazy, Suspense, useState } from "react";
import { useHousehold } from "../household/HouseholdContext";

const AccountMenuDialog = lazy(() =>
  import("./AccountMenuDialog").then((module) => ({
    default: module.AccountMenuDialog,
  })),
);

function accountName(
  displayName: string | null | undefined,
  email: string | undefined,
): string {
  return displayName?.trim() || email?.split("@")[0] || "Account";
}

function nameInitials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part.charAt(0))
    .join("")
    .toLocaleUpperCase();
}

export function AccountMenu() {
  const { me } = useHousehold();
  const [open, setOpen] = useState(false);
  const name = accountName(me?.display_name, me?.email);
  const firstName = name.split(/\s+/)[0] || "Account";

  return (
    <>
      <button
        type="button"
        // The name starts with what the button shows (initials, first name) so voice
        // control users can say what they see (WCAG 2.5.3, Label in Name).
        aria-label={`${nameInitials(name)} ${firstName}, account menu for ${name}`}
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen(true)}
        className="motion-safe:active:scale-[.98] flex min-h-11 shrink-0 items-center gap-2 rounded-full border border-line bg-background px-1.5 pr-2 text-sm font-medium transition hover:bg-muted"
      >
        <span className="grid size-8 place-items-center rounded-full bg-brand/10 text-xs font-semibold text-brand">
          {nameInitials(name)}
        </span>
        <span className="hidden max-w-24 truncate min-[400px]:inline">
          {firstName}
        </span>
      </button>
      {open ? (
        <Suspense fallback={null}>
          <AccountMenuDialog open={open} onOpenChange={setOpen} />
        </Suspense>
      ) : null}
    </>
  );
}
