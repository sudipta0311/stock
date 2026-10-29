import { CircleUserRound } from "lucide-react";
import { Link, Outlet, useLocation } from "react-router";
import { BottomNavigation } from "./BottomNavigation";
import { OfflineStatus } from "./OfflineStatus";
import { InstallExperience } from "./InstallExperience";
import { HouseholdControls } from "../household/HouseholdControls";

const pageTitles: Array<[string, string]> = [
  ["/money", "Money"],
  ["/plan", "Plan"],
  ["/ideas", "Ideas"],
  ["/inbox", "Inbox"],
  ["/settings", "Settings"],
  ["/import", "Import"],
];

function titleForPath(pathname: string): string {
  if (pathname === "/") return "Home";
  return (
    pageTitles.find(
      ([prefix]) => pathname === prefix || pathname.startsWith(`${prefix}/`),
    )?.[1] ?? "Wealth Copilot"
  );
}

export function AppShell() {
  return <AppShellLayout />;
}

function AppShellLayout() {
  const location = useLocation();
  const title = titleForPath(location.pathname);

  return (
    <div className="min-h-screen bg-background pb-[calc(5.5rem+env(safe-area-inset-bottom))] text-foreground">
      <header className="sticky top-0 z-20 h-14 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-full max-w-3xl items-center justify-between gap-3 px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <img
              src="/wealth-mark.svg"
              alt=""
              aria-hidden="true"
              className="size-6 shrink-0"
            />
            <h1 className="truncate text-page-title font-semibold">{title}</h1>
          </div>
          <Link
            to="/settings"
            aria-label="Settings"
            className="motion-safe:active:scale-[.98] grid size-11 shrink-0 place-items-center rounded-full border border-line bg-background text-foreground transition hover:bg-muted"
          >
            <CircleUserRound aria-hidden="true" className="size-5" />
          </Link>
        </div>
      </header>
      <OfflineStatus />
      <InstallExperience />
      <main className="mx-auto grid w-full max-w-3xl gap-4 px-4 py-4 sm:px-6 sm:py-6">
        <HouseholdControls />
        <Outlet />
      </main>
      <BottomNavigation />
    </div>
  );
}
