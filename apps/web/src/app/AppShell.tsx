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
      <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex min-h-16 max-w-3xl items-center justify-between gap-3 px-4">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              Wealth Copilot
            </p>
            <h1 className="text-lg font-semibold leading-tight">{title}</h1>
          </div>
          <div className="flex items-center gap-2">
            <HouseholdControls />
            <Link
              to="/settings"
              aria-label="Settings"
              className="grid size-12 place-items-center rounded-full border border-line bg-background text-foreground hover:bg-muted"
            >
              <CircleUserRound aria-hidden="true" className="size-6" />
            </Link>
          </div>
        </div>
      </header>
      <OfflineStatus />
      <InstallExperience />
      <main className="mx-auto grid w-full max-w-3xl gap-4 px-4 py-5 sm:px-6 sm:py-8">
        <Outlet />
      </main>
      <BottomNavigation />
    </div>
  );
}
