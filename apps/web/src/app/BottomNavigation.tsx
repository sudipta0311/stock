import { Compass, House, Inbox, Lightbulb, WalletCards } from "lucide-react";
import { Link, useLocation } from "react-router";

const tabs = [
  { to: "/", label: "Home", Icon: House, end: true },
  { to: "/money", label: "Money", Icon: WalletCards },
  { to: "/plan", label: "Plan", Icon: Compass },
  { to: "/ideas", label: "Ideas", Icon: Lightbulb },
  { to: "/inbox", label: "Inbox", Icon: Inbox },
];

export function BottomNavigation() {
  const { pathname } = useLocation();
  const importPage = pathname === "/import" || pathname.startsWith("/import/");
  return (
    <nav
      aria-label="Primary"
      className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-surface/95 pb-[env(safe-area-inset-bottom)] backdrop-blur"
    >
      <div className="mx-auto grid max-w-3xl grid-cols-5">
        {tabs.map(({ to, label, Icon, end }) => {
          const active =
            pathname === to ||
            (!end && pathname.startsWith(to + "/")) ||
            (to === "/money" && importPage);
          return (
            <Link
              key={to}
              to={to}
              aria-current={active ? "page" : undefined}
              className={[
                "motion-safe:active:scale-[.98] flex min-h-16 flex-col items-center justify-center gap-0.5 px-1 text-nav font-medium transition",
                active
                  ? "text-brand"
                  : "text-muted-foreground hover:text-foreground",
              ].join(" ")}
            >
              <span
                className={[
                  "grid size-9 place-items-center rounded-full",
                  active ? "bg-brand/10 text-brand" : "",
                ].join(" ")}
              >
                <Icon
                  aria-hidden="true"
                  className="size-5"
                  strokeWidth={active ? 2.5 : 2}
                />
              </span>
              <span>{label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
