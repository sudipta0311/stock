import { Compass, House, Inbox, Lightbulb, WalletCards } from "lucide-react";
import { NavLink } from "react-router";

const tabs = [
  { to: "/", label: "Home", Icon: House, end: true },
  { to: "/money", label: "Money", Icon: WalletCards },
  { to: "/plan", label: "Plan", Icon: Compass },
  { to: "/ideas", label: "Ideas", Icon: Lightbulb },
  { to: "/inbox", label: "Inbox", Icon: Inbox },
];

export function BottomNavigation() {
  return (
    <nav
      aria-label="Primary"
      className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-surface/95 pb-[env(safe-area-inset-bottom)] backdrop-blur"
    >
      <div className="mx-auto grid max-w-3xl grid-cols-5">
        {tabs.map(({ to, label, Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              `motion-safe:active:scale-[.98] flex min-h-16 flex-col items-center justify-center gap-0.5 px-1 text-nav font-medium transition ${isActive ? "text-brand" : "text-muted-foreground hover:text-foreground"}`
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={`grid size-9 place-items-center rounded-full ${isActive ? "bg-brand/10 text-brand" : ""}`}
                >
                  <Icon
                    aria-hidden="true"
                    className="size-5"
                    strokeWidth={isActive ? 2.5 : 2}
                  />
                </span>
                <span>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
