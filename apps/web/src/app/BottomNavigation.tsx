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
              `flex min-h-16 flex-col items-center justify-center gap-1 px-1 text-xs transition ${isActive ? "font-semibold text-brand" : "text-muted-foreground hover:text-foreground"}`
            }
          >
            {({ isActive }) => (
              <>
                <Icon
                  aria-hidden="true"
                  className="size-5"
                  strokeWidth={isActive ? 2.5 : 2}
                />
                <span>{label}</span>
              </>
            )}
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
