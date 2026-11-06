import { Outlet, useLocation } from "react-router";
import { AccountMenu } from "../account/AccountMenu";
import { HouseholdProvider } from "../household/HouseholdContext";
import * as englishCopy from "../copy/sections/index";

/**
 * Onboarding and join screens sit outside the app shell, but people still need to see
 * which Google account is signed in and be able to sign out (e.g. they picked the wrong one).
 */
export function OnboardingFrame() {
  const location = useLocation();
  const returnTo = (location.state as { returnTo?: unknown } | null)?.returnTo;
  const addingGoal =
    location.pathname === "/onboarding/goals" &&
    typeof returnTo === "string" &&
    returnTo.startsWith("/");
  const addingRisk =
    location.pathname === "/onboarding/risk" &&
    typeof returnTo === "string" &&
    returnTo.startsWith("/");
  return (
    <HouseholdProvider>
      <header className="sticky top-0 z-20 h-14 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-full max-w-xl items-center justify-between gap-3 px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <img
              src="/wealth-mark.svg"
              alt=""
              aria-hidden="true"
              className="size-6 shrink-0"
            />
            <span className="truncate text-sm font-semibold">
              {addingGoal
                ? englishCopy.household_ui.add_goal_title
                : addingRisk
                  ? englishCopy.household_ui.risk_home_title
                  : "Getting started"}
            </span>
          </div>
          <AccountMenu />
        </div>
      </header>
      <Outlet />
    </HouseholdProvider>
  );
}
