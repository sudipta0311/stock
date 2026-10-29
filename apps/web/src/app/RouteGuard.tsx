import { Navigate, useLocation } from "react-router";
import type { ReactNode } from "react";
import { EmptyState, Skeleton } from "../ui";
import { AUTH_RETURN_PATH_KEY, useAuth } from "../auth/AuthContext";

export function RouteGuard({
  children,
  requiresProfile = true,
}: {
  children: ReactNode;
  requiresProfile?: boolean;
}) {
  const { state, hasProfile } = useAuth();
  const location = useLocation();
  if (state === "loading") {
    return (
      <main
        className="mx-auto grid min-h-screen max-w-lg content-center gap-4 p-6"
        aria-label="Loading your account"
      >
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-24 w-full" />
      </main>
    );
  }
  if (state === "anonymous") {
    try {
      window.sessionStorage.setItem(
        AUTH_RETURN_PATH_KEY,
        `${location.pathname}${location.search}${location.hash}`,
      );
    } catch {
      // Sign-in can still continue to the default screen.
    }
    return <Navigate to="/login" replace />;
  }
  if (requiresProfile && hasProfile === null) {
    return (
      <main className="mx-auto grid min-h-screen max-w-lg content-center p-6">
        <EmptyState
          title="Checking your account"
          description="We’re getting your account ready. This can take a moment when the service has been idle."
        />
      </main>
    );
  }
  if (requiresProfile && hasProfile === false)
    return <Navigate to="/onboarding" replace />;
  return children;
}
