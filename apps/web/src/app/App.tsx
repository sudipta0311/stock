import { Component, lazy, Suspense } from "react";
import type { ErrorInfo, ReactNode } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router";
import { AuthCallbackPage, LoginPage } from "./AuthPages";
import { PwaLifecycle } from "./PwaLifecycle";
import { RouteGuard } from "./RouteGuard";
import { WakeIndicator } from "./WakeIndicator";
import { Button, Card } from "../ui";

const OfflineWriteSync = lazy(() =>
  import("../money/OfflineWriteSync").then((module) => ({
    default: module.OfflineWriteSync,
  })),
);

const ProtectedHouseholdApp = lazy(() =>
  import("./ProtectedHouseholdApp").then((module) => ({
    default: module.ProtectedHouseholdApp,
  })),
);

const HouseholdJoinPage = lazy(() =>
  import("../household/HouseholdJoinPage").then((module) => ({
    default: module.HouseholdJoinPage,
  })),
);

const MoneyMapPage = lazy(() =>
  import("../money/MoneyMapPage").then((module) => ({
    default: module.MoneyMapPage,
  })),
);
const AddAssetPage = lazy(() =>
  import("../money/AssetFormPage").then((module) => ({
    default: module.AddAssetPage,
  })),
);
const AssetDetailPage = lazy(() =>
  import("../money/AssetFormPage").then((module) => ({
    default: module.AssetDetailPage,
  })),
);
const LoanDetailPage = lazy(() =>
  import("../money/LoanDetailPage").then((module) => ({
    default: module.LoanDetailPage,
  })),
);
const OnboardingFrame = lazy(() =>
  import("../onboarding/OnboardingFrame").then((module) => ({
    default: module.OnboardingFrame,
  })),
);
const OnboardingPage = lazy(() =>
  import("../onboarding/OnboardingPage").then((module) => ({
    default: module.OnboardingPage,
  })),
);
const ImportPage = lazy(() =>
  import("../imports/ImportPage").then((module) => ({
    default: module.ImportPage,
  })),
);
const ImportReviewPage = lazy(() =>
  import("../imports/ImportPage").then((module) => ({
    default: module.ImportReviewPage,
  })),
);
const HomePage = lazy(() =>
  import("../home/HomePage").then((module) => ({ default: module.HomePage })),
);
const PlanPage = lazy(() =>
  import("../plan/PlanPage").then((module) => ({ default: module.PlanPage })),
);
const IdeasPage = lazy(() =>
  import("../ideas/IdeasPage").then((module) => ({
    default: module.IdeasPage,
  })),
);
const WorldPage = lazy(() =>
  import("../ideas/WorldPage").then((module) => ({
    default: module.WorldPage,
  })),
);
const HoldingPage = lazy(() =>
  import("../money/HoldingPage").then((module) => ({
    default: module.HoldingPage,
  })),
);
const InboxPage = lazy(() =>
  import("../inbox/InboxPage").then((module) => ({
    default: module.InboxPage,
  })),
);
const BriefPage = lazy(() =>
  import("../inbox/InboxPage").then((module) => ({
    default: module.BriefPage,
  })),
);
const SettingsPage = lazy(() =>
  import("../settings/SettingsPage").then((module) => ({
    default: module.SettingsPage,
  })),
);
const GmailStatementsSettingsPage = lazy(() =>
  import("../settings/GmailStatementsSettingsPage").then((module) => ({
    default: module.GmailStatementsSettingsPage,
  })),
);

const HelpPage = lazy(() =>
  import("../help/HelpPage").then((module) => ({
    default: module.HelpPage,
  })),
);

class AppErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError(): { hasError: boolean } {
    return { hasError: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error(
      "Wealth Copilot screen error",
      error.name,
      info.componentStack,
    );
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="grid min-h-screen place-items-center p-5">
          <Card className="w-full max-w-md text-center">
            <h1 className="text-xl font-semibold">Something didn’t load</h1>
            <p className="mt-2 text-sm text-muted-foreground">
              Your information is safe. Try loading this screen again.
            </p>
            <Button
              className="mt-5 w-full"
              onClick={() => this.setState({ hasError: false })}
            >
              Try again
            </Button>
          </Card>
        </main>
      );
    }
    return this.props.children;
  }
}

function ProtectedApp() {
  return (
    <RouteGuard>
      <Outlet />
    </RouteGuard>
  );
}

function ProfileProtectedApp() {
  return (
    <RouteGuard requiresProfile={false}>
      <OnboardingFrame />
    </RouteGuard>
  );
}

function RouteLoading() {
  return (
    <main
      className="mx-auto grid min-h-48 w-full max-w-xl content-center p-4"
      aria-label="Loading page"
    >
      <Card>
        <p className="text-sm text-muted-foreground">Loading this screen…</p>
      </Card>
    </main>
  );
}

export function App() {
  return (
    <AppErrorBoundary>
      <BrowserRouter>
        <PwaLifecycle />
        <Suspense fallback={null}>
          <OfflineWriteSync />
        </Suspense>
        <WakeIndicator />
        <Suspense fallback={<RouteLoading />}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/auth/callback" element={<AuthCallbackPage />} />
            <Route element={<ProfileProtectedApp />}>
              <Route path="/onboarding/*" element={<OnboardingPage />} />
              <Route path="/join" element={<HouseholdJoinPage />} />
            </Route>
            <Route element={<ProtectedApp />}>
              <Route element={<ProtectedHouseholdApp />}>
                <Route index element={<HomePage />} />
                <Route path="money" element={<MoneyMapPage />} />
                <Route path="money/add/:type" element={<AddAssetPage />} />
                <Route path="money/asset/:id" element={<AssetDetailPage />} />
                <Route path="money/loan/:id" element={<LoanDetailPage />} />
                <Route path="money/holding/:id" element={<HoldingPage />} />
                <Route path="import" element={<ImportPage />} />
                <Route path="import/:jobId" element={<ImportReviewPage />} />
                <Route path="plan" element={<PlanPage />} />
                <Route path="ideas" element={<IdeasPage />} />
                <Route path="ideas/world" element={<WorldPage />} />
                <Route path="inbox" element={<InboxPage />} />
                <Route path="inbox/brief/:week" element={<BriefPage />} />
                <Route path="settings" element={<SettingsPage />} />
                <Route
                  path="settings/gmail"
                  element={<GmailStatementsSettingsPage />}
                />
                <Route path="help" element={<HelpPage />} />
              </Route>
            </Route>
            <Route path="/" element={<Navigate to="/login" replace />} />
            <Route
              path="*"
              element={
                <main className="mx-auto grid min-h-screen max-w-lg content-center p-6">
                  <Card>
                    <h1 className="text-xl font-semibold">Page not found</h1>
                    <p className="mt-2 text-sm text-muted-foreground">
                      That page may have moved. Go back to your account to
                      continue.
                    </p>
                    <Button
                      className="mt-4 w-full"
                      onClick={() => window.location.assign("/")}
                    >
                      Try again
                    </Button>
                  </Card>
                </main>
              }
            />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AppErrorBoundary>
  );
}
