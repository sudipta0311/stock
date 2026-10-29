import { useEffect, useState } from "react";
import { AlertCircle } from "lucide-react";
import { Link, Navigate, useNavigate } from "react-router";
import { Button, Card } from "../ui";
import {
  AUTH_NOTICE_STORAGE_KEY,
  AUTH_RETURN_PATH_KEY,
  useAuth,
} from "../auth/AuthContext";

function consumeReturnPath(fallback: string): string {
  try {
    const stored = window.sessionStorage.getItem(AUTH_RETURN_PATH_KEY);
    window.sessionStorage.removeItem(AUTH_RETURN_PATH_KEY);
    return stored?.startsWith("/") && !stored.startsWith("//")
      ? stored
      : fallback;
  } catch {
    return fallback;
  }
}

function consumeLoginNotice(): string | null {
  try {
    const stored = window.sessionStorage.getItem(AUTH_NOTICE_STORAGE_KEY);
    window.sessionStorage.removeItem(AUTH_NOTICE_STORAGE_KEY);
    if (stored === "expired") return "Please sign in again.";
    if (stored === "signed-out") return "You've signed out.";
  } catch {
    return null;
  }
  return null;
}

export function LoginPage() {
  const {
    mode,
    state,
    hasProfile,
    configurationError,
    loginWithDevToken,
    loginWithOidc,
  } = useAuth();
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice] = useState(consumeLoginNotice);
  if (state === "loading")
    return (
      <main className="grid min-h-screen place-items-center p-5">
        <p role="status">Checking your sign-in…</p>
      </main>
    );
  if (state === "authenticated" && hasProfile === null)
    return (
      <main className="grid min-h-screen place-items-center p-5">
        <p role="status">Checking your account…</p>
      </main>
    );
  if (state === "authenticated")
    return (
      <Navigate
        to={consumeReturnPath(hasProfile ? "/" : "/onboarding")}
        replace
      />
    );

  return (
    <main className="mx-auto grid min-h-screen w-full max-w-md content-center gap-5 p-5">
      <div>
        <p className="text-sm font-semibold uppercase tracking-[0.16em] text-brand">
          Wealth Copilot
        </p>
        <h1 className="mt-2 text-page-title font-semibold tracking-tight">
          A clearer view of your money.
        </h1>
        <p className="mt-2 text-muted-foreground">
          Sign in to see your Money Map and next steps.
        </p>
      </div>
      <Card>
        {notice ? (
          <p role="status" className="mb-4 text-sm text-muted-foreground">
            {notice}
          </p>
        ) : null}
        {configurationError ? (
          <p role="alert" className="mb-4 text-sm text-danger-foreground">
            <AlertCircle
              aria-hidden="true"
              className="mr-1.5 inline size-4 align-text-bottom"
            />
            {configurationError}
          </p>
        ) : null}
        {error ? (
          <p role="alert" className="mb-4 text-sm text-danger-foreground">
            <AlertCircle
              aria-hidden="true"
              className="mr-1.5 inline size-4 align-text-bottom"
            />
            {error}
          </p>
        ) : null}
        {mode === "dev" ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              setError(null);
              if (!token.trim())
                setError("Paste a local development token to continue.");
              else loginWithDevToken(token);
            }}
            className="grid gap-3"
          >
            <label htmlFor="dev-token" className="text-sm font-medium">
              Development token
            </label>
            <textarea
              id="dev-token"
              value={token}
              onChange={(event) => setToken(event.currentTarget.value)}
              rows={4}
              autoComplete="off"
              spellCheck={false}
              className="min-h-24 rounded-xl border border-line bg-surface p-3 text-sm outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30"
              placeholder="Paste a token generated using the API’s local development instructions"
            />
            <p className="text-sm text-muted-foreground">
              The app keeps this token only in memory and this browser session.
            </p>
            <Button type="submit">Continue</Button>
          </form>
        ) : (
          <div className="grid gap-3">
            <Button
              onClick={() =>
                void loginWithOidc().catch((reason: unknown) =>
                  setError(
                    reason instanceof Error
                      ? reason.message
                      : "Unable to start sign-in.",
                  ),
                )
              }
            >
              Continue with Google
            </Button>
            <p className="text-sm text-muted-foreground">
              Only invited family members can sign in.
            </p>
          </div>
        )}
      </Card>
      <Link
        className="min-h-12 text-center text-sm text-muted-foreground underline underline-offset-4"
        to="/"
      >
        Back to the app
      </Link>
    </main>
  );
}

export function AuthCallbackPage() {
  const { completeOidcLogin } = useAuth();
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void completeOidcLogin()
      .then(() => {
        if (active) navigate(consumeReturnPath("/"), { replace: true });
      })
      .catch((reason: unknown) => {
        if (active)
          setError(
            reason instanceof Error
              ? reason.message
              : "Sign-in could not be completed.",
          );
      });
    return () => {
      active = false;
    };
  }, [completeOidcLogin, navigate]);
  return (
    <main className="grid min-h-screen place-items-center p-5">
      <p
        role={error ? "alert" : "status"}
        className={error ? "text-danger-foreground" : ""}
      >
        {error ? (
          <AlertCircle
            aria-hidden="true"
            className="mr-1.5 inline size-4 align-text-bottom"
          />
        ) : null}
        {error ?? "Completing secure sign-in…"}
      </p>
    </main>
  );
}
