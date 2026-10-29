import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router";
import { ScreenError, ScreenLoading } from "../features/api";
import { apiPathAvailable } from "../api/feature-support";
import {
  createStatementInboxKey,
  forgetStatementPasswords,
  listStatementInboxKeys,
  revokeStatementInboxKey,
} from "./api";
import type { StatementInboxKeyCreated } from "./api";
import { useHousehold } from "../household/HouseholdContext";
import { Button, Card } from "../ui";

function formatDate(value: string | null): string {
  if (!value) return "No statements received yet.";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Last received recently.";
  return `Last received ${new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date)}.`;
}

async function scriptForKey(key: string): Promise<string> {
  const response = await fetch(
    `${import.meta.env.BASE_URL}gmail-statements/Code.gs`,
    { cache: "no-store" },
  );
  if (!response.ok) throw new Error("Script unavailable");
  const template = await response.text();
  const apiUrl = new URL(
    "/v1/statement-inbox/uploads",
    import.meta.env.VITE_API_URL || window.location.origin,
  ).href;
  const withApi = template.replace(
    /const API_URL = "[^"]*";/,
    `const API_URL = ${JSON.stringify(apiUrl)};`,
  );
  return withApi.replace(
    /const UPLOAD_KEY = "[^"]*";/,
    `const UPLOAD_KEY = ${JSON.stringify(key)};`,
  );
}

export function GmailStatementsSettingsPage() {
  const queryClient = useQueryClient();
  const { role } = useHousehold();
  const supported = apiPathAvailable("/v1/statement-inbox/keys");
  const canManage = role === "owner" || role === "partner";
  const keysQuery = useQuery({
    queryKey: ["statement-inbox", "keys"],
    queryFn: listStatementInboxKeys,
    enabled: supported,
  });
  const [label, setLabel] = useState("My Gmail");
  const [createdKey, setCreatedKey] = useState<StatementInboxKeyCreated | null>(
    null,
  );
  const [scriptText, setScriptText] = useState("");
  const [scriptVisible, setScriptVisible] = useState(false);
  const [notice, setNotice] = useState("");

  const createMutation = useMutation({
    mutationFn: () => createStatementInboxKey(label),
    onSuccess: async (key) => {
      setCreatedKey(key);
      setScriptText("");
      setScriptVisible(false);
      setNotice("Copy the setup script now. This key will not be shown again.");
      await queryClient.invalidateQueries({
        queryKey: ["statement-inbox", "keys"],
      });
    },
    onError: () => setNotice("The upload key could not be created. Try again."),
  });
  const revokeMutation = useMutation({
    mutationFn: revokeStatementInboxKey,
    onSuccess: async (_result, id) => {
      if (createdKey?.id === id) {
        setCreatedKey(null);
        setScriptText("");
        setScriptVisible(false);
      }
      setNotice("The upload key was revoked.");
      await queryClient.invalidateQueries({
        queryKey: ["statement-inbox", "keys"],
      });
    },
    onError: () => setNotice("The upload key could not be revoked. Try again."),
  });
  const forgetMutation = useMutation({
    mutationFn: forgetStatementPasswords,
    onSuccess: () => setNotice("Saved statement passwords were removed."),
    onError: () =>
      setNotice("Saved passwords could not be removed. Try again."),
  });

  async function revealScript(copy: boolean): Promise<void> {
    if (!createdKey) return;
    try {
      const text = scriptText || (await scriptForKey(createdKey.key));
      setScriptText(text);
      setScriptVisible(true);
      if (copy) {
        await navigator.clipboard.writeText(text);
        setNotice("The setup script was copied.");
      }
    } catch {
      setNotice(
        copy
          ? "The setup script could not be copied. Try showing it and copy the text."
          : "The setup script could not load. Try again.",
      );
    }
  }

  if (!supported) {
    return (
      <main className="mx-auto grid w-full max-w-xl gap-4 p-4">
        <Card className="grid gap-3">
          <h1 className="text-xl font-semibold">Get statements from Gmail</h1>
          <p className="text-sm text-muted-foreground">
            This feature is not available in this app version. Upload a
            statement in Money instead.
          </p>
          <Link
            className="min-h-12 text-sm font-semibold text-brand"
            to="/import"
          >
            Upload a statement in Money
          </Link>
        </Card>
      </main>
    );
  }
  if (keysQuery.isPending)
    return <ScreenLoading label="Loading Gmail settings" />;
  if (keysQuery.error)
    return <ScreenError title="Gmail settings could not load" />;

  return (
    <main className="mx-auto grid w-full max-w-xl gap-4 p-4">
      <header className="grid gap-2">
        <Link className="min-h-10 text-sm text-brand" to="/settings">
          Back to Settings
        </Link>
        <h1 className="text-xl font-semibold">Get statements from Gmail</h1>
        <p className="text-sm text-muted-foreground">
          Your own Google script sends statement PDFs. It cannot read or change
          your saved holdings.
        </p>
      </header>

      {canManage ? (
        <Card className="grid gap-3">
          <h2 className="text-base font-semibold">Connect one Gmail account</h2>
          <label
            className="grid gap-1 text-sm font-medium"
            htmlFor="gmail-label"
          >
            Name this connection
            <input
              id="gmail-label"
              maxLength={80}
              value={label}
              onChange={(event) => setLabel(event.currentTarget.value)}
              className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
            />
          </label>
          <Button
            className="w-full"
            loading={createMutation.isPending}
            disabled={!label.trim()}
            onClick={() => createMutation.mutate()}
          >
            Create an upload key
          </Button>
          <p className="text-xs text-muted-foreground">
            The key can send new statements. You can revoke it here at any time.
          </p>
        </Card>
      ) : null}

      {createdKey ? (
        <Card className="grid gap-3" aria-live="polite">
          <h2 className="text-base font-semibold">Your new key is ready</h2>
          <p className="text-sm text-muted-foreground">
            It appears in the script below. Copy it before leaving this screen.
          </p>
          <Button className="w-full" onClick={() => void revealScript(true)}>
            Copy setup script
          </Button>
          <Button
            variant="secondary"
            className="w-full"
            onClick={() => void revealScript(false)}
          >
            {scriptVisible ? "Refresh setup script" : "Show setup script"}
          </Button>
          {scriptVisible && scriptText ? (
            <pre className="max-h-96 overflow-auto rounded-xl bg-muted p-3 text-xs leading-5">
              <code>{scriptText}</code>
            </pre>
          ) : null}
        </Card>
      ) : null}

      <Card className="grid gap-3">
        <h2 className="text-base font-semibold">Finish the setup</h2>
        <ol className="grid list-decimal gap-2 pl-5 text-sm">
          <li>
            <a
              className="text-brand underline"
              href="https://script.google.com"
              target="_blank"
              rel="noreferrer"
            >
              Open script.google.com
            </a>
            .
          </li>
          <li>Start a project and paste the copied script.</li>
          <li>
            Choose setup from the run menu, then select your Google account.
          </li>
          <li>
            Approve access so the script can find and send statement PDFs.
          </li>
          <li>
            It checks your Gmail right away, then every morning. New statements
            appear in Inbox.
          </li>
        </ol>
        <p className="text-sm text-muted-foreground">
          Google may show an unverified app notice. It is your own script, so
          Google asks you to confirm access.
        </p>
      </Card>

      {keysQuery.data.length ? (
        <Card className="grid gap-3">
          <h2 className="text-base font-semibold">Gmail connections</h2>
          {keysQuery.data.map((key) => (
            <div
              key={key.id}
              className="flex items-center justify-between gap-3 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold">{key.label}</p>
                <p className="text-xs text-muted-foreground">
                  {formatDate(key.last_used_at)}
                </p>
              </div>
              {canManage ? (
                <Button
                  variant="secondary"
                  onClick={() => revokeMutation.mutate(key.id)}
                  loading={
                    revokeMutation.isPending &&
                    revokeMutation.variables === key.id
                  }
                  aria-label={`Revoke ${key.label}`}
                >
                  Revoke
                </Button>
              ) : null}
            </div>
          ))}
        </Card>
      ) : null}

      {canManage ? (
        <Card className="grid gap-3">
          <h2 className="text-base font-semibold">Saved statement passwords</h2>
          <p className="text-sm text-muted-foreground">
            Saved passwords are encrypted and used only to prepare a review.
          </p>
          <Button
            variant="secondary"
            className="w-full"
            loading={forgetMutation.isPending}
            onClick={() => forgetMutation.mutate()}
          >
            Forget saved passwords
          </Button>
        </Card>
      ) : null}

      {notice ? (
        <p role="status" className="text-sm text-muted-foreground">
          {notice}
        </p>
      ) : null}
      <Link className="min-h-10 text-sm text-brand underline" to="/import">
        Upload a statement in Money
      </Link>
    </main>
  );
}
