import { useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router";
import { ApiError, apiFetch } from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import { Button, Card, Checkbox } from "../ui";

type SenderType = "cams" | "kfintech" | "mfcentral" | "nsdl" | "cdsl" | "other";
type ItemStatus = "needs_password" | "review" | "unreadable";

interface StatementInboxItem {
  id: string;
  sender_type: SenderType;
  received_at: string;
  status: ItemStatus;
  error_code: string | null;
  import_job_id: string | null;
  subject?: string;
  full_statement?: boolean;
  saved_password?: boolean;
}

interface StatementInboxData {
  items: StatementInboxItem[];
  cleanup_notice: string | null;
}

interface ProcessResponse {
  status: "review";
  job_id: string;
  password_saved: boolean;
  notice: string | null;
}

const senderNames: Record<SenderType, string> = {
  cams: "CAMS",
  kfintech: "KFintech",
  mfcentral: "MFCentral",
  nsdl: "NSDL",
  cdsl: "CDSL",
  other: "your provider",
};

const NOT_A_FULL_STATEMENT =
  "This file isn’t a full account statement, so there’s nothing to import.";

function statementDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
      }).format(date);
}

// Fund houses also email purchase notes and single-fund statements. Only the full
// (consolidated) statement can be imported, so the others wait in a closed group.
function isOtherEmail(item: StatementInboxItem): boolean {
  return (
    item.error_code === "not_a_full_statement" ||
    (item.full_statement === false && item.status !== "review")
  );
}

export function GmailStatementInbox() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const supported = apiPathAvailable("/v1/statement-inbox/items");
  const query = useQuery({
    queryKey: ["statement-inbox", "items"],
    queryFn: () => apiFetch<StatementInboxData>("/v1/statement-inbox/items"),
    enabled: supported,
  });
  const [openId, setOpenId] = useState<string | null>(null);
  const [passwords, setPasswords] = useState<Record<string, string>>({});
  const [remember, setRemember] = useState<Record<string, boolean>>({});
  const [feedback, setFeedback] = useState<Record<string, string>>({});
  const [results, setResults] = useState<Record<string, ProcessResponse>>({});
  const refresh = () =>
    queryClient.invalidateQueries({ queryKey: ["statement-inbox", "items"] });
  const processMutation = useMutation({
    mutationFn: ({
      id,
      password,
      rememberPassword,
      useSaved = false,
    }: {
      id: string;
      password: string;
      rememberPassword: boolean;
      useSaved?: boolean;
    }) =>
      apiFetch<ProcessResponse>(
        `/v1/statement-inbox/items/${encodeURIComponent(id)}/process`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            useSaved
              ? { use_saved_password: true }
              : { password, remember: rememberPassword },
          ),
        },
      ),
    onSuccess: async (result, variables) => {
      setResults((current) => ({ ...current, [variables.id]: result }));
      setPasswords((current) => ({ ...current, [variables.id]: "" }));
      setFeedback((current) => ({
        ...current,
        [variables.id]:
          result.notice ??
          (result.password_saved
            ? `Password saved for ${
                senderNames[
                  query.data?.items.find((item) => item.id === variables.id)
                    ?.sender_type ?? "other"
                ]
              } statements.`
            : ""),
      }));
      setOpenId(null);
      await refresh();
    },
    onError: async (error, variables) => {
      const code = error instanceof ApiError ? error.code : "";
      const message =
        code === "wrong_password"
          ? variables.useSaved
            ? "The saved password didn’t work. Enter it to continue."
            : "That password didn’t work. Please try again."
          : code === "no_saved_password"
            ? "There’s no saved password for these statements. Enter it instead."
            : code === "not_a_full_statement"
              ? NOT_A_FULL_STATEMENT
              : code === "unreadable"
                ? "We couldn’t read this file. Upload it in Money instead."
                : "The statement could not be opened. Try again.";
      setFeedback((current) => ({ ...current, [variables.id]: message }));
      if (code === "not_a_full_statement") {
        setOpenId(null);
        await refresh();
      }
    },
  });
  const removeMutation = useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/statement-inbox/items/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: refresh,
    onError: (_error, id) =>
      setFeedback((current) => ({
        ...current,
        [id]: "This email couldn’t be removed. Try again.",
      })),
  });

  if (!supported) return null;
  if (query.isPending) return null;
  if (query.isError || !query.data) {
    return (
      <Card className="grid gap-2">
        <p className="text-sm text-muted-foreground">
          New statements couldn’t load. Upload a statement in Money instead.
        </p>
        <Link className="text-sm font-semibold text-brand" to="/import">
          Upload in Money
        </Link>
      </Card>
    );
  }
  if (query.data.items.length === 0 && !query.data.cleanup_notice) return null;

  const fullItems = query.data.items.filter((item) => !isOtherEmail(item));
  const otherItems = query.data.items.filter(isOtherEmail);

  const submitPassword = (event: FormEvent<HTMLFormElement>, id: string) => {
    event.preventDefault();
    processMutation.mutate({
      id,
      password: passwords[id] ?? "",
      rememberPassword: remember[id] ?? false,
    });
  };

  const passwordForm = (item: StatementInboxItem): ReactNode => (
    <form
      className="grid gap-3"
      onSubmit={(event) => submitPassword(event, item.id)}
    >
      <label
        className="grid gap-1 text-sm font-medium"
        htmlFor={`statement-password-${item.id}`}
      >
        Statement password
        <input
          id={`statement-password-${item.id}`}
          type="password"
          autoComplete="current-password"
          required
          value={passwords[item.id] ?? ""}
          onChange={(event) => {
            const value = event.currentTarget.value;
            setPasswords((current) => ({ ...current, [item.id]: value }));
          }}
          className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
        />
      </label>
      <Checkbox
        id={`remember-statement-password-${item.id}`}
        label={`Remember this password for ${senderNames[item.sender_type]} statements`}
        checked={remember[item.id] ?? false}
        onChange={(event) => {
          const checked = event.currentTarget.checked;
          setRemember((current) => ({ ...current, [item.id]: checked }));
        }}
      />
      <Button
        type="submit"
        className="w-full"
        loading={processMutation.isPending}
        disabled={!passwords[item.id]}
      >
        Review statement
      </Button>
    </form>
  );

  const openForm = (id: string) => {
    setOpenId(id);
    setFeedback((current) => ({ ...current, [id]: "" }));
  };

  const removeButton = (item: StatementInboxItem): ReactNode => (
    <Button
      variant="ghost"
      onClick={() => removeMutation.mutate(item.id)}
      loading={removeMutation.isPending && removeMutation.variables === item.id}
    >
      Remove
    </Button>
  );

  const heading = (item: StatementInboxItem, title: string): ReactNode => {
    const date = statementDate(item.received_at);
    return (
      <div className="min-w-0">
        <h3 className="text-sm font-semibold">{title}</h3>
        {item.subject ? (
          <p className="mt-1 truncate text-xs text-muted-foreground">
            {item.subject}
          </p>
        ) : null}
        {date ? (
          <p className="mt-1 text-xs text-muted-foreground">Received {date}</p>
        ) : null}
      </div>
    );
  };

  const message = (item: StatementInboxItem): ReactNode => {
    const savedPasswordMessage =
      item.error_code === "wrong_password"
        ? "A saved password didn’t work. Enter it to continue."
        : item.error_code === "parse_timeout"
          ? "This statement took too long to read. Try again."
          : item.error_code === "unreadable" && item.status === "needs_password"
            ? "We couldn’t read this statement before. Try again, or upload it in Money."
            : "";
    if (feedback[item.id]) {
      return (
        <p role="alert" className="text-sm text-danger-foreground">
          {feedback[item.id]}
        </p>
      );
    }
    if (!savedPasswordMessage) return null;
    return (
      <p
        role={item.error_code === "wrong_password" ? "alert" : "status"}
        className={`text-sm ${item.error_code === "wrong_password" ? "text-danger-foreground" : "text-muted-foreground"}`}
      >
        {savedPasswordMessage}
      </p>
    );
  };

  return (
    <section
      aria-labelledby="gmail-statement-inbox-heading"
      className="grid gap-3"
    >
      <header>
        <h2
          id="gmail-statement-inbox-heading"
          className="text-base font-semibold"
        >
          New statements
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Enter each password, then review the changes.
        </p>
      </header>
      {query.data.cleanup_notice ? (
        <p role="status" className="text-sm text-muted-foreground">
          {query.data.cleanup_notice}
        </p>
      ) : null}
      {fullItems.length === 0 && otherItems.length > 0 ? (
        <p className="text-sm text-muted-foreground">
          No full statements yet. Your monthly statement from NSDL, CAMS or
          KFintech will show here.
        </p>
      ) : null}
      {fullItems.map((item) => {
        const result = results[item.id];
        const jobId = result?.job_id ?? item.import_job_id;
        return (
          <Card key={item.id} className="grid gap-3">
            {heading(
              item,
              `New statement from ${senderNames[item.sender_type]}`,
            )}
            {item.status === "needs_password" &&
            openId !== item.id &&
            item.saved_password ? (
              <div className="grid gap-2">
                <Button
                  className="w-full"
                  loading={
                    processMutation.isPending &&
                    processMutation.variables?.id === item.id
                  }
                  onClick={() =>
                    processMutation.mutate({
                      id: item.id,
                      password: "",
                      rememberPassword: false,
                      useSaved: true,
                    })
                  }
                >
                  Open with saved password
                </Button>
                <Button variant="ghost" onClick={() => openForm(item.id)}>
                  Enter password
                </Button>
              </div>
            ) : item.status === "needs_password" && openId !== item.id ? (
              <Button className="w-full" onClick={() => openForm(item.id)}>
                Enter password
              </Button>
            ) : null}
            {item.status !== "review" && openId === item.id
              ? passwordForm(item)
              : null}
            {item.status === "review" && jobId ? (
              <Link
                className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
                to={`/import/${encodeURIComponent(jobId)}`}
              >
                Review
              </Link>
            ) : null}
            {item.status === "unreadable" && openId !== item.id ? (
              <div className="grid gap-2">
                <p className="text-sm text-muted-foreground">
                  We couldn’t read this file. Upload it in Money instead.
                </p>
                <Link
                  className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
                  to="/import"
                >
                  Upload in Money
                </Link>
                <div className="flex flex-wrap gap-2">
                  <Button variant="ghost" onClick={() => openForm(item.id)}>
                    Try again
                  </Button>
                  {removeButton(item)}
                </div>
              </div>
            ) : null}
            {message(item)}
            {result?.job_id && item.status === "needs_password" ? (
              <Button
                variant="secondary"
                className="w-full"
                onClick={() =>
                  navigate(`/import/${encodeURIComponent(result.job_id)}`)
                }
              >
                Open review
              </Button>
            ) : null}
          </Card>
        );
      })}
      {otherItems.length > 0 ? (
        <details className="rounded-2xl border border-line bg-surface p-4">
          <summary className="min-h-10 cursor-pointer text-sm font-semibold">
            Other emails ({otherItems.length})
          </summary>
          <p className="mt-2 text-sm text-muted-foreground">
            These look like purchase notes or single-fund statements, not your
            full account statement. They can’t be imported.
          </p>
          <ul className="mt-2 grid gap-3">
            {otherItems.map((item) => (
              <li
                key={item.id}
                className="grid gap-2 border-t border-line pt-3"
              >
                {heading(item, `From ${senderNames[item.sender_type]}`)}
                {item.error_code === "not_a_full_statement" ? (
                  <p className="text-sm text-muted-foreground">
                    {NOT_A_FULL_STATEMENT}
                  </p>
                ) : null}
                {openId === item.id ? passwordForm(item) : null}
                {feedback[item.id] &&
                item.error_code !== "not_a_full_statement" ? (
                  <p role="alert" className="text-sm text-danger-foreground">
                    {feedback[item.id]}
                  </p>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  {item.error_code !== "not_a_full_statement" &&
                  openId !== item.id ? (
                    <Button variant="ghost" onClick={() => openForm(item.id)}>
                      Try it
                    </Button>
                  ) : null}
                  {removeButton(item)}
                </div>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}
