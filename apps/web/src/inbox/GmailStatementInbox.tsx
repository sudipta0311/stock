import { useState } from "react";
import type { FormEvent } from "react";
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

function statementDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? ""
    : new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
      }).format(date);
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
  const processMutation = useMutation({
    mutationFn: ({
      id,
      password,
      rememberPassword,
    }: {
      id: string;
      password: string;
      rememberPassword: boolean;
    }) =>
      apiFetch<ProcessResponse>(
        `/v1/statement-inbox/items/${encodeURIComponent(id)}/process`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password, remember: rememberPassword }),
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
      await queryClient.invalidateQueries({
        queryKey: ["statement-inbox", "items"],
      });
    },
    onError: (error, variables) => {
      const message =
        error instanceof ApiError && error.code === "wrong_password"
          ? "That password didn’t work. Please try again."
          : error instanceof ApiError && error.code === "unreadable"
            ? "We couldn’t read this file. Upload it in Money instead."
            : "The statement could not be opened. Try again.";
      setFeedback((current) => ({ ...current, [variables.id]: message }));
    },
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

  const submitPassword = (event: FormEvent<HTMLFormElement>, id: string) => {
    event.preventDefault();
    processMutation.mutate({
      id,
      password: passwords[id] ?? "",
      rememberPassword: remember[id] ?? false,
    });
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
      {query.data.items.map((item) => {
        const title = `New statement from ${senderNames[item.sender_type]}`;
        const date = statementDate(item.received_at);
        const result = results[item.id];
        const jobId = result?.job_id ?? item.import_job_id;
        const savedPasswordMessage =
          item.error_code === "wrong_password"
            ? "A saved password didn’t work. Enter it to continue."
            : item.error_code === "parse_timeout"
              ? "This statement took too long to read. Try again."
              : "";
        return (
          <Card key={item.id} className="grid gap-3">
            <div>
              <h3 className="text-sm font-semibold">{title}</h3>
              {date ? (
                <p className="mt-1 text-xs text-muted-foreground">
                  Received {date}
                </p>
              ) : null}
            </div>
            {item.status === "needs_password" && openId !== item.id ? (
              <Button
                className="w-full"
                onClick={() => {
                  setOpenId(item.id);
                  setFeedback((current) => ({ ...current, [item.id]: "" }));
                }}
              >
                Enter password
              </Button>
            ) : null}
            {item.status === "needs_password" && openId === item.id ? (
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
                      setPasswords((current) => ({
                        ...current,
                        [item.id]: value,
                      }));
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
                    setRemember((current) => ({
                      ...current,
                      [item.id]: checked,
                    }));
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
            ) : null}
            {item.status === "review" && jobId ? (
              <Link
                className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
                to={`/import/${encodeURIComponent(jobId)}`}
              >
                Review
              </Link>
            ) : null}
            {item.status === "unreadable" ? (
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
              </div>
            ) : null}
            {feedback[item.id] ? (
              <p role="status" className="text-sm text-muted-foreground">
                {feedback[item.id]}
              </p>
            ) : savedPasswordMessage ? (
              <p role="status" className="text-sm text-muted-foreground">
                {savedPasswordMessage}
              </p>
            ) : null}
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
    </section>
  );
}
