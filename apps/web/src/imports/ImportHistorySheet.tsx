import { Link } from "react-router";
import type { ImportJob } from "./api";
import { Badge, Card } from "../ui";

function sourceName(source: string | null): string {
  if (source === "nsdl") return "NSDL fund statement";
  if (source === "cdsl") return "CDSL fund statement";
  if (source === "cams_kfin") return "CAMS / KFintech statement";
  return "Statement import";
}

function jobDate(job: ImportJob): string {
  const value = job.statement_date ?? job.confirmed_at ?? job.created_at;
  const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
  if (Number.isNaN(date.getTime())) return value.slice(0, 10);
  return new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

export function ImportHistorySheet({ jobs }: { jobs: readonly ImportJob[] }) {
  return (
    <details className="rounded-xl border border-line px-3">
      <summary className="min-h-12 cursor-pointer py-3 text-sm font-semibold">
        Imports
      </summary>
      {jobs.length ? (
        <ul className="grid gap-2 pb-3">
          {jobs.map((job) => (
            <li key={job.id}>
              <Link
                to={`/import/${encodeURIComponent(job.id)}`}
                className="flex min-h-14 items-center justify-between gap-3 rounded-xl border border-line px-3 py-2 hover:bg-muted"
              >
                <span className="min-w-0">
                  <strong className="block truncate text-sm">
                    {sourceName(job.source)}
                  </strong>
                  <span className="text-xs text-muted-foreground">
                    {jobDate(job)}
                  </span>
                </span>
                <Badge
                  tone={
                    job.status === "confirmed"
                      ? "success"
                      : job.status === "failed"
                        ? "danger"
                        : "neutral"
                  }
                >
                  {job.status.replaceAll("_", " ")}
                </Badge>
              </Link>
            </li>
          ))}
        </ul>
      ) : (
        <Card className="mb-3 grid gap-3">
          <p className="text-sm text-muted-foreground">
            Upload a statement to see its review here.
          </p>
          <Link
            to="/import"
            className="inline-flex min-h-12 items-center justify-center rounded-xl border border-line px-4 py-3 text-sm font-semibold"
          >
            Upload a statement
          </Link>
        </Card>
      )}
    </details>
  );
}
