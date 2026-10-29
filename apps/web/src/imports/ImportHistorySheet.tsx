import { useState } from "react";
import { Link } from "react-router";
import type { ImportJob } from "./api";
import { Badge, Button, Card, Sheet } from "../ui";

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
  const [open, setOpen] = useState(false);
  return (
    <>
      <Button
        type="button"
        variant="secondary"
        className="w-full"
        onClick={() => setOpen(true)}
      >
        Imports
      </Button>
      <Sheet
        open={open}
        onOpenChange={setOpen}
        title="Statement imports"
        description="Recent statements and their review status."
      >
        {jobs.length ? (
          <ul className="grid gap-2">
            {jobs.map((job) => (
              <li key={job.id}>
                <Link
                  to={`/import/${encodeURIComponent(job.id)}`}
                  onClick={() => setOpen(false)}
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
          <Card className="grid gap-3">
            <p className="text-sm text-muted-foreground">
              Upload a statement to see its review here.
            </p>
            <Link
              to="/import"
              onClick={() => setOpen(false)}
              data-primary="true"
              className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-brand-foreground"
            >
              Upload a statement
            </Link>
          </Card>
        )}
      </Sheet>
    </>
  );
}
