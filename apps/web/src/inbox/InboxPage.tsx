import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router";
import { apiFetch } from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import { asArray, asRecord, textValue } from "../features/data";
import {
  ComingSoon,
  isMissingFeature,
  ScreenError,
  ScreenLoading,
} from "../features/api";
import { Badge, Card } from "../ui";

interface InboxEntry {
  id?: string;
  week?: string;
  subject?: string;
  title?: string;
  status?: string;
  created_at?: string;
  date?: string;
  due_at?: string;
  kind?: string;
}

export interface InboxData {
  briefs?: InboxEntry[];
  reminders?: InboxEntry[];
  alerts?: InboxEntry[];
  items?: InboxEntry[];
}

export type FeatureCheck = (path: string) => boolean;

function entries(
  data: unknown,
  group: "briefs" | "reminders" | "alerts",
): InboxEntry[] {
  const record = asRecord(data);
  const grouped = asArray(record[group]);
  const general = asArray(record.items);
  const selected = grouped.length
    ? grouped
    : general.filter((item) =>
        textValue(asRecord(item).kind)
          .toLowerCase()
          .startsWith(group.slice(0, -1)),
      );
  return selected
    .map((item) => asRecord(item) as InboxEntry)
    .sort((left, right) => {
      const leftDate = left.created_at ?? left.date ?? left.due_at ?? "";
      const rightDate = right.created_at ?? right.date ?? right.due_at ?? "";
      return rightDate.localeCompare(leftDate);
    });
}

function displayDate(entry: InboxEntry): string {
  const value = entry.created_at ?? entry.date ?? entry.due_at;
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en-IN", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }).format(date);
}

export function InboxPage({
  isSupported = apiPathAvailable,
}: { isSupported?: FeatureCheck } = {}) {
  const supported = isSupported("/v1/inbox");
  const query = useQuery({
    queryKey: ["c22", "inbox"],
    queryFn: () => apiFetch<InboxData>("/v1/inbox"),
    enabled: supported,
  });
  if (!supported || isMissingFeature(query.error))
    return (
      <ComingSoon
        title="Your inbox is coming soon"
        description="Briefs, reminders and alerts will collect here when the inbox service is available."
      />
    );
  if (query.isPending) return <ScreenLoading label="Loading your inbox" />;
  if (query.error) return <ScreenError title="Your inbox couldn’t load" />;
  const briefs = entries(query.data, "briefs");
  const reminders = entries(query.data, "reminders");
  const alerts = entries(query.data, "alerts");
  return (
    <div className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">Inbox</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Briefs, reminders and updates, newest first.
        </p>
      </header>
      <InboxGroup title="Briefs" entries={briefs} kind="brief" />
      <InboxGroup title="Reminders" entries={reminders} kind="reminder" />
      <InboxGroup title="Alerts" entries={alerts} kind="alert" />
    </div>
  );
}

function InboxGroup({
  title,
  entries: items,
  kind,
}: {
  title: string;
  entries: InboxEntry[];
  kind: string;
}) {
  return (
    <Card className="grid gap-3">
      <h2 className="text-base font-semibold">{title}</h2>
      {items.length ? (
        items.map((entry, index) => {
          const subject = textValue(entry.subject ?? entry.title, title);
          const week = textValue(entry.week);
          const date = displayDate(entry);
          const content = (
            <div className="flex min-h-12 items-center justify-between gap-3 border-t border-line pt-2 first:border-0 first:pt-0">
              <span>
                <strong className="block text-sm">{subject}</strong>
                {date ? (
                  <span className="text-xs text-muted-foreground">
                    {date}
                    {week ? ` · ${week}` : ""}
                  </span>
                ) : week ? (
                  <span className="text-xs text-muted-foreground">{week}</span>
                ) : null}
              </span>
              <Badge>{textValue(entry.status, "New")}</Badge>
            </div>
          );
          return kind === "brief" && week ? (
            <Link
              key={entry.id ?? `${week}-${index}`}
              to={`/inbox/brief/${encodeURIComponent(week)}`}
              className="rounded-lg hover:text-brand"
            >
              {content}
            </Link>
          ) : (
            <div key={entry.id ?? `${subject}-${index}`}>{content}</div>
          );
        })
      ) : (
        <p className="text-sm text-muted-foreground">Nothing here right now.</p>
      )}
    </Card>
  );
}

function safeSource(value: unknown): string | null {
  if (typeof value !== "string") return null;
  try {
    const url = new URL(value);
    return url.protocol === "http:" || url.protocol === "https:"
      ? url.href
      : null;
  } catch {
    return null;
  }
}

export function StructuredBrief({ value }: { value: unknown }) {
  const brief = asRecord(asRecord(value).brief ?? value);
  const title = textValue(brief.subject ?? brief.title, "Weekly brief");
  const summary = textValue(brief.summary ?? brief.introduction);
  const sections = asArray(brief.sections);
  const bullets = asArray(brief.bullets);
  const sources = asArray(brief.sources);
  return (
    <article className="grid gap-4">
      <header>
        <h2 className="text-xl font-semibold">{title}</h2>
        {summary ? (
          <p className="mt-2 text-sm leading-6 text-muted-foreground">
            {summary}
          </p>
        ) : null}
      </header>
      {sections.map((raw, index) => {
        const section = asRecord(raw);
        const items = asArray(section.bullets ?? section.items);
        return (
          <Card
            key={`${textValue(section.title ?? section.heading)}-${index}`}
            className="grid gap-2"
          >
            <h3 className="font-semibold">
              {textValue(section.title ?? section.heading, "Update")}
            </h3>
            {textValue(section.body ?? section.text) ? (
              <p className="text-sm leading-6">
                {textValue(section.body ?? section.text)}
              </p>
            ) : null}
            {items.length ? (
              <ul className="list-disc space-y-1 pl-5 text-sm">
                {items.map((item, itemIndex) => (
                  <li key={itemIndex}>
                    {typeof item === "string"
                      ? item
                      : textValue(asRecord(item).text)}
                  </li>
                ))}
              </ul>
            ) : null}
          </Card>
        );
      })}
      {bullets.length ? (
        <Card>
          <h3 className="font-semibold">This week</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {bullets.map((item, index) => (
              <li key={index}>
                {typeof item === "string"
                  ? item
                  : textValue(asRecord(item).text)}
              </li>
            ))}
          </ul>
        </Card>
      ) : null}
      {sources.length ? (
        <Card className="grid gap-2">
          <h3 className="font-semibold">Sources</h3>
          {sources.map((raw) => {
            const source = asRecord(raw);
            const href = safeSource(typeof raw === "string" ? raw : source.url);
            if (!href) return null;
            return (
              <a
                key={href}
                className="min-h-10 text-sm font-medium text-brand"
                href={href}
                target="_blank"
                rel="noreferrer"
              >
                {textValue(source.title, href)}
              </a>
            );
          })}
        </Card>
      ) : null}
    </article>
  );
}

export function BriefPage({
  isSupported = apiPathAvailable,
}: { isSupported?: FeatureCheck } = {}) {
  const { week = "" } = useParams();
  const path = "/v1/brief/{week}";
  const supported = isSupported(path);
  const query = useQuery({
    queryKey: ["c22", "brief", week],
    queryFn: () => apiFetch<unknown>(`/v1/brief/${encodeURIComponent(week)}`),
    enabled: supported && Boolean(week),
  });
  if (!supported || isMissingFeature(query.error))
    return <ComingSoon title="Weekly briefs are coming soon" />;
  if (query.isPending)
    return <ScreenLoading label="Loading your weekly brief" />;
  if (query.error) return <ScreenError title="This brief couldn’t load" />;
  return <StructuredBrief value={query.data} />;
}

export { entries as inboxEntries };
