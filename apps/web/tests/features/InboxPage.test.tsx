import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { apiPathAvailable } from "../../src/api/feature-support";
import { BriefPage, InboxPage } from "../../src/inbox/InboxPage";
import * as englishCopy from "../../src/copy/sections/index";
import { renderWithQueries } from "./render";
import { server } from "../msw";

vi.mock("../../src/api/feature-support", () => ({
  apiPathAvailable: vi.fn(),
}));

function renderInbox(supported = false) {
  vi.mocked(apiPathAvailable).mockReturnValue(supported);
  return renderWithQueries(
    <MemoryRouter initialEntries={["/inbox"]}>
      <Routes>
        <Route path="/inbox" element={<InboxPage />} />
        <Route path="/inbox/brief/:week" element={<BriefPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => vi.mocked(apiPathAvailable).mockReturnValue(true));

describe("Inbox and brief", () => {
  it("shows the upload link with an empty inbox", async () => {
    server.use(
      http.get(/\/v1\/inbox$/, () => HttpResponse.json({})),
      http.get(/\/v1\/statement-inbox\/items$/, () =>
        HttpResponse.json({ items: [], cleanup_notice: null }),
      ),
    );
    renderInbox(true);
    expect(
      await screen.findByRole("link", { name: "Upload a file" }),
    ).toHaveAttribute("href", "/import");
    expect(screen.getByRole("link", { name: "Email settings" })).toBeVisible();
  });

  it("keeps the upload link visible when emails are present", async () => {
    server.use(
      http.get(/\/v1\/inbox$/, () =>
        HttpResponse.json({
          briefs: [
            {
              id: "brief-upload-link",
              week: "2026-W44",
              subject: "Synthetic update",
              status: "New",
              created_at: "2026-10-28T00:00:00Z",
            },
          ],
        }),
      ),
      http.get(/\/v1\/statement-inbox\/items$/, () =>
        HttpResponse.json({ items: [], cleanup_notice: null }),
      ),
    );
    renderInbox(true);
    expect(
      await screen.findByRole("link", { name: "Upload a file" }),
    ).toHaveAttribute("href", "/import");
  });

  it("shows a coming soon state when inbox paths are missing from generated OpenAPI types", async () => {
    let requested = false;
    server.use(
      http.get(/\/v1\/inbox$/, () => {
        requested = true;
        return HttpResponse.json({});
      }),
    );
    vi.mocked(apiPathAvailable).mockReturnValue(false);
    renderInbox();
    expect(
      await screen.findByText("Your inbox is coming soon"),
    ).toBeInTheDocument();
    expect(requested).toBe(false);
  });

  it("lists briefs, reminders, and alerts, then renders a structured brief", async () => {
    const user = userEvent.setup();
    server.use(
      http.get(/\/v1\/inbox$/, () =>
        HttpResponse.json({
          briefs: [
            {
              id: "brief-old",
              week: "2026-W43",
              subject: "Older brief",
              status: "Read",
              created_at: "2026-10-20T00:00:00Z",
            },
            {
              id: "brief-new",
              week: "2026-W44",
              subject: "Newest brief",
              status: "New",
              created_at: "2026-10-27T00:00:00Z",
            },
          ],
          reminders: [
            {
              id: "reminder-demo",
              title: "Review a maturity",
              status: "Upcoming",
              due_at: "2026-10-29T00:00:00Z",
            },
          ],
          alerts: [
            {
              id: "alert-demo",
              title: "A plan update",
              status: "New",
              created_at: "2026-10-28T00:00:00Z",
            },
          ],
        }),
      ),
      http.get(/\/v1\/brief\/2026-W44$/, () =>
        HttpResponse.json({
          title: "Your week in review",
          summary: "A short summary.",
          sections: [
            {
              heading: "Your plan",
              body: "Your target mix is ready.",
              bullets: ["Review the global share."],
            },
          ],
          sources: [
            { title: "Source note", url: "https://example.test/brief" },
          ],
        }),
      ),
    );
    const { container } = renderInbox(true);
    const newest = await screen.findByRole("link", { name: /Newest brief/ });
    expect(newest).toBeInTheDocument();
    expect(screen.getByText("Older brief")).toBeInTheDocument();
    expect(screen.getByText("Review a maturity")).toBeInTheDocument();
    expect(screen.getByText("A plan update")).toBeInTheDocument();
    const briefLinks = screen.getAllByRole("link", { name: /brief/ });
    expect(briefLinks[0]).toHaveTextContent("Newest brief");
    expect((await axe(container)).violations).toEqual([]);
    await user.click(newest);
    expect(
      await screen.findByRole("heading", { name: "Your week in review" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Your target mix is ready.")).toBeInTheDocument();
    expect(screen.getByText("Review the global share.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Source note" })).toHaveAttribute(
      "href",
      "https://example.test/brief",
    );
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows the updates heading and a statement month without the raw subject", async () => {
    const rawSubject =
      "Your provider statement for September 2026 (reference note)";
    server.use(
      http.get(/\/v1\/inbox$/, () => HttpResponse.json({})),
      http.get(/\/v1\/statement-inbox\/items$/, () =>
        HttpResponse.json({
          items: [
            {
              id: "statement-month-demo",
              sender_type: "nsdl",
              received_at: "2026-10-16T08:00:00Z",
              status: "needs_password",
              error_code: null,
              import_job_id: null,
              subject: rawSubject,
              statement_month: "2026-09",
              full_statement: true,
              saved_password: false,
            },
          ],
          cleanup_notice: null,
        }),
      ),
    );
    renderInbox(true);

    expect(
      await screen.findByRole("heading", {
        name: englishCopy.inbox.updates_heading,
      }),
    ).toBeVisible();
    expect(
      await screen.findByText("Statement for September 2026"),
    ).toBeVisible();
    expect(screen.getByText("Received 16 Oct")).toBeVisible();
    expect(screen.queryByText(rawSubject)).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Inbox" })).toBeNull();
  });

  it("hides the statement subject line when no month is available", async () => {
    const rawSubject = "A provider statement with no month";
    server.use(
      http.get(/\/v1\/inbox$/, () => HttpResponse.json({})),
      http.get(/\/v1\/statement-inbox\/items$/, () =>
        HttpResponse.json({
          items: [
            {
              id: "statement-no-month-demo",
              sender_type: "nsdl",
              received_at: "2026-10-16T08:00:00Z",
              status: "needs_password",
              error_code: null,
              import_job_id: null,
              subject: rawSubject,
              statement_month: null,
              full_statement: true,
              saved_password: false,
            },
          ],
          cleanup_notice: null,
        }),
      ),
    );
    renderInbox(true);

    await screen.findByRole("button", { name: "Enter password" });
    expect(screen.queryByText(rawSubject)).not.toBeInTheDocument();
    expect(screen.queryByText(/Statement for /)).not.toBeInTheDocument();
    expect(screen.getByText("Received 16 Oct")).toBeVisible();
  });

  it("does not inject API strings as HTML anywhere in the app source", () => {
    const sourceRoot = path.join(process.cwd(), "src");
    const files: string[] = [];
    const visit = (directory: string) => {
      for (const entry of readdirSync(directory, { withFileTypes: true })) {
        const fullPath = path.join(directory, entry.name);
        if (entry.isDirectory()) visit(fullPath);
        else if (/\.(?:ts|tsx)$/.test(entry.name)) files.push(fullPath);
      }
    };
    visit(sourceRoot);
    const unsafe = files.filter((file) =>
      readFileSync(file, "utf8").includes("dangerouslySetInnerHTML"),
    );
    expect(unsafe).toEqual([]);
  });
});
