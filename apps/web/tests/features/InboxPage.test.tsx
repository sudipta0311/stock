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
