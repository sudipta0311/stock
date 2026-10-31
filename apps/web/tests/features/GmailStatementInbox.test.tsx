import { fireEvent, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GmailStatementInbox } from "../../src/inbox/GmailStatementInbox";
import { renderWithQueries } from "./render";

const api = vi.hoisted(() => ({ fetch: vi.fn() }));

vi.mock("../../src/api/client", () => ({ apiFetch: api.fetch }));

describe("Gmail statement inbox", () => {
  beforeEach(() => api.fetch.mockReset());

  it("asks for a password and opens the existing review flow", async () => {
    let status: "needs_password" | "review" = "needs_password";
    api.fetch.mockImplementation(async (path: string) => {
      if (path === "/v1/statement-inbox/items") {
        return {
          items: [
            {
              id: "synthetic-item-id",
              sender_type: "nsdl",
              received_at: "2026-10-01T08:00:00Z",
              status,
              error_code: null,
              import_job_id: status === "review" ? "synthetic-job-id" : null,
            },
          ],
          cleanup_notice: null,
        };
      }
      status = "review";
      return {
        status: "review",
        job_id: "synthetic-job-id",
        password_saved: false,
        notice: null,
      };
    });

    renderWithQueries(
      <MemoryRouter>
        <GmailStatementInbox />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("New statement from NSDL"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Enter password" }));
    fireEvent.change(screen.getByLabelText("Statement password"), {
      target: { value: "synthetic-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Review statement" }));

    await waitFor(() =>
      expect(api.fetch).toHaveBeenCalledWith(
        "/v1/statement-inbox/items/synthetic-item-id/process",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({
            password: "synthetic-password",
            remember: false,
          }),
        }),
      ),
    );
    expect(await screen.findByRole("link", { name: "Review" })).toHaveAttribute(
      "href",
      "/import/synthetic-job-id",
    );
  });

  it("links unreadable statements to manual upload", async () => {
    api.fetch.mockResolvedValue({
      items: [
        {
          id: "synthetic-unreadable-item",
          sender_type: "cdsl",
          received_at: "2026-10-01T08:00:00Z",
          status: "unreadable",
          error_code: "unreadable",
          import_job_id: null,
        },
      ],
      cleanup_notice: null,
    });

    renderWithQueries(
      <MemoryRouter>
        <GmailStatementInbox />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(/couldn’t read this file/),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Upload in Money" }),
    ).toHaveAttribute("href", "/import");
  });

  it("keeps other fund-house emails in a closed group that can be cleared", async () => {
    api.fetch.mockImplementation(async (path: string, init?: RequestInit) => {
      if (init?.method === "DELETE") return undefined;
      if (path === "/v1/statement-inbox/items") {
        return {
          items: [
            {
              id: "synthetic-full-statement",
              sender_type: "nsdl",
              received_at: "2026-10-12T08:00:00Z",
              status: "needs_password",
              error_code: null,
              import_job_id: null,
              subject: "Consolidated Account Statement for September",
              full_statement: true,
            },
            {
              id: "synthetic-purchase-note",
              sender_type: "kfintech",
              received_at: "2026-10-14T08:00:00Z",
              status: "unreadable",
              error_code: "not_a_full_statement",
              import_job_id: null,
              subject: "Transaction confirmation",
              full_statement: false,
            },
          ],
          cleanup_notice: null,
        };
      }
      return undefined;
    });

    renderWithQueries(
      <MemoryRouter>
        <GmailStatementInbox />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("New statement from NSDL"),
    ).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: "Enter password" }),
    ).toHaveLength(1);
    const group = screen.getByText("Other emails (1)").closest("details");
    expect(group).not.toHaveAttribute("open");
    expect(
      screen.getByText(/isn’t a full account statement/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Try it" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Remove" }));

    await waitFor(() =>
      expect(api.fetch).toHaveBeenCalledWith(
        "/v1/statement-inbox/items/synthetic-purchase-note",
        expect.objectContaining({ method: "DELETE" }),
      ),
    );
  });

  it("shows a saved wrong password as an error", async () => {
    api.fetch.mockResolvedValue({
      items: [
        {
          id: "synthetic-wrong-password-item",
          sender_type: "nsdl",
          received_at: "2026-10-01T08:00:00Z",
          status: "needs_password",
          error_code: "wrong_password",
          import_job_id: null,
        },
      ],
      cleanup_notice: null,
    });

    renderWithQueries(
      <MemoryRouter>
        <GmailStatementInbox />
      </MemoryRouter>,
    );

    const error = await screen.findByRole("alert");
    expect(error).toHaveTextContent(
      "A saved password didn’t work. Enter it to continue.",
    );
    expect(error).toHaveClass("text-danger-foreground");
  });
});
