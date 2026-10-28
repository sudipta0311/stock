import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { ImportPage, ImportReviewPage } from "../../src/imports/ImportPage";
import type { ImportDetail, ImportItem } from "../../src/imports/api";
import { importFeatureSupport } from "../../src/imports/features";
import { server } from "../msw";

const jobId = "job-test-1";

function uploadedJob() {
  return HttpResponse.json(
    {
      job_id: jobId,
      status: "ready",
      source: "nsdl",
      statement_date: "2026-10-01",
      counts: { new: 1, changed: 0, duplicate: 0, unknown: 0 },
      warnings: [],
    },
    { status: 201 },
  );
}

function renderUpload() {
  return render(
    <MemoryRouter initialEntries={["/import"]}>
      <Routes>
        <Route path="/import" element={<ImportPage />} />
        <Route path="/import/:jobId" element={<p>Review route loaded</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

function pdfFile(size = 4): File {
  return new File([new Uint8Array(size).fill(65)], "statement.pdf", {
    type: "application/pdf",
  });
}

function item(overrides: Partial<ImportItem> = {}): ImportItem {
  return {
    id: "item-new",
    kind: "equity",
    isin: "IN0000000001",
    name: "Example Equity Fund",
    quantity: "12.00",
    statement_value: "1200.00",
    account_ref: "account-0123",
    sleeve: "equity_india",
    status: "new",
    diff: null,
    match: {},
    ...overrides,
  };
}

function reviewDetail(overrides: Partial<ImportDetail> = {}): ImportDetail {
  return {
    id: jobId,
    doc_type: "nsdl_cas",
    status: "ready",
    error_code: null,
    source: "nsdl",
    statement_date: "2026-10-01",
    counts: { new: 1, changed: 1, duplicate: 1, unknown: 1 },
    created_at: "2026-10-02T00:00:00Z",
    confirmed_at: null,
    items: [
      item({ id: "item-new" }),
      item({
        id: "item-changed",
        status: "changed",
        name: "Changed Fund",
        diff: { old_quantity: "10.00", new_quantity: "12.00" },
      }),
      item({
        id: "item-duplicate",
        status: "duplicate",
        name: "Duplicate Fund",
      }),
      item({
        id: "item-unknown",
        status: "unknown",
        kind: "unknown",
        sleeve: null,
        name: "Unclassified Holding",
      }),
    ],
    ...overrides,
  };
}

function renderReview(path = `/import/${jobId}`) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/import/:jobId" element={<ImportReviewPage />} />
        <Route path="/money" element={<p>Money screen</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("statement upload screen", () => {
  it("rejects files larger than ten megabytes before uploading", async () => {
    const user = userEvent.setup();
    const post = vi.fn();
    server.use(
      http.post("/v1/imports", () => {
        post();
        return uploadedJob();
      }),
    );
    renderUpload();

    await user.upload(
      screen.getByLabelText(/Statement file/),
      pdfFile(10 * 1024 * 1024 + 1),
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "This file is over 10 MB",
    );
    expect(post).not.toHaveBeenCalled();
  });

  it("clears the password after submit and never writes it to browser storage", async () => {
    const user = userEvent.setup();
    let sentPassword: FormDataEntryValue | null = null;
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      sentPassword =
        (init?.body as FormData | undefined)?.get("password") ?? null;
      return Promise.resolve(
        new Response(
          JSON.stringify({
            job_id: jobId,
            status: "ready",
            source: "nsdl",
            statement_date: "2026-10-01",
            counts: { new: 1 },
            warnings: [],
          }),
          { status: 201, headers: { "Content-Type": "application/json" } },
        ),
      );
    });
    const localStorageWrite = vi.spyOn(localStorage, "setItem");
    const sessionStorageWrite = vi.spyOn(sessionStorage, "setItem");
    const indexedDbWrite = vi.spyOn(IDBObjectStore.prototype, "put");
    renderUpload();

    await user.upload(screen.getByLabelText(/Statement file/), pdfFile());
    await user.type(
      screen.getByLabelText(/Statement password/),
      "demo-passphrase",
    );
    await user.click(screen.getByRole("button", { name: "Upload statement" }));

    expect(screen.getByLabelText(/Statement password/)).toHaveValue("");
    expect(await screen.findByText("Review route loaded")).toBeInTheDocument();
    expect(sentPassword).toBe("demo-passphrase");
    expect(localStorageWrite).not.toHaveBeenCalled();
    expect(sessionStorageWrite).not.toHaveBeenCalled();
    expect(indexedDbWrite).not.toHaveBeenCalled();
  });

  it("aborts the upload when the user cancels", async () => {
    let signal: AbortSignal | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      signal = init?.signal as AbortSignal | undefined;
      return new Promise<Response>((_resolve, reject) => {
        signal?.addEventListener(
          "abort",
          () => reject(new DOMException("Aborted", "AbortError")),
          { once: true },
        );
      });
    });
    renderUpload();
    fireEvent.change(screen.getByLabelText(/Statement file/), {
      target: { files: [pdfFile()] },
    });
    fireEvent.submit(
      screen.getByRole("button", { name: "Upload statement" }).closest("form")!,
    );

    await waitFor(() => expect(signal).toBeDefined());
    fireEvent.click(screen.getByRole("button", { name: "Cancel upload" }));

    expect(signal?.aborted).toBe(true);
    expect(
      screen.getByRole("button", { name: "Upload statement" }),
    ).toBeEnabled();
  });

  it("shows upload and review screens without accessibility violations", async () => {
    server.use(
      http.get(`/v1/imports/${jobId}`, () => HttpResponse.json(reviewDetail())),
    );
    const upload = renderUpload();
    expect((await axe(upload.container)).violations).toHaveLength(0);
    upload.unmount();

    const review = renderReview();
    expect(await screen.findByText("Review your import")).toBeInTheDocument();
    expect((await axe(review.container)).violations).toHaveLength(0);
  });
});

describe("statement import review", () => {
  it("groups counts, patches an unknown override, confirms, and shows success", async () => {
    const user = userEvent.setup();
    let overrideBody: Record<string, unknown> | null = null;
    server.use(
      http.get(`/v1/imports/${jobId}`, () => HttpResponse.json(reviewDetail())),
      http.patch(
        `/v1/imports/${jobId}/items/:itemId`,
        async ({ params, request }) => {
          const body = (await request.json()) as Record<string, unknown>;
          if (params.itemId === "item-unknown") overrideBody = body;
          const original = reviewDetail().items.find(
            (candidate) => candidate.id === params.itemId,
          )!;
          const override = body.override as
            { kind?: string; sleeve?: string } | undefined;
          return HttpResponse.json({
            ...original,
            status: body.status ?? original.status,
            kind: override?.kind ?? original.kind,
            sleeve: override?.sleeve ?? original.sleeve,
          });
        },
      ),
      http.post(`/v1/imports/${jobId}/confirm`, () =>
        HttpResponse.json({
          job_id: jobId,
          status: "confirmed",
          counts: { new: 3 },
          net_worth: "10000.00",
        }),
      ),
    );
    renderReview();

    expect(await screen.findByText("New · 1")).toBeInTheDocument();
    expect(screen.getByText("Changed · 1")).toBeInTheDocument();
    expect(screen.getByText("Duplicate · 1")).toBeInTheDocument();
    expect(screen.getByText("Unknown · 1")).toBeInTheDocument();
    expect(screen.getByText("Already up to date")).toBeInTheDocument();
    expect(importFeatureSupport.mapping).toBe(false);
    expect(
      screen.queryByText("Column mapping is available for this API."),
    ).not.toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText("Kind"), "mf");
    await user.selectOptions(screen.getByLabelText("Sleeve"), "gold");
    await user.click(screen.getByRole("button", { name: "Save decision" }));

    await waitFor(() =>
      expect(overrideBody).toEqual({
        status: "accepted",
        override: { kind: "mf", sleeve: "gold" },
      }),
    );
    expect(await screen.findByText("Unknown · 0")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Confirm import (3 items)" }),
    );

    expect(
      await screen.findByText("Added 3 holdings · Net worth updated"),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to Money" })).toHaveAttribute(
      "href",
      "/money",
    );
  });

  it("shows the confirmed state for an already-confirmed response", async () => {
    const user = userEvent.setup();
    server.use(
      http.get(`/v1/imports/${jobId}`, () =>
        HttpResponse.json(
          reviewDetail({
            items: [item()],
            counts: { new: 1 },
          }),
        ),
      ),
      http.post(`/v1/imports/${jobId}/confirm`, () =>
        HttpResponse.json(
          { code: "already_confirmed", title: "Already confirmed", errors: [] },
          { status: 409 },
        ),
      ),
    );
    renderReview();

    await user.click(
      await screen.findByRole("button", { name: "Confirm import (1 items)" }),
    );
    expect(
      await screen.findByText("This import is already confirmed"),
    ).toBeInTheDocument();
  });

  it("asks before discarding and deletes the import", async () => {
    const user = userEvent.setup();
    const deleteRequest = vi.fn();
    server.use(
      http.get(`/v1/imports/${jobId}`, () =>
        HttpResponse.json(
          reviewDetail({
            items: [item()],
            counts: { new: 1 },
          }),
        ),
      ),
      http.delete(`/v1/imports/${jobId}`, () => {
        deleteRequest();
        return new HttpResponse(null, { status: 204 });
      }),
    );
    renderReview();

    await user.click(await screen.findByRole("button", { name: "Discard" }));
    expect(screen.getByText("Discard this import?")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Discard import" }));

    expect(await screen.findByText("Money screen")).toBeInTheDocument();
    expect(deleteRequest).toHaveBeenCalledOnce();
  });
});
