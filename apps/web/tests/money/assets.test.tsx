import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { AddAssetPage, AssetDetailPage } from "../../src/money/AssetFormPage";
import type { AssetDetailResponse, AssetResponse } from "../../src/money/api";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";

function assetResponse(details: Record<string, unknown>): AssetResponse {
  return {
    id: "asset-test-1",
    asset_type: "fd",
    member_id: "member-test-1",
    account_id: null,
    joint: false,
    details,
    valuation: {
      value_today: "546200.00",
      maturity_value: "575900.00",
      maturity_date: "2027-03",
    },
    valuation_note: "Estimated from the details entered.",
    valuation_status: "valued",
    value_cached: "546200.00",
    value_as_of: "2026-10-28",
    status: "active",
    created_at: "2026-10-28T00:00:00Z",
    updated_at: "2026-10-28T00:00:00Z",
  };
}

function fdForm() {
  server.use(http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })));
  return renderWithQueries(
    <MemoryRouter initialEntries={["/money/add/fd"]}>
      <main>
        <Routes>
          <Route path="/money/add/:type" element={<AddAssetPage />} />
          <Route path="/money/asset/:id" element={<AssetDetailPage />} />
        </Routes>
      </main>
    </MemoryRouter>,
  );
}

async function fillFdForm(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText(/Bank or post office/), "Demo Bank");
  await user.type(screen.getByLabelText(/Amount deposited/), "5L");
  await user.type(screen.getByLabelText(/Interest rate \(%\)/), "7.10");
  fireEvent.change(screen.getByLabelText(/Matures \(month and year\)/), {
    target: { value: "2027-03" },
  });
}

function memberList() {
  return http.get(/\/v1\/members$/, () => HttpResponse.json([]));
}

describe("asset forms and autosave", () => {
  it("posts a valid fixed deposit after the debounce and shows valuation", async () => {
    const user = userEvent.setup();
    let postBody: Record<string, unknown> | null = null;
    server.use(
      memberList(),
      http.post(/\/v1\/assets$/, async ({ request }) => {
        postBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(assetResponse(postBody), { status: 201 });
      }),
    );
    fdForm();
    await fillFdForm(user);
    await waitFor(() => expect(postBody).not.toBeNull(), { timeout: 2500 });
    expect(postBody).toMatchObject({
      asset_type: "fd",
      bank: "Demo Bank",
      principal: "500000.00",
      rate_pct: "7.10",
      maturity: "2027-03",
    });
    expect(screen.getByText(/Value today/)).toBeInTheDocument();
    expect(screen.getByText(/At maturity/)).toBeInTheDocument();
    expect(screen.getByText("Saved")).toBeInTheDocument();
  });

  it("patches an edited principal only after the debounce", async () => {
    const user = userEvent.setup();
    const original = assetResponse({
      asset_type: "fd",
      bank: "Demo Bank",
      principal: "500000.00",
      rate_pct: "7.10",
      maturity: "2027-03",
      compounding: "quarterly",
      payout: "cumulative",
    });
    let patchBody: Record<string, unknown> | null = null;
    server.use(
      memberList(),
      http.post(/\/v1\/assets$/, () =>
        HttpResponse.json(original, { status: 201 }),
      ),
      http.get(/\/v1\/assets\/asset-test-1$/, () =>
        HttpResponse.json({
          ...original,
          history: [],
        } satisfies AssetDetailResponse),
      ),
      http.patch(/\/v1\/assets\/asset-test-1$/, async ({ request }) => {
        patchBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({
          ...original,
          details: { ...original.details, ...patchBody },
        });
      }),
    );
    fdForm();
    await fillFdForm(user);
    await screen.findByText("Saved");
    await user.click(screen.getByRole("button", { name: "View asset" }));
    const principal = await screen.findByLabelText(/Amount deposited/);
    await user.clear(principal);
    await user.type(principal, "6L");
    fireEvent.blur(principal);
    await waitFor(() => expect(patchBody).not.toBeNull(), { timeout: 2500 });
    expect(patchBody).toMatchObject({ principal: "600000.00" });
  });

  it("shows the certificate prompt for a rate_needed response", async () => {
    const user = userEvent.setup();
    server.use(
      memberList(),
      http.post(/\/v1\/assets$/, () =>
        HttpResponse.json(
          { code: "rate_needed", title: "Rate needed", errors: [] },
          { status: 422 },
        ),
      ),
    );
    fdForm();
    await fillFdForm(user);
    expect(
      await screen.findByText(
        "Enter the interest rate printed on your certificate",
      ),
    ).toBeInTheDocument();
  });

  it("maps a 422 field error to the matching field", async () => {
    const user = userEvent.setup();
    server.use(
      memberList(),
      http.post(/\/v1\/assets$/, () =>
        HttpResponse.json(
          {
            code: "validation_error",
            title: "Check the form",
            errors: [
              { field: "rate_pct", message: "Use the certificate rate." },
            ],
          },
          { status: 422 },
        ),
      ),
    );
    fdForm();
    await fillFdForm(user);
    const rate = screen.getByLabelText(/Interest rate \(%\)/);
    await waitFor(() => expect(rate).toHaveAttribute("aria-invalid", "true"), {
      timeout: 2500,
    });
    expect(screen.getByText("Use the certificate rate.")).toBeInTheDocument();
    expect(screen.getByLabelText(/Bank or post office/)).not.toHaveAttribute(
      "aria-invalid",
      "true",
    );
  });

  it("has no axe violations on the fixed deposit form", async () => {
    server.use(memberList());
    const { container } = fdForm();
    await screen.findByRole("heading", { name: "Fixed deposit" });
    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
