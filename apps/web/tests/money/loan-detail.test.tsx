import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { HouseholdProvider } from "../../src/household/HouseholdContext";
import { LoanDetailPage } from "../../src/money/LoanDetailPage";
import type { LoanDetailResponse, LoanEventWrite } from "../../src/money/api";
import { renderWithQueries } from "../features/render";
import { server } from "../msw";

const loanId = "loan-test-1";

function loanDetail(
  overrides: Partial<LoanDetailResponse> = {},
): LoanDetailResponse {
  return {
    id: loanId,
    name: "Home loan",
    lender: "Sample Bank",
    currency: "INR",
    outstanding: "2800000.00",
    outstanding_inr: "2800000.00",
    rate_pct: "8.5",
    rate_type: "floating",
    tax_relief: "in_home_self_occupied",
    emi: "26034.70",
    emi_inr: "26034.70",
    months_left: 204,
    payoff_month: { year: 2043, month: 10 },
    percent_repaid: "6.67",
    never_pays_off: false,
    fx_missing: false,
    principal_paid: "200000.00",
    interest_paid: "100000.00",
    interest_to_pay: "2491774.43",
    next_emi_date: "2026-11-10",
    yearly_balances: [
      { month: { year: 2025, month: 10 }, outstanding: "3000000.00" },
      { month: { year: 2026, month: 10 }, outstanding: "2800000.00" },
    ],
    scheduled_payments: [
      {
        on_date: "2026-11-10",
        amount: "26034.70",
        principal: "6223.03",
        interest: "19811.67",
        outstanding: "2793776.97",
      },
    ],
    events: [
      {
        id: "event-test-1",
        kind: "prepayment",
        on_date: "2026-10-01",
        amount: "200000.00",
        rate_pct: null,
        emi: null,
        note: "October statement",
      },
    ],
    advice: "Pay extra toward this loan.",
    advice_kind: "PAY_EXTRA",
    suggested_amount_inr: "5000.00",
    reason_codes: ["loan_cost_above_safe_return"],
    tax_source_urls: [
      "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1?mobile-app=1",
    ],
    tax_relief_needs_review: true,
    ...overrides,
  };
}

function handlers(role: "owner" | "viewer" = "owner") {
  return [
    http.get(/\/v1\/me$/, () =>
      HttpResponse.json({
        id: "user-test",
        email: "test@example.test",
        created_at: "2026-10-28T00:00:00Z",
        display_name: "Test member",
        household_owner_id: "household-test",
        me_member_id: "member-test",
        checklist: [],
        profile: null,
        residency: {
          country: "IN",
          status: "RI",
          settle_plan: "undecided",
          us_person: false,
        },
        role,
      }),
    ),
    http.get(/\/v1\/members$/, () => HttpResponse.json([])),
    http.get(/\/v1\/households$/, () => HttpResponse.json([])),
    http.get(new RegExp(`/v1/loans/${loanId}$`), () =>
      HttpResponse.json(loanDetail()),
    ),
  ];
}

function renderLoan() {
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[`/money/loan/${loanId}`]}>
        <Routes>
          <Route path="/money/loan/:id" element={<LoanDetailPage />} />
        </Routes>
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

describe("Loan detail", () => {
  it("shows a repayment chart, one main action, and sourced tax review details", async () => {
    server.use(...handlers());
    const { container } = renderLoan();

    expect(await screen.findByText("Paid off in Oct 2043")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Loan balance over time" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        "Home loan interest may apply. Please check before acting.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Official tax information" }),
    ).toHaveAttribute(
      "href",
      "https://www.incometax.gov.in/iec/foportal/help/individual/return-applicable-1?mobile-app=1",
    );
    expect(container.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    const visibleCopy = container.textContent?.toLowerCase() ?? "";
    for (const term of [
      "sleeve",
      "satellite",
      "tranche",
      "route",
      "pfic",
      "abis",
      "priips",
      "lrs",
      "tcs",
      "xirr",
      "rnor",
      "rebalance",
      "allocation",
      "isin",
      "ucits",
    ]) {
      expect(visibleCopy).not.toContain(term);
    }
  });

  it("records an extra payment from the payment sheet", async () => {
    const user = userEvent.setup();
    let saved: LoanEventWrite | undefined;
    server.use(
      ...handlers(),
      http.post(
        new RegExp(`/v1/loans/${loanId}/events$`),
        async ({ request }) => {
          saved = (await request.json()) as LoanEventWrite;
          return HttpResponse.json(
            {
              id: "event-test-new",
              kind: saved.kind,
              on_date: saved.on_date,
              amount: saved.amount ?? null,
              rate_pct: saved.rate_pct ?? null,
              emi: saved.emi ?? null,
              note: saved.note ?? null,
            },
            { status: 201 },
          );
        },
      ),
    );
    renderLoan();
    await user.click(
      await screen.findByRole("button", { name: "Record a payment" }),
    );
    const dialog = await screen.findByRole("dialog");
    await user.type(within(dialog).getByLabelText(/Amount \(INR\)/), "2500");
    await user.click(
      within(dialog).getByRole("button", { name: "Save update" }),
    );

    await waitFor(() => expect(saved?.kind).toBe("prepayment"));
    expect(saved?.amount).toBe("2500");
  });

  it("shows both server-calculated prepayment choices", async () => {
    const user = userEvent.setup();
    let previewed = "";
    server.use(
      ...handlers(),
      http.post(
        new RegExp(`/v1/loans/${loanId}/prepay-preview$`),
        async ({ request }) => {
          previewed = JSON.stringify(await request.json());
          return HttpResponse.json({
            keep_emi_months_saved: 36,
            keep_emi_interest_saved: "756578.72",
            keep_end_date_new_emi: "24299.05",
            keep_end_date_interest_saved: "216553.78",
          });
        },
      ),
    );
    renderLoan();
    await user.click(
      await screen.findByRole("button", { name: "What if I pay extra?" }),
    );
    const dialog = await screen.findByRole("dialog");
    await user.type(
      within(dialog).getByLabelText(/Extra amount \(INR\)/),
      "200000",
    );
    await user.click(
      within(dialog).getByRole("button", { name: "Compare options" }),
    );

    expect(
      await within(dialog).findByText("Finish 36 months sooner"),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        (_, element) =>
          element?.textContent?.replace(/\s+/g, " ").trim() ===
          "Save \u20b97.57 L in interest.",
      ),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        (_, element) =>
          element?.textContent?.replace(/\s+/g, " ").trim() ===
          "Lower the monthly payment to \u20b924,299",
      ),
    ).toBeInTheDocument();
    expect(previewed).toBe('{"amount":"200000"}');
  });

  it("keeps payment and delete controls read-only for a viewer", async () => {
    server.use(...handlers("viewer"));
    renderLoan();

    await screen.findByText("Paid off in Oct 2043");
    expect(
      screen.queryByRole("button", { name: "Record a payment" }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Delete Extra payment" }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: "What if I pay extra?" }),
    ).toBeEnabled();
  });
});
