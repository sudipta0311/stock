import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import { MoneyMapPage } from "../../src/money/MoneyMapPage";
import type {
  AssetResponse,
  LoanListResponse,
  NetworthResponse,
} from "../../src/money/api";
import { CHECKLIST_STORAGE_KEY } from "../../src/money/config";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";
import { loanSummary } from "../features/fixtures";
import { HouseholdProvider } from "../../src/household/HouseholdContext";
import { HouseholdControls } from "../../src/household/HouseholdControls";

const memberId = "member-test-1";

function networth(overrides: Partial<NetworthResponse> = {}): NetworthResponse {
  return {
    net_worth: "1000000.00",
    assets_total: "1000000.00",
    liabilities_total: "0.00",
    investable_total: "900000.00",
    by_sleeve: [],
    liquidity_ladder: {
      instant: "100000.00",
      within_1y: "300000.00",
      locked: "600000.00",
    },
    debt_insight: {
      message_key: "debt_share",
      debt_pct_including_locked: "41",
      locked_debt_amount: "410000.00",
    },
    maturity_calendar: [],
    unvalued_assets: [],
    completeness: { ticked: [], done: [], pct: "80" },
    as_of: "2026-10-28",
    market_data_as_of: null,
    holdings: [],
    ...overrides,
  };
}

function savedAsset(
  body: Record<string, unknown>,
  id = "asset-test-1",
): AssetResponse {
  return {
    id,
    asset_type: String(body.asset_type),
    member_id: typeof body.member_id === "string" ? body.member_id : memberId,
    account_id: null,
    joint: false,
    details: body,
    valuation: {
      value_today: "500000.00",
      maturity_value: "550000.00",
      maturity_date: "2027-03",
    },
    valuation_note: "Estimate based on your details.",
    valuation_status: "valued",
    value_cached: "500000.00",
    value_as_of: "2026-10-28",
    status: "active",
    created_at: "2026-10-28T00:00:00Z",
    updated_at: "2026-10-28T00:00:00Z",
  };
}

function baseHandlers(
  networthResponse = networth(),
  loansResponse: LoanListResponse = { ...loanSummary(), loans: [] },
) {
  return [
    http.get(/\/v1\/me$/, () =>
      HttpResponse.json({
        id: "user-test",
        email: "test@example.test",
        created_at: "2026-10-28T00:00:00Z",
        display_name: "Test member",
        household_owner_id: "household-test",
        me_member_id: memberId,
        checklist: [],
        profile: null,
        residency: {
          country: "IN",
          status: "RI",
          settle_plan: "undecided",
          us_person: false,
        },
        role: "owner",
      }),
    ),
    http.get(/\/v1\/assets$/, () => HttpResponse.json([])),
    http.get(/\/v1\/networth$/, () => HttpResponse.json(networthResponse)),
    http.get(/\/v1\/members$/, () => HttpResponse.json([])),
    http.get(/\/v1\/households$/, () => HttpResponse.json([])),
    http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })),
    http.get(/\/v1\/loans$/, () => HttpResponse.json(loansResponse)),
    http.get(/\/v1\/imports$/, () => HttpResponse.json([])),
    http.put(/\/v1\/me\/checklist$/, async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
  ];
}

function renderMap(initial = "/money") {
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[initial]}>
        <HouseholdControls />
        <Routes>
          <Route path="/money" element={<MoneyMapPage />} />
          <Route path="/money/add/:type" element={<AddAssetPage />} />
        </Routes>
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

describe("Money Map", () => {
  it("shows loan balances and links each row to loan details", async () => {
    server.use(...baseHandlers(networth(), loanSummary()));
    renderMap();

    expect(await screen.findByText("Loans")).toBeInTheDocument();
    expect(screen.getByText(/Total left/)).toHaveTextContent("per month");
    const loanLink = screen.getByRole("link", {
      name: /Home loan.*Sample Bank.*Ends Mar 2031/,
    });
    expect(loanLink).toHaveAttribute("href", "/money/loan/loan-test-1");
    expect(
      within(loanLink).getByLabelText("Amount repaid"),
    ).toBeInTheDocument();
  });

  it("defaults Denmark savings to DKK and posts the normalized balance", async () => {
    const user = userEvent.setup();
    let postBody: Record<string, unknown> | null = null;
    const member = {
      id: memberId,
      created_at: "2026-10-28T00:00:00Z",
      display_name: "Test member",
      is_self: true,
      relationship: "self",
      residency: {
        country: "DK",
        status: "NRI",
        settle_plan: "undecided",
        us_person: false,
      },
      tax: {},
      investing: {},
    };
    server.use(
      http.get(/\/v1\/members$/, () => HttpResponse.json([member])),
      http.get(/\/v1\/fx\/dkk-inr$/, () =>
        HttpResponse.json({ date: "2026-10-28", rate: "12.5" }),
      ),
      http.post(/\/v1\/assets$/, async ({ request }) => {
        postBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(savedAsset(postBody), { status: 201 });
      }),
      ...baseHandlers(),
    );
    renderMap("/money/add/savings");

    const bank = await screen.findByLabelText(/Bank/);
    const currency = screen.getByLabelText(/Currency/);
    const balance = screen.getByLabelText(/Current balance/);
    await waitFor(() => expect(currency).toHaveValue("DKK"));
    expect(
      bank.compareDocumentPosition(currency) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      currency.compareDocumentPosition(balance) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    await user.type(bank, "Synthetic Bank");
    await user.type(balance, "85,000");

    await waitFor(() => expect(postBody).not.toBeNull(), { timeout: 2500 });
    expect(postBody).toMatchObject({ currency: "DKK", balance: "85000" });
  });

  it("shows a DKK savings balance with the household-currency value", async () => {
    const asset = savedAsset({
      asset_type: "savings",
      bank: "Synthetic Bank",
      balance: "85000.00",
      currency: "DKK",
    });
    server.use(
      http.get(/\/v1\/assets$/, () =>
        HttpResponse.json([
          {
            ...asset,
            id: "asset-dkk-savings",
            value_cached: "125000.00",
            details: {
              ...asset.details,
              balance: "85000.00",
              currency: "DKK",
            },
          },
        ]),
      ),
      ...baseHandlers(),
    );
    renderMap();

    const row = await screen.findByRole("link", { name: /Synthetic Bank/ });
    expect(within(row).getByText("DKK 85,000")).toBeInTheDocument();
    expect(within(row).getByText("₹1.25 L")).toBeInTheDocument();
  });

  it("quick-adds a Danish bank balance with DKK formatting", async () => {
    const user = userEvent.setup();
    server.use(
      ...baseHandlers(),
      http.post(/\/v1\/assets\/quick-add$/, () =>
        HttpResponse.json({
          asset_type: "savings",
          draft: {
            asset_type: "savings",
            bank: "Danske Bank",
            currency: "DKK",
            balance: "85000.00",
          },
          missing: [],
          confidence: "high",
        }),
      ),
    );
    renderMap();

    await user.type(
      await screen.findByLabelText("Quick add"),
      "Danske Bank savings 85000 kr",
    );
    await user.click(screen.getByRole("button", { name: "Add details" }));

    expect(await screen.findByLabelText(/Bank/)).toHaveValue("Danske Bank");
    expect(screen.getByLabelText(/Currency/)).toHaveValue("DKK");
    expect(screen.getByLabelText(/Current balance/)).toHaveValue("85,000");
  });

  it.each([
    {
      phrase: "FD 5L Demo Bank 7.1% till Mar 2027",
      result: {
        asset_type: "fd",
        draft: {
          asset_type: "fd",
          bank: "Demo Bank",
          principal: "500000.00",
          rate_pct: "7.1",
          maturity: "2027-03",
        },
        missing: [],
        confidence: "high",
      },
      label: "Bank or post office",
      expected: "Demo Bank",
    },
    {
      phrase: "PPF 6.5 lakh",
      result: {
        asset_type: "ppf",
        draft: { asset_type: "ppf", balance: "650000.00" },
        missing: ["opened_fy_start_year"],
        confidence: "medium",
      },
      label: "Current PPF balance",
      missingLabel: "Year opened",
      expected: "6,50,000",
    },
    {
      phrase: "LIC 10L premium 52000",
      result: {
        asset_type: "traditional_policy",
        draft: {
          asset_type: "traditional_policy",
          insurer: "LIC",
          sum_assured: "1000000.00",
          annual_premium: "52000.00",
        },
        missing: ["plan", "start_year", "term_years", "policy_kind"],
        confidence: "medium",
      },
      label: "Plan name or number",
      missingLabel: "Started in year",
      expected: "",
    },
  ])(
    "uses quick-add to prefill the $phrase card and marks missing fields",
    async ({ phrase, result, label, missingLabel, expected }) => {
      const user = userEvent.setup();
      server.use(
        ...baseHandlers(),
        http.post(/\/v1\/assets\/quick-add$/, () => HttpResponse.json(result)),
      );
      renderMap();
      const input = await screen.findByLabelText("Quick add");
      await user.type(input, phrase);
      await user.click(screen.getByRole("button", { name: "Add details" }));
      const field = await screen.findByLabelText(
        new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
      );
      expect(field).toHaveValue(expected);
      if (result.missing.length) {
        const missingField = await screen.findByLabelText(
          new RegExp(missingLabel!.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
        );
        expect(missingField.className).toContain("border-warning");
        expect(screen.getByText(/Quick add left/)).toBeInTheDocument();
      }
    },
  );

  it("saves selected tiles and walks the selected types in checklist order", async () => {
    const user = userEvent.setup();
    const created: string[] = [];
    server.use(
      ...baseHandlers(),
      http.post(/\/v1\/assets$/, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        created.push(String(body.asset_type));
        return HttpResponse.json(savedAsset(body, `asset-${created.length}`), {
          status: 201,
        });
      }),
    );
    renderMap();
    await user.click(await screen.findByRole("button", { name: "NPS" }));
    await user.click(screen.getByRole("button", { name: "Fixed deposits" }));
    await user.click(screen.getByRole("button", { name: "EPF" }));
    await user.click(
      screen.getByRole("button", { name: "Continue (3 selected)" }),
    );
    expect(
      await screen.findByRole("heading", { name: "EPF / VPF" }),
    ).toBeInTheDocument();
    await user.type(screen.getByLabelText(/Current EPF balance/), "8L");
    await screen.findByText("Saved");
    await user.click(
      screen.getByRole("button", { name: "Continue to next item" }),
    );
    expect(
      await screen.findByRole("heading", { name: "Fixed deposit" }),
    ).toBeInTheDocument();
    expect(screen.getByText("2 of 3")).toBeInTheDocument();
    await user.type(screen.getByLabelText(/Bank or post office/), "Demo Bank");
    await user.type(screen.getByLabelText(/Amount deposited/), "5L");
    await user.type(screen.getByLabelText(/Interest rate \(%\)/), "7.1");
    fireEvent.change(screen.getByLabelText(/Matures \(month and year\)/), {
      target: { value: "2027-03" },
    });
    await waitFor(() => expect(created).toHaveLength(2), { timeout: 2500 });
    await user.click(
      screen.getByRole("button", { name: "Continue to next item" }),
    );
    expect(
      await screen.findByRole("heading", { name: "NPS" }),
    ).toBeInTheDocument();
    expect(created).toEqual(["epf", "fd"]);
    expect(
      JSON.parse(localStorage.getItem(CHECKLIST_STORAGE_KEY) ?? "[]"),
    ).toEqual(["nps", "fd", "epf"]);
  });

  it("shows only the percentage when no checklist categories were selected", async () => {
    server.use(...baseHandlers());
    renderMap();
    expect(
      await screen.findByText("Money Map 80% complete"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/add EPF to finish/)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/everything is up to date/),
    ).not.toBeInTheDocument();
  });

  it("names the next incomplete category the household selected", async () => {
    server.use(
      ...baseHandlers(
        networth({
          completeness: {
            ticked: ["property", "savings"],
            done: [],
            pct: "0",
          },
        }),
      ),
    );
    renderMap();

    expect(
      await screen.findByText("Money Map 0% complete · add Savings to finish"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/add EPF to finish/)).not.toBeInTheDocument();
  });

  it("shows plain names and whole percentages instead of codes", async () => {
    server.use(
      ...baseHandlers(
        networth({
          by_sleeve: [
            { sleeve: "debt", amount: "600000.00", pct: "46.88" },
            { sleeve: "cash", amount: "100000.00", pct: "9.87" },
            { sleeve: "real_estate", amount: "200000.00", pct: "18.75" },
            { sleeve: "other", amount: "50000.00", pct: "4.69" },
            { sleeve: "gold", amount: "0.00", pct: "0.00" },
          ],
          maturity_calendar: [
            {
              date: "2026-11-05",
              asset_id: "asset-test-1",
              kind: "loan_payment",
              amount: "34500.00",
              label: "Loan monthly payment",
            },
          ],
        }),
      ),
    );
    renderMap();

    expect(await screen.findByText("Loan monthly payment")).toBeInTheDocument();
    expect(screen.queryByText(/loan_payment/)).not.toBeInTheDocument();
    expect(screen.queryByText(/real_estate/)).not.toBeInTheDocument();
    expect(screen.getByText("Cash")).toBeInTheDocument();
    expect(screen.getAllByText("Other holdings")).toHaveLength(1);
    // Kinds with nothing in them aren't listed.
    const split = screen.getByRole("heading", { name: "Split by type" })
      .parentElement as HTMLElement;
    expect(within(split).queryByText("Gold")).not.toBeInTheDocument();
    expect(screen.getByText(/· 47%/)).toBeInTheDocument();
    expect(screen.queryByText(/46\.88/)).not.toBeInTheDocument();
  });

  it("puts one upload action first and closes the completed checklist", async () => {
    server.use(
      ...baseHandlers(
        networth({
          completeness: { ticked: ["epf"], done: ["epf"], pct: "80" },
        }),
      ),
    );
    const { container } = renderMap();
    const root = await screen
      .findByText("Upload your statement")
      .then(() => container.querySelector('[data-copy-screen="money"]'));
    expect(root?.firstElementChild).toHaveTextContent("Upload your statement");
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    const checklist = Array.from(root?.querySelectorAll("details") ?? []).find(
      (details) =>
        details.querySelector("summary")?.textContent ===
        "What else do you have?",
    );
    expect(checklist).not.toHaveAttribute("open");
  });

  it("passes the selected household member as member_id", async () => {
    const user = userEvent.setup();
    const assetRequests: string[] = [];
    const networthRequests: string[] = [];
    server.use(
      http.get(/\/v1\/assets$/, ({ request }) => {
        assetRequests.push(
          new URL(request.url).searchParams.get("member_id") ?? "",
        );
        return HttpResponse.json([]);
      }),
      http.get(/\/v1\/networth$/, ({ request }) => {
        networthRequests.push(
          new URL(request.url).searchParams.get("member_id") ?? "",
        );
        return HttpResponse.json(networth());
      }),
      http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })),
      http.get(/\/v1\/households$/, () => HttpResponse.json([])),
      http.get(/\/v1\/members$/, () =>
        HttpResponse.json([
          {
            id: memberId,
            display_name: "Household member",
            created_at: "2026-10-28T00:00:00Z",
            is_self: true,
            relationship: "self",
            residency: {
              country: "IN",
              status: "RI",
              settle_plan: "undecided",
              us_person: false,
              planned_return_year: null,
            },
            tax: {},
            investing: {
              monthly_sip_capacity: "0.00",
              existing_sips: [],
              new_money_this_week: "0.00",
              has_lrs_broker: false,
              prefers_no_paperwork: false,
              satellite_pct: "0.00",
              birth_year: null,
              ask_used_this_fy_dkk: "0.00",
              lrs_used_this_fy: "0.00",
            },
          },
          {
            id: "member-partner-test",
            display_name: "My name",
            created_at: "2026-10-28T00:00:00Z",
            is_self: false,
            relationship: "spouse",
            residency: {
              country: "IN",
              status: "RI",
              settle_plan: "undecided",
              us_person: false,
              planned_return_year: null,
            },
            tax: {},
            investing: {
              monthly_sip_capacity: "0.00",
              existing_sips: [],
              new_money_this_week: "0.00",
              has_lrs_broker: false,
              prefers_no_paperwork: false,
              satellite_pct: "0.00",
              birth_year: null,
              ask_used_this_fy_dkk: "0.00",
              lrs_used_this_fy: "0.00",
            },
          },
        ]),
      ),
      http.get(/\/v1\/households$/, () => HttpResponse.json([])),
      http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })),
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          email: "test@example.test",
          created_at: "2026-10-28T00:00:00Z",
          display_name: "Test member",
          household_owner_id: "household-test",
          me_member_id: "member-partner-test",
          checklist: [],
          profile: null,
          residency: {
            country: "IN",
            status: "RI",
            settle_plan: "undecided",
            us_person: false,
          },
          role: "partner",
        }),
      ),
      http.get(/\/v1\/imports$/, () => HttpResponse.json([])),
      http.put(/\/v1\/me\/checklist$/, async ({ request }) =>
        HttpResponse.json(await request.json()),
      ),
    );
    renderMap();
    await screen.findByRole("combobox", { name: "View" });
    await user.selectOptions(
      screen.getByRole("combobox", { name: "View" }),
      memberId,
    );
    await waitFor(() => {
      expect(assetRequests).toContain(memberId);
      expect(networthRequests).toContain(memberId);
    });
  });

  it("has no axe violations on the Money tab", async () => {
    server.use(...baseHandlers());
    const { container } = render(
      <main>
        <MemoryRouter>
          <MoneyMapPage />
        </MemoryRouter>
      </main>,
    );
    await screen.findByText("What else do you have?");
    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
