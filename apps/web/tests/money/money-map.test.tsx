import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import { MoneyMapPage } from "../../src/money/MoneyMapPage";
import type { AssetResponse, NetworthResponse } from "../../src/money/api";
import { CHECKLIST_STORAGE_KEY } from "../../src/money/config";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";
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

function baseHandlers(networthResponse = networth()) {
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
      expected: "650000.00",
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

  it("shows the completeness sentence from the API response", async () => {
    server.use(...baseHandlers());
    renderMap();
    expect(
      await screen.findByText(/Money Map 80% complete · add EPF to finish/),
    ).toBeInTheDocument();
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
