import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import englishCopy from "../../src/copy/en.json";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import { assetFieldCopy } from "../../src/money/labels.en";
import { MoneyMapPage } from "../../src/money/MoneyMapPage";
import {
  clearOfflineAssetWrites,
  listQueuedAssetWrites,
} from "../../src/money/offline-queue";
import type {
  AssetResponse,
  LoanListResponse,
  NetworthResponse,
} from "../../src/money/api";
import { CHECKLIST_STORAGE_KEY } from "../../src/money/config";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";
import { loanSummary } from "../features/fixtures";
import {
  HouseholdProvider,
  PRIMARY_CURRENCY_STORAGE_KEY,
} from "../../src/household/HouseholdContext";
import { HouseholdControls } from "../../src/household/HouseholdControls";
import { bannedTermsIncludingDisclosuresAndFields } from "../plain-language";

const memberId = "member-test-1";

function localDateFromToday(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() + days);
  return [
    date.getFullYear(),
    String(date.getMonth() + 1).padStart(2, "0"),
    String(date.getDate()).padStart(2, "0"),
  ].join("-");
}

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

function MapHarness() {
  const [renderCount, setRenderCount] = useState(0);
  const location = useLocation();
  return (
    <>
      <button
        type="button"
        aria-label="Rerender map"
        className="sr-only"
        onClick={() => setRenderCount((current) => current + 1)}
      >
        {renderCount}
      </button>
      <output data-testid="map-search">{location.search}</output>
      <HouseholdControls />
      <Routes>
        <Route path="/money" element={<MoneyMapPage />} />
        <Route path="/money/add/:type" element={<AddAssetPage />} />
      </Routes>
    </>
  );
}

function renderMap(initial = "/money") {
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[initial]}>
        <MapHarness />
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

async function openWhatElse(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByText("Add manually"));
}

async function openQuickAdd(user: ReturnType<typeof userEvent.setup>) {
  await openWhatElse(user);
  await user.click(screen.getByText("Quick add", { selector: "summary" }));
}

describe("Money Map", () => {
  it("uses plain labels in manual bond and NPS forms", () => {
    expect(assetFieldCopy.bond?.isin).toEqual({
      label: "Investment code",
      help: "Copy the code from your statement.",
    });
    expect(assetFieldCopy.bond?.issuer?.label).toBe("Issued by");
    expect(assetFieldCopy.bond?.coupon_pct?.label).toBe("Interest rate (%)");
    expect(assetFieldCopy.nps?.equity_pct?.label).toBe("Shares (%)");
    expect(assetFieldCopy.nps?.corporate_debt_pct?.label).toBe(
      "Company bonds (%)",
    );
    expect(assetFieldCopy.nps?.govt_debt_pct?.label).toBe(
      "Government bonds (%)",
    );
  });
  it("shows loan balances and links each row to loan details", async () => {
    server.use(...baseHandlers(networth(), loanSummary()));
    renderMap();

    const loanLink = await screen.findByRole("link", {
      name: /Home loan.*Sample Bank.*Ends Mar 2031/,
    });
    expect(screen.getByText(/Monthly payments/)).toBeInTheDocument();
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
    await openQuickAdd(user);

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
      await openQuickAdd(user);
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
    await openWhatElse(user);
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
    await user.click(screen.getByRole("button", { name: "Save and continue" }));
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
    await user.click(screen.getByRole("button", { name: "Save and continue" }));
    expect(
      await screen.findByRole("heading", { name: "NPS" }),
    ).toBeInTheDocument();
    expect(created).toEqual(["epf", "fd"]);
    expect(
      JSON.parse(localStorage.getItem(CHECKLIST_STORAGE_KEY) ?? "[]"),
    ).toEqual(["nps", "fd", "epf"]);
  });

  it("keeps checklist progress plain without a completion percentage", async () => {
    const user = userEvent.setup();
    server.use(...baseHandlers());
    renderMap();
    await openWhatElse(user);
    await screen.findByText("Add manually");
    expect(
      screen.queryByText(/Money Map \d+% complete/),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/add EPF to finish/)).not.toBeInTheDocument();
    expect(
      screen.queryByText(/everything is up to date/),
    ).not.toBeInTheDocument();
  });

  it("hides the step count and returns after one selected item", async () => {
    const user = userEvent.setup();
    localStorage.removeItem(CHECKLIST_STORAGE_KEY);
    server.use(...baseHandlers());
    renderMap();
    await openWhatElse(user);
    await user.click(
      await screen.findByRole("button", { name: "Fixed deposits" }),
    );
    await user.click(
      screen.getByRole("button", { name: "Continue (1 selected)" }),
    );

    expect(
      await screen.findByRole("heading", { name: "Fixed deposit" }),
    ).toBeVisible();
    expect(screen.queryByText("1 of 1")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Save and return" }),
    ).toBeVisible();
    expect(
      screen.queryByRole("button", { name: "Save and continue" }),
    ).toBeNull();
  });

  it("names the next incomplete category the household selected", async () => {
    const user = userEvent.setup();
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
    await openWhatElse(user);

    await screen.findByText("Add manually");
    expect(screen.getByText("Add Savings to finish")).toBeInTheDocument();
    expect(screen.queryByText(/add EPF to finish/)).not.toBeInTheDocument();
  });

  it("shows plain names and whole percentages instead of codes", async () => {
    const user = userEvent.setup();
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
              // Due today: it must still show after midnight has passed.
              date: localDateFromToday(0),
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
    await user.click(await screen.findByText("More details"));

    expect(await screen.findByText("Loan monthly payment")).toBeInTheDocument();
    expect(screen.queryByText(/loan_payment/)).not.toBeInTheDocument();
    expect(screen.queryByText(/real_estate/)).not.toBeInTheDocument();
    expect(screen.getByText("Cash")).toBeInTheDocument();
    expect(screen.getAllByText("Other holdings")).toHaveLength(1);
    // Kinds with nothing in them aren't listed.
    const split = screen.getByRole("heading", { name: "Split by type" })
      .parentElement as HTMLElement;
    expect(within(split).queryByText("Gold")).not.toBeInTheDocument();
    expect(screen.getByText("47%")).toBeInTheDocument();
    expect(screen.queryByText(/46\.88/)).not.toBeInTheDocument();
  });

  it("shows one Add holdings action and keeps rare details closed", async () => {
    server.use(
      ...baseHandlers(
        networth({
          completeness: { ticked: ["epf"], done: ["epf"], pct: "80" },
        }),
      ),
    );
    const user = userEvent.setup();
    const { container } = renderMap();
    await screen.findByText("Net worth");
    const root = await screen
      .findByRole("button", { name: "Add holdings" })
      .then(() => container.querySelector('[data-copy-screen="money"]'));
    expect(root?.firstElementChild).toHaveTextContent("Net worth");
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    for (const group of ["Funds", "Shares", "Bonds and gold bonds", "Other"]) {
      expect(
        screen.queryByRole("heading", { name: group }),
      ).not.toBeInTheDocument();
    }
    const disclosureTitles = [
      "Add manually",
      "Previous uploads",
      "Check an investment",
      "More details",
    ];
    const disclosures = Array.from(
      root?.querySelectorAll("details") ?? [],
    ).filter((item) =>
      disclosureTitles.includes(
        item.querySelector("summary")?.textContent?.trim() ?? "",
      ),
    );
    expect(
      disclosures.map((item) =>
        item.querySelector("summary")?.textContent?.trim(),
      ),
    ).toEqual(disclosureTitles);
    for (const details of disclosures) {
      expect(details).not.toHaveAttribute("open");
      expect(details.className).toBe("rounded-xl border border-line px-3");
      const summary = details.querySelector("summary");
      expect(summary).toHaveClass(
        "min-h-12",
        "py-3",
        "text-sm",
        "font-semibold",
      );
      if (summary?.textContent?.trim() === "Add manually") {
        expect(summary).toHaveClass("focus-visible:outline-2");
      }
    }
    expect(screen.queryByText("Monthly amounts")).toBeNull();
    await user.click(screen.getByText("Add manually", { selector: "summary" }));
    expect(bannedTermsIncludingDisclosuresAndFields(container)).toEqual([]);
    await user.click(screen.getByRole("button", { name: "Add holdings" }));
    const sheet = screen.getByRole("dialog", { name: "Add holdings" });
    expect(
      within(sheet).getByRole("link", { name: "Upload a file" }),
    ).toBeVisible();
    expect(
      within(sheet).getByRole("link", { name: "Get statements from Gmail" }),
    ).toBeVisible();
    expect(
      within(sheet).getByRole("button", { name: "Add manually" }),
    ).toBeVisible();
  });

  it("opens the same add sheet from the empty action and the add query", async () => {
    const user = userEvent.setup();
    server.use(...baseHandlers(networth(), { ...loanSummary(), loans: [] }));
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json({
          members: [
            {
              id: memberId,
              name: "Synthetic family member",
              net_worth: "0.00",
            },
          ],
        }),
      ),
    );
    renderMap();
    await screen.findByRole("heading", { name: "Add your first holding" });
    expect(screen.queryByText("Monthly amounts")).not.toBeInTheDocument();
    expect(screen.queryByText("By family member")).not.toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", { name: "Add holdings" }),
    );
    expect(screen.getByRole("dialog", { name: "Add holdings" })).toBeVisible();
  });

  it("opens the add sheet for ?add=1", async () => {
    const user = userEvent.setup();
    server.use(...baseHandlers(networth(), { ...loanSummary(), loans: [] }));
    renderMap("/money?add=1");
    const dialog = await screen.findByRole("dialog", { name: "Add holdings" });
    expect(dialog).toBeVisible();
    expect(
      within(dialog).getByRole("heading", { name: "Add holdings" }),
    ).toHaveClass("outline-none");
    expect(
      within(dialog).getByRole("link", { name: "Upload a file" }),
    ).toHaveAttribute("href", "/import");
    await waitFor(() =>
      expect(screen.getByTestId("map-search")).toHaveTextContent(""),
    );
    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog", { name: "Add holdings" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Rerender map" }));
    expect(screen.queryByRole("dialog", { name: "Add holdings" })).toBeNull();
  });

  it("opens and focuses the manual chooser for either manual entry", async () => {
    const user = userEvent.setup();
    server.use(...baseHandlers(networth(), { ...loanSummary(), loans: [] }));
    const { unmount } = renderMap();
    await screen.findByRole("heading", { name: "Add your first holding" });
    await user.click(
      await screen.findByRole("button", { name: "Add holdings" }),
    );
    await user.click(screen.getByRole("button", { name: "Add manually" }));
    const chooser = screen.getByText("Add manually").closest("details");
    await waitFor(() => expect(chooser).toHaveAttribute("open"));
    await waitFor(() =>
      expect(chooser?.querySelector("summary")).toHaveFocus(),
    );
    expect(
      screen.getByText("Quick add", { selector: "summary" }).closest("details"),
    ).not.toHaveAttribute("open");
    expect(
      screen.queryByText(/Money Map \d+% complete/),
    ).not.toBeInTheDocument();

    unmount();
    renderMap("/money?add=manual");
    await waitFor(() => {
      expect(
        screen.getByText("Add manually").closest("details"),
      ).toHaveAttribute("open");
    });
    await waitFor(() =>
      expect(
        screen.getByText("Add manually", { selector: "summary" }),
      ).toHaveFocus(),
    );
  });

  it("groups holdings by their stored security type", async () => {
    const holdings = [
      ["equity", "Equity holding"],
      ["etf", "ETF holding"],
      ["mf", "Mutual fund holding"],
      ["bond", "Bond holding"],
      ["sgb", "Gold bond holding"],
      ["unknown", "Other security holding"],
    ] as const;
    server.use(
      ...baseHandlers(
        networth({
          holdings: holdings.map(([securityType, name], index) => ({
            added_by_you: false,
            currency: "INR",
            fx_as_of: null,
            fx_missing: false,
            name,
            price_as_of: null,
            price_source: "test",
            quantity: "1",
            security_id: `00000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`,
            security_type: securityType,
            status: "priced",
            units_entered: false,
            symbol: `TEST${index + 1}`,
            value: "100.00",
            value_usd: null,
          })),
        }),
      ),
    );
    renderMap();

    const expectedGroups = [
      ["Funds", ["ETF holding", "Mutual fund holding"]],
      ["Shares", ["Equity holding"]],
      ["Bonds and gold bonds", ["Bond holding", "Gold bond holding"]],
      ["Other", ["Other security holding"]],
    ] as const;
    for (const [title, names] of expectedGroups) {
      const group = await screen.findByRole("region", { name: title });
      expect(within(group).getByRole("heading", { name: title })).toBeVisible();
      for (const name of names) {
        expect(within(group).getByText(name)).toBeVisible();
      }
      expect(within(group).queryByText(/^TEST\d+$/)).toBeNull();
    }
    expect(screen.getByText("6 items")).toBeVisible();
  });

  it("shows a recorded purchase, its value, and its edit and removal details", async () => {
    const user = userEvent.setup();
    const buy = {
      added_by_you: true,
      amount: "1000.00",
      amount_currency: "INR",
      bought_on: "2026-10-27",
      currency: "INR",
      fx_as_of: null,
      fx_missing: false,
      name: "Indian index fund",
      price_as_of: null,
      price_source: "recorded_buy",
      quantity: "",
      recorded_buy_id: "recorded-buy-test",
      security_id: null,
      security_type: "mf",
      status: "statement_value_used",
      symbol: "",
      units_entered: false,
      value: "1000.00",
      value_usd: null,
    } as const;
    let updateBody: Record<string, unknown> | null = null;
    let deleteCalled = false;
    server.use(
      ...baseHandlers(networth({ holdings: [buy] })),
      http.patch(
        /\/v1\/recorded-buys\/recorded-buy-test$/,
        async ({ request }) => {
          updateBody = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json({
            id: "recorded-buy-test",
            action_id: "action-buy-test",
            member_id: memberId,
            line: 0,
            bought_on: "2026-10-27",
            amount: "1250.00",
            currency: "INR",
            amount_inr: "1250.00",
            units: "25",
            units_entered: true,
            security_id: null,
            security_type: "mf",
            security_symbol: null,
            security_isin: null,
            name: "Indian index fund",
            sleeve: "equity_india",
            instrument_category: "in_core_index",
            account_type: null,
            status: "waiting",
          });
        },
      ),
      http.delete(/\/v1\/recorded-buys\/recorded-buy-test$/, () => {
        deleteCalled = true;
        return HttpResponse.json({ ...buy, status: "removed" });
      }),
    );
    renderMap();

    const group = await screen.findByRole("region", { name: "Funds" });
    expect(within(group).getByText("Added by you")).toBeVisible();
    expect(within(group).getAllByText("₹1,000")).toHaveLength(2);
    await user.click(
      within(group).getByRole("button", {
        name: /Indian index fund.*Added by you/,
      }),
    );
    const sheet = screen.getByRole("dialog", { name: "Added by you" });
    expect(
      within(sheet).getByText(englishCopy.money_ui.buy_details),
    ).toBeVisible();
    expect(within(sheet).getByRole("button", { name: "Edit" })).toBeVisible();
    expect(
      within(sheet).getByRole("button", {
        name: englishCopy.money_ui.remove_buy,
      }),
    ).toBeVisible();

    await user.click(within(sheet).getByRole("button", { name: "Edit" }));
    await user.clear(within(sheet).getByLabelText("Amount (INR)"));
    await user.type(within(sheet).getByLabelText("Amount (INR)"), "1250.00");
    await user.type(within(sheet).getByLabelText("Units (optional)"), "25");
    await user.click(
      within(sheet).getByRole("button", { name: "Save changes" }),
    );
    await waitFor(() =>
      expect(updateBody).toEqual({
        amount: "1250.00",
        bought_on: "2026-10-27",
        units: "25",
      }),
    );

    await user.click(
      within(group).getByRole("button", {
        name: /Indian index fund.*Added by you/,
      }),
    );
    const reopened = screen.getByRole("dialog", { name: "Added by you" });
    await user.click(
      within(reopened).getByRole("button", {
        name: englishCopy.money_ui.remove_buy,
      }),
    );
    expect(
      within(reopened).getByText(englishCopy.money_ui.remove_buy_confirm),
    ).toBeVisible();
    await user.click(
      within(reopened).getByRole("button", {
        name: englishCopy.money_ui.remove_buy,
      }),
    );
    await waitFor(() => expect(deleteCalled).toBe(true));
  });

  it("formats a group total using every holding, including hidden rows", async () => {
    window.localStorage.clear();
    window.localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, "DKK");
    const holdings: NetworthResponse["holdings"] = [
      ...Array.from({ length: 8 }, (_value, index) => ({
        added_by_you: false,
        currency: "INR",
        fx_as_of: null,
        fx_missing: false,
        name: `Visible fund ${index + 1}`,
        price_as_of: null,
        price_source: "test",
        quantity: "1",
        security_id: `00000000-0000-4000-8000-${String(index + 1).padStart(12, "0")}`,
        security_type: "mf",
        status: "priced" as const,
        units_entered: false,
        symbol: `SYNTHETIC-FUND-${index + 1}`,
        value: "1000.00",
        value_usd: null,
      })),
      {
        added_by_you: false,
        currency: "INR",
        fx_as_of: null,
        fx_missing: false,
        name: "Hidden fund",
        price_as_of: null,
        price_source: "test",
        quantity: "1",
        security_id: "00000000-0000-4000-8000-000000000099",
        security_type: "mf",
        status: "priced" as const,
        units_entered: false,
        symbol: "SYNTHETIC-HIDDEN-FUND",
        value: "125.00",
        value_usd: null,
      },
    ];
    server.use(
      http.get(/\/v1\/members$/, () =>
        HttpResponse.json([
          {
            id: memberId,
            display_name: "Test member",
            created_at: "2026-10-28T00:00:00Z",
            is_self: true,
            relationship: "self",
            residency: { country: "DK" },
          },
        ]),
      ),
      ...baseHandlers(networth({ holdings })),
      http.get(/\/v1\/fx\/dkk-inr$/, () =>
        HttpResponse.json({ date: "2026-10-28", rate: "12.5" }),
      ),
    );
    renderMap();

    const group = await screen.findByRole("region", { name: "Funds" });
    expect(await within(group).findByText("DKK 650")).toBeVisible();
    expect(
      within(group).getByRole("button", { name: "Show all 9" }),
    ).toBeVisible();
    expect(within(group).queryByText("Hidden fund")).not.toBeInTheDocument();
  });

  it("formats deposit group totals with the household money formatter", async () => {
    window.localStorage.clear();
    const deposits = [
      {
        ...savedAsset({ asset_type: "fd", label: "Synthetic deposit one" }),
        value_cached: "125000.00",
      },
      {
        ...savedAsset(
          { asset_type: "fd", label: "Synthetic deposit two" },
          "asset-deposit-two",
        ),
        value_cached: "250000.00",
      },
    ];
    server.use(...baseHandlers());
    server.use(http.get(/\/v1\/assets$/, () => HttpResponse.json(deposits)));
    renderMap();

    expect(
      await screen.findByRole("heading", { name: "Deposits" }),
    ).toBeVisible();
    expect(screen.getByText("₹3.75 L")).toBeVisible();
  });

  it("counts listed assets and holdings and shows a plain unit line", async () => {
    server.use(
      ...baseHandlers(
        networth({
          holdings: [
            {
              added_by_you: false,
              currency: "INR",
              fx_as_of: null,
              fx_missing: false,
              name: "Synthetic index fund",
              price_as_of: null,
              price_source: "test",
              quantity: "38647.6",
              security_id: "security-synthetic-fund",
              security_type: "mf",
              status: "priced",
              units_entered: false,
              symbol: "SYNTHETIC-INDEX-CODE",
              value: "10000.00",
              value_usd: null,
            },
          ],
        }),
      ),
    );
    server.use(
      http.get(/\/v1\/assets$/, () =>
        HttpResponse.json([
          savedAsset({ asset_type: "fd", label: "Synthetic deposit" }),
        ]),
      ),
    );
    renderMap();

    expect(await screen.findByText("2 items")).toBeVisible();
    const fundName = await screen.findByText("Synthetic index fund");
    const row = fundName.closest("a");
    expect(row).toHaveTextContent("38,648 units");
    expect(row).not.toHaveTextContent("SYNTHETIC-INDEX-CODE");
  });

  it("opens monthly fund amounts beside the funds list", async () => {
    const user = userEvent.setup();
    server.use(
      http.get(/\/v1\/fund-check$/, () =>
        HttpResponse.json({
          details: {
            funds: [
              {
                security_id: "security-fund-test",
                member_id: memberId,
                name: "Example index fund",
              },
            ],
            sips: [],
          },
        }),
      ),
      ...baseHandlers(
        networth({
          holdings: [
            {
              added_by_you: false,
              currency: "INR",
              fx_as_of: null,
              fx_missing: false,
              name: "Recorded synthetic fund",
              price_as_of: null,
              price_source: "test",
              quantity: "10",
              security_id: "security-recorded-fund",
              security_type: "mf",
              status: "priced",
              units_entered: false,
              symbol: "SYNTH",
              value: "10000.00",
              value_usd: null,
            },
          ],
        }),
      ),
    );
    renderMap();

    await user.click(await screen.findByText("Monthly amounts"));
    expect(await screen.findByLabelText("Example index fund")).toBeVisible();
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
      http.get(/\/v1\/loans$/, () =>
        HttpResponse.json({ ...loanSummary(), loans: [] }),
      ),
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

  it("queues an offline deposit and leaves with the waiting status", async () => {
    const user = userEvent.setup();
    const wasOnline = navigator.onLine;
    Object.defineProperty(navigator, "onLine", {
      configurable: true,
      value: false,
    });
    await clearOfflineAssetWrites();
    server.use(...baseHandlers(networth(), { ...loanSummary(), loans: [] }));
    try {
      renderMap("/money/add/fd");
      await user.type(
        screen.getByLabelText(/Bank or post office/),
        "Demo Bank",
      );
      await user.type(screen.getByLabelText(/Amount deposited/), "5L");
      await user.type(screen.getByLabelText(/Interest rate \(%\)/), "7.1");
      fireEvent.change(screen.getByLabelText(/Matures \(month and year\)/), {
        target: { value: "2027-03" },
      });
      await user.click(screen.getByRole("button", { name: "Save and return" }));

      await screen.findByText("Net worth");
      expect(
        await screen.findByText(
          "Saved on this device · waiting for connection",
        ),
      ).toBeVisible();
      const queued = await listQueuedAssetWrites();
      expect(queued).toHaveLength(1);
      expect(queued[0]?.body).toMatchObject({
        bank: "Demo Bank",
        principal: "500000.00",
        rate_pct: "7.1",
        maturity: "2027-03",
      });
    } finally {
      await clearOfflineAssetWrites();
      Object.defineProperty(navigator, "onLine", {
        configurable: true,
        value: wasOnline,
      });
    }
  });

  it("has no axe violations on the Money tab", async () => {
    server.use(...baseHandlers());
    const { container } = renderMap();
    await screen.findByText("Add manually");
    const results = await axe(container);
    expect(results.violations).toEqual([]);
  });
});
