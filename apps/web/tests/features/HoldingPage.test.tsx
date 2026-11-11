import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import {
  HouseholdProvider,
  PRIMARY_CURRENCY_STORAGE_KEY,
} from "../../src/household/HouseholdContext";
import { HoldingPage } from "../../src/money/HoldingPage";
import type { NetworthResponse } from "../../src/money/api";
import { networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";
import { bannedTermsIncludingDisclosuresAndFields } from "../plain-language";

const holding = {
  added_by_you: false,
  security_id: "security-test",
  name: "Demo Company",
  symbol: "DEMO",
  security_type: "equity",
  quantity: "0.423700",
  value: "20000.00",
  value_usd: null,
  cost_inr: null,
  avg_cost_inr: null,
  profit_loss_inr: null,
  currency: "INR",
  fx_as_of: null,
  fx_missing: false,
  price_as_of: "2026-10-28",
  price_source: "fixture",
  units_entered: false,
  status: "priced",
} as const;

function reviewPlan() {
  return weeklyPlan({
    holding_reviews: [
      {
        security_id: "security-test",
        member_id: "member-test-1",
        verdict: "sell_some",
        reasons: [
          { code: "valuation_stretch", params: {} },
          { code: "revenue_down_quarters", params: { count: "2" } },
        ],
        tax: { estimated_tax: "250.00" },
        checked_on: "2026-10-28",
        add_more_room_inr: null,
        checks: [
          { key: "company_results", status: "ok" },
          { key: "company_strength", status: "watch" },
        ],
      },
    ],
  });
}

function pageHandlers(
  withDanishMember = false,
  fxRate?: string,
  currentHolding: NetworthResponse["holdings"][number] = holding,
) {
  return [
    http.get(/\/v1\/networth$/, () =>
      HttpResponse.json(networth({ holdings: [currentHolding] })),
    ),
    http.get(/\/v1\/plan$/, () => HttpResponse.json(reviewPlan())),
    ...(withDanishMember
      ? [
          http.get(/\/v1\/me$/, () =>
            HttpResponse.json({
              id: "user-test",
              me_member_id: "member-test-1",
              household_owner_id: "owner-test",
              display_name: "Synthetic user",
              role: "owner",
            }),
          ),
          http.get(/\/v1\/members$/, () =>
            HttpResponse.json([
              {
                id: "member-test-1",
                display_name: "Synthetic user",
                relationship: "self",
                residency: {
                  country: "DK",
                  status: "NRI",
                  settle_plan: "undecided",
                  us_person: false,
                  planned_return_year: null,
                },
              },
            ]),
          ),
          http.get(/\/v1\/households$/, () => HttpResponse.json([])),
          http.get(/\/v1\/fx\/dkk-inr$/, () =>
            fxRate
              ? HttpResponse.json({ date: "2026-10-28", rate: fxRate })
              : HttpResponse.json({ date: "2026-10-28" }),
          ),
        ]
      : []),
  ];
}

function renderHolding(withHousehold = false) {
  const route = (
    <MemoryRouter initialEntries={["/money/holding/security-test"]}>
      <Routes>
        <Route path="/money/holding/:id" element={<HoldingPage />} />
      </Routes>
    </MemoryRouter>
  );
  return renderWithQueries(
    withHousehold ? <HouseholdProvider>{route}</HouseholdProvider> : route,
  );
}

describe("holding review", () => {
  it("shows a holding's buying cost and profit percentage", async () => {
    const boughtHolding = {
      ...holding,
      value: "422000.00",
      cost_inr: "132000.00",
      avg_cost_inr: "1000.00",
      profit_loss_inr: "290000.00",
    } as const;
    const user = userEvent.setup();
    server.use(...pageHandlers(false, undefined, boughtHolding));
    renderHolding();

    expect(
      await screen.findByRole("heading", { name: "Demo Company" }),
    ).toBeVisible();
    expect(screen.getByText("Profit or loss").parentElement).toHaveTextContent(
      "+₹2.9 L · +220%",
    );

    await user.click(screen.getByText("Holding details"));
    expect(screen.getByText("Bought for")).toBeVisible();
    expect(screen.getByText("₹1.32 L")).toBeVisible();
    expect(screen.getByText("Average buying price")).toBeVisible();
    expect(screen.getByText("₹1,000.00")).toBeVisible();
  });

  it("shows the value and leading reason before closed supporting sections", async () => {
    const user = userEvent.setup();
    const networthQueries: string[] = [];
    server.use(
      ...pageHandlers(),
      http.get(/\/v1\/networth$/, ({ request }) => {
        networthQueries.push(new URL(request.url).search);
        return HttpResponse.json(networth({ holdings: [holding] }));
      }),
    );
    const { container } = renderHolding();
    expect(
      await screen.findByRole("heading", { name: "Demo Company" }),
    ).toBeInTheDocument();
    expect(networthQueries.every((query) => query === "")).toBe(true);
    expect(screen.getByRole("link", { name: "Back to Money" })).toHaveAttribute(
      "href",
      "/money",
    );
    expect(screen.getByText("0.4237 shares")).toBeVisible();
    expect(screen.getByText("₹20,000")).toBeVisible();
    const buyingPrice = screen.getByText("Buying price not recorded");
    expect(buyingPrice).toBeVisible();
    expect(buyingPrice).toHaveClass("text-muted-foreground");
    expect(buyingPrice).not.toHaveClass("font-semibold");

    const valueCard = screen.getByText("Value").closest("section");
    const viewCard = screen
      .getByRole("heading", { name: "Our view" })
      .closest("section");
    expect(valueCard?.compareDocumentPosition(viewCard!)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    expect(screen.getByText("Sell some", { exact: true })).toBeVisible();
    expect(
      screen.getByText("The price looks high against its own history."),
    ).toBeVisible();
    expect(
      screen.getByText("Company revenue fell for 2 quarters in a row."),
    ).not.toBeVisible();
    expect(screen.getByText(/Estimated tax if sold/)).toHaveTextContent("250");
    expect(screen.getByText("Checked on 28 Oct 2026")).toBeVisible();

    const why = screen.getByText("Why this suggestion?").closest("details");
    const details = screen.getByText("Holding details").closest("details");
    expect(why).not.toHaveAttribute("open");
    expect(details).not.toHaveAttribute("open");
    await user.click(screen.getByText("Why this suggestion?"));
    expect(
      screen.getByText("Company revenue fell for 2 quarters in a row."),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "How we check", level: 3 }),
    ).toBeVisible();
    const companyResultsCheck = screen
      .getAllByRole("listitem")
      .find((row) => row.textContent?.includes("Company results"));
    expect(companyResultsCheck).toBeVisible();
    expect(companyResultsCheck?.textContent).toContain(
      "Company results Looks okay",
    );
    await user.click(screen.getByText("Holding details"));
    expect(screen.getByText("DEMO")).toBeVisible();
    expect(screen.getByText("0.4237")).toBeVisible();
    expect(
      screen.getByText("Profit or loss needs your buying price."),
    ).toBeVisible();
    expect(
      screen.queryByText(/percentile|threshold|TRIM_TO_TARGET/i),
    ).toBeNull();
    expect(bannedTermsIncludingDisclosuresAndFields(container)).toEqual([]);
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows DKK with the INR value when a household rate is available", async () => {
    window.localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, "DKK");
    server.use(...pageHandlers(true, "12.5"));
    renderHolding(true);

    expect(await screen.findByText("DKK 1,600")).toBeVisible();
    expect(screen.getByText("₹20,000")).toBeVisible();
  });

  it("falls back to the INR value when the household rate is unavailable", async () => {
    window.localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, "DKK");
    server.use(...pageHandlers(true));
    renderHolding(true);

    expect(await screen.findByText("₹20,000")).toBeVisible();
    expect(screen.queryByText(/DKK/)).not.toBeInTheDocument();
  });
});
