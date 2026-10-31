import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { IdeasPage } from "../../src/ideas/IdeasPage";
import { WorldPage } from "../../src/ideas/WorldPage";
import { action, networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

function ideaHandlers(plan = weeklyPlan()) {
  return [
    http.get(/\/v1\/research\/ideas$/, () =>
      HttpResponse.json({
        week: "2026-W44",
        ideas: [
          {
            security_id: "security-demo",
            symbol: "DEMO",
            market: "IN",
            sector: "Energy",
            price: "100.00",
            quant_decile: 8,
            dossier_status: "APPROVED",
            valuation_percentile: "35",
            confidence: "HIGH",
          },
        ],
      }),
    ),
    http.get(/\/v1\/actions$/, () =>
      HttpResponse.json([
        action({
          kind: "BUY_STOCK_TRANCHE",
          payload: {
            kind: "BUY_STOCK_TRANCHE",
            security_id: "security-demo",
            symbol: "DEMO",
            amount: "10000.00",
            instrument_category: "stock",
            reason_code: "approved_satellite_idea",
            tranche: { n: "1", of: "3" },
          },
        }),
      ]),
    ),
    http.get(/\/v1\/plan$/, () => HttpResponse.json(plan)),
    http.get(/\/v1\/networth$/, () => HttpResponse.json(networth())),
  ];
}

describe("Ideas page", () => {
  it("shows research ideas, confidence, valuation words, and the disclaimer", async () => {
    server.use(...ideaHandlers());
    const { container } = renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText("DEMO")).toBeInTheDocument();
    expect(screen.getByText("Energy · Below the middle")).toBeInTheDocument();
    expect(screen.getByText("How sure we are: High")).toBeInTheDocument();
    expect(screen.getByText("Part 1 of 3")).toBeInTheDocument();
    expect(
      screen.getByText("Research, not personal advice"),
    ).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows a US idea's reference price in dollars", async () => {
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          ideas: [
            {
              security_id: "security-us-demo",
              symbol: "USDEMO",
              market: "US",
              sector: "Industrials",
              price: "1333.40",
              quant_decile: 10,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              confidence: "MEDIUM",
            },
          ],
        }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText("USDEMO")).toBeInTheDocument();
    expect(screen.getByText("Reference price $1,333")).toBeInTheDocument();
  });

  it("keeps selector details closed and explains fund terms when opened", async () => {
    const user = userEvent.setup();
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          ideas: [],
          core_picks: {
            in_nifty50: [
              {
                category_label: "India large companies",
                symbol: "SYNTH",
                name: "Synthetic ETF fund",
                rank: 1,
                metrics: { expense_ratio_pct: null },
                reason_codes: ["ter_unavailable"],
              },
            ],
          },
          track_record: {
            markets: {
              IN: {
                status: "available",
                summary: "Research ideas since 2026-10-01: +1.0% vs Nifty 50.",
              },
            },
          },
        }),
      ),
    );
    const { container } = renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    const moreResearch = await screen.findByText("More research");
    expect(screen.queryByText("How past ideas did")).not.toBeVisible();
    expect(screen.getByText("Review your plan")).toBeVisible();
    expect(container.querySelectorAll('[data-primary="true"]')).toHaveLength(1);

    await user.click(moreResearch);
    expect(screen.getByText("How past ideas did")).toBeVisible();
    expect(screen.getByText("Funds to compare")).toBeVisible();
    await user.click(screen.getByText("Funds to compare"));
    expect(screen.getByText("India large companies")).toBeVisible();
    expect(screen.getByText("Synthetic ETF fund")).toBeVisible();
    expect(screen.getByRole("button", { name: "What's this?" })).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows a sourced Denmark buy line with details closed by default", async () => {
    server.use(
      ...ideaHandlers(
        weeklyPlan({
          residency: {
            status: "NRI",
            country: "DK",
            settle_plan: "undecided",
            us_person: false,
          },
          abroad_buy_list: [
            {
              member_id: "member-test-1",
              account: "ask",
              broker: "nordnet",
              instrument: {
                name: "Example UCITS ETF Fund",
                ticker: "WLDX",
                exchange: "Xetra",
                isin: "IE00ABC00001",
                currency: "EUR",
              },
              amount_dkk: "10000",
              amount_inr: "125000.00",
              reason_codes: ["ask_first"],
              how_to_key: "how_to.nordnet",
              details: {
                tax_verified: false,
                sources: [
                  { label: "factsheet", url: "https://example.test/factsheet" },
                  { label: "tax_list", url: "https://example.test/tax-list" },
                ],
              },
            },
          ],
        }),
      ),
    );
    const { container } = renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByRole("heading", { name: "Buy from Denmark" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Example UCITS ETF Fund")).toBeInTheDocument();
    expect(
      screen.getByText("We use your share savings account first when it fits."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Core picks for you")).not.toBeInTheDocument();
    expect(container.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    expect(screen.getByText("DKK 10,000")).toBeInTheDocument();
    expect(screen.getByText(/Share savings account/)).toBeInTheDocument();
    expect(screen.getByText("IE00ABC00001")).toBeInTheDocument();
    expect(screen.getByText("How?")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Choose your Aktiesparekonto (share savings account) or your ordinary account.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Why?")).toBeInTheDocument();
    expect(
      screen.getByText("Tax details still need a check."),
    ).not.toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows a Danish member's buy line and broker check in a shared household", async () => {
    const user = userEvent.setup();
    server.use(
      ...ideaHandlers(
        weeklyPlan({
          residency: {
            status: "RI",
            country: "IN",
            settle_plan: "india",
            us_person: false,
          },
          members: [
            {
              id: "member-test-1",
              name: "Account owner",
              relationship: "self",
              net_worth: "0.00",
              residency: {
                status: "RI",
                country: "IN",
                settle_plan: "india",
                us_person: false,
              },
              notes: [],
            },
            {
              id: "member-dk",
              name: "Denmark resident",
              relationship: "spouse",
              net_worth: "0.00",
              residency: {
                status: "NRI",
                country: "DK",
                settle_plan: "undecided",
                us_person: false,
              },
              notes: [],
            },
          ],
          abroad_buy_list: [
            {
              member_id: "member-dk",
              account: "ordinary",
              broker: "other",
              instrument: {
                name: "European world fund",
                ticker: "WLDX",
                exchange: "Xetra",
                isin: "IE00ABC00001",
                currency: "EUR",
              },
              amount_dkk: "10000",
              amount_inr: "125000.00",
              reason_codes: ["broker_availability_unchecked"],
              how_to_key: "how_to.other",
              details: {
                tax_verified: false,
                us_comparison: {
                  annual_difference_pct: "0.15",
                  annual_difference_direction: "less",
                },
                sources: [],
              },
            },
          ],
        }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByRole("heading", { name: "Buy from Denmark" }),
    ).toBeInTheDocument();
    expect(screen.getByText("For Denmark resident")).toBeInTheDocument();
    await user.click(screen.getByText("Why?"));
    expect(
      screen.getByText("Check that your broker offers this fund."),
    ).toBeVisible();
    expect(
      screen.getByText("Costs about 0.15% a year less than the US version."),
    ).toBeVisible();
  });

  it("formats stress impacts for current holdings and the target plan", async () => {
    server.use(
      http.get(/\/v1\/stress$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          as_of: "2026-10-28",
          stress: weeklyPlan().stress,
        }),
      ),
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json({
          ...weeklyPlan(),
          geo_tilts: {
            what_changed: "Global target rose by 2 points.",
            why: "The portfolio has room to diversify.",
            sources: ["https://example.test/research"],
          },
        }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <WorldPage />
      </MemoryRouter>,
    );
    expect(
      await screen.findByText("If Indian shares fall 25%"),
    ).toBeInTheDocument();
    const indiaScenario = screen
      .getByRole("heading", { name: "If Indian shares fall 25%" })
      .closest("article");
    expect(indiaScenario).toHaveTextContent("Your money: down ₹25,000");
    expect(indiaScenario).toHaveTextContent(
      "With your target plan: down ₹21,000",
    );
    expect(screen.getByText("What we changed and why")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Source" })).toHaveAttribute(
      "href",
      "https://example.test/research",
    );
  });
});
