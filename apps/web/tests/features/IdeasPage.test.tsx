import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import {
  HouseholdProvider,
  PRIMARY_CURRENCY_STORAGE_KEY,
} from "../../src/household/HouseholdContext";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import * as englishCopy from "../../src/copy/sections/index";
import { IdeasPage } from "../../src/ideas/IdeasPage";
import { WorldPage } from "../../src/ideas/WorldPage";
import { action, networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";
import { bannedTermsIncludingDisclosuresAndFields } from "../plain-language";

function ideaHandlers(
  plan = weeklyPlan(),
  name: string | null = "Example Company",
) {
  return [
    http.get(/\/v1\/research\/ideas$/, () =>
      HttpResponse.json({
        week: "2026-W44",
        ideas: [
          {
            security_id: "security-demo",
            symbol: "DEMO",
            name,
            market: "IN",
            sector: "Energy",
            price: "100.00",
            quant_decile: 8,
            dossier_status: "APPROVED",
            valuation_percentile: "35",
            valuation_label: "fair",
            confidence: "HIGH",
            entry_hint: {
              code: "buy_in_parts",
              parts: 3,
              reasons: ["us_price_history_unavailable"],
            },
            why: {
              reasons: [
                "A durable business has steady demand.",
                "Its balance sheet is manageable.",
              ],
              risk: "Costs could rise faster than sales.",
            },
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
    http.get(/\/v1\/me\/investing$/, () =>
      HttpResponse.json({ monthly_sip_capacity: "0.00" }),
    ),
  ];
}

function indexFundRows() {
  return [
    {
      exposure: "global_all_world",
      role: "plan_main",
      instrument: {
        id: "world-fund",
        name: "Example world fund",
        listed_name: "Example world listing",
        isin: "TEST00000001",
        accumulating: true,
        currency: "EUR",
        brokers: ["nordnet", "saxo"],
        ask_eligible: true,
      },
      market: null,
    },
    {
      exposure: "global_sp500",
      role: "option",
      instrument: {
        id: "sp500-fund",
        name: "iShares Core S&P 500 fund",
        listed_name: "iShares Core S&P 500 UCITS ETF USD (Acc)",
        isin: "TEST00000002",
        accumulating: true,
        currency: "EUR",
        brokers: ["nordnet"],
        ask_eligible: true,
      },
      market: "US",
    },
    {
      exposure: "global_nasdaq100",
      role: "option",
      instrument: {
        id: "nasdaq-fund",
        name: "Example technology fund",
        listed_name: "Example technology listing",
        isin: "TEST00000003",
        accumulating: false,
        currency: "EUR",
        brokers: ["saxo"],
        ask_eligible: false,
      },
      market: "US",
    },
  ];
}

function indexFundResearch(usMarket: unknown = null) {
  return {
    week: "2026-W44",
    ideas: [],
    index_funds: indexFundRows(),
    us_market: usMarket,
  };
}

function denmarkPlan() {
  return weeklyPlan({
    residency: {
      status: "NRI",
      country: "DK",
      settle_plan: "undecided",
      us_person: false,
      planned_return_year: null,
    },
    abroad_buy_list: [],
  });
}

describe("Ideas page", () => {
  it("offers to set a monthly amount when Denmark has nothing to buy", async () => {
    server.use(...ideaHandlers(denmarkPlan()));
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json(indexFundResearch()),
      ),
    );

    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(englishCopy.denmark.no_buy_short),
    ).toBeInTheDocument();
    expect(
      screen.queryByText("Monthly buying: keep going."),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", {
        name: englishCopy.denmark.set_monthly_amount,
      }),
    ).toHaveAttribute("href", "/");
    expect(
      screen.getAllByRole("link", {
        name: englishCopy.denmark.set_monthly_amount,
      }),
    ).toHaveLength(1);
  });

  it("shows ideas and keeps confidence details closed", async () => {
    const user = userEvent.setup();
    server.use(...ideaHandlers());
    const { container } = renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );
    expect(await screen.findByText("DEMO")).toBeInTheDocument();
    const ideaHeading = screen.getByRole("heading", {
      name: "Example Company",
    });
    expect(ideaHeading.parentElement).toHaveTextContent("DEMO");
    const marketHeading = screen.getByText("India", { exact: true });
    expect(marketHeading).toHaveClass("text-xs", "text-muted-foreground");
    expect(marketHeading).not.toHaveClass("text-sm", "text-base");
    expect(screen.getByText("Energy")).not.toBeVisible();
    expect(screen.getByText("Fairly priced")).not.toBeVisible();
    expect(screen.getByText("Higher confidence")).not.toBeVisible();
    expect(screen.getByText("Part 1 of 3")).toBeVisible();
    const ideaArticle = ideaHeading.closest("article");
    if (!(ideaArticle instanceof HTMLElement))
      throw new Error("Expected the research idea row");
    expect(ideaArticle.querySelectorAll("details")).toHaveLength(1);
    const ideaWhy = within(ideaArticle)
      .getByText("Why this idea?")
      .closest("details");
    expect(ideaWhy).not.toHaveAttribute("open");
    expect(
      screen.getByText("A durable business has steady demand."),
    ).not.toBeVisible();
    expect(
      within(ideaArticle).getByText(
        (_, element) =>
          element?.tagName === "P" &&
          element.textContent?.includes(
            "We have under a year of US prices, so we skip the trend check.",
          ) === true,
      ),
    ).not.toBeVisible();
    expect(within(ideaArticle).queryByText("Why?")).not.toBeInTheDocument();
    await user.click(within(ideaArticle).getByText("Why this idea?"));
    expect(screen.getByText("Energy")).toBeVisible();
    expect(screen.getByText("Fairly priced")).toBeVisible();
    expect(screen.getByText("Higher confidence")).toBeVisible();
    expect(
      screen.getByText("A durable business has steady demand."),
    ).toBeVisible();
    expect(screen.getByText("Buy in 3 parts over 3 months")).toBeVisible();
    const articleText = ideaArticle.textContent ?? "";
    expect(articleText.indexOf("Buy in 3 parts over 3 months")).toBeLessThan(
      articleText.indexOf("Reference price"),
    );
    expect(
      screen.getByText("Ideas are research to review, not a reason to rush."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Ideas" }),
    ).not.toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
    expect(bannedTermsIncludingDisclosuresAndFields(container)).toEqual([]);
  });
  it("shows a long company name in full without a clamp", async () => {
    const longName =
      "Synthetic Agricultural Equipment Company With A Long Name";
    server.use(...ideaHandlers(weeklyPlan(), longName));
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    const heading = await screen.findByRole("heading", { name: longName });
    expect(heading).toBeVisible();
    expect(heading).not.toHaveClass(/line-clamp/);
    expect(heading).not.toHaveAttribute("title");
  });

  it("shows monthly buying only when a monthly amount is set", async () => {
    server.use(...ideaHandlers(denmarkPlan()));
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json(indexFundResearch()),
      ),
      http.get(/\/v1\/me\/investing$/, () =>
        HttpResponse.json({ monthly_sip_capacity: "2500.00" }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText("Monthly buying: keep going."),
    ).toBeVisible();
  });
  it("uses the ticker as the title when the company name is missing", async () => {
    server.use(...ideaHandlers(weeklyPlan(), null));
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(await screen.findByRole("heading", { name: "DEMO" })).toBeVisible();
    expect(screen.queryByText("Example Company")).not.toBeInTheDocument();
  });

  it("keeps the plan fund visible and groups other funds closed", async () => {
    const user = userEvent.setup();
    server.use(...ideaHandlers(denmarkPlan()));
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json(indexFundResearch()),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    const title = await screen.findByRole("heading", {
      name: englishCopy.ideas.index_funds_title,
    });
    const card = title.closest("div.grid");
    if (!(card instanceof HTMLElement))
      throw new Error("Expected the index fund card");
    expect(screen.getByRole("heading", { name: "World fund" })).toBeVisible();
    expect(screen.getByText(englishCopy.ideas.index_plan_main)).toBeVisible();
    const otherFunds = screen.getByText(englishCopy.ideas.index_other_funds);
    expect(otherFunds.closest("details")).not.toHaveAttribute("open");
    expect(
      screen.getByRole("heading", { name: "S&P 500 fund" }),
    ).not.toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Nasdaq 100 fund" }),
    ).not.toBeVisible();
    await user.click(otherFunds);
    expect(within(card).getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByRole("heading", { name: "S&P 500 fund" })).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "Nasdaq 100 fund" }),
    ).toBeVisible();
    expect(screen.getByText("iShares Core S&P 500 fund")).toBeVisible();
    expect(
      screen.getByText("Share savings account, Nordnet, Saxo"),
    ).toBeVisible();
    expect(screen.getByText("Share savings account, Nordnet")).toBeVisible();

    const why = screen.getByText(englishCopy.ideas.index_why);
    expect(why.closest("details")).not.toHaveAttribute("open");
    const sp500Row = screen
      .getByRole("heading", { name: "S&P 500 fund" })
      .closest("li");
    if (!(sp500Row instanceof HTMLElement))
      throw new Error("Expected the S&P 500 fund row");
    const findAtBroker = within(sp500Row).getByText(
      englishCopy.ideas.index_find_it_at_broker,
    );
    expect(findAtBroker.closest("details")).not.toHaveAttribute("open");
    expect(
      within(sp500Row).getByText("iShares Core S&P 500 UCITS ETF USD (Acc)"),
    ).not.toBeVisible();
    expect(
      within(sp500Row).getByText("Search code: TEST00000002"),
    ).not.toBeVisible();
    expect(
      within(sp500Row).getByText("It reinvests dividends for you."),
    ).not.toBeVisible();
  });
  it.each([
    [
      "cheap",
      "33.00",
      "US shares look cheap compared with the last 3 years.",
      3,
    ],
    [
      "fair",
      "50.00",
      "US shares look fairly priced compared with the last 3 years.",
      3,
    ],
    [
      "expensive",
      "80.00",
      "US shares look expensive compared with the last 3 years.",
      6,
    ],
  ])(
    "shows the %s US market message and extra-buy parts",
    async (label, percentile, message, parts) => {
      server.use(...ideaHandlers(denmarkPlan()));
      server.use(
        http.get(/\/v1\/research\/ideas$/, () =>
          HttpResponse.json(
            indexFundResearch({
              percentile,
              history_label: "long_history",
              label,
              extra_money_parts: parts,
            }),
          ),
        ),
      );
      renderWithQueries(
        <MemoryRouter>
          <IdeasPage />
        </MemoryRouter>,
      );

      expect(await screen.findByText(message)).toBeVisible();
      expect(
        screen.getByText(`Extra money: buy in ${parts} parts.`),
      ).toBeVisible();
      expect(
        screen.queryByText("Monthly buying: keep going."),
      ).not.toBeInTheDocument();
    },
  );

  it("shows the missing US market reading without extra parts", async () => {
    server.use(...ideaHandlers(denmarkPlan()));
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json(indexFundResearch(null)),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(englishCopy.ideas.index_market_unavailable),
    ).toBeVisible();
    expect(screen.queryByText(/Extra money:/)).not.toBeInTheDocument();
  });

  it("hides the index fund card for households without a Denmark resident", async () => {
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          ideas: [],
          index_funds: [],
          us_market: null,
        }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    expect(
      await screen.findByText(englishCopy.ideas.weekly_title),
    ).toBeVisible();
    expect(
      screen.queryByRole("heading", {
        name: englishCopy.ideas.index_funds_title,
      }),
    ).not.toBeInTheDocument();
  });

  it("shows buy timing lines and keeps each explanation closed", async () => {
    const user = userEvent.setup();
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          ideas: [
            {
              security_id: "security-three-parts",
              symbol: "THREE",
              market: "IN",
              sector: "Energy",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              entry_hint: {
                code: "buy_in_parts",
                parts: 3,
                reasons: ["regular_entry_schedule"],
              },
            },
            {
              security_id: "security-six-parts",
              symbol: "SIX",
              market: "US",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "85",
              valuation_label: "expensive",
              confidence: "HIGH",
              entry_hint: {
                code: "buy_in_parts",
                parts: 6,
                reasons: [
                  "us_price_history_unavailable",
                  "market_valuation_elevated",
                ],
              },
            },
            {
              security_id: "security-wait",
              symbol: "WAIT",
              market: "IN",
              sector: "Health Care",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "45",
              valuation_label: "fair",
              confidence: "MEDIUM",
              entry_hint: {
                code: "wait",
                parts: null,
                reasons: ["price_far_above_average"],
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
      await screen.findByText("Buy in 3 parts over 3 months"),
    ).toBeVisible();
    expect(
      screen.getByText("Buy in 6 parts. Markets look expensive."),
    ).toBeVisible();
    expect(screen.getByText("Wait for a better price")).toBeVisible();
    const waitIdea = screen.getByText("WAIT").closest("article");
    expect(waitIdea).not.toBeNull();
    expect(within(waitIdea!).queryByText(/confidence/i)).toBeNull();

    const whyDetails = screen
      .getAllByText(englishCopy.ideas.why_this_idea)
      .map((summary) => summary.closest("details"));
    expect(whyDetails).toHaveLength(3);
    expect(whyDetails.every((details) => details && !details.open)).toBe(true);
    expect(screen.queryByText("Why?")).not.toBeInTheDocument();
    expect(
      screen.getByText("The price is far above its recent average."),
    ).not.toBeVisible();
    const waitCard = screen.getByText("WAIT").closest("article");
    if (!waitCard) throw new Error("Expected the wait idea card");
    expect(waitCard.querySelectorAll("details")).toHaveLength(1);
    await user.click(
      within(waitCard).getByText(englishCopy.ideas.why_this_idea),
    );
    expect(
      screen.getByText("The price is far above its recent average."),
    ).toBeVisible();
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
              valuation_label: "fair",
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

  it("shows owned amounts quietly and keeps enough ideas closed", async () => {
    const user = userEvent.setup();
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({
          week: "2026-W44",
          ideas: [
            {
              security_id: "security-funds-only",
              symbol: "FUNDSONLY",
              name: "Synthetic Funds Company",
              market: "IN",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              entry_hint: { code: "buy_in_parts", parts: 3, reasons: [] },
              owned: {
                direct_inr: "0.00",
                through_funds_inr: "270000.00",
                status: "some",
                funds: [
                  {
                    name: "Synthetic Nifty 50 index fund",
                    amount_inr: "250000.00",
                    estimated: true,
                  },
                ],
              },
            },
            {
              security_id: "security-direct-and-funds",
              symbol: "BOTH",
              name: "Synthetic Direct Company",
              market: "IN",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              owned: {
                direct_inr: "50000.00",
                through_funds_inr: "271000.00",
                status: "some",
                funds: [],
              },
            },
            {
              security_id: "security-enough",
              symbol: "ENOUGH",
              name: "Synthetic Enough Company",
              market: "IN",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              owned: {
                direct_inr: "0.00",
                through_funds_inr: "1000000.00",
                status: "enough",
                funds: [],
              },
            },
            {
              security_id: "security-enough-direct-only",
              symbol: "DIRECTONLY",
              name: "Synthetic Direct Only Company",
              market: "IN",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              owned: {
                direct_inr: "300000.00",
                through_funds_inr: "0.00",
                status: "enough",
                funds: [],
              },
            },
            {
              security_id: "security-dkk-some",
              symbol: "DKKDEMO",
              name: "Synthetic Denmark Company",
              market: "US",
              sector: "Technology",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              owned: {
                currency: "DKK",
                direct_inr: "0.00",
                through_funds_inr: "40000.00",
                direct: "0.00",
                through_funds: "3200.00",
                status: "some",
                funds: [
                  {
                    name: "Synthetic global fund",
                    amount_inr: "40000.00",
                    amount: "3200.00",
                    estimated: true,
                  },
                ],
              },
            },
            {
              security_id: "security-none",
              symbol: "NONE",
              name: "Synthetic New Company",
              market: "IN",
              sector: "Industrials",
              price: "100.00",
              quant_decile: 8,
              dossier_status: "APPROVED",
              valuation_percentile: "35",
              valuation_label: "fair",
              confidence: "HIGH",
              owned: {
                direct_inr: "0.00",
                through_funds_inr: "0.00",
                status: "none",
                funds: [],
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
      await screen.findByText(
        "You already own about ₹2.7 L of this through your funds.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        "You already own about ₹3.21 L of this, ₹2.71 L through your funds.",
      ),
    ).toBeVisible();
    const ownedIdea = screen.getByText("FUNDSONLY").closest("article")!;
    const ownedLine = within(ownedIdea).getByText(/You already own about/);
    const entryLine = within(ownedIdea).getByText(
      "Buy in 3 parts over 3 months",
    );
    expect(ownedLine.compareDocumentPosition(entryLine)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
    const noneIdea = screen.getByText("NONE").closest("article");
    expect(noneIdea).not.toHaveTextContent("You already own about");
    expect(
      screen.getByText(
        "You already own about DKK 3,200 of this through your funds.",
      ),
    ).toBeVisible();
    const alreadyOwned = screen.getByText("You already own enough (2)");
    expect(alreadyOwned.closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText("ENOUGH")).not.toBeVisible();
    expect(screen.getByText("NONE")).toBeVisible();
    await user.click(alreadyOwned);
    expect(
      screen.getByText(
        "You own enough of these, directly or through your funds, so we won't suggest more.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        "You already own about ₹10 L of this through your funds.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText("You already own about ₹3 L of this."),
    ).toBeVisible();
    await user.click(screen.getAllByText(englishCopy.ideas.why_this_idea)[0]!);
    expect(
      screen.getByText(
        "Synthetic Nifty 50 index fund: about ₹2.5 L (estimate)",
      ),
    ).toBeVisible();
    const dkkIdea = screen.getByText("DKKDEMO").closest("article");
    if (!dkkIdea) throw new Error("Expected the Denmark idea card");
    await user.click(
      within(dkkIdea).getByText(englishCopy.ideas.why_this_idea),
    );
    expect(
      screen.getByText("Synthetic global fund: about DKK 3,200 (estimate)"),
    ).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
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

    expect(await screen.findByText("This week's research ideas")).toBeVisible();
    expect(screen.queryByText("How past ideas did")).not.toBeInTheDocument();
    expect(screen.queryByText("Review your plan")).not.toBeInTheDocument();
    const fundsSummary = screen.getByText("Funds to compare");
    expect(fundsSummary.closest("details")).not.toHaveAttribute("open");
    expect(
      screen.getByRole("link", { name: "Explore the wider picture" }),
    ).toHaveAttribute("href", "/ideas/world");
    await user.click(fundsSummary);
    expect(screen.getByText("India large companies")).toBeVisible();
    expect(screen.getByText("Synthetic ETF fund")).toBeVisible();
    expect(screen.getByRole("button", { name: "What's this?" })).toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("shows three ideas per market and opens the remaining ideas", async () => {
    const user = userEvent.setup();
    const ideas = ["IN1", "IN2", "IN3", "IN4", "US1", "US2", "US3", "US4"].map(
      (symbol) => ({
        security_id: `security-${symbol}`,
        symbol,
        market: symbol.startsWith("IN") ? "IN" : "US",
        sector: "Energy",
        price: "100.00",
        quant_decile: 8,
        dossier_status: "APPROVED",
        valuation_percentile: "35",
        valuation_label: "fair",
        confidence: "HIGH",
      }),
    );
    server.use(...ideaHandlers());
    server.use(
      http.get(/\/v1\/research\/ideas$/, () =>
        HttpResponse.json({ week: "2026-W44", ideas }),
      ),
    );
    renderWithQueries(
      <MemoryRouter>
        <IdeasPage />
      </MemoryRouter>,
    );

    for (const symbol of ["IN1", "IN2", "IN3", "US1", "US2", "US3"]) {
      expect(await screen.findByText(symbol)).toBeVisible();
    }
    expect(screen.getByText("IN4")).not.toBeVisible();
    expect(screen.getByText("US4")).not.toBeVisible();
    expect(screen.queryByText("Helpful notes")).not.toBeInTheDocument();
    await user.click(screen.getByText("See all ideas"));
    expect(screen.getByText("IN4")).toBeVisible();
    expect(screen.getByText("US4")).toBeVisible();
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
                name: "Example world fund",
                listed_name: "Example World UCITS ETF USD (Acc)",
                ticker: "WLDX",
                exchange: "Xetra",
                isin: "IE00ABC00001",
                accumulating: true,
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
    expect(screen.getByText("Example world fund")).toBeVisible();
    expect(
      screen.getByText("We use your share savings account first when it fits."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Core picks for you")).not.toBeInTheDocument();
    expect(screen.getByText("DKK 10,000")).toBeInTheDocument();
    expect(screen.getByText(/Share savings account/)).toBeInTheDocument();
    const findAtBroker = screen.getByText("Find it at your broker");
    expect(findAtBroker.closest("details")).not.toHaveAttribute("open");
    expect(
      screen.getByText("Example World UCITS ETF USD (Acc)"),
    ).not.toBeVisible();
    expect(screen.getByText("Search code: IE00ABC00001")).not.toBeVisible();
    expect(
      screen.getByText("It reinvests dividends for you."),
    ).not.toBeVisible();
    expect(screen.getByText("How?")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Choose your Aktiesparekonto (share savings account) or your ordinary account.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("Why?")).toBeInTheDocument();
    expect(
      screen.queryByText(englishCopy.denmark.ask_unverified_note),
    ).not.toBeInTheDocument();
    expect(
      screen.getByText("Tax details still need a check."),
    ).not.toBeVisible();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("keeps the unverified share-account note inside the closed Why details", async () => {
    const user = userEvent.setup();
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
                name: "Synthetic share",
                ticker: "SYNTH",
                exchange: "NASDAQ",
                isin: "US0378331005",
                currency: "USD",
              },
              amount_dkk: "3000",
              amount_inr: "37500.00",
              reason_codes: ["approved_us_stock_idea", "ask_room_remaining"],
              how_to_key: "how_to.nordnet",
              details: { ask_unverified: true },
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

    const note = englishCopy.denmark.ask_unverified_note;
    expect(
      await screen.findByRole("heading", { name: "Synthetic share" }),
    ).toBeInTheDocument();
    expect(screen.getByText(note)).not.toBeVisible();
    const why = screen.getByText("Why?");
    expect(why.closest("details")).not.toHaveAttribute("open");
    await user.click(why);
    expect(screen.getByText(note)).toBeVisible();
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
    let fxRequests = 0;
    localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, "INR");
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          display_name: "Synthetic user",
          me_member_id: "member-dk-test",
          household_owner_id: "household-test",
          role: "owner",
        }),
      ),
      http.get(/\/v1\/members$/, () =>
        HttpResponse.json([
          {
            id: "member-dk-test",
            display_name: "Synthetic member",
            is_self: true,
            relationship: "self",
            residency: { country: "DK", status: "NRI" },
          },
        ]),
      ),
      http.get(/\/v1\/households$/, () => HttpResponse.json([])),
      http.get(/\/v1\/fx\/dkk-inr$/, () => {
        fxRequests += 1;
        return HttpResponse.json({ date: "2026-10-28", rate: "12.35" });
      }),
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
      <HouseholdProvider>
        <MemoryRouter>
          <WorldPage />
        </MemoryRouter>
      </HouseholdProvider>,
    );
    await screen.findByText("If Indian shares fall 25%");
    await waitFor(() => expect(fxRequests).toBeGreaterThan(0));
    expect(screen.getByRole("link", { name: "Back to Ideas" })).toHaveAttribute(
      "href",
      "/ideas",
    );
    expect(
      screen.getByText("These examples show how your holdings might move."),
    ).toBeVisible();
    const indiaScenario = screen
      .getByRole("heading", { name: "If Indian shares fall 25%" })
      .closest("article");
    expect(indiaScenario).toHaveTextContent("Your money: down ₹25,000");
    expect(indiaScenario).toHaveTextContent(
      "With your target plan: down ₹21,000",
    );
    expect(indiaScenario).not.toHaveTextContent("DKK");
    expect(screen.getByText("What we changed and why")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Source" })).toHaveAttribute(
      "href",
      "https://example.test/research",
    );
  });
});
