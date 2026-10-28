import { screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { IdeasPage } from "../../src/ideas/IdeasPage";
import { WorldPage } from "../../src/ideas/WorldPage";
import { action, networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

function ideaHandlers() {
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
    http.get(/\/v1\/plan$/, () => HttpResponse.json(weeklyPlan())),
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
    expect(indiaScenario).toHaveTextContent("Your money: -₹25,000");
    expect(indiaScenario).toHaveTextContent("With your target plan: -₹21,000");
    expect(screen.getByText("What we changed and why")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Source" })).toHaveAttribute(
      "href",
      "https://example.test/research",
    );
  });
});
