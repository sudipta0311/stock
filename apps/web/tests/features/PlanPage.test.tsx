import { screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import type { WeeklyPlan } from "../../src/planning/api";
import { PlanPage } from "../../src/plan/PlanPage";
import { networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

function policyAsset() {
  return {
    id: "policy-test",
    asset_type: "traditional_policy",
    member_id: null,
    account_id: null,
    details: {
      label: "Demo traditional policy",
      insurer: "Example insurer",
      surrender_value: "125000.00",
      review: {
        irr_pct: "5.1",
        continue_value: "500000.00",
        paid_up_value: "250000.00",
        surrender_value: "125000.00",
      },
    },
    valuation: null,
    valuation_note: null,
    valuation_status: "valued",
    value_cached: "125000.00",
    value_as_of: "2026-10-28",
    status: "active",
    created_at: "2026-10-01T00:00:00Z",
    updated_at: "2026-10-01T00:00:00Z",
  };
}

function renderPlan() {
  return renderWithQueries(
    <MemoryRouter>
      <PlanPage />
    </MemoryRouter>,
  );
}

function planHandlers(planOverrides: Partial<WeeklyPlan> = {}) {
  return [
    http.get(/\/v1\/plan$/, () =>
      HttpResponse.json(
        weeklyPlan({
          route: {
            route: "C",
            vehicle: "us_etf",
            reasons: [],
            warnings: [
              "tcs_applies",
              "schedule_fa_reporting",
              "us_estate_tax_threshold_near",
            ],
            alternatives: [],
            warning_params: {},
          },
          ...planOverrides,
        }),
      ),
    ),
    http.get(/\/v1\/assets$/, () => HttpResponse.json([policyAsset()])),
    http.get(/\/v1\/goals$/, () =>
      HttpResponse.json([
        {
          id: "goal-test",
          name: "Education",
          goal_type: "education_india",
          target_amount: "500000.00",
          currency: "INR",
          target_year: 2032,
          priority: 1,
        },
      ]),
    ),
    http.get(/\/v1\/networth$/, () => HttpResponse.json(networth())),
  ];
}

describe("Plan page", () => {
  it("shows text values beside allocation bars and explains the debt answer", async () => {
    server.use(...planHandlers());
    renderPlan();
    expect(await screen.findByText(/Indian shares/)).toBeInTheDocument();
    expect(screen.getByText(/38% actual → 45% target/)).toBeInTheDocument();
    expect(document.body).toHaveTextContent(/Actual ₹3.8 L · target ₹4.5 L/);
    expect(
      screen.getByText(
        "You already hold enough debt and deposits for your plan.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Shares abroad: US-listed index fund"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/tax may be collected on larger transfers abroad/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/report accounts and investments held outside India/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/may be subject to estate tax/i)).toBeNull();
  });

  it("shows eligible debt options when their yields are unavailable", async () => {
    server.use(
      ...planHandlers({
        unranked_options: [
          { instrument: "local_bank_deposit", reason: "yield_unavailable" },
          { instrument: "nre_fd", reason: "yield_unavailable" },
        ],
      }),
    );
    renderPlan();
    expect(
      await screen.findByText(
        /Local DKK\/EUR bank deposit: Rate not available: check your bank\./,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        /NRE fixed deposit: Rate not available: check your bank\./,
      ),
    ).toBeInTheDocument();
  });

  it("compares insurance options and never tells someone to surrender", async () => {
    server.use(...planHandlers());
    renderPlan();
    const title = await screen.findByText("Your LIC / insurance policies");
    const card = title.closest("section");
    expect(card).not.toBeNull();
    expect(card).toHaveTextContent("Compare options before deciding");
    expect(card).toHaveTextContent("Estimated return 5.1%");
    expect(card).toHaveTextContent("Paid-up value");
    expect(card?.textContent?.toLowerCase()).not.toContain("surrender now");
  });

  it("shows goals with target dates and passes axe", async () => {
    server.use(...planHandlers());
    const { container } = renderPlan();
    expect(await screen.findByText("Education")).toBeInTheDocument();
    expect(document.body).toHaveTextContent(/Target ₹5 L/);
    expect(screen.getByText("By 2032")).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });
});
