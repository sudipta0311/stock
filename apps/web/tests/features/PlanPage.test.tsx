import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import type { WeeklyPlan } from "../../src/planning/api";
import { PlanPage } from "../../src/plan/PlanPage";
import { loanSummary, networth, weeklyPlan } from "./fixtures";
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
    http.get(/\/v1\/loans$/, () => HttpResponse.json(loanSummary())),
  ];
}

describe("Plan page", () => {
  it("shows the money mix first and keeps supporting detail closed", async () => {
    const user = userEvent.setup();
    server.use(...planHandlers());
    const { container } = renderPlan();
    expect(
      await screen.findByRole("heading", { name: "Your money vs your plan" }),
    ).toBeInTheDocument();
    expect(
      screen.getByText("A bit more is in safe money than your plan suggests."),
    ).toBeInTheDocument();
    expect(screen.getByText(/Your safe money target is/)).toBeInTheDocument();
    expect(
      screen.getByText(/Your safe money target is/).textContent?.trim(),
    ).toMatch(/\.$/);
    expect(container.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    const details = screen.getByText("Details").closest("details");
    expect(details).not.toHaveAttribute("open");
    expect(
      screen.getByText("Shares abroad: US-listed index fund"),
    ).not.toBeVisible();

    await user.click(screen.getByText("Show amounts"));
    expect(document.body).toHaveTextContent(/Now: ₹3.8 L \(38%\)/);
    expect(document.body).toHaveTextContent(/Plan: ₹4.5 L \(45%\)/);
    await user.click(screen.getByText("Details"));
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

  it("shows the missing year-start account value as one Denmark note", async () => {
    const user = userEvent.setup();
    server.use(
      ...planHandlers({
        residency_notes: ["dk_aktiesparekonto"],
        members: [
          {
            id: "member-test-1",
            name: "Member",
            relationship: "self",
            net_worth: "0.00",
            residency: {
              status: "NRI",
              country: "DK",
              settle_plan: "undecided",
              us_person: false,
            },
            notes: ["ask_start_value_missing", "dk_share_income"],
          },
        ],
        highlights_by_screen: {
          home: [],
          plan: [
            {
              key: "ask_start_value_missing",
              screen: "plan",
              title: "Add your account value",
              body: "Add its value from the start of the year.",
              learn_more_key: "ask_start_value_missing",
              severity: "attention",
            },
          ],
          ideas: [],
          money: [],
        },
      }),
    );
    renderPlan();
    await user.click(await screen.findByText("Why?"));
    expect(
      screen.getByText(
        "Add your share savings account's value on 1 January so we can use it.",
      ),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/share savings account/).length).toBe(1);
  });

  it("shows eligible debt options when their yields are unavailable", async () => {
    const user = userEvent.setup();
    server.use(
      ...planHandlers({
        unranked_options: [
          { instrument: "local_bank_deposit", reason: "yield_unavailable" },
          { instrument: "nre_fd", reason: "yield_unavailable" },
        ],
      }),
    );
    renderPlan();
    await user.click(await screen.findByText("Details"));
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
    const user = userEvent.setup();
    server.use(...planHandlers());
    renderPlan();
    await user.click(await screen.findByText("Details"));
    const title = await screen.findByText("Your LIC / insurance policies");
    const card = title.closest("section");
    expect(card).not.toBeNull();
    expect(card).toHaveTextContent("Compare options before deciding");
    expect(card).toHaveTextContent("Estimated return 5.1%");
    expect(card).toHaveTextContent("Paid-up value");
    expect(card?.textContent?.toLowerCase()).not.toContain("surrender now");
  });

  it("shows goals with target dates and passes axe", async () => {
    const user = userEvent.setup();
    server.use(...planHandlers());
    const { container } = renderPlan();
    await user.click(await screen.findByText("Details"));
    expect(await screen.findByText("Education")).toBeInTheDocument();
    expect(document.body).toHaveTextContent(/Target ₹5 L/);
    expect(screen.getByText("By 2032")).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("keeps loan totals and the recommendation inside collapsed details", async () => {
    const user = userEvent.setup();
    server.use(...planHandlers());
    renderPlan();
    await screen.findByRole("heading", { name: "Your money vs your plan" });
    await user.click(screen.getByText("Details"));

    const loans = screen.getByText("Loans").closest("details");
    expect(loans).not.toHaveAttribute("open");
    await user.click(screen.getByText("Loans"));
    expect(screen.getByText("Total left:")).toBeInTheDocument();
    expect(screen.getByText("Pay extra toward this loan.")).toBeInTheDocument();
  });
});
