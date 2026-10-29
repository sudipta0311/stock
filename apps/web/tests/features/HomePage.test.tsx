import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import englishCopy from "../../src/copy/en.json";
import { HomePage } from "../../src/home/HomePage";
import type { ActionResponse } from "../../src/planning/api";
import { action, baseHomeHandlers, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

function renderHome() {
  return renderWithQueries(
    <MemoryRouter>
      <HomePage />
    </MemoryRouter>,
  );
}

function withRoute(route: string, id = `action-${route}`): ActionResponse {
  return action({
    id,
    payload: {
      kind: "BUY_LUMPSUM",
      sleeve: "equity_global",
      instrument_category: "in_international_index_fund",
      amount: "10000.00",
      route,
      reason_code: "target_gap",
      params: { market: "US" },
    },
  });
}

describe("Home this week", () => {
  it("shows a positive no-action state when the API returns zero actions", async () => {
    server.use(...baseHomeHandlers([]));
    renderHome();
    const emptyMessage = await screen.findByText("Nothing to do this week");
    expect(emptyMessage).toBeInTheDocument();
    expect(emptyMessage).not.toHaveTextContent("✓");
    expect(screen.getByText("Net worth")).toBeInTheDocument();
    const root = document.querySelector<HTMLElement>(
      '[data-copy-screen="home"]',
    );
    expect(root?.firstElementChild).toHaveClass("home-hero");
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    expect(
      root?.querySelectorAll(":scope > section").length,
    ).toBeLessThanOrEqual(4);
  });

  it("shows one action and removes it after Mark done succeeds", async () => {
    const user = userEvent.setup();
    const onlyAction = action();
    let patched = false;
    server.use(
      ...baseHomeHandlers([onlyAction]),
      http.patch(/\/v1\/actions\/action-test-1$/, async ({ request }) => {
        patched =
          ((await request.json()) as { status: string }).status === "done";
        return HttpResponse.json({ ...onlyAction, status: "done" });
      }),
    );
    renderHome();
    expect(
      await screen.findByRole("article", {
        name: /Invest now.*Indian index fund/,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("₹10,000")).toBeInTheDocument();
    const root = document.querySelector<HTMLElement>(
      '[data-copy-screen="home"]',
    );
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Mark done" }));
    await waitFor(() => expect(patched).toBe(true));
    expect(
      await screen.findByText(/Nothing to do this week/),
    ).toBeInTheDocument();
  });

  it("shows up to three actions", async () => {
    const actions = [
      action({ id: "action-one", rank: 1 }),
      action({
        id: "action-two",
        rank: 2,
        kind: "BUY_STP",
        payload: {
          kind: "BUY_STP",
          amount: "20000.00",
          instrument_category: "in_core_index",
          reason_code: "target_gap",
          tranche: { n: "1", of: "3" },
        },
      }),
      action({
        id: "action-three",
        rank: 3,
        kind: "BUILD_EMERGENCY_FUND",
        payload: {
          kind: "BUILD_EMERGENCY_FUND",
          amount: "30000.00",
          instrument_category: "sweep_fd",
          reason_code: "emergency_fund_gap",
        },
      }),
      action({ id: "action-four", rank: 4 }),
    ];
    server.use(...baseHomeHandlers(actions));
    renderHome();
    const articles = await screen.findAllByRole("article");
    expect(articles).toHaveLength(3);
    const emergencyFundLabel = `${englishCopy.labels.action_kinds.BUILD_EMERGENCY_FUND} · ${englishCopy.labels.instrument_categories.sweep_fd}`;
    expect(articles[2]).toHaveTextContent(emergencyFundLabel);
    expect(screen.queryByText("action-four")).not.toBeInTheDocument();
  });

  it("rounds the action summary and keeps the full amount in How", async () => {
    const user = userEvent.setup();
    const rawAction = action({
      payload: {
        kind: "BUY_LUMPSUM",
        sleeve: "equity_india",
        instrument_category: "in_core_index",
        amount: "12345.67",
        route: "A",
        security_id: null,
        tranche: null,
        reason_code: "target_gap",
        params: { market: "IN" },
      },
    });
    server.use(...baseHomeHandlers([rawAction]));
    renderHome();

    const article = await screen.findByRole("article", {
      name: /Invest now.*Indian index fund/,
    });
    expect(within(article).getByText("₹12,300")).toBeInTheDocument();

    await user.click(within(article).getByRole("button", { name: "How to" }));
    expect(
      screen.getByRole("dialog", { name: "How to take this step" }),
    ).toHaveTextContent("monthly investment for ₹12,345.67");
  });

  it("shows the action details inside How and Why", async () => {
    const user = userEvent.setup();
    const sell = action({
      kind: "SELL_TRIM",
      payload: {
        kind: "SELL_TRIM",
        security_id: "security-test",
        quantity: "2.500",
        tax_estimate: { estimated_tax: "250.00" },
        reason_code: "valuation_stretch",
        params: { security_id: "security-test" },
      },
    });
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(
          weeklyPlan({
            exit_cards: [
              {
                security_id: "security-test",
                action: "TRIM_TO_TARGET",
                triggers: {
                  valuation_stretch: {
                    status: "triggered",
                    value: "95th percentile",
                    threshold: "90th percentile",
                  },
                  governance_break: {
                    status: "watch",
                    value: "No change",
                    threshold: "Review if changed",
                  },
                },
              },
            ],
          }),
        ),
      ),
      ...baseHomeHandlers([sell]),
    );
    renderHome();
    expect(
      await screen.findByText(englishCopy.labels.action_kinds.SELL_TRIM),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "How to" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    expect(
      within(dialog).getByText(
        englishCopy.home.sell_quantity.replace("{quantity}", "2.500"),
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("₹250")).toBeInTheDocument();
    await user.click(within(dialog).getByText("Why?"));
    expect(
      await screen.findByText(englishCopy.home.triggered_rows),
    ).toBeInTheDocument();
    expect(
      screen.getByText(englishCopy.labels.trigger_keys.valuation_stretch),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(englishCopy.labels.trigger_keys.governance_break),
    ).not.toBeInTheDocument();
  });

  it.each([
    [
      "A",
      /mutual fund app.*S&P 500 index fund.*monthly investment for ₹10,000/,
    ],
    [
      "B",
      /Indian broker app.*fund that tracks shares abroad.*buy order for ₹10,000/,
    ],
    ["C", /overseas investment account.*broad index fund.*invest for ₹10,000/],
  ])("shows route %s steps in the How to sheet", async (route, expected) => {
    const user = userEvent.setup();
    server.use(...baseHomeHandlers([withRoute(route)]));
    renderHome();
    await user.click(await screen.findByRole("button", { name: "How to" }));
    expect(
      screen.getByRole("dialog", { name: "How to take this step" }),
    ).toHaveTextContent(expected);
  });

  it("explains the reason and action parameters inside How", async () => {
    const user = userEvent.setup();
    server.use(...baseHomeHandlers([action()]));
    renderHome();
    await user.click(await screen.findByRole("button", { name: "How to" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    await user.click(within(dialog).getByText("Why?"));
    expect(dialog).toHaveTextContent(
      "This brings this part of your plan closer to its target.",
    );
    expect(within(dialog).getByText("market")).toBeInTheDocument();
  });

  it("has no axe violations", async () => {
    server.use(...baseHomeHandlers([]));
    const { container } = renderHome();
    await screen.findByText(/Nothing to do this week/);
    expect((await axe(container)).violations).toEqual([]);
  });
});
