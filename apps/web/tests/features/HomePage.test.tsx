import { act, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import * as englishCopy from "../../src/copy/sections/index";
import {
  HouseholdProvider,
  PRIMARY_CURRENCY_STORAGE_KEY,
} from "../../src/household/HouseholdContext";
import { actionHowTo, HomePage } from "../../src/home/HomePage";
import type { ActionResponse } from "../../src/planning/api";
import { action, baseHomeHandlers, networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";
import { WakeIndicator } from "../../src/app/WakeIndicator";

function renderHome() {
  return renderWithQueries(
    <MemoryRouter>
      <>
        <HomePage />
        <WakeIndicator />
      </>
    </MemoryRouter>,
  );
}

function renderDkkHome(
  investingPrefs: Record<string, unknown>,
  fxRate: string | undefined,
) {
  window.localStorage.clear();
  window.localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, "DKK");
  let markFxRequested!: () => void;
  const fxRequested = new Promise<void>((resolve) => {
    markFxRequested = resolve;
  });
  server.use(
    http.get(/\/v1\/me$/, () =>
      HttpResponse.json({
        id: "user-test",
        me_member_id: "member-dk-test",
      }),
    ),
    ...baseHomeHandlers([], investingPrefs),
    http.get(/\/v1\/members$/, () =>
      HttpResponse.json([
        {
          id: "member-dk-test",
          display_name: "Test Denmark member",
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
    http.get(/\/v1\/fx\/dkk-inr$/, () => {
      markFxRequested();
      return fxRate
        ? HttpResponse.json({ date: "2026-10-28", rate: fxRate })
        : HttpResponse.json({ date: "2026-10-28" });
    }),
  );
  const view = renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter>
        <>
          <HomePage />
          <WakeIndicator />
        </>
      </MemoryRouter>
    </HouseholdProvider>,
  );
  return { ...view, fxRequested };
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
  it("uses loan-specific plain wording for a pay-extra action", () => {
    const payLoan = action({
      payload: {
        kind: "PAY_EXTRA_ON_LOAN",
        amount: "5000.00",
        reason_code: "loan_cost_above_safe_return",
        params: {
          advice:
            "This loan costs more than safe money is expected to earn. Pay extra toward it.",
        },
      },
    });

    expect(
      actionHowTo(payLoan, undefined, [], undefined, "INR", undefined),
    ).toBe(
      "This loan costs more than safe money is expected to earn. Pay extra toward it.",
    );
  });

  it("asks for a monthly amount when none is set", async () => {
    server.use(...baseHomeHandlers([]));
    renderHome();
    expect(
      await screen.findByText(englishCopy.home.amount_question),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Save" })).toBeVisible();
    expect(
      screen.getByRole("button", { name: englishCopy.home.extra_money_link }),
    ).toHaveClass("min-h-11", "w-fit", "px-0");
    expect(
      screen.queryByText(englishCopy.home.nothing_to_buy),
    ).not.toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "No holdings added yet" }),
    ).toBeVisible();
    const addHoldings = screen.getByRole("link", { name: "Add holdings" });
    expect(addHoldings).toHaveAttribute("href", "/money?add=1");
    expect(addHoldings).toHaveAttribute("data-primary", "true");
    const root = document.querySelector<HTMLElement>(
      '[data-copy-screen="home"]',
    );
    expect(root?.firstElementChild).toHaveClass("home-hero");
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    expect(
      root?.querySelectorAll(":scope > section").length,
    ).toBeLessThanOrEqual(4);
  });

  it("keeps Home steps and the normal Money link when assets fail", async () => {
    server.use(...baseHomeHandlers([action()]));
    server.use(
      http.get(/\/v1\/assets$/, () =>
        HttpResponse.json({ detail: "Synthetic failure" }, { status: 500 }),
      ),
    );
    renderHome();

    expect(
      await screen.findByRole("button", { name: "Show steps" }),
    ).toBeVisible();
    expect(
      await screen.findByText(englishCopy.home.amount_question),
    ).toBeVisible();
    expect(screen.queryByText("Your week couldn't load")).toBeNull();
    expect(screen.getByRole("link", { name: /Money:/ })).toHaveAttribute(
      "href",
      "/money",
    );
  });

  it("hides Coming up when there are no due items", async () => {
    server.use(...baseHomeHandlers([]));
    renderHome();

    await screen.findByText(englishCopy.home.amount_question);
    expect(
      screen.queryByRole("heading", { name: "Coming up" }),
    ).not.toBeInTheDocument();
  });

  it("shows the positive no-buy state after a monthly amount is set", async () => {
    server.use(
      ...baseHomeHandlers([], {
        monthly_sip_capacity: "20000.00",
        new_money_this_week: "0.00",
      }),
    );
    renderHome();
    expect(
      await screen.findByRole("link", { name: "Add holdings" }),
    ).toHaveAttribute("href", "/money?add=1");
    expect(
      screen.queryByLabelText(englishCopy.home.monthly_amount),
    ).not.toBeInTheDocument();
  });

  it("saves the monthly amount through the existing settings API", async () => {
    const user = userEvent.setup();
    let saved: Record<string, unknown> | null = null;
    server.use(...baseHomeHandlers([]));
    server.use(
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        saved = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(saved);
      }),
    );
    renderHome();
    await screen.findByText(englishCopy.home.amount_question);
    await user.type(
      screen.getByRole("spinbutton", {
        name: new RegExp(englishCopy.home.monthly_amount),
      }),
      "25000",
    );
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(saved).toMatchObject({ monthly_sip_capacity: "25000" }),
    );
    expect(
      await screen.findByText(englishCopy.home.amount_saved),
    ).toBeVisible();
  });

  it("shows rebuilt plan actions as soon as the monthly amount is saved", async () => {
    const user = userEvent.setup();
    const seededAction = action({
      id: "seeded-emergency-fund",
      kind: "BUILD_EMERGENCY_FUND",
      payload: {
        kind: "BUILD_EMERGENCY_FUND",
        amount: "5000.00",
        member_id: "member-test-1",
      },
    });
    const rebuiltAction = action({
      id: "rebuilt-monthly-investments",
      kind: "REBALANCE_SIPS",
      payload: {
        kind: "REBALANCE_SIPS",
        amount: "25000.00",
        member_id: "member-test-1",
        instrument_category: "monthly_investment",
      },
    });
    let amountSaved = false;
    let actionReads = 0;
    let planReads = 0;
    server.use(...baseHomeHandlers([seededAction]));
    server.use(
      http.get(/\/v1\/actions$/, () => {
        actionReads += 1;
        return HttpResponse.json(
          amountSaved ? [rebuiltAction] : [seededAction],
        );
      }),
      http.patch(/\/v1\/actions\/seeded-emergency-fund$/, () =>
        HttpResponse.json({ ...seededAction, status: "done" }),
      ),
      http.get(/\/v1\/plan$/, () => {
        planReads += 1;
        return HttpResponse.json(weeklyPlan());
      }),
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        amountSaved = true;
        return HttpResponse.json(await request.json());
      }),
    );
    renderHome();

    await screen.findByRole("article", { name: /Build your emergency fund/ });
    await user.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("article", { name: /Build your emergency fund/ }),
      ).not.toBeInTheDocument(),
    );
    await user.type(
      screen.getByRole("spinbutton", {
        name: new RegExp(englishCopy.home.monthly_amount),
      }),
      "25000",
    );
    await user.click(screen.getByRole("button", { name: "Save" }));

    const rebuiltCard = await screen.findByRole("article", {
      name: /Change your monthly investments/,
    });
    expect(rebuiltCard).toHaveTextContent("25,000");
    expect(actionReads).toBeGreaterThan(1);
    expect(planReads).toBeGreaterThan(1);
  });

  it("converts a monthly DKK amount to rupees before saving", async () => {
    const user = userEvent.setup();
    let saved: Record<string, unknown> | null = null;
    renderDkkHome(
      { monthly_sip_capacity: "0.00", new_money_this_week: "0.00" },
      "12.5",
    );
    server.use(
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        saved = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(saved);
      }),
    );

    await screen.findByText(englishCopy.home.amount_question);
    const amount = screen.getByRole("spinbutton", {
      name: new RegExp(englishCopy.home.monthly_amount),
    });
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: "kr" })).toHaveAttribute(
        "aria-checked",
        "true",
      ),
    );
    await user.type(amount, "5000");
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(saved).toMatchObject({ monthly_sip_capacity: "62500.00" }),
    );
  });

  it("shows a DKK prefill and converts extra money before saving", async () => {
    const user = userEvent.setup();
    let saved: Record<string, unknown> | null = null;
    renderDkkHome(
      {
        monthly_sip_capacity: "125000.00",
        new_money_this_week: "62500.00",
      },
      "12.5",
    );
    server.use(
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        saved = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(saved);
      }),
    );

    expect(
      await screen.findByRole("link", { name: "No holdings added yet" }),
    ).toBeInTheDocument();
    await user.click(
      await screen.findByRole("button", {
        name: englishCopy.home.extra_money_link,
      }),
    );
    const amount = screen.getByRole("spinbutton", {
      name: new RegExp(englishCopy.home.weekly_amount),
    });
    expect(amount).toHaveValue(5000);
    expect(screen.getByRole("radio", { name: "kr" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await user.clear(amount);
    await user.type(amount, "5000");
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(saved).toMatchObject({
        monthly_sip_capacity: "125000.00",
        new_money_this_week: "62500.00",
      }),
    );
  });

  it("uses rupee entry and saves the typed value when the DKK rate is missing", async () => {
    const user = userEvent.setup();
    let saved: Record<string, unknown> | null = null;
    const { queryClient, fxRequested } = renderDkkHome(
      { monthly_sip_capacity: "0.00", new_money_this_week: "0.00" },
      undefined,
    );
    server.use(
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        saved = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(saved);
      }),
    );

    await screen.findByText(englishCopy.home.amount_question);
    await fxRequested;
    await waitFor(() =>
      expect(
        queryClient.getQueryState(["household-context", "fx", "DKK", "INR"])
          ?.fetchStatus,
      ).toBe("idle"),
    );
    expect(
      queryClient.getQueryState(["household-context", "fx", "DKK", "INR"])
        ?.status,
    ).toBe("error");
    const amount = screen.getByRole("spinbutton", {
      name: new RegExp(englishCopy.home.monthly_amount),
    });
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: "kr" })).toBeDisabled(),
    );
    expect(screen.getByText(englishCopy.home.dkk_rate_missing)).toBeVisible();
    expect(screen.getByRole("radio", { name: "₹" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await user.type(amount, "5000");
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(saved).toMatchObject({ monthly_sip_capacity: "5000" }),
    );
  });

  it("converts the amount when switching between kroner and rupees", async () => {
    const user = userEvent.setup();
    let saved: Record<string, unknown> | null = null;
    renderDkkHome(
      { monthly_sip_capacity: "0.00", new_money_this_week: "0.00" },
      "12.5",
    );
    server.use(
      http.put(/\/v1\/me\/investing$/, async ({ request }) => {
        saved = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(saved);
      }),
    );

    const amount = await screen.findByRole("spinbutton", {
      name: new RegExp(englishCopy.home.monthly_amount),
    });
    await waitFor(() =>
      expect(screen.getByRole("radio", { name: "kr" })).toHaveAttribute(
        "aria-checked",
        "true",
      ),
    );
    await user.type(amount, "5000");
    await user.click(screen.getByRole("radio", { name: "₹" }));
    expect(amount).toHaveValue(62500);
    await user.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() =>
      expect(saved).toMatchObject({ monthly_sip_capacity: "62500.00" }),
    );
  });

  it("keeps cached content visible with an updating chip during a slow refresh", async () => {
    server.use(...baseHomeHandlers([]));
    const { queryClient } = renderHome();
    expect(
      await screen.findByRole("link", { name: "No holdings added yet" }),
    ).toBeVisible();
    let requestStarted = false;
    let releaseRequest: (() => void) | undefined;
    server.use(
      http.get(/\/v1\/networth$/, async () => {
        requestStarted = true;
        await new Promise<void>((resolve) => {
          releaseRequest = resolve;
        });
        return HttpResponse.json(networth());
      }),
    );
    const refresh = queryClient.invalidateQueries({
      queryKey: ["c27b", "networth"],
    });
    await waitFor(() => expect(requestStarted).toBe(true));
    act(() =>
      window.dispatchEvent(
        new CustomEvent("wealth:api-waking", { detail: { waking: true } }),
      ),
    );
    expect(screen.getByText("This week")).toBeVisible();
    expect(screen.getByText(englishCopy.offline.updating)).toBeVisible();
    releaseRequest?.();
    await refresh;
  });

  it("shows one action and removes it after Done succeeds", async () => {
    const user = userEvent.setup();
    const onlyAction = action({
      kind: "BUILD_EMERGENCY_FUND",
      payload: {
        kind: "BUILD_EMERGENCY_FUND",
        sleeve: "debt",
        instrument_category: "sweep_fd",
        amount: "10000.00",
      },
    });
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
    expect(await screen.findByRole("button", { name: "Done" })).toBeVisible();
    expect(screen.getByText("₹10,000")).toBeInTheDocument();
    const root = document.querySelector<HTMLElement>(
      '[data-copy-screen="home"]',
    );
    expect(root?.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: "Done" }));
    await waitFor(() => expect(patched).toBe(true));
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Done" }),
      ).not.toBeInTheDocument(),
    );
    expect(
      screen.getByText(englishCopy.home.amount_question),
    ).toBeInTheDocument();
  });

  it("records only the checked Denmark purchases with their prefilled amounts", async () => {
    const user = userEvent.setup();
    const abroadAction = action({
      kind: "BUY_ABROAD",
      payload: {
        kind: "BUY_ABROAD",
        sleeve: "equity_global",
        instrument_category: "abroad_etf",
        amount: "30000.00",
        member_id: "member-test-1",
        params: {
          abroad_buy_list: [
            {
              member_id: "member-test-1",
              account: "ask",
              instrument: {
                name: "Synthetic global fund",
                isin: "US0000000001",
              },
              amount_dkk: "1000",
            },
            {
              member_id: "member-test-1",
              account: "ordinary",
              instrument: { name: "Synthetic company share", ticker: "SYNTH" },
              amount_dkk: "2000",
              reason_codes: ["approved_us_stock_idea"],
            },
          ],
        },
      },
    });
    let savedLines: unknown;
    let saved = false;
    const completed = {
      ...abroadAction,
      status: "done" as const,
      has_recorded_buys: true,
    };
    server.use(...baseHomeHandlers([abroadAction]));
    server.use(
      http.get(/\/v1\/actions$/, () =>
        HttpResponse.json(saved ? [completed] : [abroadAction]),
      ),
      http.post(
        /\/v1\/actions\/action-test-1\/bought$/,
        async ({ request }) => {
          saved = true;
          savedLines = ((await request.json()) as { lines: unknown[] }).lines;
          return HttpResponse.json({
            action: completed,
            recorded_buys: [],
          });
        },
      ),
    );
    renderHome();

    await user.click(
      await screen.findByRole("button", { name: "I bought it" }),
    );
    const sheet = screen.getByRole("dialog", {
      name: englishCopy.home.buy_sheet_title,
    });
    expect(
      within(sheet).getByText(englishCopy.home.buy_sheet_multiple_description),
    ).toBeVisible();
    const fundRow = within(sheet).getByTestId("buy-row-0");
    const shareRow = within(sheet).getByTestId("buy-row-1");
    expect(fundRow).toHaveAccessibleName(
      /Synthetic global fund.*Share savings account/,
    );
    expect(
      within(fundRow).getByText("Synthetic global fund", { exact: true }),
    ).toBeVisible();
    expect(
      within(fundRow).getByText(englishCopy.denmark.account_ask, {
        exact: true,
      }),
    ).toBeVisible();
    expect(shareRow).toHaveAccessibleName(
      /Synthetic company share.*Ordinary account/,
    );
    expect(
      within(shareRow).getByText("Synthetic company share", { exact: true }),
    ).toBeVisible();
    expect(
      within(shareRow).getByText(englishCopy.denmark.account_ordinary, {
        exact: true,
      }),
    ).toBeVisible();
    expect(
      within(sheet).getByRole("group", {
        name: /Synthetic global fund.*Share savings account/,
      }),
    ).toBeVisible();
    expect(
      within(sheet).getByRole("group", {
        name: /Synthetic company share.*Ordinary account/,
      }),
    ).toBeVisible();
    expect(within(sheet).getAllByRole("checkbox")).toHaveLength(2);
    expect(within(sheet).getAllByLabelText("Amount (DKK)")).toHaveLength(2);
    expect(within(sheet).getAllByLabelText("Amount (DKK)")[0]).toHaveValue(
      "1000",
    );
    const details = within(sheet)
      .getByText(englishCopy.home.buy_more_details)
      .closest("details");
    expect(details).not.toHaveAttribute("open");
    await user.click(within(sheet).getAllByRole("checkbox")[1]!);
    await user.click(
      within(sheet).getByRole("button", { name: englishCopy.home.buy_save }),
    );

    await waitFor(() =>
      expect(savedLines).toEqual([
        { line: 0, amount: "1000", currency: "DKK" },
      ]),
    );
    expect(await screen.findByText(englishCopy.home.buy_done)).toBeVisible();
    expect(screen.getByText(englishCopy.home.buy_saved)).toBeVisible();
  });

  it("shows the stock name, step quantity, and amount in the buy sheet", async () => {
    const user = userEvent.setup();
    const stockAction = action({
      kind: "BUY_STOCK_TRANCHE",
      payload: {
        kind: "BUY_STOCK_TRANCHE",
        name: "Synthetic company share",
        amount: "12000.00",
        quantity: "2.5",
        member_id: "member-test-1",
        params: { symbol: "SYNTH" },
      },
    });
    server.use(...baseHomeHandlers([stockAction]));
    renderHome();

    await user.click(
      await screen.findByRole("button", { name: "I bought it" }),
    );
    const sheet = screen.getByRole("dialog", {
      name: englishCopy.home.buy_sheet_title,
    });
    expect(
      within(sheet).getByText(englishCopy.home.buy_sheet_description),
    ).toBeVisible();
    expect(
      within(sheet).queryByText(
        englishCopy.home.buy_sheet_multiple_description,
      ),
    ).not.toBeInTheDocument();
    const buyRow = within(sheet).getByTestId("buy-row-0");
    expect(
      within(buyRow).getByText("Synthetic company share", { exact: true }),
    ).toBeVisible();
    expect(within(sheet).getByText("2.5 units from this step")).toBeVisible();
    expect(within(sheet).getByLabelText("Amount (INR)")).toHaveValue(
      "12000.00",
    );
    expect(within(sheet).queryByRole("checkbox")).not.toBeInTheDocument();
    expect(
      within(sheet).queryByLabelText(englishCopy.home.buy_units_optional),
    ).toBeNull();
  });

  it("refreshes action rows when a fresh plan replaces their ids", async () => {
    const staleAction = action({ id: "stale-plan-action" });
    let actionReads = 0;
    server.use(...baseHomeHandlers([staleAction]));
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(weeklyPlan({ fresh: true })),
      ),
      http.get(/\/v1\/actions$/, () => {
        actionReads += 1;
        return HttpResponse.json(actionReads === 1 ? [staleAction] : []);
      }),
    );
    renderHome();

    await screen.findByText(englishCopy.home.amount_question);
    await waitFor(() => expect(actionReads).toBeGreaterThan(1));
    expect(
      screen.queryByRole("button", { name: "I bought it" }),
    ).not.toBeInTheDocument();
  });

  it("closes a stale buy sheet and refreshes actions after a 404", async () => {
    const user = userEvent.setup();
    const staleBuy = action({
      id: "stale-buy-action",
      payload: {
        kind: "BUY_LUMPSUM",
        sleeve: "equity_india",
        instrument_category: "in_core_index",
        amount: "10000.00",
        reason_code: "target_gap",
      },
    });
    let actionReads = 0;
    server.use(...baseHomeHandlers([staleBuy]));
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(weeklyPlan({ fresh: false })),
      ),
      http.get(/\/v1\/actions$/, () => {
        actionReads += 1;
        return HttpResponse.json(actionReads === 1 ? [staleBuy] : []);
      }),
      http.post(/\/v1\/actions\/stale-buy-action\/bought$/, () =>
        HttpResponse.json(
          { detail: "The action was not found." },
          { status: 404 },
        ),
      ),
    );
    renderHome();

    await user.click(
      await screen.findByRole("button", { name: "I bought it" }),
    );
    const sheet = screen.getByRole("dialog", {
      name: englishCopy.home.buy_sheet_title,
    });
    await user.click(
      within(sheet).getByRole("button", { name: englishCopy.home.buy_save }),
    );

    expect(await screen.findByRole("status")).toHaveTextContent(
      englishCopy.home.step_updated,
    );
    expect(
      screen.queryByRole("dialog", { name: englishCopy.home.buy_sheet_title }),
    ).not.toBeInTheDocument();
    await waitFor(() => expect(actionReads).toBeGreaterThan(1));
    expect(
      screen.queryByRole("button", { name: "I bought it" }),
    ).not.toBeInTheDocument();
  });

  it("refreshes actions and shows one line after marking a stale step done", async () => {
    const user = userEvent.setup();
    const staleAction = action({
      id: "stale-done-action",
      kind: "BUILD_EMERGENCY_FUND",
      payload: {
        kind: "BUILD_EMERGENCY_FUND",
        amount: "5000.00",
        instrument_category: "sweep_fd",
        reason_code: "emergency_fund_gap",
      },
    });
    let actionReads = 0;
    server.use(...baseHomeHandlers([staleAction]));
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(weeklyPlan({ fresh: false })),
      ),
      http.get(/\/v1\/actions$/, () => {
        actionReads += 1;
        return HttpResponse.json(actionReads === 1 ? [staleAction] : []);
      }),
      http.patch(/\/v1\/actions\/stale-done-action$/, () =>
        HttpResponse.json(
          { detail: "The action was not found." },
          { status: 404 },
        ),
      ),
    );
    renderHome();

    await user.click(await screen.findByRole("button", { name: "Done" }));

    expect(await screen.findByRole("status")).toHaveTextContent(
      englishCopy.home.step_updated,
    );
    await waitFor(() => expect(actionReads).toBeGreaterThan(1));
    expect(
      screen.queryByRole("button", { name: "Done" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByText(englishCopy.home.step_updated)).toHaveLength(1);
  });

  it("wraps a long fund title in the buy sheet", async () => {
    const user = userEvent.setup();
    const longName = "Synthetic Developed World Equity Index Fund Acc";
    const buyAction = action({
      kind: "BUY_ABROAD",
      payload: {
        kind: "BUY_ABROAD",
        amount: "12345.60",
        member_id: "member-test-1",
        instrument_category: "abroad_etf",
        params: {
          abroad_buy_list: [
            {
              member_id: "member-test-1",
              account: "ask",
              instrument: { name: longName, isin: "DK0000000001" },
              amount_dkk: "1000.00",
            },
          ],
        },
      },
    });
    server.use(...baseHomeHandlers([buyAction]));
    renderHome();

    await user.click(
      await screen.findByRole("button", { name: "I bought it" }),
    );
    const sheet = screen.getByRole("dialog", {
      name: englishCopy.home.buy_sheet_title,
    });
    const title = within(sheet)
      .getByTestId("buy-row-0")
      .querySelector(".line-clamp-2");
    expect(title).toHaveTextContent(longName);
    expect(title).not.toHaveClass("truncate");
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
    expect(
      within(articles[0]!).getByRole("button", { name: "Show steps" }),
    ).toHaveAttribute("data-primary", "true");
    expect(
      within(articles[1]!).getByRole("button", { name: "Show steps" }),
    ).not.toHaveAttribute("data-primary");
    expect(articles[2]).toHaveTextContent(
      englishCopy.labels.action_kinds.BUILD_EMERGENCY_FUND,
    );
    expect(articles[2]).toHaveTextContent(
      englishCopy.labels.instrument_categories.sweep_fd,
    );
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

    await user.click(
      within(article).getByRole("button", { name: "Show steps" }),
    );
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
        quantity: "2.000000",
        tax_estimate: { estimated_tax: "250.00" },
        reason_code: "valuation_stretch",
        params: { security_id: "security-test" },
      },
    });
    server.use(
      http.get(/\/v1\/networth$/, () =>
        HttpResponse.json(
          networth({
            holdings: [
              {
                added_by_you: false,
                currency: "INR",
                fx_as_of: null,
                fx_missing: false,
                name: "Synthetic holding name",
                price_as_of: null,
                price_source: "test",
                quantity: "1",
                security_id: "security-test",
                security_type: "equity",
                status: "priced",
                units_entered: false,
                symbol: "SYNTH",
                value: "500.00",
                value_usd: null,
              },
            ],
          }),
        ),
      ),
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
    expect(await screen.findByText("Sell some")).toBeInTheDocument();
    expect(
      await screen.findByText("Synthetic holding name"),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show steps" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    expect(
      within(dialog).getByText("Quantity: 2 shares", { exact: true }),
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

  it("shows a singular share quantity in the sell steps sheet", async () => {
    const user = userEvent.setup();
    const sell = action({
      kind: "SELL_TRIM",
      payload: {
        kind: "SELL_TRIM",
        security_id: "security-one-share",
        quantity: "1.000000",
        reason_code: "valuation_stretch",
      },
    });
    server.use(...baseHomeHandlers([sell]));
    renderHome();

    await user.click(await screen.findByRole("button", { name: "Show steps" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    expect(
      within(dialog).getByText("Quantity: 1 share", { exact: true }),
    ).toBeInTheDocument();
  });

  it.each([
    ["SELL_EXIT", "Sell", "I sold it", englishCopy.home.sell_steps],
    ["SELL_TRIM", "Sell some", "I sold it", englishCopy.home.sell_steps],
    [
      "REVIEW_EXIT",
      "Think about selling",
      "Done",
      englishCopy.home.review_steps,
    ],
  ])(
    "shows matching instructions and buttons for %s",
    async (kind, title, completion, instructions) => {
      const user = userEvent.setup();
      const exit = action({
        kind,
        payload: {
          kind,
          security_id: "security-sell-step",
          quantity: "2",
          amount: "100.00",
          reason_code: "valuation_stretch",
        },
      });
      server.use(...baseHomeHandlers([exit]));
      renderHome();

      const article = await screen.findByRole("article");
      expect(within(article).getByText(title)).toBeVisible();
      expect(within(article).queryByText("security-sell-step")).toBeNull();
      const showSteps = within(article).getByRole("button", {
        name: "Show steps",
      });
      const completionButton = within(article).getByRole("button", {
        name: completion,
      });
      expect(showSteps).toHaveAttribute("data-primary", "true");
      expect(completionButton).not.toHaveAttribute("data-primary");

      await user.click(showSteps);
      const dialog = screen.getByRole("dialog", {
        name: "How to take this step",
      });
      expect(dialog).toHaveTextContent(instructions);
      expect(dialog).not.toHaveTextContent(
        /invest as planned|monthly investment|buy order/i,
      );
    },
  );

  it("uses a symbol only when no matching holding name is loaded", async () => {
    const symbolAction = action({
      kind: "SELL_EXIT",
      payload: {
        kind: "SELL_EXIT",
        symbol: "SYNTH",
        security_id: "security-unlisted",
      },
    });
    server.use(...baseHomeHandlers([symbolAction]));
    renderHome();
    const article = await screen.findByRole("article");
    expect(within(article).getByText("SYNTH")).toBeVisible();
    expect(within(article).queryByText("security-unlisted")).toBeNull();
  });

  it("keeps the monthly editor open below steps when no amount is set", async () => {
    server.use(...baseHomeHandlers([action()]));
    renderHome();
    await screen.findByRole("article");
    expect(screen.getByText(englishCopy.home.amount_question)).toBeVisible();
    expect(
      screen.getByRole("button", { name: englishCopy.home.amount_save }),
    ).not.toHaveAttribute("data-primary");
  });

  it("marks a sell step done with the existing action update", async () => {
    const user = userEvent.setup();
    const sell = action({
      kind: "SELL_EXIT",
      payload: { kind: "SELL_EXIT", security_id: "security-sell-step" },
    });
    let updatedStatus = "";
    server.use(
      ...baseHomeHandlers([sell]),
      http.patch(/\/v1\/actions\/action-test-1$/, async ({ request }) => {
        updatedStatus = ((await request.json()) as { status: string }).status;
        return HttpResponse.json({ ...sell, status: "done" });
      }),
    );
    renderHome();
    await user.click(await screen.findByRole("button", { name: "I sold it" }));
    await waitFor(() => expect(updatedStatus).toBe("done"));
  });

  it("does not borrow trigger rows from a different holding", async () => {
    const user = userEvent.setup();
    const exit = action({
      kind: "SELL_EXIT",
      payload: {
        kind: "SELL_EXIT",
        security_id: "security-current-step",
        reason_code: "valuation_stretch",
      },
    });
    server.use(
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(
          weeklyPlan({
            exit_cards: [
              {
                security_id: "security-other-holding",
                action: "EXIT",
                triggers: {
                  valuation_stretch: {
                    status: "triggered",
                    value: "Synthetic value",
                    threshold: "Synthetic threshold",
                  },
                },
              },
            ],
          }),
        ),
      ),
      ...baseHomeHandlers([exit]),
    );
    renderHome();
    await user.click(await screen.findByRole("button", { name: "Show steps" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    await user.click(within(dialog).getByText("Why?"));
    expect(
      screen.queryByText(englishCopy.home.triggered_rows),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("Synthetic value")).not.toBeInTheDocument();
  });

  it("closes the monthly editor below actions when an amount is set", async () => {
    const user = userEvent.setup();
    server.use(
      ...baseHomeHandlers([action()], {
        monthly_sip_capacity: "25000.00",
        new_money_this_week: "0.00",
      }),
    );
    renderHome();
    await screen.findByRole("article");
    const monthlySummary = screen.getByText("Monthly investment");
    expect(monthlySummary.closest("details")).not.toHaveAttribute("open");
    expect(
      screen.queryByLabelText(englishCopy.home.monthly_amount),
    ).not.toBeInTheDocument();
    await user.click(monthlySummary);
    expect(
      await screen.findByLabelText(englishCopy.home.monthly_amount),
    ).toBeVisible();
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
    await user.click(await screen.findByRole("button", { name: "Show steps" }));
    expect(
      screen.getByRole("dialog", { name: "How to take this step" }),
    ).toHaveTextContent(expected);
  });

  it("shows the reason without raw action parameters inside Why", async () => {
    const user = userEvent.setup();
    const parameterAction = action({
      payload: {
        ...action().payload,
        params: { market: "IN", account_type: "ASK", unknown_key: "synthetic" },
      },
    });
    server.use(...baseHomeHandlers([parameterAction]));
    renderHome();
    await user.click(await screen.findByRole("button", { name: "Show steps" }));
    const dialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    await user.click(within(dialog).getByText("Why?"));
    expect(dialog).toHaveTextContent(
      "This brings this part of your plan closer to its target.",
    );
    expect(within(dialog).queryByText("Market")).not.toBeInTheDocument();
    expect(within(dialog).queryByText("Account type")).not.toBeInTheDocument();
    expect(within(dialog).queryByText("IN", { exact: true })).toBeNull();
    expect(within(dialog).queryByText("ASK", { exact: true })).toBeNull();
    expect(dialog).not.toHaveTextContent("unknown_key");
  });

  it("has no axe violations", async () => {
    server.use(...baseHomeHandlers([]));
    const { container } = renderHome();
    await screen.findByText(englishCopy.home.amount_question);
    expect((await axe(container)).violations).toEqual([]);
  });
});
