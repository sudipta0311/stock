import type { LoanListResponse, NetworthResponse } from "../../src/money/api";
import type { ActionResponse, WeeklyPlan } from "../../src/planning/api";
import { HttpResponse, http } from "msw";

export function networth(
  overrides: Partial<NetworthResponse> = {},
): NetworthResponse {
  return {
    as_of: "2026-10-28",
    assets_total: "1000000.00",
    by_sleeve: [],
    completeness: { ticked: [], done: [], pct: "100" },
    debt_insight: {
      message_key: "debt_share",
      debt_pct_including_locked: "0",
      locked_debt_amount: "0.00",
    },
    holdings: [],
    investable_total: "900000.00",
    liabilities_total: "0.00",
    liquidity_ladder: {
      instant: "100000.00",
      within_1y: "100000.00",
      locked: "0.00",
    },
    market_data_as_of: null,
    maturity_calendar: [],
    net_worth: "1000000.00",
    unvalued_assets: [],
    ...overrides,
  };
}

export function loanSummary(
  overrides: Partial<LoanListResponse> = {},
): LoanListResponse {
  return {
    advice: "Pay extra toward this loan.",
    advice_kind: "PAY_EXTRA",
    debt_load_level: "fine",
    debt_load_pct: "25.0",
    loans: [
      {
        id: "loan-test-1",
        name: "Home loan",
        lender: "Sample Bank",
        currency: "INR",
        outstanding: "1000000.00",
        outstanding_inr: "1000000.00",
        rate_pct: "8.5",
        emi: "10000.00",
        emi_inr: "10000.00",
        months_left: 54,
        payoff_month: { year: 2031, month: 3 },
        percent_repaid: "25.00",
        never_pays_off: false,
        fx_missing: false,
      },
    ],
    remaining_inr: "1000000.00",
    monthly_emis_inr: "10000.00",
    suggested_amount_inr: "5000.00",
    reason_codes: [],
    ...overrides,
  };
}

export function weeklyPlan(overrides: Partial<WeeklyPlan> = {}): WeeklyPlan {
  return {
    actions: [],
    carry_forward: "0.00",
    debt_picks: [],
    disclaimer: "",
    allocation: {
      equity_india: {
        target: "450000.00",
        current: "380000.00",
        gap: "70000.00",
        target_pct: "45",
        current_pct: "38",
      },
      equity_global: {
        target: "120000.00",
        current: "40000.00",
        gap: "80000.00",
        target_pct: "12",
        current_pct: "4",
      },
      debt: {
        target: "230000.00",
        current: "390000.00",
        gap: "-160000.00",
        target_pct: "23",
        current_pct: "39",
      },
      gold: {
        target: "100000.00",
        current: "100000.00",
        gap: "0.00",
        target_pct: "10",
        current_pct: "10",
      },
      cash: {
        target: "100000.00",
        current: "90000.00",
        gap: "10000.00",
        target_pct: "10",
        current_pct: "9",
      },
    },
    as_of: "2026-10-28",
    band_breaches: [],
    deployment_mode: "personal",
    debt_answer: {
      target_debt: "230000.00",
      current_debt: "390000.00",
      locked_debt: "100000.00",
      gap: "-160000.00",
      status: "enough",
      new_money_order: ["equity_india", "equity_global"],
      message_key: "debt_target_met",
      params: { current_pct: "39", target_pct: "23" },
    },
    foundation: [
      {
        key: "term_cover",
        status: "gap",
        required: "10000000.00",
        have: "5000000.00",
        gap: "5000000.00",
      },
      {
        key: "health_cover",
        status: "ok",
        required: "5000000.00",
        have: "5000000.00",
        gap: "0.00",
      },
      {
        key: "emergency_fund",
        status: "ok",
        required: "100000.00",
        have: "100000.00",
        gap: "0.00",
      },
      {
        key: "high_interest_debt",
        status: "ok",
        required: "0.00",
        have: "0.00",
        gap: "0.00",
      },
    ],
    fresh: true,
    inputs_digest: "a".repeat(64),
    me_member_id: "member-test-1",
    notes: [],
    plan_health: 100,
    route: {
      route: "A",
      vehicle: "in_international_index_fund",
      reasons: [],
      warnings: [],
      alternatives: [],
      warning_params: {},
    },
    unranked_options: [],
    exit_cards: [],
    stress: {
      india_equity_minus_25: {
        current: { total: "-25000.00" },
        target: { total: "-21000.00" },
        difference: {
          current_total: "-25000.00",
          target_total: "-21000.00",
          difference: "4000.00",
        },
      },
      us_equity_minus_30: {
        current: { total: "-12000.00" },
        target: { total: "-18000.00" },
        difference: {
          current_total: "-12000.00",
          target_total: "-18000.00",
          difference: "-6000.00",
        },
      },
    },
    week: "2026-W44",
    ...overrides,
  };
}

export function action(
  overrides: Partial<ActionResponse> = {},
): ActionResponse {
  return {
    id: "action-test-1",
    week: "2026-W44",
    rank: 1,
    kind: "BUY_LUMPSUM",
    payload: {
      kind: "BUY_LUMPSUM",
      sleeve: "equity_india",
      instrument_category: "in_core_index",
      amount: "10000.00",
      route: "A",
      security_id: null,
      tranche: null,
      reason_code: "target_gap",
      params: { market: "IN" },
    },
    status: "proposed",
    created_at: "2026-10-28T00:00:00Z",
    ...overrides,
  };
}

export function baseHomeHandlers(actions: ActionResponse[] = []) {
  return [
    http.get(/\/v1\/me$/, () =>
      HttpResponse.json({
        id: "user-test",
        email: "test@example.test",
        display_name: "Test Member",
        created_at: "2026-10-01T00:00:00Z",
        profile: null,
      }),
    ),
    http.get(/\/v1\/networth$/, () => HttpResponse.json(networth())),
    http.get(/\/v1\/plan$/, () => HttpResponse.json(weeklyPlan())),
    http.get(/\/v1\/actions$/, () => HttpResponse.json(actions)),
  ];
}
