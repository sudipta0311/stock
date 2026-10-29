import type { ReactNode } from "react";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { HouseholdProvider } from "../../src/household/HouseholdContext";
import { HouseholdControls } from "../../src/household/HouseholdControls";
import { HouseholdJoinPage } from "../../src/household/HouseholdJoinPage";
import { HomePage } from "../../src/home/HomePage";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import type { MemberResponse } from "../../src/money/api";
import { PlanPage } from "../../src/plan/PlanPage";
import type { ActionResponse, WeeklyPlan } from "../../src/planning/api";
import englishCopy from "../../src/copy/en.json";
import { networth, weeklyPlan } from "../features/fixtures";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";

const ownerId = "member-owner-test";
const partnerId = "member-partner-test";
const ownerHouseholdId = "household-owner-test";
const primaryCurrencyKey = "wealth.settings.primary-currency";
const memberFilterKey = "wealth.household.member-filter";

function member(
  id: string,
  displayName: string,
  relationship: string,
  country: MemberResponse["residency"]["country"],
  status: MemberResponse["residency"]["status"],
  isSelf: boolean,
): MemberResponse {
  return {
    id,
    display_name: displayName,
    relationship,
    is_self: isSelf,
    created_at: "2026-10-28T00:00:00Z",
    residency: {
      country,
      status,
      settle_plan: country === "DK" ? "undecided" : "india",
      us_person: false,
      planned_return_year: null,
    },
    tax: {
      annual_income: null,
      annual_income_currency: "INR",
      birth_year: null,
      employment_type: "other",
      fy_realised_equity_ltcg: "0.00",
      fy_realised_equity_stcg: "0.00",
      surcharge_pct: "0.00",
      tax_regime: "new",
      tax_slab_pct: null,
    },
    investing: {
      monthly_sip_capacity: "0.00",
      existing_sips: [],
      new_money_this_week: "0.00",
      has_lrs_broker: false,
      prefers_no_paperwork: false,
      satellite_pct: "0.00",
      birth_year: null,
      ask_used_this_fy_dkk: "0.00",
      lrs_used_this_fy: id === partnerId ? "250000.00" : "0.00",
      broker: "nordnet",
      has_indian_nri_share_account: false,
      has_danish_pension_account: false,
      us_stock_picks_opt_in: false,
      us_stock_picks_pct: "10.00",
    },
  };
}

const members = [
  member(ownerId, "Test owner", "self", "IN", "RI", true),
  member(partnerId, "Test partner", "spouse", "DK", "NRI", false),
];

function meResponse() {
  return {
    id: "user-partner-test",
    email: "partner@example.test",
    display_name: "Test partner",
    created_at: "2026-10-28T00:00:00Z",
    household_owner_id: ownerHouseholdId,
    me_member_id: partnerId,
    members,
    checklist: [],
    profile: null,
    residency: members[1]?.residency,
    role: "partner",
  };
}

function planResponse(): WeeklyPlan {
  return weeklyPlan({
    me_member_id: partnerId,
    members: [
      {
        id: ownerId,
        name: "Test owner",
        relationship: "self",
        net_worth: "500000.00",
        sleeves: { equity_india: "400000.00", debt: "100000.00" },
        eligibility_status: {
          ppf: "not_allowed",
          nsc: "not_allowed",
        },
        eligibility_conditions: {
          ppf: "ppf_nri_eligibility",
          nsc: "small_savings_nri_eligibility",
        },
        limits_room: {
          ppf_annual_inr: "125000.00",
          ask_deposit_dkk: "5000.00",
        },
        notes: [],
      },
      {
        id: partnerId,
        name: "Test partner",
        relationship: "spouse",
        net_worth: "1000000.00",
        sleeves: { equity_global: "600000.00", debt: "400000.00" },
        eligibility_status: { ppf: "conditional" },
        eligibility_conditions: { ppf: "ppf_nri_eligibility" },
        limits_room: {
          ppf_annual_inr: "0.00",
          ask_deposit_dkk: "8000.00",
        },
        notes: [],
      },
    ],
    placement_notes: ["clubbing_check"],
  });
}

function actionFor(
  id: string,
  memberId: string,
  memberName: string,
  status: "proposed" | "done" = "proposed",
): ActionResponse {
  return {
    id,
    week: "2026-W44",
    rank: memberId === ownerId ? 1 : 2,
    kind: "BUY_LUMPSUM",
    payload: {
      kind: "BUY_LUMPSUM",
      sleeve: "equity_india",
      instrument_category: "in_core_index",
      amount: "10000.00",
      route: "A",
      reason_code: "target_gap",
      member_id: memberId,
      member_name: memberName,
      member_relationship: "spouse",
      clubbing_check: memberId === ownerId,
      params: { account_type: "ORDINARY" },
    },
    status,
    done_by: status === "done" ? "Test partner" : undefined,
    created_at: "2026-10-28T00:00:00Z",
  } as ActionResponse;
}

function householdHandlers(
  actions: ActionResponse[] = [
    actionFor("action-owner-test", ownerId, "Test owner"),
    actionFor("action-partner-test", partnerId, "Test partner"),
    actionFor("action-done-test", partnerId, "Test partner", "done"),
  ],
  requests?: { assets: string[]; networth: string[] },
) {
  return [
    http.get(/\/v1\/me$/, () => HttpResponse.json(meResponse())),
    http.get(/\/v1\/members$/, () => HttpResponse.json(members)),
    http.get(/\/v1\/households$/, () =>
      HttpResponse.json([
        {
          owner_id: ownerHouseholdId,
          owner_name: "Test owner",
          member_count: 2,
          role: "partner",
        },
      ]),
    ),
    http.get("https://api.frankfurter.dev/v2/rate/dkk/inr", () =>
      HttpResponse.json({ date: "2026-10-27", rate: 100 }),
    ),
    http.get(/\/v1\/plan$/, () => HttpResponse.json(planResponse())),
    http.get(/\/v1\/assets$/, ({ request }) => {
      requests?.assets.push(
        new URL(request.url).searchParams.get("member_id") ?? "",
      );
      return HttpResponse.json([]);
    }),
    http.get(/\/v1\/networth$/, ({ request }) => {
      const memberId = new URL(request.url).searchParams.get("member_id") ?? "";
      requests?.networth.push(memberId);
      const amount = memberId === ownerId ? "500000.00" : "1500000.00";
      return HttpResponse.json(
        networth({
          net_worth: amount,
          assets_total: amount,
          investable_total: amount,
        }),
      );
    }),
    http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
    http.get(/\/v1\/imports$/, () => HttpResponse.json([])),
    http.get(/\/v1\/actions$/, () => HttpResponse.json(actions)),
    http.put(/\/v1\/me\/checklist$/, async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
  ];
}

function renderHouseholdPage(page: ReactNode, initial = "/") {
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[initial]}>
        <HouseholdControls />
        {page}
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("family plan screens", () => {
  it("labels the owner relative to a partner viewer and filters Home by the selected person", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem(primaryCurrencyKey, "DKK");
    server.use(...householdHandlers());
    const { container } = renderHouseholdPage(<HomePage />);

    const actionCards = await screen.findAllByRole("article", {
      name: /Invest now/,
    });
    const ownerCard = actionCards.find((card) =>
      card.textContent?.includes("For Test owner"),
    );
    expect(ownerCard).toBeDefined();
    if (!ownerCard) throw new Error("The owner's action card was not rendered");
    await screen.findByText(/Reference rate: 1 DKK/);
    const freshOwnerCard = screen
      .getAllByRole("article", { name: /Invest now/ })
      .find((card) => card.textContent?.includes("For Test owner"));
    expect(freshOwnerCard).toBeDefined();
    if (!freshOwnerCard)
      throw new Error("The owner's action card was not rendered");
    expect(freshOwnerCard).toHaveTextContent("For Test owner");
    expect(within(freshOwnerCard).getByText(/DKK 100.00/)).toBeInTheDocument();
    expect(within(freshOwnerCard).getByText(/₹10,000/)).toBeInTheDocument();
    expect(freshOwnerCard).not.toHaveTextContent("For you");
    expect(
      screen.getByText(/Reference rate: 1 DKK = ₹100/),
    ).toBeInTheDocument();
    await user.click(
      within(freshOwnerCard).getByRole("button", { name: "How to" }),
    );
    const howTo = await screen.findByRole("dialog", {
      name: "How to take this step",
    });
    expect(howTo).toHaveTextContent(
      "Open Nordnet and choose your standard investment account.",
    );
    expect(howTo).toHaveTextContent("Test owner");
    await user.click(within(howTo).getByText("Why?"));
    expect(howTo).toHaveTextContent(
      "Income from money you give your spouse can be taxed as yours.",
    );
    await user.click(within(howTo).getByRole("button", { name: "Close" }));

    expect(document.body).not.toHaveTextContent("clubbing_check");
    expect((await axe(container)).violations).toEqual([]);

    await user.selectOptions(
      screen.getByRole("combobox", {
        name: englishCopy.household.member_switcher,
      }),
      ownerId,
    );
    await waitFor(() => {
      expect(screen.queryByText("For you")).not.toBeInTheDocument();
      expect(screen.queryByText("For Test partner")).not.toBeInTheDocument();
    });
  });

  it("shows family holdings, member limits, and translated placement notes", async () => {
    const user = userEvent.setup();
    server.use(...householdHandlers([]));
    const { container } = renderHouseholdPage(<PlanPage />);

    expect(await screen.findByText("Who holds what")).toBeInTheDocument();
    await screen.findByText(/Reference rate: 1 DKK/);
    expect(
      screen.getByRole("heading", { name: "Test owner" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "You" })).toBeInTheDocument();
    expect(screen.getAllByText("DKK 5,000.00").length).toBeGreaterThan(0);
    const ownerSection = screen
      .getByRole("heading", { name: "Test owner" })
      .closest("section");
    const partnerSection = screen
      .getByRole("heading", { name: "You" })
      .closest("section");
    expect(ownerSection).not.toBeNull();
    expect(partnerSection).not.toBeNull();
    await user.click(within(ownerSection!).getByText("Details"));
    await user.click(within(partnerSection!).getByText("Details"));
    expect(
      screen.getAllByText(englishCopy.household_ui.ppf_room).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("Sent abroad this year").length).toBeGreaterThan(
      0,
    );
    expect(
      screen.getAllByText(englishCopy.household_ui.ask_room).length,
    ).toBeGreaterThan(0);
    await user.click(
      screen.getByText(englishCopy.household_ui.placement_details),
    );
    expect(
      screen.getByText(
        "Income from money you give your spouse can be taxed as yours.",
      ),
    ).toBeInTheDocument();
    expect(document.body).not.toHaveTextContent("clubbing_check");
    expect(
      screen.getByText(englishCopy.household_ui.personal_disclaimer),
    ).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
    await user.selectOptions(
      screen.getByRole("combobox", {
        name: englishCopy.household.member_switcher,
      }),
      ownerId,
    );
    expect(
      await screen.findByRole("heading", { name: "Plan for Test owner" }),
    ).toBeInTheDocument();
    expect(screen.queryByText("Who holds what")).not.toBeInTheDocument();
  });

  it("shows the selected member's conditional PPF note on a new asset", async () => {
    window.sessionStorage.setItem(memberFilterKey, partnerId);
    server.use(...householdHandlers([]));
    const { container } = renderWithQueries(
      <HouseholdProvider>
        <MemoryRouter initialEntries={["/money/add/ppf"]}>
          <HouseholdControls />
          <Routes>
            <Route path="/money/add/:type" element={<AddAssetPage />} />
          </Routes>
        </MemoryRouter>
      </HouseholdProvider>,
    );
    expect(
      await screen.findByText("Possible with the account's terms"),
    ).toBeInTheDocument();
    expect(screen.getByText("What can fit for You")).toBeInTheDocument();
    expect(
      screen.queryByText(/Not available for new deposits/),
    ).not.toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });

  it("joins from an invite and brings existing investments by default", async () => {
    const user = userEvent.setup();
    let joinBody: Record<string, unknown> | null = null;
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({ ...meResponse(), profile: {} }),
      ),
      http.get(/\/v1\/assets$/, () =>
        HttpResponse.json([{ id: "asset-test" }]),
      ),
      http.post(/\/v1\/households\/join$/, async ({ request }) => {
        joinBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({ owner_id: ownerHouseholdId });
      }),
    );
    const client = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    const { container } = render(
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={["/join?token=invite-test"]}>
          <Routes>
            <Route path="/join" element={<HouseholdJoinPage />} />
            <Route path="/" element={<p>Family home</p>} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
    const checkbox = await screen.findByLabelText(
      /Bring my existing investments/,
    );
    expect(checkbox).toBeChecked();
    expect((await axe(container)).violations).toEqual([]);
    await user.click(
      screen.getByRole("button", { name: englishCopy.household.join_button }),
    );
    expect(await screen.findByText("Family home")).toBeInTheDocument();
    await waitFor(() =>
      expect(joinBody).toEqual({ token: "invite-test", bring_my_data: true }),
    );
  });
});
