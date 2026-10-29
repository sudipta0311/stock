import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Link, Route, Routes } from "react-router";
import { beforeEach, describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { HouseholdControls } from "../../src/household/HouseholdControls";
import { HouseholdJoinPage } from "../../src/household/HouseholdJoinPage";
import { HouseholdProvider } from "../../src/household/HouseholdContext";
import { HomePage } from "../../src/home/HomePage";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import { MoneyMapPage } from "../../src/money/MoneyMapPage";
import type { MemberResponse, MeResponse } from "../../src/money/api";
import { PlanPage } from "../../src/plan/PlanPage";
import type { ActionResponse, WeeklyPlan } from "../../src/planning/api";
import { MemberDetailsEditor } from "../../src/settings/MemberDetailsEditor";
import type { SettingsMember } from "../../src/settings/api";
import englishCopy from "../../src/copy/en.json";
import { server } from "../msw";
import { action, networth, weeklyPlan } from "../features/fixtures";
import { renderWithQueries } from "../features/render";

const ownerId = "member-demo-owner";
const partnerId = "member-demo-partner";
const householdId = "household-demo";

function member(
  id: string,
  displayName: string,
  relationship: "self" | "spouse",
  country: "DK" | "IN",
  status: "NRI" | "RI",
  isSelf: boolean,
): MemberResponse {
  return {
    id,
    display_name: displayName,
    created_at: "2026-10-28T00:00:00Z",
    is_self: isSelf,
    relationship,
    residency: {
      country,
      status,
      settle_plan: status === "NRI" ? "undecided" : "india",
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
      ask_used_this_fy_dkk: "1000.00",
      birth_year: null,
      existing_sips: [],
      has_lrs_broker: false,
      lrs_used_this_fy: id === partnerId ? "25000.00" : "0.00",
      broker: "nordnet",
      has_indian_nri_share_account: false,
      has_danish_pension_account: false,
      us_stock_picks_opt_in: false,
      us_stock_picks_pct: "10.00",
      monthly_sip_capacity: "0.00",
      new_money_this_week: "0.00",
      prefers_no_paperwork: false,
      satellite_pct: "20.00",
    },
  };
}

function members(): MemberResponse[] {
  return [
    member(ownerId, "Demo Owner", "self", "DK", "NRI", true),
    member(partnerId, "Demo Partner", "spouse", "IN", "RI", false),
  ];
}

function me(
  viewerId = ownerId,
  role: "owner" | "partner" | "viewer" = "owner",
): MeResponse {
  const current = members().find((candidate) => candidate.id === viewerId);
  return {
    id: `user-${viewerId}`,
    email: "demo@example.test",
    created_at: "2026-10-28T00:00:00Z",
    display_name: current?.display_name ?? "Demo viewer",
    household_owner_id: householdId,
    me_member_id: viewerId,
    role,
    checklist: [],
    profile: null,
    residency: current?.residency ?? members()[0]!.residency,
    members: members(),
  };
}

function planData(overrides: Partial<WeeklyPlan> = {}): WeeklyPlan {
  return weeklyPlan({
    me_member_id: ownerId,
    members: [
      {
        id: ownerId,
        name: "Demo Owner",
        relationship: "self",
        net_worth: "1000000.00",
        sleeves: {
          equity_india: "400000.00",
          equity_global: "200000.00",
          debt: "300000.00",
        },
        limits_room: {
          ppf_annual_inr: "120000.00",
          lrs_annual_usd: "10000.00",
          ask_deposit_dkk: "50000.00",
        },
        notes: [],
      },
      {
        id: partnerId,
        name: "Demo Partner",
        relationship: "spouse",
        net_worth: "250000.00",
        sleeves: { equity_india: "200000.00", debt: "50000.00" },
        limits_room: {
          ppf_annual_inr: "100000.00",
          lrs_annual_usd: "0.00",
          ask_deposit_dkk: "0.00",
        },
        notes: ["clubbing_check"],
      },
    ],
    placement_notes: ["clubbing_check"],
    ...overrides,
  });
}

function actionForMember(
  id: string,
  rank: number,
  memberId: string,
  memberName: string,
  accountType: "ASK" | "ORDINARY",
  extra: Record<string, unknown> = {},
): ActionResponse {
  return Object.assign(
    action({
      id,
      rank,
      payload: {
        kind: "BUY_LUMPSUM",
        instrument_category: "in_core_index",
        amount: "10000.00",
        reason_code: "target_gap",
        params: { account_type: accountType },
      },
    }),
    { member_id: memberId, member_name: memberName, ...extra },
  );
}

function householdHandlers({
  viewerId = ownerId,
  role = "owner",
  actions = [],
  plan = planData(),
  onNetworth,
  onAssets,
  onFxRequest,
}: {
  viewerId?: string;
  role?: "owner" | "partner" | "viewer";
  actions?: ActionResponse[];
  plan?: WeeklyPlan;
  onNetworth?(memberId: string): void;
  onAssets?(memberId: string): void;
  onFxRequest?(): void;
} = {}) {
  const householdMembers = members();
  return [
    http.get(/\/v1\/me$/, () => HttpResponse.json(me(viewerId, role))),
    http.get(/\/v1\/members$/, () => HttpResponse.json(householdMembers)),
    http.get(/\/v1\/households$/, () =>
      HttpResponse.json([
        {
          owner_id: householdId,
          owner_name: "Demo Owner",
          member_count: householdMembers.length,
          role,
        },
      ]),
    ),
    http.get(/\/v1\/plan$/, () => HttpResponse.json(plan)),
    http.get(/\/v1\/actions$/, () => HttpResponse.json(actions)),
    http.get(/\/v1\/networth$/, ({ request }) => {
      const memberId = new URL(request.url).searchParams.get("member_id") ?? "";
      onNetworth?.(memberId);
      return HttpResponse.json(
        memberId === partnerId
          ? networth({ net_worth: "250000.00", assets_total: "250000.00" })
          : networth(),
      );
    }),
    http.get(/\/v1\/assets$/, ({ request }) => {
      onAssets?.(new URL(request.url).searchParams.get("member_id") ?? "");
      return HttpResponse.json([]);
    }),
    http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
    http.get(/\/v1\/imports$/, () => HttpResponse.json([])),
    http.get("https://api.frankfurter.dev/v2/rate/dkk/inr", () => {
      onFxRequest?.();
      return HttpResponse.json({ date: "2026-10-28", rate: 20 });
    }),
  ];
}

function renderHouseholdApp(initialPath = "/") {
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[initialPath]}>
        <HouseholdControls />
        <nav aria-label="Test pages" className="flex gap-3">
          <Link to="/">Home</Link>
          <Link to="/plan">Plan</Link>
          <Link to="/money">Money</Link>
        </nav>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/plan" element={<PlanPage />} />
          <Route path="/money" element={<MoneyMapPage />} />
          <Route path="/money/add/:type" element={<AddAssetPage />} />
        </Routes>
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

describe("household screens", () => {
  it("filters Home, Plan, and Money by the selected family member", async () => {
    const user = userEvent.setup();
    const networthRequests: string[] = [];
    const assetRequests: string[] = [];
    server.use(
      ...householdHandlers({
        onNetworth: (id) => networthRequests.push(id),
        onAssets: (id) => assetRequests.push(id),
      }),
    );
    renderHouseholdApp();
    const memberFilter = await screen.findByRole("combobox", { name: "View" });
    await user.selectOptions(memberFilter, partnerId);
    await waitFor(() => expect(networthRequests).toContain(partnerId));

    await user.click(screen.getByRole("link", { name: "Plan" }));
    expect(
      await screen.findByRole("heading", { name: "Plan for Demo Partner" }),
    ).toBeInTheDocument();
    await user.click(screen.getByRole("link", { name: "Money" }));
    await waitFor(() => expect(assetRequests).toContain(partnerId));
  });

  it("uses the signed-in member ID for You and gives account-specific steps", async () => {
    const user = userEvent.setup();
    const ownerAsk = actionForMember(
      "action-owner-ask",
      1,
      ownerId,
      "Demo Owner",
      "ASK",
      { clubbing_check: "clubbing_check" },
    );
    const partnerOrdinary = actionForMember(
      "action-partner-ordinary",
      2,
      partnerId,
      "Demo Partner",
      "ORDINARY",
    );
    const completed = Object.assign(
      actionForMember(
        "action-completed",
        3,
        partnerId,
        "Demo Partner",
        "ORDINARY",
      ),
      { status: "done" as const, done_by: "Demo Partner" },
    );
    server.use(
      ...householdHandlers({
        viewerId: partnerId,
        role: "partner",
        actions: [ownerAsk, partnerOrdinary, completed],
      }),
    );
    renderHouseholdApp();
    const actionCards = await screen.findAllByRole("article", {
      name: /Invest now/,
    });
    const ownerCard = actionCards.find((card) =>
      card.textContent?.includes("For Demo Owner"),
    );
    const partnerCard = actionCards.find((card) =>
      card.textContent?.includes("For you"),
    );
    expect(ownerCard).toBeDefined();
    expect(partnerCard).toBeDefined();
    expect(
      screen.queryByText(englishCopy.household_ui.clubbing_check),
    ).not.toBeInTheDocument();
    expect(screen.queryByText("clubbing_check")).not.toBeInTheDocument();

    const howToButtons = await screen.findAllByRole("button", {
      name: "How to",
    });
    await user.click(howToButtons[0]!);
    const askDialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    expect(askDialog).toHaveTextContent("For Demo Owner");
    expect(askDialog).toHaveTextContent("Nordnet");
    expect(askDialog).toHaveTextContent("share savings account");
    await user.click(within(askDialog).getByText("Why?"));
    expect(
      within(askDialog).getByText(englishCopy.household_ui.clubbing_check),
    ).toBeInTheDocument();
    await user.click(within(askDialog).getByRole("button", { name: "Close" }));

    await user.click(screen.getAllByRole("button", { name: "How to" })[1]!);
    const ordinaryDialog = screen.getByRole("dialog", {
      name: "How to take this step",
    });
    expect(ordinaryDialog).toHaveTextContent("For you");
    expect(ordinaryDialog).toHaveTextContent("standard investment account");
    await user.click(
      within(ordinaryDialog).getByRole("button", { name: "Close" }),
    );
  });

  it("shows member holdings, limits, and plain clubbing guidance with INR and DKK", async () => {
    const user = userEvent.setup();
    let fxRequests = 0;
    localStorage.setItem("wealth.settings.primary-currency", "DKK");
    server.use(...householdHandlers({ onFxRequest: () => fxRequests++ }));
    const { container } = renderHouseholdApp("/plan");
    expect(
      await screen.findByRole("heading", { name: "Your plan" }),
    ).toBeInTheDocument();
    expect(document.body.textContent).toContain("DKK 50,000.00");
    expect(document.body.textContent).toContain("₹10 L");
    expect(
      await screen.findByText(/Reference rate: 1 DKK = ₹20/),
    ).toBeInTheDocument();
    expect(fxRequests).toBeGreaterThan(0);
    expect((await axe(container)).violations).toEqual([]);

    await user.click(screen.getAllByText(englishCopy.household_ui.details)[0]!);
    expect(
      screen.getAllByText(englishCopy.household_ui.sent_abroad).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByText(englishCopy.household_ui.ppf_room).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getAllByText(englishCopy.household_ui.ask_room).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByText(englishCopy.household_ui.clubbing_check),
    ).toBeInTheDocument();
    expect(screen.queryByText("clubbing_check")).not.toBeInTheDocument();

    await user.click(
      screen.getByText(englishCopy.household_ui.placement_details),
    );
    expect(
      screen.getAllByText(englishCopy.household_ui.clubbing_check).length,
    ).toBeGreaterThan(0);
  });

  it("uses the selected member's residency note in the asset form", async () => {
    const user = userEvent.setup();
    const plan = planData({
      members: [
        {
          ...planData().members![0]!,
          eligibility_status: { ppf: "not_allowed" },
          eligibility_conditions: { ppf: "ppf_nri_eligibility" },
        },
        {
          ...planData().members![1]!,
          eligibility_status: { ppf: "allowed" },
        },
      ],
    });
    server.use(...householdHandlers({ plan }));
    renderHouseholdApp("/money/add/ppf");
    expect(
      await screen.findByText(
        "Not available for new deposits: the account's terms.",
      ),
    ).toBeInTheDocument();
    await user.selectOptions(
      screen.getByLabelText(englishCopy.household_ui.whose_is_this),
      partnerId,
    );
    expect(
      await screen.findByText(englishCopy.household_ui.eligibility_allowed),
    ).toBeInTheDocument();
  });

  it("asks before changing tax residence and invalidates household plan data", async () => {
    const user = userEvent.setup();
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    const editorMember = {
      ...members()[1]!,
      residency: { ...members()[1]!.residency, country: "IN", status: "RI" },
    } as SettingsMember;
    queryClient.setQueryData(["c27b", "plan", null, undefined], planData());
    let patchBody: Record<string, unknown> | null = null;
    server.use(
      http.patch(/\/v1\/members\/member-demo-partner$/, async ({ request }) => {
        patchBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(editorMember);
      }),
    );
    render(
      <QueryClientProvider client={queryClient}>
        <MemberDetailsEditor member={editorMember} disabled={false} />
      </QueryClientProvider>,
    );
    await user.click(screen.getByRole("button", { name: "Edit details" }));
    await user.selectOptions(screen.getByLabelText("Tax residence"), "NRI");
    await user.click(screen.getByRole("button", { name: "Save details" }));
    const confirmDialog = screen.getByRole("dialog", {
      name: englishCopy.household.residence_confirm_title,
    });
    expect(confirmDialog).toHaveTextContent(
      "Demo Partner is nri living in denmark",
    );
    expect((await axe(confirmDialog)).violations).toEqual([]);
    await user.click(
      within(confirmDialog).getByRole("button", { name: "Confirm" }),
    );
    await waitFor(() => expect(patchBody).not.toBeNull());
    expect(patchBody).toMatchObject({
      residency: { status: "NRI", country: "DK" },
    });
    expect(
      queryClient.getQueryState(["c27b", "plan", null, undefined])
        ?.isInvalidated,
    ).toBe(true);
  });

  it("shows the join screen accessibly and moves data only after confirmation", async () => {
    const user = userEvent.setup();
    let joinBody: Record<string, unknown> | null = null;
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({ ...me(), profile: { display_name: "Demo Owner" } }),
      ),
      http.get(/\/v1\/assets$/, () => HttpResponse.json([])),
      http.post(/\/v1\/households\/join$/, async ({ request }) => {
        joinBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({ owner_id: householdId, role: "partner" });
      }),
    );
    const { container } = renderWithQueries(
      <MemoryRouter initialEntries={["/join?token=invite-demo"]}>
        <HouseholdJoinPage />
      </MemoryRouter>,
    );
    expect(
      await screen.findByRole("heading", {
        name: englishCopy.household.join_title.replace(
          "{role}",
          englishCopy.household.role_partner,
        ),
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("checkbox", {
        name: new RegExp(englishCopy.household.join_data),
      }),
    ).toBeChecked();
    expect((await axe(container)).violations).toEqual([]);
    await user.click(
      screen.getByRole("button", { name: englishCopy.household.join_button }),
    );
    await waitFor(() =>
      expect(joinBody).toMatchObject({
        token: "invite-demo",
        bring_my_data: true,
      }),
    );
  });
});
