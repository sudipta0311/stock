import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { server } from "../msw";
import englishCopy from "../../src/copy/en.json";
import { SettingsPage, deleteWarning } from "../../src/settings/SettingsPage";
import { downloadSettingsExport } from "../../src/settings/api";
import { ThemeProvider } from "../../src/ui/theme";
import type { SettingsMe } from "../../src/settings/api";
import { HouseholdProvider } from "../../src/household/HouseholdContext";

const { logoutMock, enablePushMock } = vi.hoisted(() => ({
  logoutMock: vi.fn(),
  enablePushMock: vi.fn().mockResolvedValue("push-subscription-test"),
}));

vi.mock("../../src/auth/AuthContext", () => ({
  useAuth: () => ({ logout: logoutMock }),
}));
vi.mock("../../src/pwa/push", () => ({
  enablePush: enablePushMock,
  disablePush: vi.fn().mockResolvedValue(undefined),
  PushSetupError: class PushSetupError extends Error {
    constructor(
      readonly reason:
        "unavailable" | "permission" | "configuration" | "subscription",
    ) {
      super(reason);
    }
  },
}));

const householdOwnerId = "00000000-0000-4000-8000-000000000001";
const settingsMe: SettingsMe = {
  id: "user-test-owner",
  email: "owner@example.test",
  created_at: "2026-10-01T00:00:00Z",
  display_name: "Test Owner",
  household_owner_id: householdOwnerId,
  me_member_id: "member-self-test",
  checklist: [],
  profile: null,
  residency: {
    country: "IN",
    status: "RI",
    settle_plan: "undecided",
    us_person: false,
    planned_return_year: null,
  },
  role: "owner",
};
const settingsMembers = [
  {
    id: "member-self-test",
    display_name: "Test Owner",
    created_at: "2026-10-01T00:00:00Z",
    is_self: true,
    relationship: "self",
    tax: { tax_regime: "new", tax_slab_pct: "20" },
    residency: {
      country: "IN",
      status: "RI",
      settle_plan: "undecided",
      us_person: false,
      planned_return_year: null,
    },
    investing: {
      monthly_sip_capacity: "2500.00",
      existing_sips: [],
      new_money_this_week: "0.00",
      has_lrs_broker: false,
      prefers_no_paperwork: false,
      satellite_pct: "20.00",
      birth_year: null,
      ask_used_this_fy_dkk: "0.00",
      lrs_used_this_fy: "0.00",
    },
  },
  {
    id: "member-child-test",
    display_name: "Test Child",
    created_at: "2026-10-01T00:00:00Z",
    is_self: false,
    relationship: "child",
    tax: {},
    residency: {
      country: "IN",
      status: "RI",
      settle_plan: "undecided",
      us_person: false,
      planned_return_year: null,
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
      lrs_used_this_fy: "0.00",
    },
  },
];
const consentValues = {
  weekly_email: false,
  reminder_email: false,
  push: false,
  urgent_alerts: false,
};
const investingPrefs = {
  monthly_sip_capacity: "2500.00",
  existing_sips: [],
  new_money_this_week: "0.00",
  has_lrs_broker: false,
  prefers_no_paperwork: false,
  satellite_pct: "20.00",
  birth_year: null,
  ask_used_this_fy_dkk: "0.00",
  lrs_used_this_fy: "0.00",
};

function settingsHandlers() {
  return [
    http.get("*/v1/me", () => HttpResponse.json(settingsMe)),
    http.get("*/v1/auth/sessions", () =>
      HttpResponse.json([
        {
          id: "00000000-0000-4000-8000-000000000010",
          device_label: "iPhone",
          created_at: "2026-10-01T00:00:00Z",
          last_used_at: "2026-10-29T00:00:00Z",
          current: true,
        },
      ]),
    ),
    http.post(
      "*/v1/auth/logout-all",
      () => new HttpResponse(null, { status: 204 }),
    ),
    http.get("*/v1/members", () => HttpResponse.json(settingsMembers)),
    http.get("*/v1/households", () =>
      HttpResponse.json([
        {
          owner_id: householdOwnerId,
          owner_name: "Test Household",
          member_count: 2,
          role: "owner",
        },
      ]),
    ),
    http.get("*/v1/households/access", () =>
      HttpResponse.json([
        {
          user_id: "user-test-viewer",
          member_id: "member-child-test",
          role: "viewer",
          display_name: "Test Viewer",
          created_at: "2026-10-01T00:00:00Z",
          revoked_at: null,
        },
      ]),
    ),
    http.get("*/v1/households/invites", () => HttpResponse.json([])),
    http.get("*/v1/highlights/dismissals", () =>
      HttpResponse.json([
        {
          key: "synthetic_notice",
          fingerprint: "a".repeat(24),
          dismissed_at: "2026-10-28T00:00:00Z",
        },
      ]),
    ),
    http.get("*/v1/me/consents", () => HttpResponse.json(consentValues)),
    http.put("*/v1/me/consents", async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
    http.get("*/v1/me/investing", () => HttpResponse.json(investingPrefs)),
    http.put("*/v1/me/investing", async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
    http.put("*/v1/me/profile", async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
    http.patch("*/v1/members/member-self-test", async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
    http.post("*/v1/members", async ({ request }) =>
      HttpResponse.json(
        { id: "member-new-test", ...((await request.json()) as object) },
        { status: 201 },
      ),
    ),
    http.delete("*/v1/members/member-child-test", () =>
      HttpResponse.json(
        {
          code: "member_has_access",
          title: "Member still has access",
          errors: [],
        },
        { status: 409 },
      ),
    ),
    http.get("*/v1/meta/risk-questions", () =>
      HttpResponse.json([
        {
          id: "comfort-test",
          text: "How would you feel about a short fall?",
          options: [
            { id: "calm", text: "I could wait" },
            { id: "worried", text: "I would worry" },
          ],
        },
      ]),
    ),
    http.put("*/v1/me/risk-answers", async ({ request }) =>
      HttpResponse.json(await request.json()),
    ),
    http.get(/\/v1\/(accounts|assets|goals|imports|actions|inbox)$/, () =>
      HttpResponse.json([]),
    ),
  ];
}

function renderSettings() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0, staleTime: 0 } },
  });
  const view = render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <ThemeProvider>
          <HouseholdProvider>
            <SettingsPage />
          </HouseholdProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </MemoryRouter>,
  );
  return { ...view, queryClient };
}

beforeEach(() => {
  logoutMock.mockClear();
  enablePushMock.mockClear();
  enablePushMock.mockResolvedValue("push-subscription-test");
  window.localStorage.clear();
  window.sessionStorage.clear();
  server.use(...settingsHandlers());
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Settings", () => {
  it("lists signed-in devices and confirms sign-out everywhere", async () => {
    const user = userEvent.setup();
    renderSettings();

    expect(await screen.findByText("iPhone · This phone")).toBeInTheDocument();
    expect(
      screen.getByText("You stay signed in on this phone for 30 days."),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Sign out on all devices" }),
    );
    expect(
      screen.getByRole("heading", { name: "Sign out on all devices?" }),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "Sign out everywhere" }),
    );
    await waitFor(() =>
      expect(logoutMock).toHaveBeenCalledWith({ serverRevoked: true }),
    );
  });

  it("shows four simple groups, keeps plan details closed, and passes the accessibility scan", async () => {
    renderSettings();
    expect(
      await screen.findByRole("heading", {
        name: englishCopy.household.settings.household,
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: englishCopy.settings.notifications.title,
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: englishCopy.settings.privacy.title,
        level: 2,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", {
        name: englishCopy.settings.about.title,
        level: 2,
      }),
    ).toBeInTheDocument();
    const details = screen
      .getByText(englishCopy.settings.family.fine_tune)
      .closest("details");
    expect(details).not.toHaveAttribute("open");
    expect(
      screen.queryByRole("button", { name: englishCopy.household.leave }),
    ).not.toBeInTheDocument();
    expect(await axe(document.body)).toEqual(
      expect.objectContaining({ violations: [] }),
    );
  });

  it.each([
    {
      code: "member_is_owner",
      memberId: "member-self-test",
      message: englishCopy.household.invite_member_owner,
    },
    {
      code: "member_already_linked",
      memberId: "member-child-test",
      message: englishCopy.household.invite_member_linked,
    },
  ])(
    "explains an invite conflict for $code",
    async ({ code, memberId, message }) => {
      const user = userEvent.setup();
      let submitted: Record<string, unknown> | null = null;
      server.use(
        http.post("*/v1/households/invites", async ({ request }) => {
          submitted = (await request.json()) as Record<string, unknown>;
          return HttpResponse.json(
            { code, title: "Invite cannot be created", errors: [] },
            { status: 409 },
          );
        }),
      );
      renderSettings();
      await screen.findByRole("heading", {
        name: englishCopy.household.share_with_partner,
      });
      await user.type(
        screen.getByLabelText(englishCopy.household.invite_email),
        "partner@example.test",
      );
      await user.selectOptions(
        screen.getByLabelText(englishCopy.household.invite_member),
        memberId,
      );
      await user.click(
        screen.getByRole("button", {
          name: englishCopy.household.invite_create,
        }),
      );
      expect(await screen.findByRole("status")).toHaveTextContent(message);
      expect(submitted).toMatchObject({ member_id: memberId });
    },
  );

  it("saves the selected main display currency", async () => {
    const user = userEvent.setup();
    renderSettings();
    await screen.findByText("Test Child");
    await user.selectOptions(
      screen.getByLabelText(englishCopy.household.primary_currency),
      "DKK",
    );
    expect(
      window.localStorage.getItem("wealth.settings.primary-currency"),
    ).toBe("DKK");
  });

  it("saves profile and investing settings to their supported API paths", async () => {
    const user = userEvent.setup();
    const saved: Record<string, unknown> = {};
    server.use(
      http.put("*/v1/me/profile", async ({ request }) => {
        saved.profile = await request.json();
        return HttpResponse.json({});
      }),
      http.put("*/v1/me/investing", async ({ request }) => {
        saved.investing = await request.json();
        return HttpResponse.json(investingPrefs);
      }),
      http.patch("*/v1/members/member-self-test", async ({ request }) => {
        saved.member = await request.json();
        return HttpResponse.json(settingsMembers[0]);
      }),
    );
    renderSettings();
    await screen.findByText("Test Child");
    await user.click(screen.getByText(englishCopy.settings.family.fine_tune));
    await user.clear(
      screen.getByLabelText(englishCopy.settings.family.display_name),
    );
    await user.type(
      screen.getByLabelText(englishCopy.settings.family.display_name),
      "Updated Test Owner",
    );
    await user.clear(
      screen.getByLabelText(englishCopy.settings.family.monthly_sip),
    );
    await user.type(
      screen.getByLabelText(englishCopy.settings.family.monthly_sip),
      "3500",
    );
    await user.click(
      screen.getByRole("button", {
        name: englishCopy.settings.family.save_plan,
      }),
    );

    await screen.findByText(englishCopy.settings.family.save_profile_success);
    expect(saved.member).toEqual({ display_name: "Updated Test Owner" });
    expect(saved.profile).toEqual(
      expect.objectContaining({
        tax_regime: "new",
        language: "en",
        timezone: "Asia/Kolkata",
      }),
    );
    expect(saved.investing).toEqual(
      expect.objectContaining({
        monthly_sip_capacity: "3500",
        satellite_pct: "20.00",
        has_lrs_broker: false,
      }),
    );
  });

  it("shows the optional year-start account value for a Denmark resident", async () => {
    const user = userEvent.setup();
    const selfMember = settingsMembers[0]!;
    let submitted: Record<string, unknown> | null = null;
    server.use(
      http.get("*/v1/members", () =>
        HttpResponse.json([
          {
            ...selfMember,
            residency: {
              country: "DK",
              status: "NRI",
              settle_plan: "undecided",
              us_person: false,
              planned_return_year: null,
            },
            investing: {
              ...selfMember.investing,
              ask_value_start_of_year_dkk: null,
            },
          },
          settingsMembers[1],
        ]),
      ),
      http.patch("*/v1/members/member-self-test", async ({ request }) => {
        submitted = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(selfMember);
      }),
    );
    renderSettings();
    await screen.findByText("Test Child");
    await user.click(
      screen.getAllByText(englishCopy.household.settings.edit_member)[0]!,
    );
    expect(
      screen.getByLabelText(englishCopy.denmark.ask_start_value_label),
    ).toBeInTheDocument();
    expect(
      screen.getByText(englishCopy.denmark.ask_start_value_hint),
    ).toBeInTheDocument();
    await user.type(
      screen.getByLabelText(englishCopy.denmark.ask_start_value_label),
      "105000",
    );
    await user.click(
      screen.getByRole("button", {
        name: englishCopy.household.settings.save_member,
      }),
    );
    await screen.findByText(englishCopy.household.settings.details_saved);
    expect(submitted).toMatchObject({
      investing: { ask_value_start_of_year_dkk: "105000" },
    });
  });

  it("asks for push setup only after the push toggle is tapped", async () => {
    const user = userEvent.setup();
    const savedConsents: unknown[] = [];
    server.use(
      http.put("*/v1/me/consents", async ({ request }) => {
        const body = await request.json();
        savedConsents.push(body);
        return HttpResponse.json(body);
      }),
    );
    renderSettings();
    const pushToggle = await screen.findByRole("checkbox", {
      name: englishCopy.settings.notifications.push,
    });
    expect(enablePushMock).not.toHaveBeenCalled();
    await user.click(pushToggle);
    await waitFor(() => expect(enablePushMock).toHaveBeenCalledOnce());
    expect(savedConsents).toEqual([expect.objectContaining({ push: true })]);
  });

  it("sends a new member with self status off and explains a removal conflict", async () => {
    const user = userEvent.setup();
    const created: unknown[] = [];
    server.use(
      http.post("*/v1/members", async ({ request }) => {
        created.push(await request.json());
        return HttpResponse.json(
          {
            id: "member-new-test",
            display_name: "Test Guest",
            relationship: "other",
            is_self: false,
          },
          { status: 201 },
        );
      }),
    );
    renderSettings();
    await screen.findByText("Test Child");
    await user.click(
      screen.getByRole("button", {
        name: englishCopy.settings.family.add_person,
      }),
    );
    await user.type(
      screen.getByLabelText(englishCopy.settings.family.name),
      "Test Guest",
    );
    await user.click(
      screen.getByRole("button", {
        name: englishCopy.settings.family.add_person,
      }),
    );
    await waitFor(() =>
      expect(created).toEqual([
        {
          display_name: "Test Guest",
          relationship: "other",
          is_self: false,
          residency: {
            status: "RI",
            country: "IN",
            settle_plan: "india",
            planned_return_year: null,
          },
        },
      ]),
    );

    await user.click(
      screen.getByRole("button", {
        name: englishCopy.settings.family.remove_person,
      }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: englishCopy.settings.family.remove_person,
    });
    await user.click(
      within(dialog).getByRole("button", {
        name: englishCopy.settings.family.remove_person,
      }),
    );
    expect(await within(dialog).findByRole("alert")).toHaveTextContent(
      englishCopy.settings.family.remove_conflict,
    );
  });

  it("confirms a tax residence change and invalidates the household plan", async () => {
    const user = userEvent.setup();
    let patchBody: Record<string, unknown> | null = null;
    server.use(
      ...settingsHandlers(),
      http.patch("*/v1/members/member-child-test", async ({ request }) => {
        patchBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({
          ...settingsMembers[1],
          ...patchBody,
        });
      }),
    );
    const { queryClient } = renderSettings();
    queryClient.setQueryData(["c27b", "plan"], { members: [] });
    const invalidateQueries = vi.spyOn(queryClient, "invalidateQueries");
    await screen.findByText("Test Child");
    await user.click(
      screen.getAllByRole("button", {
        name: englishCopy.household.settings.edit_member,
      })[1]!,
    );
    await user.selectOptions(
      screen.getByLabelText(englishCopy.household.residency_title),
      "NRI",
    );
    await user.click(
      screen.getByRole("button", {
        name: englishCopy.household.settings.save_member,
      }),
    );
    const confirmation = await screen.findByRole("dialog", {
      name: englishCopy.household.residence_confirm_title,
    });
    expect(confirmation).toHaveTextContent("Plan will be recalculated");
    expect(confirmation).toHaveTextContent(
      "Test Child is nri living in denmark",
    );
    await user.click(
      within(confirmation).getByRole("button", {
        name: englishCopy.household.confirm,
      }),
    );
    await waitFor(() => expect(patchBody).not.toBeNull());
    expect(patchBody).toMatchObject({
      residency: {
        status: "NRI",
        country: "DK",
        settle_plan: "undecided",
        planned_return_year: null,
      },
    });
    expect(invalidateQueries).toHaveBeenCalledWith(
      expect.objectContaining({ queryKey: ["c27b"] }),
    );
  });

  it("warns about shared access and exports the open household without auth tokens", async () => {
    const access = [
      {
        user_id: "user-test-viewer",
        member_id: "member-child-test",
        role: "viewer" as const,
        display_name: "Test Viewer",
        created_at: "2026-10-01T00:00:00Z",
        revoked_at: null,
      },
    ];
    expect(deleteWarning(settingsMe, access, [])).toContain("Test Viewer");
    const exportBlob = await downloadSettingsExport();
    const exported = JSON.parse(await exportBlob.text()) as Record<
      string,
      unknown
    >;
    expect(exported).toEqual(
      expect.objectContaining({
        household_name: "Test Household",
        me: settingsMe,
        members: settingsMembers,
        accounts: [],
        assets: [],
        goals: [],
        imports: [],
        actions: [],
        inbox: [],
        highlight_dismissals: [
          {
            key: "synthetic_notice",
            fingerprint: "a".repeat(24),
            dismissed_at: "2026-10-28T00:00:00Z",
          },
        ],
      }),
    );
    expect(JSON.stringify(exported).toLowerCase()).not.toContain("token");
  });

  it("clears browser storage, IndexedDB, and Cache Storage when deleting an account", async () => {
    const user = userEvent.setup();
    const cacheStorage = {
      keys: vi.fn().mockResolvedValue(["api-snapshots", "workbox-vite"]),
      delete: vi.fn().mockResolvedValue(true),
    };
    vi.stubGlobal("caches", cacheStorage);
    await new Promise<void>((resolve, reject) => {
      const request = indexedDB.open("wealth-offline-test", 1);
      request.onupgradeneeded = () => request.result.createObjectStore("items");
      request.onsuccess = () => {
        request.result.close();
        resolve();
      };
      request.onerror = () => reject(request.error);
    });
    window.localStorage.setItem("wealth-test", "local");
    window.sessionStorage.setItem("wealth-test", "session");
    let deleted = false;
    server.use(
      http.delete("*/v1/me", () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    renderSettings();
    await screen.findByText("Test Child");
    await user.click(
      screen.getByRole("button", { name: englishCopy.settings.privacy.delete }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: englishCopy.settings.privacy.delete,
    });
    await user.type(
      within(dialog).getByLabelText(
        englishCopy.settings.privacy.delete_confirm,
      ),
      "DELETE",
    );
    await user.click(
      within(dialog).getByRole("button", {
        name: englishCopy.settings.privacy.delete_button,
      }),
    );

    await waitFor(() => expect(logoutMock).toHaveBeenCalledOnce());
    expect(deleted).toBe(true);
    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
    expect(cacheStorage.delete).toHaveBeenCalledWith("api-snapshots");
    expect(cacheStorage.delete).toHaveBeenCalledWith("workbox-vite");
    await act(async () => {
      const databases = await indexedDB.databases();
      expect(
        databases.some((database) => database.name === "wealth-offline-test"),
      ).toBe(false);
    });
  });
});
