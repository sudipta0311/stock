import { cleanup, screen, within } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import englishCopy from "../../src/copy/en.json";
import { AppShell } from "../../src/app/AppShell";
import { HouseholdProvider } from "../../src/household/HouseholdContext";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";

function householdHandlers() {
  return [
    http.get(/\/v1\/me$/, () =>
      HttpResponse.json({
        id: "",
        email: "synthetic@example.test",
        created_at: "2026-10-28T00:00:00Z",
        display_name: "Synthetic person",
        household_owner_id: "household-own",
        me_member_id: "member-self",
        checklist: [],
        profile: null,
        residency: {
          country: "IN",
          status: "RI",
          settle_plan: "india",
          us_person: false,
        },
        role: "owner",
      }),
    ),
    http.get(/\/v1\/members$/, () =>
      HttpResponse.json([
        {
          id: "member-self",
          display_name: "Synthetic person",
          relationship: "self",
          residency: { country: "IN" },
        },
        {
          id: "member-partner",
          display_name: "Synthetic partner",
          relationship: "spouse",
          residency: { country: "IN" },
        },
      ]),
    ),
    http.get(/\/v1\/households$/, () =>
      HttpResponse.json([
        {
          owner_id: "household-own",
          owner_name: "Synthetic person",
          member_count: 2,
          role: "owner",
        },
        {
          owner_id: "household-shared",
          owner_name: "Shared family",
          member_count: 1,
          role: "viewer",
        },
      ]),
    ),
  ];
}

function renderShell(path: string) {
  server.use(...householdHandlers());
  return renderWithQueries(
    <HouseholdProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<p>Home screen</p>} />
            <Route path="money/*" element={<p>Money screen</p>} />
            <Route path="plan/*" element={<p>Plan screen</p>} />
            <Route path="ideas/*" element={<p>Ideas screen</p>} />
            <Route path="inbox/*" element={<p>Inbox screen</p>} />
            <Route path="settings/*" element={<p>Settings screen</p>} />
          </Route>
        </Routes>
      </MemoryRouter>
    </HouseholdProvider>,
  );
}

describe("AppShell household picker", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
  });

  afterEach(() => {
    cleanup();
  });

  it.each([
    ["/", "Home"],
    ["/money", "Money"],
    ["/plan", "Plan"],
  ])("places the picker in the %s header", async (path, title) => {
    renderShell(path);
    const header = screen.getByRole("banner");
    const picker = await within(header).findByLabelText(
      englishCopy.household.picker_label,
    );

    expect(picker).toBeVisible();
    expect(picker).toHaveAttribute("id", "header-member-filter");
    expect(screen.getByRole("heading", { name: title })).toHaveClass(
      "truncate",
      "whitespace-nowrap",
    );
    expect(
      screen.getByRole("main").querySelector("#header-member-filter"),
    ).toBeNull();
  });

  it.each(["/inbox", "/ideas", "/settings"])(
    "hides the picker on %s",
    async (path) => {
      renderShell(path);
      const header = screen.getByRole("banner");
      await within(header).findByRole("link", { name: /Help with/ });
      expect(
        within(header).queryByLabelText(englishCopy.household.picker_label),
      ).toBeNull();
    },
  );
});
