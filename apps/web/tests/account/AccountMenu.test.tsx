import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { AccountMenu } from "../../src/account/AccountMenu";
import { ThemeProvider } from "../../src/ui/theme";

const mocks = vi.hoisted(() => ({
  logout: vi.fn().mockResolvedValue(undefined),
  selectHousehold: vi.fn(),
  household: {
    me: {
      id: "synthetic-user",
      display_name: "Morgan Sample",
      email: "morgan@example.test",
      household_owner_id: "household-own",
      role: "partner",
    },
    households: [
      {
        owner_id: "household-own",
        owner_name: "Morgan Sample",
        member_count: 2,
        role: "partner",
      },
      {
        owner_id: "household-other",
        owner_name: "Another Example",
        member_count: 1,
        role: "viewer",
      },
    ],
    selectedHouseholdId: null,
    selectHousehold: vi.fn(),
    role: "partner",
  },
}));

vi.mock("../../src/auth/AuthContext", () => ({
  useAuth: () => ({ logout: mocks.logout }),
}));
vi.mock("../../src/household/HouseholdContext", () => ({
  useHousehold: () => mocks.household,
}));

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  mocks.logout.mockClear();
  mocks.logout.mockResolvedValue(undefined);
  mocks.household.selectHousehold = mocks.selectHousehold;
  mocks.selectHousehold.mockClear();
});

it("passes axe checks in light and dark themes", async () => {
  const user = userEvent.setup();
  for (const theme of ["light", "dark"] as const) {
    window.localStorage.setItem("wealth.theme", theme);
    const view = render(
      <ThemeProvider>
        <MemoryRouter>
          <main>
            <AccountMenu />
          </main>
        </MemoryRouter>
      </ThemeProvider>,
    );
    await waitFor(() =>
      expect(document.documentElement).toHaveAttribute("data-theme", theme),
    );
    await user.click(
      screen.getByRole("button", { name: /account menu for Morgan Sample/i }),
    );
    await screen.findByRole("dialog");
    expect((await axe(document.body)).violations).toEqual([]);
    view.unmount();
  }
});

it("shows account details and confirms a device-only sign-out", async () => {
  const user = userEvent.setup();
  render(
    <MemoryRouter>
      <AccountMenu />
    </MemoryRouter>,
  );

  const trigger = screen.getByRole("button", {
    name: /account menu for Morgan Sample/i,
  });
  expect(trigger).toHaveTextContent("MS");
  expect(trigger).toHaveTextContent("Morgan");
  await user.click(trigger);

  expect(await screen.findByText("Morgan Sample")).toBeInTheDocument();
  expect(screen.getByText("morgan@example.test")).toBeInTheDocument();
  expect(screen.getAllByText("Your family plan")).toHaveLength(2);
  expect(screen.getByText("You can edit")).toBeInTheDocument();
  expect(screen.getByLabelText("Switch family plan")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Settings" })).toHaveAttribute(
    "href",
    "/settings",
  );
  expect(screen.getByRole("link", { name: "Help and guide" })).toHaveAttribute(
    "href",
    "/help",
  );

  await user.selectOptions(
    screen.getByLabelText("Switch family plan"),
    "household-other",
  );
  expect(mocks.selectHousehold).toHaveBeenCalledWith("household-other");

  await user.click(screen.getByRole("button", { name: "Sign out" }));
  expect(
    await screen.findByText("Sign out of Wealth Copilot on this phone?"),
  ).toBeInTheDocument();
  const pushChoice = screen.getByLabelText("Turn off alerts on this phone too");
  expect(pushChoice).not.toBeChecked();
  await user.click(pushChoice);
  await user.click(screen.getByRole("button", { name: "Sign out" }));
  await waitFor(() =>
    expect(mocks.logout).toHaveBeenCalledWith({ disablePush: true }),
  );
});

it("keeps device alerts enabled unless the person asks to turn them off", async () => {
  const user = userEvent.setup();
  render(
    <MemoryRouter>
      <AccountMenu />
    </MemoryRouter>,
  );
  await user.click(
    screen.getByRole("button", { name: /account menu for Morgan Sample/i }),
  );
  await user.click(await screen.findByRole("button", { name: "Sign out" }));
  expect(
    screen.getByLabelText("Turn off alerts on this phone too"),
  ).not.toBeChecked();
  await user.click(screen.getByRole("button", { name: "Sign out" }));
  await waitFor(() =>
    expect(mocks.logout).toHaveBeenCalledWith({ disablePush: false }),
  );
});
