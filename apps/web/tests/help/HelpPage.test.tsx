import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { HelpPage } from "../../src/help/HelpPage";
import { ThemeProvider } from "../../src/ui/theme";

it("searches guide text and follows deep links", async () => {
  const user = userEvent.setup();
  const scrollIntoView = vi.fn();
  const original = Object.getOwnPropertyDescriptor(
    HTMLElement.prototype,
    "scrollIntoView",
  );
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    value: scrollIntoView,
  });

  render(
    <MemoryRouter initialEntries={["/help#loans"]}>
      <HelpPage />
    </MemoryRouter>,
  );

  expect(
    await screen.findByRole("heading", { name: "Add and update loans" }),
  ).toBeInTheDocument();
  await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
  await user.type(
    screen.getByLabelText("Search help articles"),
    "extra payment",
  );
  expect(
    screen.getByRole("heading", { name: "Add and update loans" }),
  ).toBeInTheDocument();
  expect(screen.queryByText("Add a bank deposit")).not.toBeInTheDocument();

  if (original) {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", original);
  } else {
    delete (HTMLElement.prototype as Partial<HTMLElement>).scrollIntoView;
  }
});

it("passes axe checks in light and dark themes", async () => {
  for (const theme of ["light", "dark"] as const) {
    window.localStorage.setItem("wealth.theme", theme);
    const view = render(
      <ThemeProvider>
        <MemoryRouter>
          <main>
            <HelpPage />
          </main>
        </MemoryRouter>
      </ThemeProvider>,
    );
    await waitFor(() =>
      expect(document.documentElement).toHaveAttribute("data-theme", theme),
    );
    expect((await axe(view.container)).violations).toEqual([]);
    view.unmount();
  }
});
