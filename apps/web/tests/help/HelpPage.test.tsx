import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";
import { expect, it, vi } from "vitest";
import { axe } from "vitest-axe";
import { HelpPage } from "../../src/help/HelpPage";
import { ThemeProvider } from "../../src/ui/theme";

it("shows article summaries closed by default", () => {
  render(
    <MemoryRouter>
      <HelpPage />
    </MemoryRouter>,
  );

  const article = screen
    .getByRole("heading", { name: "Add and update loans" })
    .closest("details");
  expect(article).not.toHaveAttribute("open");
});

it("opens, scrolls to and focuses an article from a hash", async () => {
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
    <MemoryRouter
      initialEntries={[
        { pathname: "/help", hash: "#loans", state: { from: "/money" } },
      ]}
    >
      <Routes>
        <Route path="/help" element={<HelpPage />} />
        <Route path="/money" element={<p>Money page returned</p>} />
      </Routes>
    </MemoryRouter>,
  );

  const heading = await screen.findByRole("heading", {
    name: "Add and update loans",
  });
  const details = heading.closest("details");
  await waitFor(() => expect(details).toHaveAttribute("open"));
  await waitFor(() => expect(heading).toHaveFocus());
  expect(scrollIntoView).toHaveBeenCalled();
  if (original) {
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", original);
  } else {
    delete (HTMLElement.prototype as Partial<HTMLElement>).scrollIntoView;
  }
});

it("searches article body text and opens the selected result", async () => {
  const user = userEvent.setup();
  render(
    <MemoryRouter>
      <HelpPage />
    </MemoryRouter>,
  );

  await user.type(
    screen.getByLabelText("Search help articles"),
    "extra payment",
  );
  const heading = screen.getByRole("heading", { name: "Add and update loans" });
  const details = heading.closest("details");
  expect(details).not.toHaveAttribute("open");
  expect(screen.queryByText("Read steps")).not.toBeInTheDocument();
  await user.click(heading);
  expect(details).toHaveAttribute("open");
  expect(
    within(details!).getByText(
      "Record an extra payment with its date and amount.",
    ),
  ).toBeVisible();
});

it("returns to the origin or Home when Help was opened directly", async () => {
  const user = userEvent.setup();
  const view = render(
    <MemoryRouter
      initialEntries={[{ pathname: "/help", state: { from: "/money" } }]}
    >
      <Routes>
        <Route path="/help" element={<HelpPage />} />
        <Route path="/money" element={<p>Money page returned</p>} />
        <Route path="/" element={<p>Home page returned</p>} />
      </Routes>
    </MemoryRouter>,
  );

  await user.click(screen.getByRole("link", { name: "Back" }));
  expect(await screen.findByText("Money page returned")).toBeVisible();
  view.unmount();

  render(
    <MemoryRouter initialEntries={["/help"]}>
      <Routes>
        <Route path="/help" element={<HelpPage />} />
        <Route path="/" element={<p>Home page returned</p>} />
      </Routes>
    </MemoryRouter>,
  );
  await user.click(screen.getByRole("link", { name: "Back to Home" }));
  expect(await screen.findByText("Home page returned")).toBeVisible();
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
