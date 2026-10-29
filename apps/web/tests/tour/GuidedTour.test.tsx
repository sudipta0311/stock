import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, expect, it } from "vitest";
import { GuidedTour, tourSeenKey } from "../../src/tour/GuidedTour";

function TourScreens() {
  return (
    <>
      <GuidedTour userId="synthetic-user" />
      <Routes>
        <Route
          path="/"
          element={
            <main>
              Home screen <Link to="/help">Open help</Link>
            </main>
          }
        />
        <Route
          path="/money"
          element={
            <main>
              Money screen <Link to="/help">Open help</Link>
            </main>
          }
        />
        <Route path="/plan" element={<main>Plan screen</main>} />
        <Route path="/ideas" element={<main>Ideas screen</main>} />
        <Route
          path="/help"
          element={
            <main>
              Help screen
              <button
                type="button"
                onClick={() =>
                  window.dispatchEvent(new Event("wealth:show-tour"))
                }
              >
                Show the tour again
              </button>
            </main>
          }
        />
      </Routes>
    </>
  );
}

beforeEach(() => window.localStorage.clear());

it("shows the four screens, skips, and replays the tour from Help", async () => {
  const user = userEvent.setup();
  render(
    <MemoryRouter initialEntries={["/"]}>
      <TourScreens />
    </MemoryRouter>,
  );

  const dialog = await screen.findByRole("dialog");
  expect(dialog).toHaveClass("motion-reduce:transition-none");
  expect(
    screen.getByText("Home keeps your next steps together."),
  ).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("Money screen")).toBeInTheDocument();
  expect(
    screen.getByText("Add what you have, even if the numbers are rough."),
  ).toBeInTheDocument();

  await user.click(screen.getByRole("button", { name: "Skip" }));
  await waitFor(() =>
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
  );
  expect(window.localStorage.getItem(tourSeenKey("synthetic-user"))).toBe(
    "true",
  );

  await user.click(screen.getByRole("link", { name: "Open help" }));
  await user.click(screen.getByRole("button", { name: "Show the tour again" }));
  expect(await screen.findByRole("dialog")).toBeInTheDocument();
  expect(await screen.findByText("Home screen")).toBeInTheDocument();
});

it("does not automatically show again after it was seen", async () => {
  window.localStorage.setItem(tourSeenKey("synthetic-user"), "true");
  render(
    <MemoryRouter initialEntries={["/"]}>
      <TourScreens />
    </MemoryRouter>,
  );
  await waitFor(() =>
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
  );
});
