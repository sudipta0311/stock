import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { expect, it } from "vitest";
import { BottomNavigation } from "../../src/app/BottomNavigation";

it.each(["/import", "/import/job-test-1"])("highlights Money at %s", (path) => {
  render(
    <MemoryRouter initialEntries={[path]}>
      <BottomNavigation />
    </MemoryRouter>,
  );
  expect(screen.getByRole("link", { name: "Money" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});

it("shows the five primary tabs with accessible names", () => {
  render(
    <MemoryRouter initialEntries={["/"]}>
      <BottomNavigation />
    </MemoryRouter>,
  );
  const navigation = screen.getByRole("navigation", { name: "Primary" });
  expect(navigation).toBeInTheDocument();
  for (const label of ["Home", "Money", "Plan", "Ideas", "Inbox"]) {
    expect(screen.getByRole("link", { name: label })).toBeInTheDocument();
  }
  expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute(
    "aria-current",
    "page",
  );
});
