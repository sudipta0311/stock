import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { expect, it } from "vitest";
import { BottomNavigation } from "../../src/app/BottomNavigation";

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
