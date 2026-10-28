import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ThemeProvider, useTheme } from "../../src/ui/theme";

function ThemeToggle() {
  const { resolved, toggleTheme } = useTheme();
  return (
    <button type="button" onClick={toggleTheme}>
      Current theme: {resolved}
    </button>
  );
}

it("persists a manual theme choice", () => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Current theme: light" }));
  expect(document.documentElement).toHaveAttribute("data-theme", "dark");
  expect(window.localStorage.getItem("wealth.theme")).toBe("dark");
});

it("still switches themes when browser storage throws", () => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("Storage unavailable");
  });
  render(
    <ThemeProvider>
      <ThemeToggle />
    </ThemeProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Current theme: light" }));
  expect(document.documentElement).toHaveAttribute("data-theme", "dark");
});
