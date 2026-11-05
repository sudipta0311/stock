import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { OurView } from "../../src/money/HoldingPage";

describe("holding review", () => {
  it.each([
    ["keep", "Keep"],
    ["add_more", "Add more"],
    ["sell_some", "Sell some"],
    ["review", "Think about selling"],
    ["sell", "Sell"],
  ])("shows the %s verdict in plain words", (verdict, label) => {
    render(<OurView value={{ verdict, reasons: [] }} />);

    expect(screen.getByRole("heading", { name: "Our view" })).toBeVisible();
    expect(screen.getByText(label, { exact: true })).toBeVisible();
  });

  it("explains unchecked shares without exposing codes", () => {
    render(
      <OurView
        value={{
          verdict: "not_checked",
          checked_on: null,
          reasons: [],
          checks: [],
        }}
      />,
    );

    expect(
      screen.getByText(
        "We don't follow this company's results yet, so we only check its price and size.",
      ),
    ).toBeVisible();
    expect(screen.queryByText("not_checked")).not.toBeInTheDocument();
  });

  it("keeps the check details closed until asked", async () => {
    const user = userEvent.setup();
    render(
      <OurView
        value={{
          verdict: "review",
          checked_on: "2026-11-07",
          reasons: [
            {
              code: "revenue_down_quarters",
              params: { count: 1 },
            },
          ],
          checks: [
            { key: "company_results", status: "watch" },
            { key: "price_history", status: "ok" },
          ],
        }}
      />,
    );

    expect(
      screen.getByText("Company results", { exact: true }),
    ).not.toBeVisible();
    expect(screen.getByText("Checked on 7 Nov 2026")).toBeVisible();
    await user.click(screen.getByText("How we check", { exact: true }));
    expect(screen.getByText("Company results", { exact: true })).toBeVisible();
    expect(
      screen.getByText(
        "Company revenue fell last quarter compared with a year earlier.",
      ),
    ).toBeVisible();
  });

  it("does not show a waiting date for a share savings account", () => {
    render(
      <OurView
        value={{
          verdict: "sell_some",
          reasons: [],
          tax: { note_code: "ask_same_tax", wait_until: "2027-03-12" },
        }}
      />,
    );

    expect(
      screen.getByText(
        "In your share savings account, tax is the same whether you sell or keep.",
      ),
    ).toBeVisible();
    expect(screen.queryByText(/12 Mar 2027/)).not.toBeInTheDocument();
  });
});
