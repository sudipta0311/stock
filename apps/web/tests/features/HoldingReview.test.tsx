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

  it("explains unchecked shares and labels the latest price separately", async () => {
    const user = userEvent.setup();
    render(
      <OurView
        value={{
          verdict: "not_checked",
          checked_on: null,
          reasons: [],
          checks: [
            { key: "latest_price", status: "ok" },
            { key: "position_size", status: "ok" },
          ],
        }}
      />,
    );

    expect(
      screen.getByText(
        "We don't follow this company's results yet, so we only check its price and size.",
      ),
    ).toBeVisible();
    expect(screen.queryByText("not_checked")).not.toBeInTheDocument();
    await user.click(screen.getByText("Why this suggestion?"));
    expect(screen.getByText("Latest price", { exact: true })).toBeVisible();
    expect(
      screen.queryByText("Price level", { exact: true }),
    ).not.toBeInTheDocument();
  });

  it("shows one reason and keeps remaining reasons and checks closed", async () => {
    const user = userEvent.setup();
    render(
      <OurView
        value={{
          verdict: "review",
          checked_on: "2026-11-07",
          reasons: [
            { code: "revenue_down_quarters", params: { count: 1 } },
            { code: "valuation_stretch", params: {} },
          ],
          tax: { note_code: "ask_same_tax" },
          checks: [
            { key: "company_results", status: "watch" },
            { key: "price_history", status: "ok" },
          ],
        }}
      />,
    );

    expect(
      screen.getByText(
        "Company revenue fell last quarter compared with a year earlier.",
      ),
    ).toBeVisible();
    expect(
      screen.getByText("The price looks high against its own history."),
    ).not.toBeVisible();
    expect(screen.getByText("Checked on 7 Nov 2026")).toBeVisible();
    expect(
      screen.getByText(
        "In your share savings account, tax is the same whether you sell or keep.",
      ),
    ).toBeVisible();
    const companyResultsCheck = screen
      .getAllByRole("listitem", { hidden: true })
      .find((row) => row.textContent?.includes("Company results"));
    expect(companyResultsCheck).toBeDefined();
    expect(companyResultsCheck?.textContent).toContain(
      "Company results Worth a look",
    );
    expect(companyResultsCheck).not.toBeVisible();

    await user.click(screen.getByText("Why this suggestion?"));
    expect(
      screen.getByText("The price looks high against its own history."),
    ).toBeVisible();
    expect(
      screen.getByRole("heading", { name: "How we check", level: 3 }),
    ).toBeVisible();
    expect(companyResultsCheck).toBeVisible();
  });

  it("summarizes known checks and waits for the missing data", async () => {
    const user = userEvent.setup();
    render(
      <OurView
        value={{
          verdict: "keep",
          reasons: [],
          checks: [
            { key: "company_results", status: "unknown" },
            { key: "company_events", status: "ok" },
            { key: "price_history", status: "unknown" },
            { key: "position_size", status: "ok" },
          ],
        }}
      />,
    );

    expect(
      screen.getByText("No warning signs in 2 of 4 checks."),
    ).toBeVisible();
    await user.click(screen.getByText("Why this suggestion?"));

    expect(
      screen.getByText("Waiting for data: Company results, Price level."),
    ).toBeVisible();
    const listItems = screen.getAllByRole("listitem");
    expect(listItems).toHaveLength(2);
    expect(listItems[0]).toHaveTextContent("Company events Looks okay");
    expect(listItems[1]).toHaveTextContent(
      "Share of your investments Looks okay",
    );
  });

  it("does not summarize when a known check is a watch without a reason", () => {
    render(
      <OurView
        value={{
          verdict: "keep",
          reasons: [],
          checks: [{ key: "company_strength", status: "watch" }],
        }}
      />,
    );

    expect(screen.queryByText(/^No warning signs/)).not.toBeInTheDocument();
  });

  it("uses the all checked summary and omits the waiting line", async () => {
    const user = userEvent.setup();
    render(
      <OurView
        value={{
          verdict: "keep",
          reasons: [],
          checks: [
            { key: "price_history", status: "ok" },
            { key: "company_strength", status: "ok" },
          ],
        }}
      />,
    );

    expect(
      screen.getByText("No warning signs in any of the 2 checks."),
    ).toBeVisible();
    await user.click(screen.getByText("Why this suggestion?"));

    expect(screen.queryByText(/^Waiting for data:/)).not.toBeInTheDocument();
    expect(screen.getByText("Price level", { exact: true })).toBeVisible();
    expect(screen.getByText("Company strength", { exact: true })).toBeVisible();
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
