import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { HoldingPage } from "../../src/money/HoldingPage";
import { networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

describe("holding review", () => {
  it("shows a plain review with details closed by default", async () => {
    const user = userEvent.setup();
    const networthQueries: string[] = [];
    server.use(
      http.get(/\/v1\/networth$/, ({ request }) => {
        networthQueries.push(new URL(request.url).search);
        return HttpResponse.json(
          networth({
            holdings: [
              {
                added_by_you: false,
                security_id: "security-test",
                name: "Demo Company",
                symbol: "DEMO",
                security_type: "equity",
                quantity: "0.423700",
                value: "20000.00",
                value_usd: null,
                currency: "INR",
                fx_as_of: null,
                fx_missing: false,
                price_as_of: "2026-10-28",
                price_source: "fixture",
                units_entered: false,
                status: "priced",
              },
            ],
          }),
        );
      }),
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(
          weeklyPlan({
            holding_reviews: [
              {
                security_id: "security-test",
                member_id: "member-test-1",
                verdict: "sell_some",
                reasons: [{ code: "valuation_stretch", params: {} }],
                tax: { estimated_tax: "250.00" },
                checked_on: "2026-10-28",
                add_more_room_inr: null,
                checks: [
                  { key: "company_results", status: "ok" },
                  { key: "company_strength", status: "watch" },
                ],
              },
            ],
          }),
        ),
      ),
    );
    const { container } = renderWithQueries(
      <MemoryRouter initialEntries={["/money/holding/security-test"]}>
        <Routes>
          <Route path="/money/holding/:id" element={<HoldingPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(
      await screen.findByRole("heading", { name: "Demo Company" }),
    ).toBeInTheDocument();
    expect(networthQueries.every((query) => query === "")).toBe(true);
    expect(screen.getByText("0.4237 shares")).toBeVisible();
    expect(screen.queryByText("DEMO")).not.toBeVisible();
    expect(screen.queryByText("0.4237")).not.toBeVisible();

    const details = screen.getByText("Details").closest("details");
    expect(details).not.toHaveAttribute("open");
    await user.click(screen.getByText("Details"));
    expect(screen.getByText("DEMO")).toBeVisible();
    expect(screen.getByText("0.4237")).toBeVisible();
    expect(screen.getByRole("heading", { name: "Our view" })).toBeVisible();
    expect(screen.getByText("Sell some", { exact: true })).toBeVisible();
    expect(
      screen.getByText("The price looks high against its own history."),
    ).toBeVisible();
    expect(screen.getByText("Checked on 28 Oct 2026")).toBeVisible();
    expect(screen.getByText(/Estimated tax if sold/)).toHaveTextContent("250");
    expect(
      screen.getByText("Company results", { exact: true }),
    ).not.toBeVisible();
    await user.click(screen.getByText("How we check", { exact: true }));
    expect(screen.getByText("Company results", { exact: true })).toBeVisible();
    expect(
      screen.queryByText(/percentile|threshold|TRIM_TO_TARGET/i),
    ).not.toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });
});
