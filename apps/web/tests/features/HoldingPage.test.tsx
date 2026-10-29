import { screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import englishCopy from "../../src/copy/en.json";
import { HoldingPage } from "../../src/money/HoldingPage";
import { networth, weeklyPlan } from "./fixtures";
import { renderWithQueries } from "./render";
import { server } from "../msw";

describe("holding Exit Card", () => {
  it("shows the holding, trigger rows, tax estimate, and plain-language action", async () => {
    server.use(
      http.get(/\/v1\/networth$/, () =>
        HttpResponse.json(
          networth({
            holdings: [
              {
                security_id: "security-test",
                name: "Demo Company",
                symbol: "DEMO",
                security_type: "equity",
                quantity: "10.000",
                value: "20000.00",
                value_usd: null,
                currency: "INR",
                fx_as_of: null,
                fx_missing: false,
                price_as_of: "2026-10-28",
                price_source: "fixture",
                status: "priced",
              },
            ],
          }),
        ),
      ),
      http.get(/\/v1\/plan$/, () =>
        HttpResponse.json(
          weeklyPlan({
            exit_cards: [
              {
                security_id: "security-test",
                symbol: "DEMO",
                action: "TRIM_TO_TARGET",
                urgency: "MEDIUM",
                triggers: {
                  thesis_break: {
                    key: "thesis_break",
                    status: "ok",
                    value: "On track",
                    threshold: "Stable",
                  },
                  valuation_stretch: {
                    key: "valuation_stretch",
                    status: "watch",
                    value: "81st percentile",
                    threshold: "Trim above 90",
                  },
                },
                quantity: "10.000",
                lots_to_sell: [],
                tax_estimate: {
                  estimated_tax: "250.00",
                  long_term_date: "2027-01-01",
                  notes: [
                    "acquisition_fx_missing",
                    "acquisition_fx_approximation",
                  ],
                },
                reason_codes: ["valuation_stretch"],
                params: {},
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
    expect(screen.getByText(/DEMO/).parentElement).toHaveTextContent(
      "10.000 shares",
    );
    expect(screen.getByText("Exit Card")).toBeInTheDocument();
    const thesisLabel = englishCopy.labels.trigger_keys.thesis_break;
    expect(screen.getByText(thesisLabel)).toBeInTheDocument();
    expect(screen.getByText(thesisLabel).parentElement).toHaveTextContent(
      englishCopy.labels.trigger_states.ok,
    );
    expect(
      screen.getByText(englishCopy.labels.trigger_states.watch),
    ).toBeInTheDocument();
    expect(
      screen.getByText(englishCopy.labels.exit_actions.TRIM_TO_TARGET),
    ).toBeInTheDocument();
    expect(screen.getByText(/Estimated tax if sold/)).toHaveTextContent("₹250");
    expect(
      screen.getByText(englishCopy.imports.tax_fx_missing),
    ).toBeInTheDocument();
    expect(
      screen.getByText(englishCopy.imports.tax_fx_approximation),
    ).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);
  });
});
