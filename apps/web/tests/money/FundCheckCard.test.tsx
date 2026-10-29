import { fireEvent, render, screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";
import { axe } from "vitest-axe";
import { FundCheckCard } from "../../src/money/FundCheckCard";
import { server } from "../msw";

function fundCheckResponse(overrides: Record<string, unknown> = {}) {
  return {
    status: "fine",
    summary_key: "fund_check.good",
    checks: [],
    early_warning: null,
    details: {
      funds: [
        {
          security_id: "synthetic-security",
          name: "Synthetic Large Cap Fund",
          sub_category: "Large Cap",
          plan_type: "direct",
          option: "growth",
          return_3y_pct: "12.50",
          median_3y_pct: "11.25",
          ter_pct: "0.50",
        },
      ],
      notes: [],
    },
    ...overrides,
  };
}

function renderCard() {
  return render(
    <MemoryRouter>
      <FundCheckCard />
    </MemoryRouter>,
  );
}

describe("Indian fund status card", () => {
  it("shows the good state, plain details, and accessible controls", async () => {
    server.use(
      http.get(/\/v1\/fund-check$/, () =>
        HttpResponse.json(fundCheckResponse()),
      ),
    );
    const { container } = renderCard();

    expect(
      await screen.findByText("Your Indian funds look good ✓"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "See why" }));
    fireEvent.click(screen.getByText("Synthetic Large Cap Fund"));
    expect(screen.getByText("Large companies")).toBeInTheDocument();
    expect(screen.getByText("Direct plan ✓")).toBeInTheDocument();
    expect((await axe(container)).violations).toEqual([]);

    const visibleCopy = container.textContent ?? "";
    const bannedTerms = [
      "sleeve",
      "satellite",
      "tranche",
      "STP",
      "route",
      "PFIC",
      "ABIS",
      "PRIIPs",
      "LRS",
      "TCS",
      "XIRR",
      "RNOR",
      "rebalance",
      "allocation",
      "lot",
      "drift",
      "ISIN",
      "UCITS",
    ];
    for (const term of bannedTerms) {
      expect(visibleCopy).not.toMatch(new RegExp(`\\b${term}\\b`, "i"));
    }
  });

  it("shows an early warning with one next step", async () => {
    server.use(
      http.get(/\/v1\/fund-check$/, () =>
        HttpResponse.json(
          fundCheckResponse({
            status: "attention",
            summary_key: "fund_check.attention",
            early_warning: {
              family: {
                bucket: "small",
                months_until: 5,
                direction: "over",
              },
            },
            checks: [
              {
                reason_code: "regular_plan",
                member_id: "synthetic-member",
                details: { yearly_saving_inr: "4200.00" },
              },
            ],
            details: {
              funds: [
                {
                  security_id: "synthetic-security",
                  name: "Synthetic Large Cap Fund",
                  sub_category: "Large Cap",
                  plan_type: "direct",
                  option: "growth",
                },
              ],
              notes: ["ter_unavailable", "tax_estimate_unavailable"],
            },
          }),
        ),
      ),
    );
    renderCard();

    expect(await screen.findByText("One change suggested")).toBeInTheDocument();
    expect(
      screen.getByText(
        /share of small companies may pass its limit in about 5 months/,
      ),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "See why" }));
    expect(screen.getByText(/Possible yearly saving/)).toBeInTheDocument();
    expect(
      screen.getByText("A current fund cost is not available."),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(
        "Tax details are incomplete, so no switch cost is shown.",
      ),
    ).not.toBeInTheDocument();
  });

  it("offers the upload action when no fund details are available", async () => {
    server.use(
      http.get(/\/v1\/fund-check$/, () =>
        HttpResponse.json(
          fundCheckResponse({
            summary_key: "fund_check.upload",
            details: { funds: [], notes: [] },
          }),
        ),
      ),
    );
    renderCard();

    expect(
      await screen.findByText("Upload a detailed statement to see more."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "Upload CAMS statement" }),
    ).toHaveAttribute("href", "/import");
  });
});
