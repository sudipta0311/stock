import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Highlights } from "../../src/features/Highlights";

describe("policy highlights", () => {
  it("uses the copied instrument name when a rate is missing", () => {
    render(
      <Highlights
        items={[
          {
            key: "missing_yield",
            screen: "plan",
            title: "Check today's rate",
            body: "We don't have today's rate for this option. Check it with the bank or post office first.",
            learn_more_key: "missing_yield",
            severity: "info",
            context: { instrument_code: "po_td_5y" },
          },
        ]}
      />,
    );

    expect(
      screen.getByText(
        "We don't have today's rate for five-year post office time deposit. Check it with the bank or post office first.",
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/po_td_5y/)).not.toBeInTheDocument();
  });

  it("keeps the next step when no copied instrument name is available", () => {
    render(
      <Highlights
        items={[
          {
            key: "missing_yield",
            screen: "plan",
            title: "Check today's rate",
            body: "We don't have today's rate for this option. Check it with the bank or post office first.",
            learn_more_key: "missing_yield",
            severity: "info",
          },
        ]}
      />,
    );

    expect(
      screen.getByText(
        "We don't have today's rate for this option. Check it with the bank or post office first.",
      ),
    ).toBeInTheDocument();
  });
});
