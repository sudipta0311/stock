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
            body: "Today's rate for this option isn't known yet. Check it before you add money.",
            learn_more_key: "missing_yield",
            severity: "info",
            context: { instrument_code: "po_td_5y" },
          },
        ]}
      />,
    );

    expect(
      screen.getByText(
        "Today's five-year post office time deposit rate isn't known yet. Check it before you add money.",
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
            body: "Today's rate for this option isn't known yet. Check it before you add money.",
            learn_more_key: "missing_yield",
            severity: "info",
          },
        ]}
      />,
    );

    expect(
      screen.getByText(
        "Today's rate for this option isn't known yet. Check it before you add money.",
      ),
    ).toBeInTheDocument();
  });
});
