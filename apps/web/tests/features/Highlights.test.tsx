import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { Highlights } from "../../src/features/Highlights";

describe("policy highlights", () => {
  it("shows the first two notes and opens the rest with See all", async () => {
    const user = userEvent.setup();
    render(
      <Highlights
        items={[1, 2, 3, 4].map((number) => ({
          key: `note-${number}`,
          screen: "home" as const,
          title: `Note ${number}`,
          body: `Body ${number}`,
          learn_more_key: "plan_details",
          severity: "info" as const,
        }))}
        onDismiss={() => undefined}
      />,
    );

    const notes = screen.getByRole("region", { name: "Helpful notes" });
    const heading = within(notes).getByRole("heading", {
      name: "Helpful notes",
    });
    expect(heading.parentElement).toBe(notes);
    expect(within(notes).getAllByRole("list")[0]?.parentElement).toBe(notes);
    expect(within(notes).getAllByText("Why?")[0]).toBeVisible();
    expect(
      within(notes).getByRole("button", { name: "Hide: Note 1" }),
    ).toBeVisible();
    expect(screen.getByText("Note 1")).toBeVisible();
    expect(screen.getByText("Note 2")).toBeVisible();
    expect(screen.getByText("Note 3")).not.toBeVisible();
    const seeAll = screen.getByText("See all notes (2)");
    expect(seeAll).toBeVisible();
    await user.click(seeAll);
    expect(screen.getByText("Note 3")).toBeVisible();
    expect(screen.getByText("Note 4")).toBeVisible();
  });

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
