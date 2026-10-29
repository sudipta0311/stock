import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MonthYearInput, Select, TileToggle } from "../../src/ui";

describe("required field markers", () => {
  it("marks required selects, month inputs, and option groups", () => {
    render(
      <>
        <Select id="currency" label="Currency" required>
          <option value="DKK">DKK</option>
        </Select>
        <MonthYearInput id="opened" label="Opened in" required />
        <TileToggle
          label="Savings type"
          required
          options={[{ value: "bank", label: "Bank account" }]}
          value="bank"
          onChange={() => undefined}
        />
      </>,
    );

    expect(screen.getByLabelText(/Currency/)).toBeRequired();
    expect(screen.getByLabelText(/Opened in/)).toBeRequired();
    expect(document.querySelector('label[for="currency"]')).toHaveTextContent(
      "Currency*",
    );
    expect(document.querySelector('label[for="opened"]')).toHaveTextContent(
      "Opened in*",
    );
    const group = screen.getByRole("group", {
      name: "Savings type (required)",
    });
    expect(group).toHaveTextContent("Savings type*");
  });
});
