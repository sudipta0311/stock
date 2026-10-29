import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import { AddAssetPage } from "../../src/money/AssetFormPage";
import {
  applyFieldRules,
  withoutHiddenValues,
} from "../../src/money/field-rules";
import { normalizeAmount } from "../../src/money/form-utils";
import { assetForms } from "../../src/money/forms.generated";
import { server } from "../msw";
import { renderWithQueries } from "../features/render";

const goldFields = assetForms.gold.fields;

function names(fields: readonly { name: string; required: boolean }[]) {
  return {
    shown: fields.map((field) => field.name),
    required: fields.filter((field) => field.required).map((f) => f.name),
  };
}

describe("fields that depend on the kind of item", () => {
  it("asks for weight and purity for physical gold and hides bond fields", () => {
    const { shown, required } = names(
      applyFieldRules("gold", goldFields, { kind: "physical" }),
    );
    expect(required).toEqual(
      expect.arrayContaining(["kind", "grams", "purity"]),
    );
    expect(shown).toContain("purpose");
    expect(shown).not.toContain("units");
    expect(shown).not.toContain("series");
    expect(shown).not.toContain("issue_price_per_gram");
  });

  it("shows only the common fields until a kind is chosen", () => {
    const { shown } = names(applyFieldRules("gold", goldFields, {}));
    for (const name of ["units", "grams", "purity", "series", "issue"]) {
      expect(shown).not.toContain(name);
    }
    expect(shown).toContain("member_id");
  });

  it("asks for the term only for a post office time deposit", () => {
    const fields = assetForms.po_scheme.fields;
    expect(
      names(applyFieldRules("po_scheme", fields, { scheme: "td" })).required,
    ).toContain("td_years");
    expect(
      names(applyFieldRules("po_scheme", fields, { scheme: "nsc" })).shown,
    ).not.toContain("td_years");
  });

  it("does not send values left over from another kind", () => {
    expect(
      withoutHiddenValues("gold", {
        kind: "physical",
        grams: "80",
        units: "10",
      }),
    ).toEqual({ kind: "physical", grams: "80", units: "" });
  });

  it("saves physical gold with its weight and purity", async () => {
    const user = userEvent.setup();
    let postBody: Record<string, unknown> | null = null;
    server.use(
      http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.post(/\/v1\/assets$/, async ({ request }) => {
        postBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          {
            id: "asset-gold-1",
            asset_type: "gold",
            member_id: null,
            account_id: null,
            joint: false,
            details: postBody,
            valuation: null,
            valuation_note: null,
            valuation_status: "valued",
            value_cached: "480000.00",
            value_as_of: "2026-10-29",
            status: "active",
            created_at: "2026-10-29T00:00:00Z",
            updated_at: "2026-10-29T00:00:00Z",
          },
          { status: 201 },
        );
      }),
    );
    renderWithQueries(
      <MemoryRouter initialEntries={["/money/add/gold"]}>
        <main>
          <Routes>
            <Route path="/money/add/:type" element={<AddAssetPage />} />
          </Routes>
        </main>
      </MemoryRouter>,
    );

    await user.selectOptions(
      await screen.findByLabelText(/Type of gold/),
      "physical",
    );
    expect(screen.queryByLabelText(/Units held/)).not.toBeInTheDocument();
    await user.type(screen.getByLabelText(/Weight in grams/), "80");
    const purity = screen.getByLabelText(/Purity/);
    await user.selectOptions(purity, "22");
    expect(purity).toBeRequired();
    expect(
      document.querySelector(`label[for="${purity.id}"]`),
    ).toHaveTextContent("Purity (karat)*");

    await waitFor(() => expect(postBody).not.toBeNull(), { timeout: 2500 });
    expect(postBody).toMatchObject({
      asset_type: "gold",
      kind: "physical",
      grams: 80,
      purity: 22,
    });
    expect(postBody).not.toHaveProperty("units");
  });
});

describe("loan amounts", () => {
  it("reads rupee shorthand and keeps kroner strict about dots", () => {
    expect(normalizeAmount("38L", "INR")).toBe("3800000.00");
    expect(normalizeAmount("34,500", undefined)).toBe("34500.00");
    expect(normalizeAmount("120,000", "DKK")).toBe("120000");
    expect(normalizeAmount("2600.50", "EUR")).toBe("2600.50");
    expect(() => normalizeAmount("120.000", "DKK")).toThrow();
    expect(() => normalizeAmount("38L", "DKK")).toThrow();
  });

  it("says why a loan didn't save and opens the section with the problem", async () => {
    const user = userEvent.setup();
    let posted = false;
    server.use(
      http.get(/\/v1\/plan$/, () => HttpResponse.json({ members: [] })),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.post(/\/v1\/assets$/, () => {
        posted = true;
        return HttpResponse.json({}, { status: 201 });
      }),
    );
    renderWithQueries(
      <MemoryRouter initialEntries={["/money/add/loan"]}>
        <main>
          <Routes>
            <Route path="/money/add/:type" element={<AddAssetPage />} />
          </Routes>
        </main>
      </MemoryRouter>,
    );

    await user.selectOptions(await screen.findByLabelText(/Loan type/), "home");
    await user.type(screen.getByLabelText(/Outstanding balance/), "38L");
    await user.type(screen.getByLabelText(/Interest rate \(%\)/), "8.6");
    await user.type(screen.getByLabelText(/Monthly payment/), "34,500");
    const original = screen.getByLabelText(/Original loan amount/);
    await user.type(original, "a lot");
    const details = original.closest("details");
    if (details) details.open = false;

    await user.click(screen.getByRole("button", { name: "Save now" }));

    expect(
      await screen.findByText("Please check the highlighted details."),
    ).toBeInTheDocument();
    expect(details?.open).toBe(true);
    expect(posted).toBe(false);
  });
});
