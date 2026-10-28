import { createHash } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useForm } from "react-hook-form";
import { describe, expect, it } from "vitest";
import { MoneyInput } from "../../src/ui";
import {
  assetForms,
  assetSchemaDigest,
  type AssetType,
} from "../../src/money/forms.generated";
import { assetFieldCopy, commonFieldCopy } from "../../src/money/labels.en";

function MoneyInputHarness() {
  const form = useForm<{ amount: string }>({ defaultValues: { amount: "" } });
  return (
    <MoneyInput control={form.control} name="amount" label="Amount" required />
  );
}

describe("schema-generated money forms", () => {
  it("stays in sync with the checked-in asset schemas", () => {
    const schemaDirectory = path.resolve(process.cwd(), "../../schemas/assets");
    const sources = readdirSync(schemaDirectory)
      .filter((name) => name.endsWith(".json") && name !== "any_asset.json")
      .sort()
      .map(
        (name) =>
          `${name}\n${readFileSync(path.join(schemaDirectory, name), "utf8")}`,
      )
      .join("\n");
    expect(createHash("sha256").update(sources).digest("hex")).toBe(
      assetSchemaDigest,
    );

    const institutionSource = readFileSync(
      path.resolve(process.cwd(), "../../rules/institutions.yaml"),
      "utf8",
    );
    const canonicalNames = [
      ...institutionSource.matchAll(/^\s*-\s+canonical:\s*(.+?)\s*$/gm),
    ].map((match) => match[1]);
    const institutions = JSON.parse(
      readFileSync("public/institutions.json", "utf8"),
    ) as Array<{ canonical: string }>;
    expect(institutions.map((institution) => institution.canonical)).toEqual(
      canonicalNames,
    );
  });

  it("has plain-language labels for each required field", () => {
    for (const [assetType, definition] of Object.entries(assetForms) as Array<
      [AssetType, (typeof assetForms)[AssetType]]
    >) {
      for (const field of definition.fields) {
        if (!field.required || field.kind === "hidden") continue;
        const label =
          assetFieldCopy[assetType]?.[field.name]?.label ??
          commonFieldCopy[field.name]?.label;
        expect(label, `${assetType}.${field.name}`).toBeTruthy();
      }
    }
  });

  it("shows an exact parsed INR amount and rejects unrecognized text", async () => {
    const user = userEvent.setup();
    render(<MoneyInputHarness />);
    const amount = screen.getByLabelText(/Amount/);
    await user.type(amount, "5L");
    expect(screen.getByText("= ₹5,00,000")).toBeInTheDocument();
    await user.clear(amount);
    await user.type(amount, "junk");
    expect(
      screen.getByText("Enter a rupee amount such as 5 lakh or 50k."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/^= ₹/)).not.toBeInTheDocument();
  });
});
