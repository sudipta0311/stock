import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, it } from "vitest";

it("generated API schema matches the current OpenAPI document hash", () => {
  const spec = readFileSync("../../schemas/openapi.json");
  const declaration = readFileSync("src/api/schema.d.ts", "utf8");
  const digest = createHash("sha256").update(spec).digest("hex");
  expect(declaration.startsWith(`// OpenAPI SHA256: ${digest}\n`)).toBe(true);
});
