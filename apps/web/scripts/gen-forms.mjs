import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const schemasDir = path.join(root, "schemas", "assets");
const outputPath = fileURLToPath(
  new URL("../src/money/forms.generated.ts", import.meta.url),
);
const institutionsModulePath = fileURLToPath(
  new URL("../src/money/institutions.generated.ts", import.meta.url),
);
const institutionsPath = path.join(root, "rules", "institutions.yaml");
const institutionsOutput = fileURLToPath(
  new URL("../public/institutions.json", import.meta.url),
);
const checkOnly = process.argv.includes("--check");

function resolveSchema(schema, rootSchema) {
  if (schema.$ref?.startsWith("#/$defs/")) {
    return rootSchema.$defs?.[schema.$ref.slice("#/$defs/".length)] ?? {};
  }
  return schema;
}

function branchesFor(schema, rootSchema) {
  const resolved = resolveSchema(schema, rootSchema);
  return resolved.anyOf
    ? resolved.anyOf
        .map((branch) => resolveSchema(branch, rootSchema))
        .filter((branch) => branch.type !== "null")
    : [resolved];
}

function fieldKind(schema, rootSchema, fieldName) {
  const resolved = resolveSchema(schema, rootSchema);
  const branches = branchesFor(schema, rootSchema);
  if (resolved.const !== undefined) return "hidden";
  const enumValues = branches.flatMap((branch) => branch.enum ?? []);
  if (enumValues.length) return "enum";
  if (branches.some((branch) => branch.type === "boolean")) return "boolean";
  if (branches.some((branch) => branch.type === "array")) return "list";
  if (
    branches.some(
      (branch) => branch.format === "decimal" && branch["x-unit"] === "INR",
    )
  )
    return "rupees";
  if (fieldName.endsWith("_pct")) return "rate_pct";
  if (branches.some((branch) => branch.format === "decimal")) return "amount";
  if (
    branches.some(
      (branch) =>
        typeof branch.pattern === "string" && /\\d\{4\}-/.test(branch.pattern),
    )
  )
    return "month_year";
  if (
    branches.some(
      (branch) => branch.type === "integer" || branch.type === "number",
    )
  )
    return "number";
  return "text";
}

function tsType(schema, rootSchema) {
  if (schema.$ref?.startsWith("#/$defs/")) {
    return tsType(resolveSchema(schema, rootSchema), rootSchema);
  }
  if (schema.const !== undefined) return JSON.stringify(schema.const);
  if (schema.enum)
    return [...new Set(schema.enum.map((value) => JSON.stringify(value)))].join(
      " | ",
    );
  if (schema.anyOf)
    return [
      ...new Set(schema.anyOf.map((branch) => tsType(branch, rootSchema))),
    ].join(" | ");
  if (schema.type === "array")
    return `${tsType(schema.items ?? {}, rootSchema)}[]`;
  if (schema.type === "object") return "Record<string, unknown>";
  if (schema.type === "integer" || schema.type === "number") return "number";
  if (schema.type === "boolean") return "boolean";
  if (schema.type === "null") return "null";
  return "string";
}

function formData(schema, assetType) {
  const required = new Set(schema.required ?? []);
  const fields = Object.entries(schema.properties ?? {})
    .map(([name, property]) => {
      const resolved = resolveSchema(property, schema);
      const enumValues = branchesFor(property, schema).flatMap(
        (branch) => branch.enum ?? [],
      );
      return {
        name,
        kind: fieldKind(property, schema, name),
        required: required.has(name),
        ...(resolved.default !== undefined
          ? { defaultValue: resolved.default }
          : {}),
        ...(enumValues.length ? { options: [...new Set(enumValues)] } : {}),
      };
    })
    .sort(
      (left, right) =>
        Number(right.required) - Number(left.required) ||
        left.name.localeCompare(right.name),
    );
  return { assetType, fields };
}

function parseInstitutions(source) {
  const institutions = [];
  let current;
  for (const line of source.split(/\r?\n/)) {
    const canonical = /^\s*-\s+canonical:\s*(.+?)\s*$/.exec(line);
    if (canonical) {
      current = { canonical: canonical[1], aliases: [] };
      institutions.push(current);
      continue;
    }
    const aliases = /^\s+aliases:\s*\[(.*)\]\s*$/.exec(line);
    if (aliases && current) {
      current.aliases = aliases[1]
        .split(",")
        .map((alias) => alias.trim().replace(/^['"]|['"]$/g, ""))
        .filter(Boolean);
    }
  }
  return institutions;
}

const filenames = (await readdir(schemasDir))
  .filter((name) => name.endsWith(".json") && name !== "any_asset.json")
  .sort();
const loaded = await Promise.all(
  filenames.map(async (filename) => ({
    filename,
    source: await readFile(path.join(schemasDir, filename), "utf8"),
  })),
);
const forms = loaded.map(({ filename, source }) => {
  const schema = JSON.parse(source);
  const assetType = schema.properties?.asset_type?.const;
  if (typeof assetType !== "string")
    throw new Error(`${filename} has no asset_type const.`);
  return { schema, source, filename, ...formData(schema, assetType) };
});
const digest = createHash("sha256")
  .update(
    loaded.map(({ filename, source }) => `${filename}\n${source}`).join("\n"),
  )
  .digest("hex");

const assetDraftTypes = forms
  .map(({ schema, assetType }) => {
    const required = new Set(schema.required ?? []);
    const properties = Object.entries(schema.properties ?? {})
      .map(
        ([name, property]) =>
          `  ${JSON.stringify(name)}${required.has(name) ? "" : "?"}: ${tsType(property, schema)};`,
      )
      .join("\n");
    return `  ${JSON.stringify(assetType)}: {\n${properties}\n  };`;
  })
  .join("\n");
const definitions = forms
  .map(
    ({ assetType, fields }) =>
      `  ${JSON.stringify(assetType)}: { assetType: ${JSON.stringify(assetType)}, fields: ${JSON.stringify(fields)} },`,
  )
  .join("\n");
const generated = `// Generated by scripts/gen-forms.mjs. Do not edit by hand.\nexport const assetSchemaDigest = ${JSON.stringify(digest)};\n\nexport interface AssetDrafts {\n${assetDraftTypes}\n}\n\nexport type AssetType = keyof AssetDrafts;\nexport type AssetDraft<T extends AssetType = AssetType> = T extends AssetType ? AssetDrafts[T] : never;\n\nexport type GeneratedFieldKind = "hidden" | "rupees" | "amount" | "rate_pct" | "month_year" | "enum" | "boolean" | "number" | "list" | "text";\nexport interface GeneratedAssetField { name: string; kind: GeneratedFieldKind; required: boolean; defaultValue?: unknown; options?: readonly (string | number)[]; }\nexport interface GeneratedAssetForm { assetType: AssetType; fields: readonly GeneratedAssetField[]; }\n\nexport const assetForms = {\n${definitions}\n} as const satisfies Record<AssetType, GeneratedAssetForm>;\n`;
const institutions = parseInstitutions(
  await readFile(institutionsPath, "utf8"),
);
const institutionsJson = `${JSON.stringify(institutions, null, 2)}\n`;
const institutionsModule = `// Generated by scripts/gen-forms.mjs. Do not edit by hand.\nexport const institutions = ${JSON.stringify(institutions, null, 2)} as const;\nexport const institutionNames = institutions.map((institution) => institution.canonical);\n`;

async function writeOrCheck(filePath, content) {
  if (checkOnly) {
    let existing = "";
    try {
      existing = await readFile(filePath, "utf8");
    } catch {
      throw new Error(
        `${path.relative(root, filePath)} is missing. Run npm run gen:forms.`,
      );
    }
    if (existing !== content)
      throw new Error(
        `${path.relative(root, filePath)} is stale. Run npm run gen:forms.`,
      );
  } else {
    await mkdir(path.dirname(filePath), { recursive: true });
    await writeFile(filePath, content, "utf8");
  }
}

await writeOrCheck(outputPath, generated);
await writeOrCheck(institutionsModulePath, institutionsModule);
await writeOrCheck(institutionsOutput, institutionsJson);
console.log(
  `${checkOnly ? "Verified" : "Generated"} ${forms.length} asset forms and ${institutions.length} institutions (${digest.slice(0, 12)}).`,
);
