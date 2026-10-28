import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString } from "openapi-typescript";

const openApiPath = fileURLToPath(
  new URL("../../../schemas/openapi.json", import.meta.url),
);
const outputPath = fileURLToPath(
  new URL("../src/api/schema.d.ts", import.meta.url),
);
const featureOutputPath = fileURLToPath(
  new URL("../src/api/features.generated.ts", import.meta.url),
);
const openApiSource = await readFile(openApiPath, "utf8");
const openApiDocument = JSON.parse(openApiSource);
const schema = await openapiTS(openApiDocument);
const digest = createHash("sha256").update(openApiSource).digest("hex");

await writeFile(
  outputPath,
  `// OpenAPI SHA256: ${digest}\n${astToString(schema)}`,
  "utf8",
);
const paths = Object.keys(openApiDocument.paths ?? {}).sort();
await writeFile(
  featureOutputPath,
  `// Generated from schemas/openapi.json. Do not edit by hand.\nexport const API_PATHS = ${JSON.stringify(paths, null, 2)} as const;\n`,
  "utf8",
);
console.log(
  `Generated src/api/schema.d.ts and features.generated.ts from schemas/openapi.json (${digest.slice(0, 12)}).`,
);
