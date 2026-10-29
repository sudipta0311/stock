import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const sourcePath = fileURLToPath(
  new URL("../../../copy/en.yaml", import.meta.url),
);
const bannedTermsSourcePath = fileURLToPath(
  new URL("../../../copy/banned_terms.yaml", import.meta.url),
);
const helpSourcePath = fileURLToPath(
  new URL("../../../copy/help.en.yaml", import.meta.url),
);
const outputPath = fileURLToPath(
  new URL("../src/copy/en.json", import.meta.url),
);
const bannedTermsOutputPath = fileURLToPath(
  new URL("../src/copy/banned-terms.json", import.meta.url),
);
const helpOutputPath = fileURLToPath(
  new URL("../src/copy/help.en.json", import.meta.url),
);
const source = await readFile(sourcePath, "utf8");
const document = yaml.load(source);
const bannedTermsSource = await readFile(bannedTermsSourcePath, "utf8");
const bannedTerms = yaml.load(bannedTermsSource);
const helpSource = await readFile(helpSourcePath, "utf8");
const helpDocument = yaml.load(helpSource);

if (
  typeof document !== "object" ||
  document === null ||
  Array.isArray(document)
) {
  throw new TypeError("English copy must be a YAML mapping.");
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(document, null, 2)}\n`, "utf8");
if (
  typeof bannedTerms !== "object" ||
  bannedTerms === null ||
  Array.isArray(bannedTerms) ||
  !Array.isArray(bannedTerms.terms)
) {
  throw new TypeError("Banned terms must contain a terms list.");
}
await writeFile(
  bannedTermsOutputPath,
  `${JSON.stringify(bannedTerms, null, 2)}\n`,
  "utf8",
);
if (
  typeof helpDocument !== "object" ||
  helpDocument === null ||
  Array.isArray(helpDocument)
) {
  throw new TypeError("Help copy must be a YAML mapping.");
}
await writeFile(
  helpOutputPath,
  `${JSON.stringify(helpDocument, null, 2)}\n`,
  "utf8",
);
console.log("Generated src/copy/en.json from copy/en.yaml.");
console.log(
  "Generated src/copy/banned-terms.json from copy/banned_terms.yaml.",
);
console.log("Generated src/copy/help.en.json from copy/help.en.yaml.");
