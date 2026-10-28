import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import yaml from "js-yaml";

const sourcePath = fileURLToPath(
  new URL("../../../copy/en.yaml", import.meta.url),
);
const outputPath = fileURLToPath(
  new URL("../src/copy/en.json", import.meta.url),
);
const source = await readFile(sourcePath, "utf8");
const document = yaml.load(source);

if (
  typeof document !== "object" ||
  document === null ||
  Array.isArray(document)
) {
  throw new TypeError("English copy must be a YAML mapping.");
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(document, null, 2)}\n`, "utf8");
console.log("Generated src/copy/en.json from copy/en.yaml.");
