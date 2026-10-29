import { copyFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const source = fileURLToPath(
  new URL("../../../tools/gmail-statements/Code.gs", import.meta.url),
);
const output = fileURLToPath(
  new URL("../public/gmail-statements/Code.gs", import.meta.url),
);

await mkdir(new URL("../public/gmail-statements/", import.meta.url), {
  recursive: true,
});
await copyFile(source, output);
console.log("Copied the Gmail helper to the static PWA assets.");
