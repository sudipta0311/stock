import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));

function run(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: root,
      stdio: "inherit",
      shell: false,
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (code === 0) resolve();
      else reject(new Error(`Build step failed (${signal ?? code}).`));
    });
  });
}

await run(["scripts/generate-icons.mjs"]);
await run(["scripts/gen-forms.mjs"]);
await run(["scripts/generate-api.mjs"]);
await run(["scripts/generate-copy.mjs"]);
await run([
  "node_modules/vite/bin/vite.js",
  "build",
  "--configLoader",
  "native",
]);

if (process.env.SANDBOX_BUILD === "1") {
  await run(["scripts/minify-assets.mjs"]);
}
