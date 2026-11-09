import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";
import { getLoginEnvironment } from "./environment.mjs";

function installedChromePath(override) {
  if (override) {
    try {
      if (statSync(override).isFile()) return override;
    } catch {
      // Keep the missing-browser message free of environment values.
    }
    return null;
  }

  const roots = [
    process.env.ProgramFiles,
    process.env["ProgramFiles(x86)"],
    process.env.LOCALAPPDATA,
  ].filter(Boolean);
  const candidates = roots.map((root) =>
    join(root, "Google", "Chrome", "Application", "chrome.exe"),
  );
  return (
    candidates.find((candidate) => {
      try {
        return existsSync(candidate) && statSync(candidate).isFile();
      } catch {
        return false;
      }
    }) ?? null
  );
}

let environment;
try {
  environment = getLoginEnvironment();
} catch (error) {
  console.error(
    error instanceof Error ? error.message : "Live login is not configured.",
  );
  process.exit(1);
}

const chromePath = installedChromePath(environment.chromePath);
if (!chromePath) {
  console.error(
    "Google Chrome was not found. Set LIVE_CHROME_PATH to its executable.",
  );
  process.exit(1);
}

console.log("Sign in, wait for Home, then close the window.");
const browserProcess = spawn(
  chromePath,
  [
    `--user-data-dir=${environment.profileDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    "--window-size=430,900",
    `${environment.pwaUrl}/login`,
  ],
  { stdio: "ignore" },
);

await new Promise((resolveProcess, rejectProcess) => {
  browserProcess.once("error", () =>
    rejectProcess(new Error("Could not start Google Chrome.")),
  );
  browserProcess.once("exit", (code) => {
    if (code === 0 || code === null) resolveProcess();
    else rejectProcess(new Error("Google Chrome closed with an error."));
  });
}).catch((error) => {
  console.error(
    error instanceof Error ? error.message : "Could not start Google Chrome.",
  );
  process.exitCode = 1;
});
