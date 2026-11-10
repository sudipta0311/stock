import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { createServer } from "node:net";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync, openSync } from "node:fs";
import { resolve, join } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(
  fileURLToPath(new URL("../../", import.meta.url)),
);
const webRoot = join(repositoryRoot, "apps", "web");
const reportDir = join(repositoryRoot, "out", "e2e");
const processes = new Set();
let databaseProcess;
let databaseSetupHandoff;

function pythonExecutable() {
  const localPython = join(repositoryRoot, ".venv", "Scripts", "python.exe");
  return (
    process.env.PYTHON ?? (existsSync(localPython) ? localPython : "python")
  );
}

function cleanEnvironment(source = process.env) {
  return Object.fromEntries(
    Object.entries(source).filter(
      ([name]) =>
        !/^(?:FOUNDRY_|EMAIL_|SMTP_|RESEND_|POSTMARK_|MAILGUN_|SENDGRID_|GOOGLE_|GMAIL_|VAPID_|MARKETDATA_|AWS_SES_)/i.test(
          name,
        ),
    ),
  );
}

function removeDatabaseEnvironment(environment) {
  for (const name of ["OWNER_DATABASE_URL", "DATABASE_URL"]) {
    delete environment[name];
  }
  return environment;
}

function removeE2eEnvironment(environment) {
  for (const name of [
    "E2E_BASE_URL",
    "E2E_API_URL",
    "E2E_DEV_AUTH_SECRET",
    "E2E_CAS_PATH",
    "E2E_CAS_PASSWORD",
    "E2E_SCHEMA_NAME",
    "E2E_SETUP_FILE_HANDOFF",
    "E2E_SETUP_OUTPUT_FILE",
    "E2E_SETUP_RELEASE_FILE",
    "E2E_EMAIL_LINK_SECRET",
    "STATEMENT_PASSWORD_KEY",
  ]) {
    delete environment[name];
  }
  return environment;
}

function apiEnvironment(
  database,
  webOrigin,
  emailLinkSecret,
  statementPasswordKey,
) {
  const environment = removeE2eEnvironment(
    removeDatabaseEnvironment(cleanEnvironment()),
  );
  delete environment.TEST_DATABASE_URL;
  return {
    ...environment,
    ENV: "test",
    AUTH_MODE: "dev",
    DEV_AUTH_SECRET: database.api_secret,
    EMAIL_LINK_SECRET: emailLinkSecret,
    STATEMENT_PASSWORD_KEY: statementPasswordKey,
    DATABASE_URL: database.api_database_url,
    ALLOWED_ORIGINS: webOrigin,
  };
}

function waitForJsonLine(child) {
  return new Promise((resolveLine, reject) => {
    let buffered = "";
    const onData = (chunk) => {
      buffered += chunk.toString("utf8");
      let newline = buffered.indexOf("\n");
      while (newline >= 0) {
        const line = buffered.slice(0, newline).trim();
        buffered = buffered.slice(newline + 1);
        newline = buffered.indexOf("\n");
        if (!line) continue;
        try {
          const value = JSON.parse(line);
          if (value.api_database_url && value.api_secret) {
            child.stdout.off("data", onData);
            resolveLine(value);
            return;
          }
        } catch {
          // Ignore non-protocol setup output; it is never forwarded to the test log.
        }
      }
    };
    child.stdout.on("data", onData);
    child.once("error", () =>
      reject(new Error("Database setup process failed.")),
    );
    child.once("exit", () =>
      reject(new Error("Database setup process stopped.")),
    );
  });
}

async function waitForSetupFile(child, outputPath, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error("Database setup process stopped.");
    }
    try {
      const value = JSON.parse(await readFile(outputPath, "utf8"));
      if (value.api_database_url && value.api_secret) {
        await rm(outputPath, { force: true });
        await rm(`${outputPath}.pending`, { force: true });
        return value;
      }
    } catch (error) {
      if (error?.code !== "ENOENT" && !(error instanceof SyntaxError))
        throw error;
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 100));
  }
  throw new Error("Database setup process did not become ready in time.");
}

function spawnQuiet(command, args, options) {
  const child = spawn(command, args, {
    ...options,
    stdio: options.stdio ?? "ignore",
    windowsHide: true,
  });
  processes.add(child);
  child.once("exit", () => processes.delete(child));
  return child;
}

async function freePort() {
  const server = createServer();
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    server.close();
    throw new Error("Could not reserve a local port.");
  }
  const port = address.port;
  await new Promise((resolveClose, reject) =>
    server.close((error) => (error ? reject(error) : resolveClose())),
  );
  return port;
}

async function waitForHttp(url, child, timeoutMs = 90_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error("A local e2e service stopped during startup.");
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(2_000) });
      if (response.ok) return;
    } catch {
      // The service is still starting.
    }
    await new Promise((resolveDelay) => setTimeout(resolveDelay, 300));
  }
  throw new Error("A local e2e service did not become ready in time.");
}

function waitForExit(child, timeoutMs = 20_000) {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolveExit) => {
    const timeout = setTimeout(() => {
      child.kill();
      resolveExit();
    }, timeoutMs);
    timeout.unref();
    child.once("exit", () => {
      clearTimeout(timeout);
      resolveExit();
    });
  });
}

async function stopStack() {
  for (const child of [...processes].reverse()) {
    if (child === databaseProcess) continue;
    if (child.exitCode === null && child.signalCode === null) child.kill();
    await waitForExit(child);
  }
  if (databaseProcess && databaseProcess.exitCode === null) {
    if (databaseSetupHandoff) {
      await writeFile(databaseSetupHandoff.releasePath, "release\n", {
        flag: "w",
      });
    } else {
      databaseProcess.stdin.end();
    }
    await waitForExit(databaseProcess, 35_000);
  }
  if (databaseSetupHandoff) {
    await rm(databaseSetupHandoff.outputPath, { force: true });
    await rm(`${databaseSetupHandoff.outputPath}.pending`, { force: true });
    await rm(databaseSetupHandoff.releasePath, { force: true });
    databaseSetupHandoff = undefined;
  }
}

async function buildWeb(apiOrigin) {
  const environment = removeE2eEnvironment(
    removeDatabaseEnvironment(cleanEnvironment()),
  );
  delete environment.TEST_DATABASE_URL;
  Object.assign(environment, {
    VITE_API_URL: apiOrigin,
    VITE_AUTH_MODE: "dev",
    VITE_E2E_DEV_AUTH: "true",
    PYTHONDONTWRITEBYTECODE: "1",
  });
  // Standard (minified) build unless a restricted sandbox sets SANDBOX_BUILD itself;
  // the load-speed test must measure what users download.
  const build = spawnQuiet(
    process.execPath,
    [join(webRoot, "scripts", "build.mjs")],
    { cwd: webRoot, env: environment },
  );
  const [code] = await new Promise((resolveExit, reject) => {
    build.once("error", reject);
    build.once("exit", (exitCode, signal) => resolveExit([exitCode, signal]));
  });
  if (code !== 0) throw new Error("The e2e PWA build failed.");
}

export default async function globalSetup() {
  process.env.PLAYWRIGHT_BROWSERS_PATH ??= join(
    repositoryRoot,
    ".playwright-browsers",
  );
  await mkdir(reportDir, { recursive: true });

  const databaseEnvironment = removeE2eEnvironment(
    removeDatabaseEnvironment(cleanEnvironment()),
  );
  databaseEnvironment.PYTHONDONTWRITEBYTECODE = "1";
  const useFileHandoff = process.env.E2E_SETUP_FILE_HANDOFF === "1";
  if (useFileHandoff) {
    const runId = `${process.pid}-${Date.now()}`;
    databaseSetupHandoff = {
      outputPath: join(reportDir, `database-setup-${runId}.json`),
      releasePath: join(reportDir, `database-release-${runId}.signal`),
    };
    databaseEnvironment.E2E_SETUP_OUTPUT_FILE = databaseSetupHandoff.outputPath;
    databaseEnvironment.E2E_SETUP_RELEASE_FILE =
      databaseSetupHandoff.releasePath;
  }
  databaseProcess = spawnQuiet(
    pythonExecutable(),
    ["-m", "scripts.e2e_database", "setup"],
    useFileHandoff
      ? { cwd: repositoryRoot, env: databaseEnvironment }
      : {
          cwd: repositoryRoot,
          env: databaseEnvironment,
          stdio: ["pipe", "pipe", "ignore"],
        },
  );

  try {
    const database = useFileHandoff
      ? await waitForSetupFile(databaseProcess, databaseSetupHandoff.outputPath)
      : await waitForJsonLine(databaseProcess);
    const apiPort = await freePort();
    const webPort = await freePort();
    const apiOrigin = `http://127.0.0.1:${apiPort}`;
    const webOrigin = `http://127.0.0.1:${webPort}`;
    const emailLinkSecret = randomBytes(32).toString("base64url");
    const statementPasswordKey = randomBytes(32).toString("base64");
    const apiEnv = apiEnvironment(
      database,
      webOrigin,
      emailLinkSecret,
      statementPasswordKey,
    );

    process.env.E2E_BASE_URL = webOrigin;
    process.env.E2E_API_URL = apiOrigin;
    process.env.E2E_DEV_AUTH_SECRET = database.api_secret;
    process.env.E2E_CAS_PATH = database.cas_path;
    process.env.E2E_CAS_PASSWORD = database.cas_password;
    process.env.E2E_SCHEMA_NAME = database.schema_name;
    process.env.E2E_EMAIL_LINK_SECRET = emailLinkSecret;

    const api = spawnQuiet(
      pythonExecutable(),
      [
        "-m",
        "uvicorn",
        "wealth.api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        String(apiPort),
        "--no-access-log",
      ],
      {
        cwd: repositoryRoot,
        env: apiEnv,
        ...(process.env.E2E_API_LOG
          ? {
              // Capture the local API output when the reviewer is diagnosing a failure.
              stdio: [
                "ignore",
                openSync(process.env.E2E_API_LOG, "a"),
                openSync(process.env.E2E_API_LOG, "a"),
              ],
            }
          : {}),
      },
    );
    await waitForHttp(`${apiOrigin}/healthz`, api);

    await buildWeb(apiOrigin);
    const previewEnvironment = removeE2eEnvironment(
      removeDatabaseEnvironment(cleanEnvironment()),
    );
    delete previewEnvironment.TEST_DATABASE_URL;
    const preview = spawnQuiet(
      process.execPath,
      [
        join(webRoot, "node_modules", "vite", "bin", "vite.js"),
        "preview",
        "--configLoader",
        "native",
        "--host",
        "127.0.0.1",
        "--port",
        String(webPort),
        "--strictPort",
      ],
      { cwd: webRoot, env: previewEnvironment },
    );
    await waitForHttp(webOrigin, preview);
    return stopStack;
  } catch (error) {
    await stopStack();
    // Startup errors are this file's own messages (no URLs or secrets); name the failed step.
    const step = error instanceof Error ? ` (${error.message})` : "";
    throw new Error(
      `The local e2e stack could not start${step}. Check TEST_DATABASE_URL and the installed Python and Node dependencies.`,
    );
  }
}
