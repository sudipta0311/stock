import { existsSync, realpathSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const DEFAULT_LIVE_PWA_URL = "https://wealth-pwa.onrender.com";
export const DEFAULT_LIVE_API_URL = "https://wealth-api-84le.onrender.com";

const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../..",
);

function resolveExistingPath(candidate) {
  let current = resolve(candidate);
  const suffix = [];
  while (!existsSync(current)) {
    const parent = dirname(current);
    if (parent === current) break;
    suffix.unshift(current.slice(parent.length + 1));
    current = parent;
  }
  const realParent = existsSync(current)
    ? realpathSync.native(current)
    : current;
  return resolve(realParent, ...suffix);
}

export function isPathInsideRepository(candidate, root = repositoryRoot) {
  const realRoot = resolveExistingPath(root);
  const realCandidate = resolveExistingPath(candidate);
  const pathFromRoot = relative(realRoot, realCandidate);
  return (
    pathFromRoot === "" ||
    (pathFromRoot !== ".." &&
      !pathFromRoot.startsWith(`..${sep}`) &&
      !isAbsolute(pathFromRoot))
  );
}

function requiredPath(name, value, root) {
  if (!value?.trim()) throw new Error(`${name} is required.`);
  if (isPathInsideRepository(value, root)) {
    throw new Error(`${name} must resolve outside the repository.`);
  }
  return resolve(value);
}

function httpUrl(name, value) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error(`${name} must be an absolute HTTP(S) URL.`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`${name} must be an absolute HTTP(S) URL.`);
  }
  return parsed.toString().replace(/\/$/, "");
}

export function getLiveEnvironment(env = process.env, root = repositoryRoot) {
  const profileDir = requiredPath(
    "LIVE_PROFILE_DIR",
    env.LIVE_PROFILE_DIR,
    root,
  );
  const outDir = requiredPath("LIVE_OUT_DIR", env.LIVE_OUT_DIR, root);
  return {
    profileDir,
    outDir,
    pwaUrl: httpUrl("LIVE_PWA_URL", env.LIVE_PWA_URL || DEFAULT_LIVE_PWA_URL),
    apiUrl: httpUrl("LIVE_API_URL", env.LIVE_API_URL || DEFAULT_LIVE_API_URL),
    chromePath: env.LIVE_CHROME_PATH?.trim() || undefined,
    headed: env.LIVE_HEADED === "1",
  };
}

export function getLoginEnvironment(env = process.env, root = repositoryRoot) {
  return {
    profileDir: requiredPath("LIVE_PROFILE_DIR", env.LIVE_PROFILE_DIR, root),
    pwaUrl: httpUrl("LIVE_PWA_URL", env.LIVE_PWA_URL || DEFAULT_LIVE_PWA_URL),
    chromePath: env.LIVE_CHROME_PATH?.trim() || undefined,
  };
}
