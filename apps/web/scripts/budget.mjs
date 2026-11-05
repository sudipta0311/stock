import { gzipSync } from "node:zlib";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const dist = path.join(root, "dist");
const manifestPath = path.join(dist, ".vite", "manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const entries = Object.values(manifest).filter(
  (entry) =>
    entry.isEntry && (entry.file.endsWith(".js") || entry.src === "index.html"),
);
const initialFiles = new Set();

function collect(entry) {
  if (entry.file.endsWith(".js")) initialFiles.add(entry.file);
  for (const imported of entry.imports ?? []) {
    const dependency = manifest[imported];
    if (dependency) collect(dependency);
  }
}

for (const entry of entries) collect(entry);

if (initialFiles.size === 0) {
  throw new Error("Vite manifest has no initial JavaScript entry.");
}

let totalBytes = 0;
for (const file of initialFiles) {
  const contents = await readFile(path.join(dist, file));
  totalBytes += gzipSync(contents).byteLength;
}

const kilobytes = totalBytes / 1024;
console.log(`Initial JS: ${kilobytes.toFixed(2)} KB gzipped (limit 170 KB).`);
if (totalBytes > 170 * 1024) {
  throw new Error(
    `Initial JavaScript is ${totalBytes} gzipped bytes; maximum is 174080.`,
  );
}

const importRoute = Object.entries(manifest).find(
  ([key, entry]) =>
    entry.isDynamicEntry && /imports[\\/]ImportPage\.tsx$/.test(key),
);
if (!importRoute) {
  throw new Error(
    "The import route must be emitted as a code-split dynamic entry.",
  );
}
const importRouteBytes = gzipSync(
  await readFile(path.join(dist, importRoute[1].file)),
).byteLength;
console.log(
  `Import route chunk: ${(importRouteBytes / 1024).toFixed(2)} KB gzipped (code-split).`,
);

const routeModules = [
  ["Home", /home[\\/]HomePage\.tsx$/],
  ["Money", /money[\\/]MoneyMapPage\.tsx$/],
  ["Plan", /plan[\\/]PlanPage\.tsx$/],
  ["Ideas", /ideas[\\/]IdeasPage\.tsx$/],
  ["Settings", /settings[\\/]SettingsPage\.tsx$/],
  ["World & your money", /ideas[\\/]WorldPage\.tsx$/],
  ["Inbox", /inbox[\\/]InboxPage\.tsx$/],
  ["Holding", /money[\\/]HoldingPage\.tsx$/],
];

const routeGuards = [
  ["Ideas", /ideas[\\/]IdeasPage\.tsx$/, "Why this idea?"],
  ["Plan", /plan[\\/]PlanPage\.tsx$/, "Your money is close to your plan."],
  ["Money", /money[\\/]MoneyMapPage\.tsx$/, "Add holdings"],
  ["Settings", /settings[\\/]SettingsPage\.tsx$/, "Delete my account"],
  ["Help", /help[\\/]HelpPage\.tsx$/, "Start with a rough picture"],
];

for (const [label, pattern] of routeModules) {
  const route = Object.entries(manifest).find(
    ([key, entry]) => entry.isDynamicEntry && pattern.test(key),
  );
  if (!route) {
    throw new Error(`${label} must be emitted as a code-split route entry.`);
  }
  const routeBytes = gzipSync(
    await readFile(path.join(dist, route[1].file)),
  ).byteLength;
  console.log(
    `${label} route chunk: ${(routeBytes / 1024).toFixed(2)} KB gzipped.`,
  );
}

function collectClosure(entry, visited = new Set()) {
  if (visited.has(entry)) return visited;
  visited.add(entry);
  for (const imported of manifest[entry]?.imports ?? []) {
    if (manifest[imported]) collectClosure(imported, visited);
  }
  return visited;
}

const initialContents = await Promise.all(
  [...initialFiles].map((file) => readFile(path.join(dist, file), "utf8")),
);
for (const [label, pattern, guardText] of routeGuards) {
  const route = Object.entries(manifest).find(
    ([key, entry]) => entry.isDynamicEntry && pattern.test(key),
  );
  if (!route) throw new Error(`${label} must have a dynamic route entry.`);
  if (initialContents.some((contents) => contents.includes(guardText))) {
    throw new Error(`${label} route copy leaked into initial JavaScript.`);
  }
  const routeChunks = [...collectClosure(route[0])]
    .map((key) => manifest[key].file)
    .filter((file) => file.endsWith(".js"));
  let matchingChunk;
  for (const file of routeChunks) {
    const contents = await readFile(path.join(dist, file), "utf8");
    if (contents.includes(guardText)) {
      matchingChunk = file;
      break;
    }
  }
  if (!matchingChunk) {
    throw new Error(
      `${label} route copy guard string is missing from its chunks.`,
    );
  }
  console.log(
    `${label} copy guard: ${JSON.stringify(guardText)} in ${matchingChunk}.`,
  );
}

const rootEntries = await readdir(dist);
if (rootEntries.length === 0) throw new Error("Build output is empty.");
