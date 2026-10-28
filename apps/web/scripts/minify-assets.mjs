import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { minify } from "terser";

const dist = fileURLToPath(new URL("../dist/", import.meta.url));

async function listJavaScript(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const files = await Promise.all(
    entries.map(async (entry) => {
      const file = path.join(directory, entry.name);
      return entry.isDirectory()
        ? listJavaScript(file)
        : /\.(?:js|mjs)$/.test(entry.name)
          ? [file]
          : [];
    }),
  );
  return files.flat();
}

const files = await listJavaScript(dist);
for (const file of files) {
  const source = await readFile(file, "utf8");
  const result = await minify(source, {
    module: true,
    compress: true,
    mangle: true,
    format: { comments: false },
  });
  if (result.code) await writeFile(file, result.code, "utf8");
}
console.log(`Minified ${files.length} built JavaScript files.`);
