import sharp from "sharp";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const source = fileURLToPath(
  new URL("../public/wealth-mark.svg", import.meta.url),
);
const output = fileURLToPath(new URL("../public/icons/", import.meta.url));

await mkdir(output, { recursive: true });
await sharp(source).resize(192, 192).png().toFile(`${output}wealth-192.png`);
await sharp(source).resize(512, 512).png().toFile(`${output}wealth-512.png`);
await sharp(source)
  .resize(512, 512)
  .png()
  .toFile(`${output}wealth-maskable-512.png`);
