import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { typescriptPlugin } from "./scripts/typescript-plugin.ts";

const sandboxBuild = process.env.SANDBOX_BUILD === "1";

export default defineConfig({
  resolve: { preserveSymlinks: true },
  plugins: [
    ...(sandboxBuild ? [typescriptPlugin()] : []),
    tailwindcss(),
    VitePWA({
      strategies: "injectManifest",
      srcDir: "src",
      filename: "sw.ts",
      manifest: {
        name: "Wealth Copilot",
        short_name: "Wealth",
        description: "A clear view of your household wealth and next steps.",
        display: "standalone",
        theme_color: "#f4f6f7",
        background_color: "#f4f6f7",
        start_url: "/",
        scope: "/",
        share_target: {
          action: "/import/share",
          method: "POST",
          enctype: "multipart/form-data",
          params: {
            files: [{ name: "file", accept: ["application/pdf"] }],
          },
        },
        icons: [
          { src: "/icons/wealth-192.png", sizes: "192x192", type: "image/png" },
          { src: "/icons/wealth-512.png", sizes: "512x512", type: "image/png" },
          {
            src: "/icons/wealth-maskable-512.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      injectManifest: {
        globPatterns: ["**/*.{js,css,html,svg,png,ico,woff2}"],
        maximumFileSizeToCacheInBytes: 3_000_000,
      },
    }),
  ],
  build: {
    manifest: true,
    sourcemap: false,
    target: "es2022",
    ...(sandboxBuild ? { minify: false } : {}),
  },
  ...(sandboxBuild ? { esbuild: false, oxc: false } : {}),
});
