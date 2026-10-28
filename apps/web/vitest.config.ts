import { defineConfig } from "vitest/config";
import { typescriptPlugin } from "./scripts/typescript-plugin.ts";

export default defineConfig({
  resolve: { preserveSymlinks: true },
  esbuild: false,
  oxc: false,
  plugins: [typescriptPlugin()],
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    pool: "threads",
    maxWorkers: 1,
    restoreMocks: true,
    clearMocks: true,
  },
});
