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
    // Shared CI runners compile and render far slower than a dev machine.
    testTimeout: 20_000,
    restoreMocks: true,
    clearMocks: true,
  },
});
