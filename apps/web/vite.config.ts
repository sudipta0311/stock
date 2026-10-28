import tailwindcss from "@tailwindcss/vite";
import { defineConfig, loadEnv, type Plugin } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import { typescriptPlugin } from "./scripts/typescript-plugin.ts";

const sandboxBuild = process.env.SANDBOX_BUILD === "1";

function sourceOrigin(
  value: string | undefined,
  label: string,
  fallback: string,
): string {
  const candidate = value?.trim() || fallback;
  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new Error(`${label} must be a valid API or sign-in URL.`);
  }
  if (
    !["http:", "https:"].includes(parsed.protocol) ||
    parsed.username ||
    parsed.password
  ) {
    throw new Error(`${label} must use HTTP or HTTPS without login details.`);
  }
  return parsed.origin;
}

function contentSecurityPolicyPlugin(
  apiUrl: string | undefined,
  authority: string | undefined,
): Plugin {
  const apiOrigin = sourceOrigin(
    apiUrl,
    "VITE_API_URL",
    "http://localhost:8000",
  );
  const authorityOrigin = sourceOrigin(
    authority,
    "VITE_OIDC_AUTHORITY",
    "https://accounts.google.com",
  );
  const connectSources = [...new Set(["'self'", apiOrigin, authorityOrigin])];
  const directives = [
    ["default-src", ["'self'"]],
    ["script-src", ["'self'"]],
    ["style-src", ["'self'"]],
    ["connect-src", connectSources],
    ["img-src", ["'self'", "data:"]],
    ["frame-src", ["'self'", authorityOrigin]],
    ["worker-src", ["'self'"]],
    ["manifest-src", ["'self'"]],
    ["base-uri", ["'self'"]],
    ["form-action", ["'self'", authorityOrigin]],
  ];
  const policy = directives
    .map(([name, values]) => `${name} ${(values as string[]).join(" ")}`)
    .join("; ");

  return {
    name: "wealth-content-security-policy",
    apply: "build",
    transformIndexHtml(html) {
      const meta = `<meta http-equiv="Content-Security-Policy" content="${policy}" />`;
      const withoutExisting = html.replace(
        /\s*<meta\s+http-equiv=["']Content-Security-Policy["'][^>]*>/gi,
        "",
      );
      return withoutExisting.replace("</head>", `    ${meta}\n  </head>`);
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const apiUrl = process.env.VITE_API_URL ?? env.VITE_API_URL;
  const oidcAuthority =
    process.env.VITE_OIDC_AUTHORITY ?? env.VITE_OIDC_AUTHORITY;

  return {
    resolve: { preserveSymlinks: true },
    plugins: [
      ...(sandboxBuild ? [typescriptPlugin()] : []),
      contentSecurityPolicyPlugin(apiUrl, oidcAuthority),
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
            {
              src: "/icons/wealth-192.png",
              sizes: "192x192",
              type: "image/png",
            },
            {
              src: "/icons/wealth-512.png",
              sizes: "512x512",
              type: "image/png",
            },
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
  };
});
