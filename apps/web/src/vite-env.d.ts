/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />

declare const __APP_VERSION__: string;
declare const __APP_BUILD_DATE__: string;

interface ImportMetaEnv {
  readonly VITE_VAPID_PUBLIC_KEY?: string;
  readonly VITE_E2E_DEV_AUTH?: string;
}
