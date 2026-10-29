// Vitest stand-in for vite-plugin-pwa's virtual module (it only exists in Vite builds).
// Tests that render PwaLifecycle replace it with vi.mock.
export function useRegisterSW(): never {
  throw new Error("Mock virtual:pwa-register/react in this test.");
}
