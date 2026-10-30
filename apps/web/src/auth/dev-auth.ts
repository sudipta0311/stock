export type DevelopmentAuthMode = "oidc" | "dev";

export function developmentTokenSignInAllowed(
  mode: DevelopmentAuthMode,
  isProductionBuild: boolean,
  e2eAuthEnabled: boolean,
): boolean {
  return mode === "dev" && (!isProductionBuild || e2eAuthEnabled);
}
