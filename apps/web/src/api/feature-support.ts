import { API_PATHS } from "./features.generated";

export function apiPathAvailable(path: string): boolean {
  return (API_PATHS as readonly string[]).includes(path);
}
