import { AppLockGate } from "./AppLockGate";
import { AppShell } from "./AppShell";
import { HouseholdProvider } from "../household/HouseholdContext";

export function ProtectedHouseholdApp() {
  return (
    <AppLockGate>
      <HouseholdProvider>
        <AppShell />
      </HouseholdProvider>
    </AppLockGate>
  );
}
