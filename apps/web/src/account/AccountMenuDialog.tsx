import { useState } from "react";
import { Link } from "react-router";
import { Button, Sheet } from "../ui";
import { useAuth } from "../auth/AuthContext";
import { useHousehold } from "../household/HouseholdContext";
import { SignOutConfirmation } from "./SignOutConfirmation";

export function AccountMenuDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange(open: boolean): void;
}) {
  const { logout } = useAuth();
  const { me, households, selectedHouseholdId, selectHousehold, role } =
    useHousehold();
  const [confirming, setConfirming] = useState(false);
  const ownHouseholdId = me?.household_owner_id;
  const viewedHouseholdId = selectedHouseholdId ?? ownHouseholdId;
  const viewedHousehold = households.find(
    (household) => household.owner_id === viewedHouseholdId,
  );
  const householdName =
    viewedHouseholdId === ownHouseholdId
      ? "Your family plan"
      : `${viewedHousehold?.owner_name ?? "Family plan"}'s family plan`;
  const roleDescription =
    role === "viewer"
      ? "You can view"
      : role === "owner" || role === "partner"
        ? "You can edit"
        : "Checking your access";
  const displayName = me?.display_name?.trim() || "Your account";

  return (
    <Sheet
      open={open}
      onOpenChange={(next) => {
        if (!next) setConfirming(false);
        onOpenChange(next);
      }}
      title={confirming ? "Sign out" : "Account"}
    >
      {confirming ? (
        <SignOutConfirmation
          onCancel={() => setConfirming(false)}
          onConfirm={async (disablePush) => {
            await logout({ disablePush });
          }}
        />
      ) : (
        <div className="grid gap-4">
          <div>
            <p className="font-semibold">{displayName}</p>
            {me?.email ? (
              <p className="mt-1 break-all text-sm text-muted-foreground">
                {me.email}
              </p>
            ) : null}
          </div>
          <div className="rounded-xl bg-muted p-3 text-sm">
            <p className="font-medium">{householdName}</p>
            <p className="mt-1 text-muted-foreground">{roleDescription}</p>
          </div>
          {households.length > 1 ? (
            <label
              className="grid gap-2 text-sm font-medium"
              htmlFor="account-household-switcher"
            >
              Switch family plan
              <select
                id="account-household-switcher"
                value={viewedHouseholdId ?? ""}
                onChange={(event) => {
                  const nextId = event.currentTarget.value;
                  selectHousehold(nextId === ownHouseholdId ? null : nextId);
                }}
                className="min-h-12 rounded-xl border border-line bg-surface px-3"
              >
                {households.map((household) => (
                  <option key={household.owner_id} value={household.owner_id}>
                    {household.owner_id === ownHouseholdId
                      ? "Your family plan"
                      : `${household.owner_name ?? "Family plan"}'s family plan`}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <div className="grid gap-2 border-t border-line pt-3">
            <Link
              to="/settings"
              onClick={() => onOpenChange(false)}
              className="flex min-h-12 items-center rounded-xl px-3 text-sm font-medium hover:bg-muted"
            >
              Settings
            </Link>
            <Link
              to="/help"
              onClick={() => onOpenChange(false)}
              className="flex min-h-12 items-center rounded-xl px-3 text-sm font-medium hover:bg-muted"
            >
              Help and guide
            </Link>
            <div className="border-t border-line pt-2">
              <Button
                type="button"
                variant="ghost"
                className="w-full justify-start px-3"
                onClick={() => setConfirming(true)}
              >
                Sign out
              </Button>
            </div>
          </div>
        </div>
      )}
    </Sheet>
  );
}
