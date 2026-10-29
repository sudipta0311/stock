import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  listSignedInDevices,
  settingsPathAvailable,
  signOutEverywhere,
} from "./api";
import { useAuth } from "../auth/AuthContext";
import { Button, Sheet } from "../ui";

function formatDeviceDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function SignedInDevicesPanel() {
  const queryClient = useQueryClient();
  const { logout } = useAuth();
  const supported = settingsPathAvailable("/v1/auth/sessions");
  const query = useQuery({
    queryKey: ["settings", "signed-in-devices"],
    queryFn: listSignedInDevices,
    enabled: supported,
  });
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");

  async function confirmSignOutEverywhere(): Promise<void> {
    setBusy(true);
    setMessage("");
    try {
      await signOutEverywhere();
      await queryClient.cancelQueries();
      queryClient.clear();
      await logout({ serverRevoked: true });
    } catch {
      setMessage("We could not sign out your other devices. Try again.");
      setBusy(false);
    }
  }

  if (!supported) return null;

  return (
    <div className="grid gap-3 border-t border-line pt-3">
      <div>
        <h3 className="text-sm font-semibold">Signed-in devices</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          You stay signed in on this phone for 30 days.
        </p>
      </div>
      {query.isPending ? (
        <p role="status" className="text-sm text-muted-foreground">
          Loading devices…
        </p>
      ) : null}
      {query.isError ? (
        <p role="alert" className="text-sm text-danger-foreground">
          Devices could not be loaded.
        </p>
      ) : null}
      {query.data?.length ? (
        <ul className="grid gap-2">
          {query.data.map((device) => (
            <li key={device.id} className="rounded-xl bg-muted p-3 text-sm">
              <p className="font-medium">
                {device.device_label}
                {device.current ? " · This phone" : ""}
              </p>
              <p className="mt-1 text-muted-foreground">
                Added {formatDeviceDate(device.created_at)} · Last used{" "}
                {formatDeviceDate(device.last_used_at)}
              </p>
            </li>
          ))}
        </ul>
      ) : query.isSuccess ? (
        <p className="text-sm text-muted-foreground">No signed-in devices.</p>
      ) : null}
      {message ? (
        <p role="alert" className="text-sm text-danger-foreground">
          {message}
        </p>
      ) : null}
      <Button
        type="button"
        variant="secondary"
        className="w-full"
        onClick={() => setConfirming(true)}
      >
        Sign out on all devices
      </Button>
      <Sheet
        open={confirming}
        onOpenChange={(open) => {
          if (!busy) setConfirming(open);
        }}
        title="Sign out on all devices?"
      >
        <div className="grid gap-3">
          <p className="text-sm">
            This signs you out here and on your other devices.
          </p>
          <Button
            type="button"
            variant="danger"
            className="w-full"
            loading={busy}
            onClick={() => void confirmSignOutEverywhere()}
          >
            Sign out everywhere
          </Button>
          <Button
            type="button"
            variant="secondary"
            className="w-full"
            disabled={busy}
            onClick={() => setConfirming(false)}
          >
            Cancel
          </Button>
        </div>
      </Sheet>
    </div>
  );
}
