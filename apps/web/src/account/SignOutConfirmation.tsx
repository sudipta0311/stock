import { useState } from "react";
import { Button, Checkbox } from "../ui";

export function SignOutConfirmation({
  onCancel,
  onConfirm,
}: {
  onCancel(): void;
  onConfirm(disablePush: boolean): Promise<void>;
}) {
  const [disablePush, setDisablePush] = useState(false);
  const [busy, setBusy] = useState(false);

  async function confirm(): Promise<void> {
    setBusy(true);
    await onConfirm(disablePush);
  }

  return (
    <div className="grid gap-3">
      <p className="text-sm">Sign out of Wealth Copilot on this phone?</p>
      <Checkbox
        id="sign-out-disable-push"
        label="Turn off alerts on this phone too"
        checked={disablePush}
        onChange={(event) => setDisablePush(event.currentTarget.checked)}
      />
      <Button
        type="button"
        variant="danger"
        className="w-full"
        loading={busy}
        onClick={() => void confirm()}
      >
        Sign out
      </Button>
      <Button
        type="button"
        variant="secondary"
        className="w-full"
        disabled={busy}
        onClick={onCancel}
      >
        Cancel
      </Button>
    </div>
  );
}
