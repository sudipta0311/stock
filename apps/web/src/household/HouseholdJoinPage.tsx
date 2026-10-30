import { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router";
import { useQueryClient } from "@tanstack/react-query";
import { getMe, listAssets } from "../money/api";
import { Button, Card, Checkbox } from "../ui";
import englishCopy from "../copy/en.json";
import { ApiError, persistSelectedHouseholdId } from "../api/client";
import { joinHousehold } from "./api";

export function HouseholdJoinPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const token = new URLSearchParams(location.search).get("token") ?? "";
  const [hasExistingData, setHasExistingData] = useState(false);
  const [bringMyData, setBringMyData] = useState(true);
  const [checkingData, setCheckingData] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    void Promise.allSettled([getMe(), listAssets()]).then(
      ([meResult, assetsResult]) => {
        if (!active) return;
        const hasProfile =
          meResult.status === "fulfilled" && meResult.value.profile !== null;
        const hasAssets =
          assetsResult.status === "fulfilled" && assetsResult.value.length > 0;
        setHasExistingData(hasProfile || hasAssets);
        setCheckingData(false);
      },
    );
    return () => {
      active = false;
    };
  }, []);

  async function acceptInvite(): Promise<void> {
    if (!token) {
      setError(englishCopy.household.join_missing);
      return;
    }
    setSaving(true);
    setError("");
    try {
      const joined = await joinHousehold({
        token,
        bring_my_data: hasExistingData && bringMyData,
      });
      persistSelectedHouseholdId(joined.owner_id);
      queryClient.clear();
      navigate("/", { replace: true });
    } catch (reason) {
      setError(
        reason instanceof ApiError
          ? englishCopy.household.join_error
          : englishCopy.household.join_error,
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="mx-auto grid min-h-screen w-full max-w-xl content-center gap-4 p-4">
      <Card className="grid gap-4">
        <div>
          <h1 className="text-xl font-semibold">
            {englishCopy.household.join_title}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {englishCopy.household.join_description}
          </p>
        </div>
        {checkingData ? (
          <p role="status" className="text-sm text-muted-foreground">
            {englishCopy.household.join_checking}
          </p>
        ) : null}
        {!checkingData && hasExistingData ? (
          <Checkbox
            id="bring-existing-data"
            label={englishCopy.household.join_data}
            help={englishCopy.household.join_data_help}
            checked={bringMyData}
            onChange={(event) => setBringMyData(event.currentTarget.checked)}
          />
        ) : null}
        {error ? (
          <p role="alert" className="text-sm text-danger-foreground">
            {error}
          </p>
        ) : null}
        <Button
          onClick={() => void acceptInvite()}
          loading={saving || checkingData}
          disabled={!token}
        >
          {englishCopy.household.join_button}
        </Button>
      </Card>
    </main>
  );
}
