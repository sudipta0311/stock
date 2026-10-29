import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError } from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import { useHousehold } from "../household/HouseholdContext";
import {
  createHouseholdInvite,
  listHouseholdAccess,
  listHouseholdInvites,
  leaveHousehold,
  revokeHouseholdAccess,
  revokeHouseholdInvite,
  updateHouseholdAccess,
} from "../household/api";
import type { HouseholdInviteCreated, HouseholdRole } from "../household/api";
import { Button, Card, Select } from "../ui";
import englishCopy from "../copy/en.json";

export function HouseholdSharingPanel() {
  const queryClient = useQueryClient();
  const {
    selectedHouseholdId,
    members,
    role: currentRole,
    isViewer,
    selectHousehold,
  } = useHousehold();
  const [email, setEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<HouseholdRole>("partner");
  const [memberId, setMemberId] = useState("");
  const [createdInvite, setCreatedInvite] =
    useState<HouseholdInviteCreated | null>(null);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const [activeChangeId, setActiveChangeId] = useState("");
  const [leaving, setLeaving] = useState(false);

  const enabled = apiPathAvailable("/v1/households/invites");
  const invitesQuery = useQuery({
    queryKey: ["settings", "household-invites", selectedHouseholdId],
    queryFn: listHouseholdInvites,
    enabled,
  });
  const accessQuery = useQuery({
    queryKey: ["settings", "household-access-panel", selectedHouseholdId],
    queryFn: listHouseholdAccess,
    enabled: apiPathAvailable("/v1/households/access"),
  });
  const activeAccess = useMemo(
    () =>
      (accessQuery.data ?? []).filter((person) => person.revoked_at === null),
    [accessQuery.data],
  );
  const pendingInvites = useMemo(
    () =>
      (invitesQuery.data ?? []).filter(
        (invite) => !invite.accepted_at && !invite.revoked_at,
      ),
    [invitesQuery.data],
  );
  const canManage = !isViewer;

  async function refreshSharing(): Promise<void> {
    await Promise.all([
      queryClient.invalidateQueries({
        queryKey: ["settings", "household-invites"],
      }),
      queryClient.invalidateQueries({
        queryKey: ["settings", "household-access-panel"],
      }),
      queryClient.invalidateQueries({
        queryKey: ["settings", "household-access"],
      }),
    ]);
  }

  async function createInvite(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    setCreatedInvite(null);
    try {
      const invite = await createHouseholdInvite({
        email: email.trim(),
        role: inviteRole,
        ...(memberId ? { member_id: memberId } : {}),
        send_email: false,
      });
      setCreatedInvite(invite);
      setMessage(englishCopy.household.invite_created);
      await refreshSharing();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        const code = `${error.code} ${error.title}`.toLowerCase();
        if (code.includes("member_is_owner")) {
          setMessage(englishCopy.household.invite_member_owner);
        } else if (code.includes("member_already_linked")) {
          setMessage(englishCopy.household.invite_member_linked);
        } else {
          setMessage(englishCopy.household.invite_error);
        }
      } else {
        setMessage(englishCopy.household.invite_error);
      }
    } finally {
      setSaving(false);
    }
  }

  async function copyInviteLink(): Promise<void> {
    if (!createdInvite) return;
    const url = new URL(createdInvite.share_link, window.location.origin).href;
    try {
      await navigator.clipboard.writeText(url);
      setMessage(englishCopy.household.invite_copied);
    } catch {
      setMessage(englishCopy.household.invite_copy_help);
    }
  }

  async function shareInviteLink(): Promise<void> {
    if (!createdInvite) return;
    const url = new URL(createdInvite.share_link, window.location.origin).href;
    if (navigator.share) {
      try {
        await navigator.share({ url });
      } catch {
        // A dismissed share sheet does not change the invite.
      }
      return;
    }
    await copyInviteLink();
  }

  async function changeAccessRole(userId: string, nextRole: HouseholdRole) {
    setActiveChangeId(userId);
    setMessage("");
    try {
      await updateHouseholdAccess(userId, nextRole);
      await refreshSharing();
    } catch {
      setMessage(englishCopy.household.access_error);
    } finally {
      setActiveChangeId("");
    }
  }

  async function removeAccess(userId: string) {
    setActiveChangeId(userId);
    setMessage("");
    try {
      await revokeHouseholdAccess(userId);
      await refreshSharing();
    } catch {
      setMessage(englishCopy.household.access_error);
    } finally {
      setActiveChangeId("");
    }
  }

  async function revokeInvite(inviteId: string) {
    setActiveChangeId(inviteId);
    setMessage("");
    try {
      await revokeHouseholdInvite(inviteId);
      await refreshSharing();
    } catch {
      setMessage(englishCopy.household.invite_error);
    } finally {
      setActiveChangeId("");
    }
  }

  async function leaveFamilyPlan(): Promise<void> {
    setLeaving(true);
    setMessage("");
    try {
      await leaveHousehold();
      selectHousehold(null);
      await queryClient.invalidateQueries();
      setMessage(englishCopy.household.leave_success);
    } catch {
      setMessage(englishCopy.household.leave_error);
    } finally {
      setLeaving(false);
    }
  }

  if (!enabled) return null;

  return (
    <Card className="grid gap-4">
      <div>
        <h2 className="text-base font-semibold">
          {englishCopy.household.share_with_partner}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.household.settings.share_explanation}
        </p>
      </div>
      {isViewer ? (
        <p className="text-sm text-muted-foreground">
          {englishCopy.household.view_only_help}
        </p>
      ) : (
        <form
          onSubmit={(event) => void createInvite(event)}
          className="grid gap-5"
        >
          <label
            className="grid gap-1 text-sm font-medium"
            htmlFor="invite-email"
          >
            {englishCopy.household.invite_email}
            <input
              id="invite-email"
              type="email"
              value={email}
              onChange={(event) => setEmail(event.currentTarget.value)}
              className="min-h-12 rounded-xl border border-line bg-background px-3"
              required
            />
          </label>
          <Select
            id="invite-role"
            label={englishCopy.household.invite_role}
            value={inviteRole}
            onChange={(event) =>
              setInviteRole(event.currentTarget.value as HouseholdRole)
            }
          >
            <option value="partner">
              {englishCopy.household.role_partner} ·{" "}
              {englishCopy.household.settings.role_partner_description}
            </option>
            <option value="viewer">
              {englishCopy.household.role_viewer} ·{" "}
              {englishCopy.household.settings.role_viewer_description}
            </option>
          </Select>
          <Select
            id="invite-member"
            label={englishCopy.household.invite_member}
            value={memberId}
            onChange={(event) => setMemberId(event.currentTarget.value)}
          >
            <option value="">{englishCopy.household.invite_member_none}</option>
            {members.map((member) => (
              <option key={member.id} value={member.id}>
                {member.display_name}
              </option>
            ))}
          </Select>
          <Button type="submit" loading={saving}>
            {englishCopy.household.invite_create}
          </Button>
        </form>
      )}
      {createdInvite ? (
        <div className="grid gap-2 rounded-xl border border-line p-3">
          <label
            className="grid gap-1 text-sm font-medium"
            htmlFor="share-link"
          >
            {englishCopy.household.invite_link}
            <input
              id="share-link"
              readOnly
              value={
                new URL(createdInvite.share_link, window.location.origin).href
              }
              className="min-h-12 w-full rounded-xl border border-line bg-background px-3 text-xs"
            />
          </label>
          <div className="flex flex-wrap gap-2">
            <Button variant="secondary" onClick={() => void copyInviteLink()}>
              {englishCopy.household.invite_copy}
            </Button>
            <Button variant="secondary" onClick={() => void shareInviteLink()}>
              {englishCopy.household.invite_share}
            </Button>
          </div>
        </div>
      ) : null}
      {pendingInvites.length ? (
        <div className="grid gap-2 border-t border-line pt-3">
          <h3 className="text-sm font-semibold">
            {englishCopy.household.invite_pending}
          </h3>
          {pendingInvites.map((invite) => (
            <div
              key={invite.id}
              className="flex flex-wrap items-center justify-between gap-2 text-sm"
            >
              <span>
                {invite.email} ·{" "}
                {invite.role === "viewer"
                  ? englishCopy.household.role_viewer
                  : englishCopy.household.role_partner}
              </span>
              {canManage ? (
                <Button
                  variant="ghost"
                  className="min-h-10 px-3 py-2"
                  loading={activeChangeId === invite.id}
                  onClick={() => void revokeInvite(invite.id)}
                >
                  {englishCopy.household.invite_revoke}
                </Button>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {activeAccess.length ? (
        <div className="grid gap-2 border-t border-line pt-3">
          <h3 className="text-sm font-semibold">
            {englishCopy.settings.family.access_people}
          </h3>
          {activeAccess.map((person) => (
            <div
              key={person.user_id}
              className="grid gap-2 rounded-xl border border-line p-3 sm:grid-cols-[1fr_auto_auto] sm:items-center"
            >
              <span className="text-sm">
                {person.display_name ||
                  englishCopy.settings.privacy.delete_other_person}
              </span>
              {canManage ? (
                <Select
                  id={`access-role-${person.user_id}`}
                  label={englishCopy.household.access_role}
                  value={person.role}
                  disabled={activeChangeId === person.user_id}
                  onChange={(event) =>
                    void changeAccessRole(
                      person.user_id,
                      event.currentTarget.value as HouseholdRole,
                    )
                  }
                >
                  <option value="partner">
                    {englishCopy.household.role_partner}
                  </option>
                  <option value="viewer">
                    {englishCopy.household.role_viewer}
                  </option>
                </Select>
              ) : (
                <span className="text-sm text-muted-foreground">
                  {person.role === "viewer"
                    ? englishCopy.household.role_viewer
                    : englishCopy.household.role_partner}
                </span>
              )}
              {canManage ? (
                <Button
                  variant="ghost"
                  className="min-h-10 px-3 py-2 text-danger-foreground"
                  loading={activeChangeId === person.user_id}
                  onClick={() => void removeAccess(person.user_id)}
                >
                  {englishCopy.household.access_revoke}
                </Button>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      {currentRole && currentRole !== "owner" ? (
        <div className="border-t border-line pt-3">
          <Button
            variant="secondary"
            className="w-full"
            loading={leaving}
            onClick={() => void leaveFamilyPlan()}
          >
            {englishCopy.household.leave}
          </Button>
        </div>
      ) : null}
      {message ? (
        <p role="status" className="text-sm text-muted-foreground">
          {message}
        </p>
      ) : null}
    </Card>
  );
}
