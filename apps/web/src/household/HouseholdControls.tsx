import { Badge } from "../ui";
import englishCopy from "../copy/en.json";
import { useHousehold } from "./HouseholdContext";

export function HouseholdControls() {
  const {
    me,
    members,
    households,
    selectedHouseholdId,
    selectedMemberId,
    role,
    selectHousehold,
    selectMember,
    memberLabel,
  } = useHousehold();
  const ownHouseholdId = me?.household_owner_id;
  const householdValue = selectedHouseholdId ?? ownHouseholdId ?? "";

  return (
    <div className="flex flex-wrap items-center justify-end gap-2">
      {members.length > 1 ? (
        <select
          id="header-member-filter"
          aria-label={englishCopy.household.member_switcher}
          value={selectedMemberId}
          onChange={(event) => selectMember(event.currentTarget.value)}
          className="min-h-10 max-w-40 rounded-full border border-line bg-background px-3 text-sm font-medium"
        >
          <option value="family">{englishCopy.household.family}</option>
          {members.map((member) => (
            <option key={member.id} value={member.id}>
              {memberLabel(member)}
            </option>
          ))}
        </select>
      ) : null}
      {households.length > 1 ? (
        <select
          id="header-household-filter"
          aria-label={englishCopy.household.household_switcher}
          value={householdValue}
          onChange={(event) => {
            const value = event.currentTarget.value;
            selectHousehold(value === ownHouseholdId ? null : value || null);
          }}
          className="min-h-10 max-w-44 rounded-full border border-line bg-background px-3 text-sm"
        >
          {households.map((household) => (
            <option key={household.owner_id} value={household.owner_id}>
              {household.owner_id === ownHouseholdId
                ? englishCopy.household.your_household
                : (household.owner_name ??
                  englishCopy.household.owner_name_fallback)}
            </option>
          ))}
        </select>
      ) : null}
      {role ? (
        <Badge>
          {role === "viewer"
            ? englishCopy.household.role_viewer
            : role === "owner"
              ? englishCopy.household.role_owner
              : englishCopy.household.role_partner}
        </Badge>
      ) : null}
      {role === "viewer" ? (
        <Badge>{englishCopy.household.read_only}</Badge>
      ) : null}
    </div>
  );
}
