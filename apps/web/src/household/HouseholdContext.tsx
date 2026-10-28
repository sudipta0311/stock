import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  getSelectedHouseholdId,
  persistSelectedHouseholdId,
} from "../api/client";
import { apiPathAvailable } from "../api/feature-support";
import { queryClient } from "../api/query-client";
import englishCopy from "../copy/en.json";
import { getMe, listMembers } from "../money/api";
import type { MeResponse, MemberResponse } from "../money/api";
import { listHouseholds } from "./api";
import type { HouseholdSummary } from "./api";
import { getDkkInrRate } from "./fx";
import type { DkkInrRate } from "./fx";

export const MEMBER_FILTER_STORAGE_KEY = "wealth.household.member-filter";
export const PRIMARY_CURRENCY_STORAGE_KEY = "wealth.settings.primary-currency";

export type PrimaryCurrency = "INR" | "DKK";

interface HouseholdContextValue {
  selectedHouseholdId: string | null;
  selectedMemberId: string;
  activeMemberId: string | undefined;
  me?: MeResponse;
  meMemberId: string | undefined;
  members: MemberResponse[];
  households: HouseholdSummary[];
  role: "owner" | "partner" | "viewer" | undefined;
  isViewer: boolean;
  primaryCurrency: PrimaryCurrency;
  fxRate: DkkInrRate | undefined;
  selectHousehold(id: string | null): void;
  selectMember(id: string): void;
  setPrimaryCurrency(currency: PrimaryCurrency): void;
  memberLabel(member: MemberResponse): string;
}

const defaultValue: HouseholdContextValue = {
  selectedHouseholdId: null,
  selectedMemberId: "family",
  activeMemberId: undefined,
  meMemberId: undefined,
  members: [],
  households: [],
  role: undefined,
  isViewer: false,
  primaryCurrency: "INR",
  fxRate: undefined,
  selectHousehold: () => undefined,
  selectMember: () => undefined,
  setPrimaryCurrency: () => undefined,
  memberLabel: (member) => member.display_name,
};

const HouseholdContext = createContext<HouseholdContextValue>(defaultValue);

function readMemberFilter(): string {
  try {
    return window.sessionStorage.getItem(MEMBER_FILTER_STORAGE_KEY) ?? "family";
  } catch {
    return "family";
  }
}

function readPrimaryCurrency(): PrimaryCurrency {
  try {
    return window.localStorage.getItem(PRIMARY_CURRENCY_STORAGE_KEY) === "DKK"
      ? "DKK"
      : "INR";
  } catch {
    return "INR";
  }
}

export function HouseholdProvider({ children }: { children: ReactNode }) {
  const [selectedHouseholdId, setSelectedHouseholdId] = useState<string | null>(
    () => getSelectedHouseholdId(),
  );
  const [selectedMemberId, setSelectedMemberId] = useState(readMemberFilter);
  const [primaryCurrency, setCurrency] = useState(readPrimaryCurrency);
  const meQuery = useQuery({
    queryKey: ["household-context", "me", selectedHouseholdId],
    queryFn: getMe,
    enabled: apiPathAvailable("/v1/me"),
  });
  const memberQuery = useQuery({
    queryKey: ["household-context", "members", selectedHouseholdId],
    queryFn: listMembers,
    enabled: apiPathAvailable("/v1/members"),
  });
  const householdsQuery = useQuery({
    queryKey: ["household-context", "households", selectedHouseholdId],
    queryFn: listHouseholds,
    enabled: apiPathAvailable("/v1/households"),
  });
  const fxQuery = useQuery({
    queryKey: ["household-context", "fx", "DKK", "INR"],
    queryFn: getDkkInrRate,
    enabled: (memberQuery.data ?? meQuery.data?.members ?? []).some(
      (member) => member.residency.country === "DK",
    ),
    staleTime: 6 * 60 * 60 * 1000,
    gcTime: 24 * 60 * 60 * 1000,
  });

  const me = meQuery.data;
  const members = useMemo(
    () => memberQuery.data ?? me?.members ?? [],
    [memberQuery.data, me?.members],
  );
  const meMemberId = me?.me_member_id;
  const households = useMemo(() => {
    const available = householdsQuery.data ?? [];
    if (!me?.household_owner_id) return available;
    if (available.some((item) => item.owner_id === me.household_owner_id))
      return available;
    return [
      {
        owner_id: me.household_owner_id,
        owner_name: me.display_name,
        member_count: members.length,
        role: me.role,
      },
      ...available,
    ];
  }, [householdsQuery.data, me, members.length]);

  useEffect(() => {
    if (
      selectedMemberId !== "family" &&
      members.length > 0 &&
      !members.some((member) => member.id === selectedMemberId)
    ) {
      setSelectedMemberId("family");
      try {
        window.sessionStorage.setItem(MEMBER_FILTER_STORAGE_KEY, "family");
      } catch {
        // Member selection is a convenience and can remain in memory.
      }
    }
  }, [members, selectedMemberId]);

  useEffect(() => {
    function resetRemovedHousehold() {
      setSelectedHouseholdId(null);
      setSelectedMemberId("family");
      void queryClient.invalidateQueries();
    }
    window.addEventListener("wealth:household-removed", resetRemovedHousehold);
    return () =>
      window.removeEventListener(
        "wealth:household-removed",
        resetRemovedHousehold,
      );
  }, []);

  const selectHousehold = useCallback((id: string | null) => {
    persistSelectedHouseholdId(id);
    setSelectedHouseholdId(id);
    setSelectedMemberId("family");
    try {
      window.sessionStorage.setItem(MEMBER_FILTER_STORAGE_KEY, "family");
    } catch {
      // Household switching continues for this session.
    }
    void queryClient.invalidateQueries();
  }, []);

  const selectMember = useCallback((id: string) => {
    setSelectedMemberId(id || "family");
    try {
      window.sessionStorage.setItem(MEMBER_FILTER_STORAGE_KEY, id || "family");
    } catch {
      // Member selection is a convenience and can remain in memory.
    }
  }, []);

  const setPrimaryCurrency = useCallback((currency: PrimaryCurrency) => {
    setCurrency(currency);
    try {
      window.localStorage.setItem(PRIMARY_CURRENCY_STORAGE_KEY, currency);
    } catch {
      // Currency preference remains active for this session.
    }
  }, []);

  const value = useMemo<HouseholdContextValue>(
    () => ({
      selectedHouseholdId,
      selectedMemberId,
      activeMemberId:
        selectedMemberId === "family" ? undefined : selectedMemberId,
      me,
      meMemberId,
      members,
      households,
      role: me?.role,
      isViewer: me?.role === "viewer",
      primaryCurrency,
      fxRate: fxQuery.data,
      selectHousehold,
      selectMember,
      setPrimaryCurrency,
      memberLabel: (member) =>
        member.id === meMemberId
          ? englishCopy.household.you
          : member.display_name,
    }),
    [
      selectedHouseholdId,
      selectedMemberId,
      me,
      meMemberId,
      members,
      households,
      primaryCurrency,
      fxQuery.data,
      selectHousehold,
      selectMember,
      setPrimaryCurrency,
    ],
  );

  return (
    <HouseholdContext.Provider value={value}>
      {children}
    </HouseholdContext.Provider>
  );
}

export function useHousehold(): HouseholdContextValue {
  return useContext(HouseholdContext);
}
