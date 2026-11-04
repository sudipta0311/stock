export const BROKER_STORAGE_PREFIX = "wealth.import.broker.";

export const BROKER_GROUPS = [
  {
    label: "India",
    options: [
      ["zerodha", "Zerodha"],
      ["hdfc_securities", "HDFC Securities"],
      ["groww", "Groww"],
      ["icici_direct", "ICICI Direct"],
      ["angel_one", "Angel One"],
      ["upstox", "Upstox"],
      ["other_indian_broker", "Other Indian broker"],
    ],
  },
  {
    label: "US shares from India",
    options: [
      ["indmoney", "INDmoney"],
      ["vested", "Vested"],
    ],
  },
  {
    label: "Denmark",
    options: [
      ["nordnet", "Nordnet"],
      ["saxo", "Saxo"],
      ["danske_bank", "Danske Bank"],
      ["other_danish_broker", "Other Danish broker"],
    ],
  },
  {
    label: "Other",
    options: [
      ["ibkr", "Interactive Brokers"],
      ["other", "Other"],
    ],
  },
] as const;

const brokerValues: ReadonlySet<string> = new Set(
  BROKER_GROUPS.flatMap((group) => group.options.map(([value]) => value)),
);

export function defaultBrokerForResidence(country: string | undefined): string {
  if (country === "IN") return "zerodha";
  if (country === "DK") return "nordnet";
  return "other";
}

export function readBrokerForMember(
  memberId: string,
  country: string | undefined,
): string {
  try {
    const saved = window.localStorage.getItem(
      `${BROKER_STORAGE_PREFIX}${memberId}`,
    );
    if (saved && brokerValues.has(saved)) return saved;
  } catch {
    // The upload keeps a usable default when browser storage is unavailable.
  }
  return defaultBrokerForResidence(country);
}

export function saveBrokerForMember(memberId: string, broker: string): void {
  if (!brokerValues.has(broker)) return;
  try {
    window.localStorage.setItem(`${BROKER_STORAGE_PREFIX}${memberId}`, broker);
  } catch {
    // Remembering a broker is optional.
  }
}
