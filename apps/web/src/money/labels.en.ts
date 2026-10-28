import type { AssetType } from "./forms.generated";

export interface FieldCopy {
  label: string;
  help?: string;
}

export const commonFieldCopy: Record<string, FieldCopy> = {
  label: { label: "Name this item" },
  notes: { label: "Notes" },
  member_id: { label: "Owned by" },
  balance_as_of: { label: "Balance as of (month and year)" },
  rate_pct: {
    label: "Interest rate (%)",
    help: "Use the rate printed on your statement or certificate.",
  },
  maturity: { label: "Matures (month and year)" },
  start: { label: "Started (month and year)" },
};

export const assetFieldCopy: Partial<
  Record<AssetType, Partial<Record<string, FieldCopy>>>
> = {
  bond: {
    kind: { label: "Bond type" },
    face_value: { label: "Amount invested", help: "A rough amount is fine." },
    issuer: { label: "Issuer" },
    coupon_pct: { label: "Coupon rate (%)" },
    coupon_frequency: { label: "Payments per year" },
    isin: { label: "ISIN" },
    maturity: { label: "Matures (month and year)" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  epf: {
    balance: {
      label: "Current EPF balance",
      help: "A recent passbook balance is enough.",
    },
    balance_as_of: commonFieldCopy.balance_as_of,
    monthly_employee: { label: "Your monthly contribution" },
    monthly_employer: { label: "Employer monthly contribution" },
    vpf_monthly: { label: "Monthly VPF contribution" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  fd: {
    bank: { label: "Bank or post office" },
    principal: { label: "Amount deposited", help: "Rough numbers are fine." },
    rate_pct: commonFieldCopy.rate_pct,
    maturity: commonFieldCopy.maturity,
    start: commonFieldCopy.start,
    compounding: { label: "Compounding" },
    payout: { label: "Interest payout" },
    joint: { label: "Joint deposit" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  gold: {
    kind: { label: "Type of gold" },
    grams: { label: "Weight in grams" },
    units: { label: "Units held" },
    purity: { label: "Purity (karat)" },
    issue: { label: "Issue month (month and year)" },
    issue_price_per_gram: { label: "Issue price per gram" },
    series: { label: "Series" },
    purpose: { label: "Purpose" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  health_insurance: {
    insurer: { label: "Insurer" },
    cover: { label: "Health cover amount" },
    renewal_month: { label: "Renewal month (1–12)" },
    members_covered: {
      label: "People covered",
      help: "Separate entries with commas.",
    },
    top_up_cover: { label: "Top-up cover amount" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  loan: {
    loan_kind: { label: "Loan type" },
    outstanding: { label: "Outstanding balance" },
    rate_pct: commonFieldCopy.rate_pct,
    emi: { label: "Monthly EMI" },
    end: { label: "Expected end (month and year)" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  nps: {
    tier1_balance: { label: "NPS Tier I balance" },
    tier2_balance: { label: "NPS Tier II balance" },
    equity_pct: { label: "Equity allocation (%)" },
    corporate_debt_pct: { label: "Corporate debt allocation (%)" },
    govt_debt_pct: { label: "Government debt allocation (%)" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  po_scheme: {
    scheme: { label: "Post Office scheme" },
    amount: { label: "Amount invested" },
    purchase: { label: "Purchased (month and year)" },
    rate_pct: commonFieldCopy.rate_pct,
    td_years: { label: "Time deposit term (years)" },
    balance_as_of: commonFieldCopy.balance_as_of,
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  ppf: {
    balance: { label: "Current PPF balance" },
    opened_fy_start_year: { label: "Year opened" },
    yearly_contribution: { label: "Yearly contribution" },
    held_at: { label: "Held at" },
    balance_as_of: commonFieldCopy.balance_as_of,
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  property: {
    property_kind: { label: "Property type" },
    estimated_value: { label: "Estimated value" },
    linked_loan_id: { label: "Linked loan" },
    monthly_rent: { label: "Monthly rent" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  rd: {
    institution: { label: "Bank or post office" },
    monthly_amount: { label: "Monthly deposit" },
    rate_pct: commonFieldCopy.rate_pct,
    maturity: commonFieldCopy.maturity,
    start: commonFieldCopy.start,
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  savings: {
    bank: { label: "Bank" },
    balance: { label: "Current balance" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  term_insurance: {
    insurer: { label: "Insurer" },
    cover: { label: "Life cover amount" },
    ends_year: { label: "Cover ends in year" },
    premium: { label: "Premium amount" },
    premium_frequency: { label: "Premium frequency" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  traditional_policy: {
    insurer: { label: "Insurer" },
    plan: { label: "Plan name or number" },
    policy_kind: { label: "Policy type" },
    sum_assured: { label: "Sum assured" },
    annual_premium: { label: "Annual premium" },
    start_year: { label: "Started in year" },
    term_years: { label: "Policy term (years)" },
    premium_paying_term: { label: "Premium paying term (years)" },
    bonus_accrued: { label: "Bonuses so far" },
    surrender_value: { label: "Current surrender value" },
    maturity_illustration: { label: "Maturity estimate from illustration" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
  ulip: {
    insurer: { label: "Insurer" },
    fund_value: { label: "Current fund value" },
    annual_premium: { label: "Annual premium" },
    equity_pct: { label: "Equity allocation (%)" },
    start_year: { label: "Started in year" },
    label: commonFieldCopy.label,
    member_id: commonFieldCopy.member_id,
    notes: commonFieldCopy.notes,
  },
};

export function fieldCopy(assetType: AssetType, field: string): FieldCopy {
  return (
    assetFieldCopy[assetType]?.[field] ??
    commonFieldCopy[field] ?? {
      label: field
        .replaceAll("_", " ")
        .replace(/^./, (letter) => letter.toUpperCase()),
    }
  );
}
