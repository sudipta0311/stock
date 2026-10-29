import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router";
import { ArrowLeft, Trash2 } from "lucide-react";
import { useHousehold } from "../household/HouseholdContext";
import {
  deleteLoanEvent,
  getLoan,
  previewLoanPrepayment,
  recordLoanEvent,
} from "./api";
import type { LoanEventWrite, PrepayPreviewResponse } from "./api";
import { Button, Card, Field, Progress, Select, Sheet, Term } from "../ui";
import { formatInr } from "../lib/money";

type EventKind = LoanEventWrite["kind"];

function money(
  value: string | null | undefined,
  currency: string,
  compact = false,
): string {
  if (value === null || value === undefined) return "Not available";
  if (currency === "INR") {
    return formatInr(value, compact ? "compact" : "full", compact ? 0 : 2);
  }
  return new Intl.NumberFormat(currency === "INR" ? "en-IN" : "en-DK", {
    style: "currency",
    currency,
    maximumFractionDigits: compact ? 0 : 2,
    minimumFractionDigits: compact ? 0 : 2,
  }).format(Number(value));
}

function monthLabel(
  value: { year: number; month: number } | null | undefined,
): string | null {
  if (!value) return null;
  const { year, month } = value;
  if (!year || !month) return null;
  return new Intl.DateTimeFormat("en-IN", {
    month: "short",
    year: "numeric",
  }).format(new Date(year, month - 1, 1));
}

function chartPath(points: { outstanding: string }[]): string {
  if (points.length < 2) return "";
  const values = points.map((point) => Number(point.outstanding));
  const maximum = Math.max(...values, 1);
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 300 + 10;
      const y = 90 - (value / maximum) * 78;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function eventLabel(kind: string): string {
  const labels: Record<string, string> = {
    prepayment: "Extra payment",
    rate_change: "New interest rate",
    emi_change: "New monthly payment",
    balance_update: "Statement balance",
    missed_emi: "Missed payment",
    closed: "Loan closed",
  };
  return labels[kind] ?? "Loan update";
}

function taxReliefLabel(value: string): string {
  const labels: Record<string, string> = {
    in_home_self_occupied: "Home loan interest",
    in_home_let_out: "Rental home loan interest",
    in_education: "Education loan interest",
    dk_interest_deduction: "Interest expenses in Denmark",
  };
  return labels[value] ?? "Tax relief";
}

export function LoanDetailPage() {
  const { id = "" } = useParams();
  const { isViewer } = useHousehold();
  const cache = useQueryClient();
  const [eventSheet, setEventSheet] = useState(false);
  const [previewSheet, setPreviewSheet] = useState(false);
  const [kind, setKind] = useState<EventKind>("prepayment");
  const [eventDate, setEventDate] = useState(
    new Date().toISOString().slice(0, 10),
  );
  const [eventAmount, setEventAmount] = useState("");
  const [eventRate, setEventRate] = useState("");
  const [eventNote, setEventNote] = useState("");
  const [extraAmount, setExtraAmount] = useState("");
  const [preview, setPreview] = useState<PrepayPreviewResponse | null>(null);
  const [formError, setFormError] = useState("");
  const loan = useQuery({
    queryKey: ["c30", "loan", id],
    queryFn: () => getLoan(id),
    enabled: Boolean(id),
  });
  const saved = () => {
    void cache.invalidateQueries({ queryKey: ["c30", "loan", id] });
    void cache.invalidateQueries({ queryKey: ["c30", "loans"] });
    void cache.invalidateQueries({ queryKey: ["c28", "plan"] });
  };
  const record = useMutation({
    mutationFn: (payload: LoanEventWrite) => recordLoanEvent(id, payload),
    onSuccess: () => {
      saved();
      setEventSheet(false);
      setEventAmount("");
      setEventRate("");
      setEventNote("");
      setFormError("");
    },
    onError: () => setFormError("We couldn't save this update. Try again."),
  });
  const remove = useMutation({
    mutationFn: (eventId: string) => deleteLoanEvent(id, eventId),
    onSuccess: saved,
  });
  const calculate = useMutation({
    mutationFn: () => previewLoanPrepayment(id, { amount: extraAmount }),
    onSuccess: (value) => {
      setPreview(value);
      setFormError("");
    },
    onError: () => setFormError("Add an amount and try again."),
  });

  if (loan.isPending)
    return <Card aria-label="Loading loan">Loading loan details…</Card>;
  if (loan.isError || !loan.data)
    return (
      <Card role="alert">
        <h2 className="font-semibold">This loan couldn't load</h2>
        <Link to="/money" className="mt-3 inline-block text-sm underline">
          Back to your Money Map
        </Link>
      </Card>
    );

  const item = loan.data;
  const points = item.yearly_balances;
  const payoff = monthLabel(item.payoff_month);
  const isAmountEvent = kind === "prepayment" || kind === "balance_update";
  const isRateEvent = kind === "rate_change";
  const isEmiEvent = kind === "emi_change";

  function submitEvent() {
    const payload: LoanEventWrite = { kind, on_date: eventDate };
    if (isAmountEvent) {
      if (!eventAmount || Number(eventAmount) < 0) {
        setFormError("Enter an amount first.");
        return;
      }
      payload.amount = eventAmount;
    }
    if (isRateEvent) {
      if (!eventRate || Number(eventRate) < 0) {
        setFormError("Enter an interest rate first.");
        return;
      }
      payload.rate_pct = eventRate;
    }
    if (isEmiEvent) {
      if (!eventAmount || Number(eventAmount) < 0) {
        setFormError("Enter the new monthly payment first.");
        return;
      }
      payload.emi = eventAmount;
    }
    if (eventNote.trim()) payload.note = eventNote.trim();
    record.mutate(payload);
  }

  return (
    <div className="grid gap-4" data-copy-screen="loan-detail">
      <Card className="grid gap-4">
        <Link
          to="/money"
          className="inline-flex min-h-12 items-center gap-2 text-sm text-muted-foreground"
        >
          <ArrowLeft aria-hidden="true" className="size-4" /> Your Money Map
        </Link>
        <header>
          <h2 className="text-xl font-semibold">{item.name}</h2>
          {item.lender ? (
            <p className="mt-1 text-sm text-muted-foreground">{item.lender}</p>
          ) : null}
        </header>
        <div>
          <p className="text-sm text-muted-foreground">Remaining</p>
          <p className="mt-1 text-3xl font-semibold tabular-nums">
            {money(item.outstanding, item.currency, true)}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {item.never_pays_off
              ? "Your monthly payment may not cover interest."
              : payoff
                ? `Paid off in ${payoff}`
                : "Add a monthly payment to estimate the end date."}
          </p>
          {item.percent_repaid !== null ? (
            <div className="mt-3 grid gap-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Amount repaid</span>
                <span>{item.percent_repaid}%</span>
              </div>
              <Progress
                value={Number(item.percent_repaid)}
                label="Amount repaid"
                quiet
              />
            </div>
          ) : null}
        </div>
        <svg
          viewBox="0 0 320 100"
          role="img"
          aria-label="Loan balance over time"
          className="h-28 w-full rounded-xl bg-muted/40 p-2"
        >
          {chartPath(points) ? (
            <path
              d={chartPath(points)}
              fill="none"
              stroke="currentColor"
              strokeWidth="3"
              className="text-brand"
            />
          ) : (
            <text
              x="12"
              y="52"
              className="fill-current text-xs text-muted-foreground"
            >
              More balance history will appear here.
            </text>
          )}
        </svg>
        {!isViewer ? (
          <Button
            type="button"
            data-primary="true"
            onClick={() => setEventSheet(true)}
          >
            Record a payment
          </Button>
        ) : null}
        <Button
          type="button"
          variant="secondary"
          onClick={() => {
            setPreview(null);
            setFormError("");
            setPreviewSheet(true);
          }}
        >
          What if I pay extra?
        </Button>
      </Card>

      <Card className="grid gap-3">
        <h3 className="text-base font-semibold">Recorded updates</h3>
        {item.events.length ? (
          item.events.map((event) => (
            <div
              key={event.id}
              className="flex items-start justify-between gap-3 border-t border-line pt-3 first:border-0 first:pt-0"
            >
              <div>
                <p className="text-sm font-medium">{eventLabel(event.kind)}</p>
                <p className="text-xs text-muted-foreground">
                  {event.on_date}
                  {event.amount
                    ? ` · ${money(event.amount, item.currency)}`
                    : ""}
                  {event.rate_pct ? ` · ${event.rate_pct}%` : ""}
                </p>
                {event.note ? (
                  <p className="mt-1 text-sm text-muted-foreground">
                    {event.note}
                  </p>
                ) : null}
              </div>
              {!isViewer ? (
                <Button
                  type="button"
                  variant="ghost"
                  aria-label={`Delete ${eventLabel(event.kind)}`}
                  disabled={remove.isPending}
                  onClick={() => remove.mutate(event.id)}
                >
                  <Trash2 aria-hidden="true" className="size-4" />
                </Button>
              ) : null}
            </div>
          ))
        ) : (
          <p className="text-sm text-muted-foreground">
            No updates have been recorded yet.
          </p>
        )}
      </Card>

      <details className="rounded-xl border border-line bg-surface p-4">
        <summary className="cursor-pointer text-sm font-semibold">
          Loan details
        </summary>
        <div className="grid gap-3 pt-4 text-sm">
          <p>
            Interest rate: <strong>{item.rate_pct}%</strong>
          </p>
          <p>
            <Term
              explanation="The fixed amount you pay each month."
              prompt="What's this?"
            >
              Monthly payment (EMI)
            </Term>
            : <strong>{money(item.emi, item.currency)}</strong>
            {item.rate_type
              ? ` · ${item.rate_type === "floating" ? "variable" : "fixed"} rate`
              : ""}
          </p>
          {item.tax_relief_needs_review ? (
            <div>
              <p>
                {taxReliefLabel(item.tax_relief)} may apply. Please check before
                acting.
              </p>
              {item.tax_source_urls.map((url) => (
                <a
                  key={url}
                  href={url}
                  target="_blank"
                  rel="noreferrer"
                  className="block break-all text-xs text-brand underline"
                >
                  Official tax information
                </a>
              ))}
            </div>
          ) : null}
          <div>
            <p className="mb-2 font-medium">Repayment schedule</p>
            {item.scheduled_payments.length ? (
              item.scheduled_payments.map((payment) => (
                <div
                  key={payment.on_date}
                  className="flex justify-between border-t border-line py-2 text-xs"
                >
                  <span>{payment.on_date}</span>
                  <span>{money(payment.amount, item.currency)}</span>
                </div>
              ))
            ) : (
              <p className="text-xs text-muted-foreground">
                Add a monthly payment to see future dates.
              </p>
            )}
          </div>
        </div>
      </details>

      <Sheet
        open={eventSheet}
        onOpenChange={setEventSheet}
        title="Record a payment or change"
        description="Use the latest statement or payment details."
      >
        <div className="grid gap-4">
          <Select
            id="loan-event-kind"
            label="What happened?"
            value={kind}
            onChange={(event) => {
              setKind(event.currentTarget.value as EventKind);
              setFormError("");
            }}
          >
            <option value="prepayment">Extra payment</option>
            <option value="rate_change">New interest rate</option>
            <option value="emi_change">New monthly payment</option>
            <option value="balance_update">Balance from a statement</option>
            <option value="missed_emi">Missed a payment</option>
          </Select>
          <Field
            id="loan-event-date"
            label="Date"
            required
            error={formError && !eventDate ? formError : undefined}
          >
            <input
              id="loan-event-date"
              type="date"
              value={eventDate}
              onChange={(event) => setEventDate(event.currentTarget.value)}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base"
            />
          </Field>
          {isAmountEvent || isEmiEvent ? (
            <Field
              id="loan-event-amount"
              label={
                isEmiEvent ? "New monthly payment" : `Amount (${item.currency})`
              }
              required
              prefix={
                item.currency === "DKK"
                  ? "kr"
                  : item.currency === "EUR"
                    ? "€"
                    : "₹"
              }
            >
              <input
                id="loan-event-amount"
                type="text"
                inputMode="decimal"
                value={eventAmount}
                onChange={(event) => setEventAmount(event.currentTarget.value)}
                className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 pl-12 text-base"
              />
            </Field>
          ) : null}
          {isRateEvent ? (
            <Field id="loan-event-rate" label="New interest rate (%)" required>
              <input
                id="loan-event-rate"
                type="text"
                inputMode="decimal"
                value={eventRate}
                onChange={(event) => setEventRate(event.currentTarget.value)}
                className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base"
              />
            </Field>
          ) : null}
          <Field id="loan-event-note" label="Note (optional)">
            <input
              id="loan-event-note"
              type="text"
              maxLength={140}
              value={eventNote}
              onChange={(event) => setEventNote(event.currentTarget.value)}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base"
            />
          </Field>
          {formError ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {formError}
            </p>
          ) : null}
          <Button
            type="button"
            disabled={record.isPending}
            onClick={submitEvent}
          >
            {record.isPending ? "Saving…" : "Save update"}
          </Button>
        </div>
      </Sheet>

      <Sheet
        open={previewSheet}
        onOpenChange={setPreviewSheet}
        title="What if you pay extra?"
        description="Compare two options before you decide."
      >
        <div className="grid gap-4">
          <Field
            id="loan-extra-amount"
            label={`Extra amount (${item.currency})`}
            required
            prefix={
              item.currency === "DKK"
                ? "kr"
                : item.currency === "EUR"
                  ? "€"
                  : "₹"
            }
          >
            <input
              id="loan-extra-amount"
              type="text"
              inputMode="decimal"
              value={extraAmount}
              onChange={(event) => setExtraAmount(event.currentTarget.value)}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 pl-12 text-base"
            />
          </Field>
          <Button
            type="button"
            variant="secondary"
            disabled={calculate.isPending || !extraAmount}
            onClick={() => calculate.mutate()}
          >
            Compare options
          </Button>
          {formError ? (
            <p role="alert" className="text-sm text-danger-foreground">
              {formError}
            </p>
          ) : null}
          {preview ? (
            <PrepayCards preview={preview} currency={item.currency} />
          ) : null}
        </div>
      </Sheet>
    </div>
  );
}

function PrepayCards({
  preview,
  currency,
}: {
  preview: PrepayPreviewResponse;
  currency: string;
}) {
  return (
    <div className="grid gap-3">
      <Card>
        <p className="font-semibold">
          Finish {preview.keep_emi_months_saved} months sooner
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          Save {money(preview.keep_emi_interest_saved, currency, true)} in
          interest.
        </p>
      </Card>
      <Card>
        <p className="font-semibold">
          Lower the monthly payment to{" "}
          {money(preview.keep_end_date_new_emi, currency, true)}
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          Save {money(preview.keep_end_date_interest_saved, currency, true)} in
          interest.
        </p>
      </Card>
    </div>
  );
}
