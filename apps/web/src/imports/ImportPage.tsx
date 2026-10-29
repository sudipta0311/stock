import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  CheckCircle2,
  FileUp,
  LoaderCircle,
  ShieldCheck,
  X,
} from "lucide-react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { ApiError } from "../api/client";
import { formatInr } from "../lib/money";
import { Badge, Button, Card, Sheet, Term } from "../ui";
import { useHousehold } from "../household/HouseholdContext";
import { listAccounts } from "../money/api";
import type { AccountResponse } from "../money/api";
import {
  confirmImport,
  discardImport,
  getImport,
  mapImportColumns,
  patchImportItem,
  uploadImport,
} from "./api";
import type {
  ImportDetail,
  ImportFormat,
  ImportItem,
  ImportMappingRequest,
} from "./api";
import { importFeatureSupport } from "./features";
import {
  categoryForItem,
  errorCode,
  importErrorCopy,
  maskAccountReference,
  quantityChange,
  sumImportCounts,
  validateImportFile,
} from "./domain";
import type { ImportCategory } from "./domain";
import {
  importProgressReducer,
  initialImportProgress,
  progressLabel,
} from "./progress";
import { takeSharedFile } from "./share-target";
import { markValueMoment } from "../pwa/install";
import englishCopy from "../copy/en.json";

const sourceCopy: Record<
  ImportFormat,
  { title: string; help: string; steps?: string[]; note?: string }
> = {
  nsdl_cas: {
    title: "Depository statement (PDF)",
    help: "Your monthly statement usually arrives by email. You can also request it from your depository.",
  },
  cams_kfin_cas: {
    title: englishCopy.fund_check.cams_title,
    help: englishCopy.fund_check.cams_request,
    steps: englishCopy.fund_check.cams_steps,
    note: englishCopy.fund_check.cams_summary_note,
  },
  broker_statement: {
    title: englishCopy.imports.broker_title,
    help: englishCopy.imports.broker_help,
  },
};

const FILE_PICKER_ACCEPT = importFeatureSupport.broker
  ? ".pdf,.csv,.xlsx,application/pdf,text/csv"
  : ".pdf,application/pdf";

const brokerOptions = [
  ["nordnet", englishCopy.imports.broker_nordnet],
  ["saxo", englishCopy.imports.broker_saxo],
  ["danske_bank", englishCopy.imports.broker_danske],
  ["ibkr", englishCopy.imports.broker_ibkr],
  ["vested", englishCopy.imports.broker_vested],
  ["indmoney", englishCopy.imports.broker_indmoney],
  ["other", englishCopy.imports.broker_other],
] as const;

function sourceLabel(source: string | null): string {
  switch (source) {
    case "nsdl":
      return "Depository statement";
    case "cdsl":
      return "Depository statement";
    case "cams_kfin":
      return "CAMS / KFintech statement";
    case "broker":
      return englishCopy.imports.broker_source_label;
    default:
      return "Statement import";
  }
}

function displayDate(value: string | null): string {
  if (!value) return "Date not available";
  const date = new Date(`${value.slice(0, 10)}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

function initialSelection(item: ImportItem): boolean {
  return ["new", "changed", "accepted", "enrich", "replace_with_lots"].includes(
    item.status,
  );
}

function readableSleeve(sleeve: string | null): string {
  if (!sleeve) return englishCopy.imports.category_needs_choice;
  const labels: Record<string, string> = {
    equity_india: "Equity India",
    equity_global: "Equity Global",
    debt: "Debt",
    gold: "Gold",
    cash: "Cash",
    real_estate: "Property",
    protection: "Protection",
  };
  return labels[sleeve] ?? sleeve.replaceAll("_", " ");
}

export function ImportPage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [format, setFormat] = useState<ImportFormat>("nsdl_cas");
  const { members, meMemberId, memberLabel } = useHousehold();
  const [accounts, setAccounts] = useState<AccountResponse[]>([]);
  const [selectedAccountId, setSelectedAccountId] = useState("");
  const [selectedMemberId, setSelectedMemberId] = useState("");
  const [broker, setBroker] = useState("other");
  const [file, setFile] = useState<File | null>(null);
  const [password, setPassword] = useState("");
  const [problem, setProblem] = useState<{
    message: string;
    nextStep: string;
  } | null>(null);
  const [progress, dispatchProgress] = useReducer(
    importProgressReducer,
    initialImportProgress,
  );
  const [canCancel, setCanCancel] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const progressTimerRef = useRef<number | undefined>(undefined);
  const sharedReadRef = useRef(false);
  const busy = progress.phase !== "idle";
  const inFlight =
    progress.phase === "uploading" || progress.phase === "reading";

  useEffect(() => {
    if (meMemberId && !selectedMemberId) setSelectedMemberId(meMemberId);
  }, [meMemberId, selectedMemberId]);

  useEffect(() => {
    if (format !== "broker_statement") return;
    let active = true;
    void listAccounts()
      .then((rows) => {
        if (active) setAccounts(rows);
      })
      .catch(() => {
        if (active) setAccounts([]);
      });
    return () => {
      active = false;
    };
  }, [format]);

  useEffect(() => {
    const onWaking = (event: Event) => {
      const detail = (event as CustomEvent<{ waking?: boolean }>).detail;
      dispatchProgress({ type: "waking", value: detail?.waking === true });
    };
    window.addEventListener("wealth:api-waking", onWaking);
    return () => window.removeEventListener("wealth:api-waking", onWaking);
  }, []);

  useEffect(() => {
    if (searchParams.get("shared") !== "1" || sharedReadRef.current) return;
    sharedReadRef.current = true;
    void takeSharedFile()
      .then((sharedFile) => {
        if (!sharedFile) return;
        const fileProblem = validateImportFile(
          sharedFile,
          importFeatureSupport.broker,
        );
        if (fileProblem) {
          setProblem(fileProblem);
          return;
        }
        setFile(sharedFile);
      })
      .catch(() => {
        setProblem({
          message: "We couldn't open the shared file.",
          nextStep: "Choose the statement with the file picker instead.",
        });
      })
      .finally(() => {
        setSearchParams({}, { replace: true });
      });
  }, [searchParams, setSearchParams]);

  useEffect(() => {
    const shareError = searchParams.get("shareError");
    if (!shareError) return;
    setProblem(
      shareError === "missing-file"
        ? {
            message: "No statement file came through.",
            nextStep:
              "Choose your statement with the file picker and try again.",
          }
        : {
            message: "We couldn't keep the shared file ready.",
            nextStep:
              "Choose your statement with the file picker and try again.",
          },
    );
    setSearchParams({}, { replace: true });
  }, [searchParams, setSearchParams]);

  function selectFile(selected: File | undefined) {
    if (!selected) return;
    setProblem(null);
    const fileProblem = validateImportFile(
      selected,
      importFeatureSupport.broker,
    );
    if (fileProblem) {
      setFile(null);
      setProblem(fileProblem);
      return;
    }
    setFile(selected);
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || busy) return;
    setProblem(null);
    let oneTimePassword = password;
    setPassword("");
    dispatchProgress({ type: "start" });
    setCanCancel(true);
    const controller = new AbortController();
    controllerRef.current = controller;
    progressTimerRef.current = window.setTimeout(
      () => dispatchProgress({ type: "reading" }),
      1200,
    );
    const request = uploadImport(
      file,
      oneTimePassword,
      format,
      controller.signal,
      format === "broker_statement"
        ? {
            accountId: selectedAccountId || undefined,
            memberId: selectedAccountId
              ? undefined
              : selectedMemberId || undefined,
            broker,
          }
        : undefined,
    );
    oneTimePassword = "";
    try {
      const result = await request;
      window.clearTimeout(progressTimerRef.current);
      setCanCancel(false);
      dispatchProgress({ type: "reading" });
      await new Promise<void>((resolve) => {
        progressTimerRef.current = window.setTimeout(resolve, 300);
      });
      dispatchProgress({ type: "ready" });
      progressTimerRef.current = window.setTimeout(
        () => navigate(`/import/${encodeURIComponent(result.job_id)}`),
        350,
      );
    } catch (error) {
      window.clearTimeout(progressTimerRef.current);
      dispatchProgress({ type: "cancel" });
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        const copy = importErrorCopy(errorCode(error));
        setProblem(copy);
      }
    } finally {
      setCanCancel(false);
      if (controllerRef.current === controller) controllerRef.current = null;
    }
  }

  function cancelUpload() {
    if (!canCancel) return;
    window.clearTimeout(progressTimerRef.current);
    controllerRef.current?.abort();
    controllerRef.current = null;
    setCanCancel(false);
    dispatchProgress({ type: "cancel" });
  }

  return (
    <div data-copy-screen="import" className="grid gap-4">
      <Card className="grid gap-4">
        <div>
          <h2 className="text-xl font-semibold">
            Upload your{" "}
            <Term
              explanation="A combined statement of your investment accounts."
              prompt="What's this?"
            >
              CAS
            </Term>{" "}
            statement
          </h2>
          <p className="mt-1 text-sm leading-6 text-muted-foreground">
            Choose a statement. We will show every holding before you save
            anything.
          </p>
        </div>

        <fieldset className="grid gap-2">
          <legend className="mb-2 text-sm font-medium">
            What are you uploading?
          </legend>
          {(Object.keys(sourceCopy) as ImportFormat[]).map((option) => {
            const disabled =
              option === "broker_statement" && !importFeatureSupport.broker;
            return (
              <div
                key={option}
                className={`rounded-2xl border p-3 ${format === option ? "border-brand bg-brand/5" : "border-line bg-background"} ${disabled ? "opacity-70" : ""}`}
              >
                <button
                  type="button"
                  aria-pressed={format === option}
                  disabled={disabled}
                  onClick={() => {
                    setFormat(option);
                    setFile(null);
                    setProblem(null);
                  }}
                  className="flex min-h-10 w-full items-center justify-between gap-3 text-left text-sm font-semibold focus-visible:outline-2 focus-visible:outline-brand disabled:cursor-not-allowed"
                >
                  <span>{sourceCopy[option].title}</span>
                  {disabled ? <Badge>Coming soon</Badge> : null}
                  {!disabled && format === option ? (
                    <CheckCircle2
                      aria-hidden="true"
                      className="size-5 text-brand"
                    />
                  ) : null}
                </button>
                <details className="mt-1 text-sm text-muted-foreground">
                  <summary className="min-h-10 cursor-pointer py-2 text-foreground">
                    How to get it
                  </summary>
                  <p className="pb-2 leading-6">{sourceCopy[option].help}</p>
                  {sourceCopy[option].steps ? (
                    <ol className="list-decimal space-y-1 pb-2 pl-5 leading-6">
                      {sourceCopy[option].steps.map((step) => (
                        <li key={step}>{step}</li>
                      ))}
                    </ol>
                  ) : null}
                  {sourceCopy[option].note ? (
                    <p className="pb-2 leading-6">{sourceCopy[option].note}</p>
                  ) : null}
                </details>
              </div>
            );
          })}
        </fieldset>

        <form className="grid gap-4" onSubmit={(event) => void submit(event)}>
          <label
            className="grid gap-1.5 text-sm font-medium"
            htmlFor="statement-file"
          >
            Statement file
            <input
              id="statement-file"
              type="file"
              accept={FILE_PICKER_ACCEPT}
              disabled={busy}
              onChange={(event) => selectFile(event.currentTarget.files?.[0])}
              className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 py-2 text-sm file:mr-3 file:min-h-8 file:rounded-lg file:border-0 file:bg-muted file:px-3 file:font-semibold"
            />
            <span className="font-normal text-muted-foreground">
              {format === "broker_statement"
                ? englishCopy.imports.broker_upload_hint
                : "PDF up to 10 MB"}
              {file ? ` · ${file.name}` : ""}
            </span>
          </label>

          {format === "broker_statement" ? (
            <>
              <label
                className="grid gap-1.5 text-sm font-medium"
                htmlFor="broker-name"
              >
                {englishCopy.imports.broker_picker_label}
                <select
                  id="broker-name"
                  value={broker}
                  onChange={(event) => setBroker(event.currentTarget.value)}
                  disabled={busy}
                  className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                >
                  {brokerOptions.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label
                className="grid gap-1.5 text-sm font-medium"
                htmlFor="broker-account"
              >
                {englishCopy.imports.broker_account_label}
                <select
                  id="broker-account"
                  value={selectedAccountId}
                  onChange={(event) =>
                    setSelectedAccountId(event.currentTarget.value)
                  }
                  disabled={busy}
                  className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                >
                  <option value="">
                    {englishCopy.imports.broker_no_account}
                  </option>
                  {accounts.map((account) => (
                    <option key={account.id} value={account.id}>
                      {account.institution} · {account.account_type}
                    </option>
                  ))}
                </select>
              </label>
              {!selectedAccountId ? (
                <label
                  className="grid gap-1.5 text-sm font-medium"
                  htmlFor="broker-member"
                >
                  {englishCopy.imports.broker_member_label}
                  <select
                    id="broker-member"
                    value={selectedMemberId || meMemberId || ""}
                    onChange={(event) =>
                      setSelectedMemberId(event.currentTarget.value)
                    }
                    disabled={busy || members.length === 0}
                    className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
                  >
                    {members.map((member) => (
                      <option key={member.id} value={member.id}>
                        {memberLabel(member)}
                      </option>
                    ))}
                  </select>
                </label>
              ) : null}
            </>
          ) : null}

          {format !== "broker_statement" ? (
            <label
              className="grid gap-1.5 text-sm font-medium"
              htmlFor="statement-password"
            >
              Statement password{" "}
              <span className="font-normal text-muted-foreground">
                (if required)
              </span>
              <input
                id="statement-password"
                type="password"
                autoComplete="off"
                value={password}
                onChange={(event) => setPassword(event.currentTarget.value)}
                disabled={busy}
                className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30"
                aria-describedby="statement-password-help"
              />
              <span
                id="statement-password-help"
                className="font-normal text-muted-foreground"
              >
                {englishCopy.imports.holdings_password_help}
              </span>
            </label>
          ) : null}

          {problem ? (
            <div
              role="alert"
              className="rounded-xl border border-danger/40 bg-danger-soft p-3 text-sm text-danger-foreground"
            >
              <p className="font-semibold">{problem.message}</p>
              <p className="mt-1">{problem.nextStep}</p>
            </div>
          ) : null}

          {busy ? (
            <div
              className="grid gap-3 rounded-xl border border-line bg-background p-3"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-center gap-2 text-sm font-semibold">
                {progress.phase === "ready" ? (
                  <CheckCircle2
                    aria-hidden="true"
                    className="size-4 text-brand"
                  />
                ) : (
                  <LoaderCircle
                    aria-hidden="true"
                    className="size-4 animate-spin text-brand"
                  />
                )}
                {progressLabel(progress.phase)}
              </div>
              <ol className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-3">
                {(["uploading", "reading", "ready"] as const).map((phase) => (
                  <li
                    key={phase}
                    className={
                      progress.phase === phase ? "font-semibold text-brand" : ""
                    }
                  >
                    {progress.phase === phase ? "● " : "○ "}
                    {progressLabel(phase)}
                  </li>
                ))}
              </ol>
              {progress.waking ? (
                <Badge tone="warning">Waking up the import service</Badge>
              ) : null}
              {canCancel ? (
                <Button
                  type="button"
                  variant="secondary"
                  onClick={cancelUpload}
                >
                  <X aria-hidden="true" className="size-4" /> Cancel upload
                </Button>
              ) : null}
            </div>
          ) : null}

          <Button
            type="submit"
            data-primary="true"
            disabled={!file || busy}
            loading={inFlight}
          >
            <FileUp aria-hidden="true" className="size-4" /> Upload statement
          </Button>
          <p className="text-center text-xs leading-5 text-muted-foreground">
            Your file and password are used once to read your holdings and are
            not stored.
          </p>
        </form>
      </Card>

      <Card className="flex gap-3 text-sm text-muted-foreground">
        <ShieldCheck
          aria-hidden="true"
          className="mt-0.5 size-5 shrink-0 text-brand"
        />
        <p>
          The file is cleared after it is read. Your password is sent only with
          this upload.
        </p>
      </Card>
    </div>
  );
}

const kindOptions = [
  ["equity", "Equity"],
  ["etf", "Exchange traded fund"],
  ["mf", "Mutual fund"],
  ["bond", "Bond"],
  ["sgb", "Sovereign gold bond"],
] as const;

const sleeveOptions = [
  ["equity_india", "Equity India"],
  ["equity_global", "Equity Global"],
  ["debt", "Debt"],
  ["gold", "Gold"],
  ["cash", "Cash"],
  ["real_estate", "Property"],
  ["protection", "Protection"],
] as const;

const categories: ImportCategory[] = ["new", "changed", "duplicate", "unknown"];

function ItemToggle({
  item,
  checked,
  disabled,
  onChange,
}: {
  item: ImportItem;
  checked: boolean;
  disabled: boolean;
  onChange(checked: boolean): void;
}) {
  const id = `include-${item.id}`;
  const actionNote =
    item.action === "enrich"
      ? englishCopy.imports.item_enrich
      : item.action === "replace_with_lots"
        ? englishCopy.imports.item_replace_with_lots
        : item.action === "quantity_mismatch"
          ? item.status === "accepted"
            ? englishCopy.imports.use_tradebook
            : item.status === "rejected"
              ? englishCopy.imports.keep_statement
              : englishCopy.imports.item_quantity_mismatch
          : null;
  return (
    <label
      htmlFor={id}
      className="flex min-h-12 items-center gap-3 border-t border-line pt-2 text-sm"
    >
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.currentTarget.checked)}
        className="size-5 accent-brand"
      />
      <span className="min-w-0 flex-1">
        <strong className="block truncate" data-security>
          {item.name}
        </strong>
        <span className="block text-xs text-muted-foreground">
          {item.quantity} units ·{" "}
          {item.statement_value
            ? formatValue(item.statement_value, item.currency)
            : "Value not provided"}
        </span>
        {actionNote ? (
          <span className="block text-xs text-muted-foreground">
            {actionNote}
          </span>
        ) : null}
      </span>
      <Badge>
        {item.action
          ? englishCopy.imports.broker_security_label
          : readableSleeve(item.sleeve)}
      </Badge>
    </label>
  );
}

function TradebookMismatchChoice({
  item,
  disabled,
  onChange,
}: {
  item: ImportItem;
  disabled: boolean;
  onChange(useTradebook: boolean): void;
}) {
  const choice =
    item.status === "accepted"
      ? "tradebook"
      : item.status === "rejected"
        ? "current"
        : "";
  return (
    <fieldset className="grid gap-2 border-t border-line pt-3 text-sm">
      <legend className="font-medium">
        {englishCopy.imports.quantity_choice_label}
      </legend>
      <label className="flex min-h-10 items-center gap-2">
        <input
          type="radio"
          name={`quantity-choice-${item.id}`}
          value="current"
          checked={choice === "current"}
          disabled={disabled}
          onChange={() => onChange(false)}
          className="size-4 accent-brand"
        />
        {englishCopy.imports.keep_statement}
      </label>
      <label className="flex min-h-10 items-center gap-2">
        <input
          type="radio"
          name={`quantity-choice-${item.id}`}
          value="tradebook"
          checked={choice === "tradebook"}
          disabled={disabled}
          onChange={() => onChange(true)}
          className="size-4 accent-brand"
        />
        {englishCopy.imports.use_tradebook}
      </label>
    </fieldset>
  );
}

function formatValue(value: string, currency: string | null = "INR"): string {
  if (currency && currency !== "INR") {
    const amount = Number(value);
    return Number.isFinite(amount)
      ? `${new Intl.NumberFormat("en", {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }).format(amount)} ${currency}`
      : `${value} ${currency}`;
  }
  try {
    return formatInr(value, "full", 2);
  } catch {
    return value;
  }
}

export function ImportReviewPage() {
  const { jobId = "" } = useParams();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<ImportDetail | null>(null);
  const [categoryById, setCategoryById] = useState<
    Record<string, ImportCategory>
  >({});
  const [selectedById, setSelectedById] = useState<Record<string, boolean>>({});
  const [kindById, setKindById] = useState<Record<string, string>>({});
  const [sleeveById, setSleeveById] = useState<Record<string, string>>({});
  const [pageError, setPageError] = useState("");
  const [itemError, setItemError] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [discardOpen, setDiscardOpen] = useState(false);
  const [confirmedCount, setConfirmedCount] = useState<number | null>(null);
  const [alreadyConfirmed, setAlreadyConfirmed] = useState(false);
  const [taxWrapper, setTaxWrapper] = useState<"ask" | "ordinary" | "pension">(
    "ordinary",
  );
  const [cashFlowsSaved, setCashFlowsSaved] = useState(false);

  useEffect(() => {
    let active = true;
    setLoading(true);
    getImport(jobId)
      .then((loaded) => {
        if (!active) return;
        setDetail(loaded);
        setTaxWrapper(loaded.tax_wrapper ?? "ordinary");
        if (loaded.status === "confirmed") {
          setAlreadyConfirmed(true);
          markValueMoment();
        }
        setCategoryById(
          Object.fromEntries(
            loaded.items.map((item) => [item.id, categoryForItem(item)]),
          ),
        );
        setSelectedById(
          Object.fromEntries(
            loaded.items.map((item) => [item.id, initialSelection(item)]),
          ),
        );
        setKindById(
          Object.fromEntries(loaded.items.map((item) => [item.id, item.kind])),
        );
        setSleeveById(
          Object.fromEntries(
            loaded.items.map((item) => [
              item.id,
              item.sleeve ?? "equity_india",
            ]),
          ),
        );
      })
      .catch(() => {
        if (active)
          setPageError("We couldn't load this import. Please try again.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [jobId]);

  const grouped = useMemo(() => {
    const groups: Record<ImportCategory, ImportItem[]> = {
      new: [],
      changed: [],
      duplicate: [],
      unknown: [],
    };
    for (const item of detail?.items ?? [])
      groups[categoryById[item.id] ?? categoryForItem(item)].push(item);
    return groups;
  }, [categoryById, detail]);
  const counts = useMemo(
    () =>
      Object.fromEntries(
        categories.map((category) => [category, grouped[category].length]),
      ) as Record<ImportCategory, number>,
    [grouped],
  );
  const selectedCount = useMemo(
    () =>
      Object.entries(selectedById).filter(([, selected]) => selected).length,
    [selectedById],
  );
  const canConfirm =
    selectedCount > 0 ||
    (detail?.cash_flow_count ?? 0) > 0 ||
    grouped.changed.some(
      (item) =>
        item.action === "quantity_mismatch" && item.status === "rejected",
    );

  async function updateStatus(item: ImportItem, accepted: boolean) {
    if (!detail) return;
    setItemError("");
    try {
      const updated = await patchImportItem(detail.id, item.id, {
        status: accepted ? "accepted" : "rejected",
      });
      setDetail(
        (current) =>
          current && {
            ...current,
            items: current.items.map((currentItem) =>
              currentItem.id === updated.id ? updated : currentItem,
            ),
          },
      );
      setSelectedById((current) => ({ ...current, [item.id]: accepted }));
    } catch {
      setItemError("We couldn't update that holding. Please try again.");
    }
  }

  async function acceptUnknown(item: ImportItem) {
    if (!detail) return;
    const kind = kindById[item.id] ?? "equity";
    const sleeve = sleeveById[item.id] ?? "equity_india";
    setItemError("");
    try {
      const updated = await patchImportItem(detail.id, item.id, {
        status: "accepted",
        override: {
          kind: kind as "equity" | "etf" | "mf" | "bond" | "sgb",
          sleeve: sleeve as
            | "equity_india"
            | "equity_global"
            | "debt"
            | "gold"
            | "cash"
            | "real_estate"
            | "protection",
        },
      });
      setDetail(
        (current) =>
          current && {
            ...current,
            items: current.items.map((currentItem) =>
              currentItem.id === updated.id ? updated : currentItem,
            ),
          },
      );
      setCategoryById((current) => ({ ...current, [item.id]: "new" }));
      setSelectedById((current) => ({ ...current, [item.id]: true }));
    } catch {
      setItemError(
        "We couldn't classify that holding. Check the selection and try again.",
      );
    }
  }

  async function selectAllNew() {
    if (!detail) return;
    const newItems = grouped.new;
    setSaving(true);
    setItemError("");
    try {
      const updated = await Promise.all(
        newItems.map((item) =>
          patchImportItem(detail.id, item.id, { status: "accepted" }),
        ),
      );
      setDetail(
        (current) =>
          current && {
            ...current,
            items: current.items.map(
              (item) =>
                updated.find((candidate) => candidate.id === item.id) ?? item,
            ),
          },
      );
      setSelectedById((current) => ({
        ...current,
        ...Object.fromEntries(newItems.map((item) => [item.id, true])),
      }));
    } catch {
      setItemError("We couldn't select every new holding. Please try again.");
    } finally {
      setSaving(false);
    }
  }

  async function confirm() {
    if (!detail || !canConfirm) return;
    setSaving(true);
    setPageError("");
    try {
      const result = await confirmImport(
        detail.id,
        detail.tax_wrapper_question ? { tax_wrapper: taxWrapper } : {},
      );
      setAlreadyConfirmed(false);
      setConfirmedCount(
        sumImportCounts(
          Object.fromEntries(
            Object.entries(result.counts).filter(
              ([key]) => key !== "cash_flows",
            ),
          ),
        ),
      );
      setCashFlowsSaved((result.counts.cash_flows ?? 0) > 0);
      markValueMoment();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setAlreadyConfirmed(true);
        setConfirmedCount(0);
      } else {
        setPageError("We couldn't confirm this import. Please try again.");
      }
    } finally {
      setSaving(false);
    }
  }

  async function discard() {
    if (!detail) return;
    setSaving(true);
    try {
      await discardImport(detail.id);
      navigate("/money", { replace: true });
    } catch {
      setPageError("We couldn't discard this import. Please try again.");
      setDiscardOpen(false);
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return <Card role="status">Loading your statement review…</Card>;
  }
  if (pageError && !detail) {
    return (
      <Card role="alert" className="text-sm">
        {pageError}
      </Card>
    );
  }
  if (!detail) return null;

  if (detail.status === "needs_mapping") {
    return (
      <BrokerMappingPanel
        detail={detail}
        onMapped={(mapped) => {
          setDetail(mapped);
          setTaxWrapper(mapped.tax_wrapper ?? "ordinary");
          setCategoryById(
            Object.fromEntries(
              mapped.items.map((item) => [item.id, categoryForItem(item)]),
            ),
          );
          setSelectedById(
            Object.fromEntries(
              mapped.items.map((item) => [item.id, initialSelection(item)]),
            ),
          );
        }}
      />
    );
  }

  if (
    alreadyConfirmed ||
    confirmedCount !== null ||
    detail.status === "confirmed"
  ) {
    const added = confirmedCount ?? 0;
    const wasAlreadyConfirmed =
      alreadyConfirmed || detail.status === "confirmed";
    return (
      <Card className="grid gap-3 text-center">
        <CheckCircle2
          aria-hidden="true"
          className="mx-auto size-10 text-success-foreground"
        />
        <h2 className="text-xl font-semibold">
          {wasAlreadyConfirmed
            ? "This import is already confirmed"
            : added > 0
              ? `Added ${added} holdings · Net worth updated`
              : cashFlowsSaved
                ? englishCopy.imports.cash_flow_saved_title
                : englishCopy.imports.confirmed_zero_title}
        </h2>
        {added === 0 ? (
          <p className="text-sm text-muted-foreground">
            {wasAlreadyConfirmed
              ? "Your Money Map already includes this statement."
              : cashFlowsSaved
                ? englishCopy.imports.cash_flow_saved_description
                : englishCopy.imports.confirmed_zero_description}
          </p>
        ) : null}
        <Link
          to="/money"
          className="inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 text-sm font-semibold text-white"
        >
          Go to Money
        </Link>
      </Card>
    );
  }

  if (detail.status === "failed") {
    const copy = importErrorCopy(detail.error_code ?? "unknown");
    return (
      <Card className="grid gap-2" role="alert">
        <h2 className="font-semibold">{copy.message}</h2>
        <p className="text-sm text-muted-foreground">{copy.nextStep}</p>
        <Link
          to="/import"
          className="mt-2 inline-flex min-h-12 items-center justify-center rounded-xl bg-brand px-4 text-sm font-semibold text-white"
        >
          Try another statement
        </Link>
      </Card>
    );
  }

  const countsList: Array<[ImportCategory, string]> = [
    ["new", "New"],
    ["changed", "Changed"],
    ["duplicate", "Duplicate"],
    ["unknown", "Unknown"],
  ];

  return (
    <div data-copy-screen="import" className="grid gap-4 pb-36">
      <Card className="grid gap-3">
        <div>
          <h2 className="text-xl font-semibold">Review your import</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {sourceLabel(detail.source)} · {displayDate(detail.statement_date)}
          </p>
        </div>
        <div className="flex flex-wrap gap-2" aria-label="Import item counts">
          {countsList.map(([category, label]) => (
            <Badge
              key={category}
              tone={
                category === "unknown" && counts[category]
                  ? "warning"
                  : "neutral"
              }
            >
              {label} {counts[category]}
            </Badge>
          ))}
        </div>
        <p className="text-sm text-muted-foreground">
          Your file and password were not stored.
        </p>
        {detail.cash_flow_count > 0 ? (
          <p className="text-sm text-muted-foreground">
            {englishCopy.imports.cash_flow_count.replace(
              "{count}",
              String(detail.cash_flow_count),
            )}
          </p>
        ) : null}
      </Card>

      {detail.tax_wrapper_question ? (
        <Card>
          <fieldset className="grid gap-2 text-sm">
            <legend className="font-medium">
              {englishCopy.imports.tax_wrapper_question}
            </legend>
            <label className="flex min-h-10 items-center gap-2">
              <input
                type="radio"
                name="tax-wrapper"
                value="ask"
                checked={taxWrapper === "ask"}
                disabled={saving}
                onChange={() => setTaxWrapper("ask")}
                className="size-4 accent-brand"
              />
              {englishCopy.imports.tax_wrapper_yes}
            </label>
            <label className="flex min-h-10 items-center gap-2">
              <input
                type="radio"
                name="tax-wrapper"
                value="ordinary"
                checked={taxWrapper === "ordinary"}
                disabled={saving}
                onChange={() => setTaxWrapper("ordinary")}
                className="size-4 accent-brand"
              />
              {englishCopy.imports.tax_wrapper_no}
            </label>
          </fieldset>
        </Card>
      ) : null}

      {pageError ? (
        <Card role="alert" className="border-danger/50 text-sm">
          {pageError}
        </Card>
      ) : null}
      {itemError ? (
        <Card role="alert" className="border-danger/50 text-sm">
          {itemError}
        </Card>
      ) : null}

      <section aria-labelledby="new-heading" className="grid gap-3">
        <Card className="grid gap-2">
          <div className="flex items-center justify-between gap-3">
            <h3 id="new-heading" className="font-semibold">
              New · {counts.new}
            </h3>
            {grouped.new.length ? (
              <Button
                type="button"
                variant="ghost"
                className="min-h-10 px-2"
                disabled={saving}
                onClick={() => void selectAllNew()}
              >
                Select all new
              </Button>
            ) : null}
          </div>
          {grouped.new.length ? (
            grouped.new.map((item) => (
              <div key={item.id} className="grid gap-1">
                <ItemToggle
                  item={item}
                  checked={selectedById[item.id] ?? false}
                  disabled={saving}
                  onChange={(checked) => void updateStatus(item, checked)}
                />
                <p className="pl-8 text-xs text-muted-foreground">
                  {maskAccountReference(item.account_ref)}
                </p>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">No new holdings.</p>
          )}
        </Card>
      </section>

      <section aria-labelledby="changed-heading" className="grid gap-3">
        <Card className="grid gap-2">
          <h3 id="changed-heading" className="font-semibold">
            Changed · {counts.changed}
          </h3>
          {grouped.changed.length ? (
            grouped.changed.map((item) => (
              <div key={item.id} className="grid gap-1">
                {item.action === "quantity_mismatch" ? (
                  <>
                    <div className="flex items-center gap-3 border-t border-line pt-2 text-sm">
                      <span className="min-w-0 flex-1">
                        <strong className="block truncate" data-security>
                          {item.name}
                        </strong>
                        <span className="block text-xs text-muted-foreground">
                          {item.quantity} units
                        </span>
                      </span>
                      <Badge>{englishCopy.imports.broker_security_label}</Badge>
                    </div>
                    <TradebookMismatchChoice
                      item={item}
                      disabled={saving}
                      onChange={(useTradebook) =>
                        void updateStatus(item, useTradebook)
                      }
                    />
                  </>
                ) : (
                  <ItemToggle
                    item={item}
                    checked={selectedById[item.id] ?? false}
                    disabled={saving}
                    onChange={(checked) => void updateStatus(item, checked)}
                  />
                )}
                <p className="pl-8 text-xs text-muted-foreground">
                  Quantity: {quantityChange(item)} ·{" "}
                  {maskAccountReference(item.account_ref)}
                </p>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">
              No changed holdings.
            </p>
          )}
        </Card>
      </section>

      <section aria-labelledby="duplicate-heading">
        <Card>
          <details>
            <summary
              className="flex min-h-12 cursor-pointer items-center justify-between gap-3 font-semibold"
              id="duplicate-heading"
            >
              <span>Duplicate · {counts.duplicate}</span>
              <span className="text-sm font-normal text-muted-foreground">
                Already up to date
              </span>
            </summary>
            {grouped.duplicate.length ? (
              <ul className="grid gap-2 border-t border-line pt-3">
                {grouped.duplicate.map((item) => (
                  <li key={item.id} className="text-sm">
                    <strong data-security>{item.name}</strong>
                    <span className="block text-xs text-muted-foreground">
                      {item.quantity} units ·{" "}
                      {maskAccountReference(item.account_ref)}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="pt-2 text-sm text-muted-foreground">
                No duplicate holdings.
              </p>
            )}
          </details>
        </Card>
      </section>

      <section aria-labelledby="unknown-heading" className="grid gap-3">
        <Card className="grid gap-3">
          <div>
            <h3 id="unknown-heading" className="font-semibold">
              Unknown · {counts.unknown}
            </h3>
            <p className="mt-1 text-sm text-muted-foreground">
              {detail.source === "broker"
                ? englishCopy.imports.mapping_unknown_help
                : "Choose the holding type, or leave it out of this import."}
            </p>
          </div>
          {grouped.unknown.length ? (
            grouped.unknown.map((item) => (
              <div
                key={item.id}
                className="grid gap-3 border-t border-line pt-3"
              >
                <div>
                  <strong className="text-sm" data-security>
                    {item.name}
                  </strong>
                  <p className="text-xs text-muted-foreground">
                    {item.quantity} units ·{" "}
                    {item.statement_value
                      ? formatValue(item.statement_value, item.currency)
                      : "Value not provided"}{" "}
                    · {maskAccountReference(item.account_ref)}
                  </p>
                </div>
                {detail.source === "broker" ? (
                  <p className="text-sm text-muted-foreground">
                    {englishCopy.imports.mapping_unknown_heading}
                  </p>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    <label
                      className="grid gap-1 text-xs font-medium"
                      htmlFor={`kind-${item.id}`}
                    >
                      Kind
                      <select
                        id={`kind-${item.id}`}
                        value={kindById[item.id] ?? "equity"}
                        onChange={(event) => {
                          const value = event.currentTarget.value;
                          setKindById((current) => ({
                            ...current,
                            [item.id]: value,
                          }));
                        }}
                        className="min-h-12 rounded-xl border border-line bg-surface px-2 text-sm"
                      >
                        {kindOptions.map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label
                      className="grid gap-1 text-xs font-medium"
                      htmlFor={`sleeve-${item.id}`}
                    >
                      {englishCopy.imports.category_label}
                      <select
                        id={`sleeve-${item.id}`}
                        value={sleeveById[item.id] ?? "equity_india"}
                        onChange={(event) => {
                          const value = event.currentTarget.value;
                          setSleeveById((current) => ({
                            ...current,
                            [item.id]: value,
                          }));
                        }}
                        className="min-h-12 rounded-xl border border-line bg-surface px-2 text-sm"
                      >
                        {sleeveOptions.map(([value, label]) => (
                          <option key={value} value={value}>
                            {label}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                )}
                {detail.source === "broker" ? (
                  <Button
                    type="button"
                    variant="secondary"
                    disabled={saving}
                    onClick={() => void updateStatus(item, false)}
                  >
                    {englishCopy.imports.mapping_leave_out}
                  </Button>
                ) : (
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      type="button"
                      disabled={saving}
                      onClick={() => void acceptUnknown(item)}
                    >
                      Save decision
                    </Button>
                    <Button
                      type="button"
                      variant="secondary"
                      disabled={saving}
                      onClick={() => void updateStatus(item, false)}
                    >
                      Reject
                    </Button>
                  </div>
                )}
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">
              No items need a decision.
            </p>
          )}
        </Card>
      </section>

      <div className="fixed inset-x-0 bottom-[calc(4.5rem+env(safe-area-inset-bottom))] z-20 mx-auto max-w-3xl border-t border-line bg-surface/95 p-3 shadow-lg backdrop-blur">
        <div className="flex gap-2">
          <Button
            type="button"
            variant="secondary"
            className="shrink-0"
            disabled={saving}
            onClick={() => setDiscardOpen(true)}
          >
            Discard
          </Button>
          <Button
            type="button"
            data-primary="true"
            className="flex-1"
            disabled={saving || !canConfirm}
            loading={saving}
            onClick={() => void confirm()}
          >
            {selectedCount > 0
              ? englishCopy.imports.confirm_import_selected.replace(
                  "{count}",
                  String(selectedCount),
                )
              : englishCopy.imports.cash_flow_confirm_only}
          </Button>
        </div>
      </div>

      <Sheet
        open={discardOpen}
        onOpenChange={setDiscardOpen}
        title="Discard this import?"
        description="This removes the review job. None of its holdings will be added."
      >
        <div className="grid grid-cols-2 gap-2">
          <Button
            type="button"
            variant="secondary"
            onClick={() => setDiscardOpen(false)}
          >
            Keep reviewing
          </Button>
          <Button
            type="button"
            variant="danger"
            loading={saving}
            onClick={() => void discard()}
          >
            Discard import
          </Button>
        </div>
      </Sheet>
    </div>
  );
}

const HOLDINGS_FIELDS = [
  ["isin", "mapping_security_code"],
  ["symbol", "mapping_symbol"],
  ["name", "mapping_name"],
  ["quantity", "mapping_quantity"],
  ["avg_price", "mapping_average_price"],
  ["last_price", "mapping_last_price"],
  ["exchange", "mapping_exchange"],
  ["market", "mapping_market"],
  ["currency", "mapping_currency"],
  ["market_value", "mapping_market_value"],
  ["security_type", "mapping_security_type"],
  ["tax_wrapper", "mapping_tax_wrapper"],
] as const;

const TRADEBOOK_FIELDS = [
  ["isin", "mapping_security_code"],
  ["symbol", "mapping_symbol"],
  ["trade_date", "mapping_trade_date"],
  ["trade_type", "mapping_trade_type"],
  ["quantity", "mapping_quantity"],
  ["price", "mapping_price"],
  ["exchange", "mapping_exchange"],
  ["market", "mapping_market"],
  ["currency", "mapping_currency"],
  ["amount", "mapping_amount"],
  ["commission", "mapping_commission"],
  ["order_id", "mapping_order_id"],
  ["trade_id", "mapping_trade_id"],
  ["security_type", "mapping_security_type"],
  ["tax_wrapper", "mapping_tax_wrapper"],
] as const;

const marketDefaultOptions = [
  ["US", englishCopy.imports.market_us],
  ["DK", englishCopy.imports.market_dk],
  ["EU", englishCopy.imports.market_eu],
  ["GB", englishCopy.imports.market_gb],
  ["IN", englishCopy.imports.mapping_india],
] as const;

const currencyDefaultOptions = [
  ["USD", englishCopy.imports.currency_usd],
  ["EUR", englishCopy.imports.currency_eur],
  ["DKK", englishCopy.imports.currency_dkk],
  ["GBP", englishCopy.imports.currency_gbp],
  ["SEK", englishCopy.imports.currency_sek],
  ["INR", englishCopy.imports.currency_inr],
] as const;

function BrokerMappingPanel({
  detail,
  onMapped,
}: {
  detail: ImportDetail;
  onMapped(mapped: ImportDetail): void;
}) {
  const [kind, setKind] = useState<"holdings" | "tradebook">("holdings");
  const [columns, setColumns] = useState<Record<string, string>>({});
  const [marketDefault, setMarketDefault] = useState<
    ImportMappingRequest["market_default"]
  >(detail.market_default);
  const [currencyDefault, setCurrencyDefault] = useState<
    ImportMappingRequest["currency_default"]
  >(detail.currency_default);
  const [saveAs, setSaveAs] = useState(
    englishCopy.imports.mapping_save_as_default,
  );
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setMarketDefault(detail.market_default);
    setCurrencyDefault(detail.currency_default);
    setColumns(
      Object.fromEntries(
        Object.entries(detail.suggested_mapping ?? {}).filter(
          (entry): entry is [string, string] => typeof entry[1] === "string",
        ),
      ),
    );
  }, [detail]);

  const fields = kind === "holdings" ? HOLDINGS_FIELDS : TRADEBOOK_FIELDS;

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    const request: ImportMappingRequest = {
      kind,
      columns: Object.fromEntries(
        Object.entries(columns).filter(
          ([, header]) => header.trim().length > 0,
        ),
      ),
      save_as: saveAs.trim() || null,
      broker: (detail.broker ?? "other") as ImportMappingRequest["broker"],
      market_default: marketDefault,
      currency_default: currencyDefault,
    };
    try {
      onMapped(await mapImportColumns(detail.id, request));
    } catch {
      setError(englishCopy.imports.mapping_error);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Card data-copy-screen="import" className="grid gap-4">
      <div>
        <h2 className="text-xl font-semibold">
          {englishCopy.imports.mapping_title}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          {englishCopy.imports.mapping_intro}
        </p>
      </div>
      <p className="text-sm text-muted-foreground">
        {englishCopy.imports.mapping_preview_note}
      </p>
      <label
        className="grid gap-1.5 text-sm font-medium"
        htmlFor="mapping-kind"
      >
        {englishCopy.imports.mapping_kind_label}
        <select
          id="mapping-kind"
          value={kind}
          onChange={(event) => {
            setKind(event.currentTarget.value as "holdings" | "tradebook");
            setColumns({});
          }}
          className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
        >
          <option value="holdings">
            {englishCopy.imports.mapping_holdings}
          </option>
          <option value="tradebook">
            {englishCopy.imports.mapping_tradebook}
          </option>
        </select>
      </label>
      <details className="rounded-xl border border-line p-3 text-sm">
        <summary className="min-h-10 cursor-pointer font-medium">
          {englishCopy.imports.mapping_preview_rows}
        </summary>
        <div className="overflow-x-auto pt-2">
          <table data-security className="min-w-full text-left text-xs">
            <thead>
              <tr>
                {(detail.headers ?? []).map((header, index) => (
                  <th
                    key={`${header}-${index}`}
                    className="px-2 py-1 font-semibold"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(detail.samples ?? []).map((sample, rowIndex) => (
                <tr key={rowIndex}>
                  {sample.map((value, index) => (
                    <td key={`${rowIndex}-${index}`} className="px-2 py-1">
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <form className="grid gap-3" onSubmit={(event) => void submit(event)}>
        <div className="grid gap-3 sm:grid-cols-2">
          <label
            className="grid gap-1.5 text-sm font-medium"
            htmlFor="mapping-market-default"
          >
            {englishCopy.imports.mapping_market_default}
            <select
              id="mapping-market-default"
              value={marketDefault}
              onChange={(event) =>
                setMarketDefault(
                  event.currentTarget
                    .value as ImportMappingRequest["market_default"],
                )
              }
              className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
            >
              {marketDefaultOptions.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label
            className="grid gap-1.5 text-sm font-medium"
            htmlFor="mapping-currency-default"
          >
            {englishCopy.imports.mapping_currency_default}
            <select
              id="mapping-currency-default"
              value={currencyDefault}
              onChange={(event) =>
                setCurrencyDefault(
                  event.currentTarget
                    .value as ImportMappingRequest["currency_default"],
                )
              }
              className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
            >
              {currencyDefaultOptions.map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          {fields.map(([field, labelKey]) => (
            <label
              key={field}
              className="grid gap-1.5 text-sm font-medium"
              htmlFor={`mapping-${field}`}
            >
              {englishCopy.imports[labelKey]}
              <select
                id={`mapping-${field}`}
                value={columns[field] ?? ""}
                onChange={(event) => {
                  const selectedHeader = event.currentTarget.value;
                  setColumns((current) => {
                    const next = { ...current };
                    if (selectedHeader) next[field] = selectedHeader;
                    else delete next[field];
                    return next;
                  });
                }}
                className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
              >
                <option value="">
                  {englishCopy.imports.mapping_skip_column}
                </option>
                {(detail.headers ?? []).map((header, index) => (
                  <option key={`${header}-${index}`} value={header}>
                    {header}
                  </option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <label
          className="grid gap-1.5 text-sm font-medium"
          htmlFor="mapping-save-as"
        >
          {englishCopy.imports.mapping_save_as}
          <input
            id="mapping-save-as"
            value={saveAs}
            onChange={(event) => setSaveAs(event.currentTarget.value)}
            maxLength={80}
            className="min-h-12 rounded-xl border border-line bg-surface px-3 text-base"
          />
        </label>
        {error ? (
          <p role="alert" className="text-sm text-danger-foreground">
            {error}
          </p>
        ) : null}
        <Button
          type="submit"
          data-primary="true"
          loading={saving}
          disabled={saving}
        >
          {englishCopy.imports.mapping_submit}
        </Button>
      </form>
    </Card>
  );
}
