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
import { Badge, Button, Card, Sheet } from "../ui";
import {
  confirmImport,
  discardImport,
  getImport,
  patchImportItem,
  uploadImport,
} from "./api";
import type { ImportDetail, ImportFormat, ImportItem } from "./api";
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
    title: "NSDL / CDSL CAS (PDF)",
    help: "Your CAS usually arrives by email each month. You can also request it from your depository.",
  },
  cams_kfin_cas: {
    title: englishCopy.fund_check.cams_title,
    help: englishCopy.fund_check.cams_request,
    steps: englishCopy.fund_check.cams_steps,
    note: englishCopy.fund_check.cams_summary_note,
  },
  broker_statement: {
    title: "Broker statement (CSV/XLSX)",
    help: "Export your holdings or tradebook from your broker's reports or portfolio page.",
  },
};

const FILE_PICKER_ACCEPT = importFeatureSupport.broker
  ? ".pdf,.csv,.xlsx,application/pdf,text/csv"
  : ".pdf,application/pdf";

function sourceLabel(source: string | null): string {
  switch (source) {
    case "nsdl":
      return "NSDL CAS";
    case "cdsl":
      return "CDSL CAS";
    case "cams_kfin":
      return "CAMS / KFintech CAS";
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
  return ["new", "changed", "accepted"].includes(item.status);
}

function readableSleeve(sleeve: string | null): string {
  if (!sleeve) return "Needs a sleeve";
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
            nextStep: "Choose your CAS with the file picker and try again.",
          }
        : {
            message: "We couldn't keep the shared file ready.",
            nextStep: "Choose your CAS with the file picker and try again.",
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
    <div className="grid gap-4">
      <Card className="grid gap-4">
        <div>
          <h2 className="text-xl font-semibold">Upload your CAS</h2>
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
              PDF up to 10 MB{file ? ` · ${file.name}` : ""}
            </span>
          </label>

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
              For NSDL CAS this is usually your PAN in capitals.
            </span>
          </label>

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

          <Button type="submit" disabled={!file || busy} loading={inFlight}>
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
  ["etf", "ETF"],
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
        <strong className="block truncate">{item.name}</strong>
        <span className="block text-xs text-muted-foreground">
          {item.quantity} units ·{" "}
          {item.statement_value
            ? formatValue(item.statement_value)
            : "Value not provided"}
        </span>
      </span>
      <Badge>{readableSleeve(item.sleeve)}</Badge>
    </label>
  );
}

function formatValue(value: string): string {
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

  useEffect(() => {
    let active = true;
    setLoading(true);
    getImport(jobId)
      .then((loaded) => {
        if (!active) return;
        setDetail(loaded);
        if (loaded.status === "confirmed") markValueMoment();
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
    if (!detail || selectedCount === 0) return;
    setSaving(true);
    setPageError("");
    try {
      const result = await confirmImport(detail.id);
      setConfirmedCount(sumImportCounts(result.counts) || selectedCount);
      markValueMoment();
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
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

  if (confirmedCount !== null || detail.status === "confirmed") {
    const added = confirmedCount ?? 0;
    return (
      <Card className="grid gap-3 text-center">
        <CheckCircle2
          aria-hidden="true"
          className="mx-auto size-10 text-success-foreground"
        />
        <h2 className="text-xl font-semibold">
          {added > 0
            ? `Added ${added} holdings · Net worth updated`
            : "This import is already confirmed"}
        </h2>
        {added === 0 ? (
          <p className="text-sm text-muted-foreground">
            Your Money Map already includes this statement.
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
    <div className="grid gap-4 pb-36">
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
      </Card>

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
                <ItemToggle
                  item={item}
                  checked={selectedById[item.id] ?? false}
                  disabled={saving}
                  onChange={(checked) => void updateStatus(item, checked)}
                />
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
                    <strong>{item.name}</strong>
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
              Choose a kind and sleeve, or leave it out of this import.
            </p>
          </div>
          {grouped.unknown.length ? (
            grouped.unknown.map((item) => (
              <div
                key={item.id}
                className="grid gap-3 border-t border-line pt-3"
              >
                <div>
                  <strong className="text-sm">{item.name}</strong>
                  <p className="text-xs text-muted-foreground">
                    {item.quantity} units ·{" "}
                    {item.statement_value
                      ? formatValue(item.statement_value)
                      : "Value not provided"}{" "}
                    · {maskAccountReference(item.account_ref)}
                  </p>
                </div>
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
                    Sleeve
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
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">
              No items need a decision.
            </p>
          )}
        </Card>
      </section>

      {importFeatureSupport.mapping ? (
        <Card className="text-sm">
          Column mapping is available for this API.
        </Card>
      ) : null}

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
            className="flex-1"
            disabled={saving || selectedCount === 0}
            loading={saving}
            onClick={() => void confirm()}
          >
            Confirm import ({selectedCount} items)
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
