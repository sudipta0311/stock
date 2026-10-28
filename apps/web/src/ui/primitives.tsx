import * as Dialog from "@radix-ui/react-dialog";
import * as ProgressPrimitive from "@radix-ui/react-progress";
import * as ToastPrimitive from "@radix-ui/react-toast";
import {
  AlertCircle,
  CheckCircle2,
  LoaderCircle,
  TriangleAlert,
  X,
} from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import type { ApiFieldError } from "../api/client";

function classes(...names: Array<string | false | undefined>): string {
  return names.filter(Boolean).join(" ");
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  loading?: boolean;
}

export function Button({
  variant = "primary",
  loading = false,
  disabled,
  className,
  children,
  ...props
}: ButtonProps) {
  const styles: Record<ButtonVariant, string> = {
    primary: "bg-brand text-white hover:bg-brand-strong",
    secondary: "border border-line bg-surface text-foreground hover:bg-muted",
    ghost: "bg-transparent text-foreground hover:bg-muted",
    danger: "bg-danger text-white hover:opacity-90",
  };
  return (
    <button
      className={classes(
        "inline-flex min-h-12 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand disabled:cursor-not-allowed disabled:opacity-60",
        styles[variant],
        className,
      )}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? (
        <LoaderCircle aria-hidden="true" className="size-4 animate-spin" />
      ) : variant === "danger" ? (
        <AlertCircle aria-hidden="true" className="size-4" />
      ) : null}
      {children}
    </button>
  );
}

export function Card({ className, ...props }: HTMLAttributes<HTMLElement>) {
  return (
    <section
      className={classes(
        "rounded-2xl border border-line bg-surface p-4 shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

export function Stat({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail?: string;
}) {
  return (
    <div className="space-y-1">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </p>
      {detail ? (
        <p className="text-sm text-muted-foreground">{detail}</p>
      ) : null}
    </div>
  );
}

type BadgeTone = "neutral" | "success" | "warning" | "danger";

export function Badge({
  tone = "neutral",
  children,
}: {
  tone?: BadgeTone;
  children: ReactNode;
}) {
  const styles: Record<BadgeTone, string> = {
    neutral: "bg-muted text-foreground",
    success: "bg-success-soft text-success-foreground",
    warning: "bg-warning-soft text-warning-foreground",
    danger: "bg-danger-soft text-danger-foreground",
  };
  const icon =
    tone === "success" ? (
      <CheckCircle2 aria-hidden="true" className="size-3.5" />
    ) : tone === "warning" ? (
      <TriangleAlert aria-hidden="true" className="size-3.5" />
    ) : tone === "danger" ? (
      <AlertCircle aria-hidden="true" className="size-3.5" />
    ) : null;
  return (
    <span
      className={classes(
        "inline-flex min-h-7 items-center gap-1.5 rounded-full px-2.5 text-xs font-medium",
        styles[tone],
      )}
    >
      {icon}
      {children}
    </span>
  );
}

export function Progress({
  value,
  max = 100,
  label,
}: {
  value: number;
  max?: number;
  label: string;
}) {
  const boundedValue = Math.min(Math.max(value, 0), max);
  return (
    <ProgressPrimitive.Root
      className="h-2.5 w-full overflow-hidden rounded-full bg-muted"
      value={boundedValue}
      max={max}
      aria-label={label}
    >
      <ProgressPrimitive.Indicator
        className="h-full rounded-full bg-brand transition-transform"
        style={{
          transform: `translateX(-${100 - (boundedValue / max) * 100}%)`,
        }}
      />
    </ProgressPrimitive.Root>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={classes("animate-pulse rounded-lg bg-muted", className)}
    />
  );
}

export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex min-h-52 flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-line bg-surface/60 p-6 text-center">
      <span className="grid size-11 place-items-center rounded-full bg-muted text-muted-foreground">
        <AlertCircle aria-hidden="true" className="size-5" />
      </span>
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="max-w-sm text-sm leading-6 text-muted-foreground">
        {description}
      </p>
      {action}
    </div>
  );
}

export function Sheet({
  open,
  onOpenChange,
  title,
  description,
  children,
}: {
  open: boolean;
  onOpenChange(open: boolean): void;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-40 bg-black/40" />
        <Dialog.Content className="fixed inset-x-0 bottom-0 z-50 max-h-[88vh] overflow-y-auto rounded-t-3xl border border-line bg-surface p-5 pb-[calc(1.25rem+env(safe-area-inset-bottom))] shadow-xl sm:inset-x-auto sm:left-1/2 sm:top-1/2 sm:bottom-auto sm:w-full sm:max-w-lg sm:-translate-x-1/2 sm:-translate-y-1/2 sm:rounded-2xl">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <Dialog.Title className="text-lg font-semibold">
                {title}
              </Dialog.Title>
              {description ? (
                <Dialog.Description className="mt-1 text-sm text-muted-foreground">
                  {description}
                </Dialog.Description>
              ) : null}
            </div>
            <Dialog.Close asChild>
              <Button
                aria-label="Close"
                variant="ghost"
                className="size-12 px-0"
              >
                <X aria-hidden="true" className="size-5" />
              </Button>
            </Dialog.Close>
          </div>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

interface ToastItem {
  id: number;
  title: string;
  description?: string;
  action?: { label: string; altText: string; onClick(): void };
}

interface ToastContextValue {
  toast(item: Omit<ToastItem, "id">): void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const toast = useCallback((item: Omit<ToastItem, "id">) => {
    setItems((current) => [
      ...current,
      { ...item, id: Date.now() + Math.random() },
    ]);
  }, []);
  const context = useMemo(() => ({ toast }), [toast]);
  return (
    <ToastContext.Provider value={context}>
      <ToastPrimitive.Provider swipeDirection="down">
        {children}
        {items.map((item) => (
          <ToastPrimitive.Root
            key={item.id}
            className="rounded-xl border border-line bg-surface p-4 text-foreground shadow-lg"
            onOpenChange={(open) => {
              if (!open)
                setItems((current) =>
                  current.filter((toastItem) => toastItem.id !== item.id),
                );
            }}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <ToastPrimitive.Title className="text-sm font-semibold">
                  {item.title}
                </ToastPrimitive.Title>
                {item.description ? (
                  <ToastPrimitive.Description className="mt-1 text-sm text-muted-foreground">
                    {item.description}
                  </ToastPrimitive.Description>
                ) : null}
              </div>
              <ToastPrimitive.Close asChild>
                <button
                  className="grid size-8 place-items-center rounded-lg text-muted-foreground hover:bg-muted"
                  aria-label="Dismiss notification"
                >
                  <X aria-hidden="true" className="size-4" />
                </button>
              </ToastPrimitive.Close>
            </div>
            {item.action ? (
              <ToastPrimitive.Action altText={item.action.altText} asChild>
                <button
                  type="button"
                  onClick={item.action.onClick}
                  className="mt-3 min-h-12 rounded-lg bg-brand px-4 text-sm font-semibold text-white"
                >
                  {item.action.label}
                </button>
              </ToastPrimitive.Action>
            ) : null}
          </ToastPrimitive.Root>
        ))}
        <ToastPrimitive.Viewport
          className="fixed inset-x-3 bottom-[calc(5.5rem+env(safe-area-inset-bottom))] z-[60] flex flex-col gap-2 outline-none sm:left-auto sm:right-4 sm:top-4 sm:bottom-auto sm:w-96"
          aria-label="Notifications"
        />
      </ToastPrimitive.Provider>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside ToastProvider.");
  return context;
}

export function fieldErrorMessage(
  error: string | ApiFieldError[] | undefined,
  name?: string,
): string | undefined {
  if (typeof error === "string") return error;
  if (!error?.length) return undefined;
  return (
    error.find((item) => item.field === name)?.message ?? error[0]?.message
  );
}
