import * as ProgressPrimitive from "@radix-ui/react-progress";
import {
  AlertCircle,
  CheckCircle2,
  LoaderCircle,
  TriangleAlert,
} from "lucide-react";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import { useId, useState } from "react";
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
    primary: "bg-brand text-brand-foreground hover:bg-brand-strong",
    secondary: "border border-line bg-surface text-foreground hover:bg-muted",
    ghost: "bg-transparent text-foreground hover:bg-muted",
    danger: "bg-danger text-danger-on-fill hover:opacity-90",
  };
  return (
    <button
      className={classes(
        "motion-safe:active:scale-[.98] inline-flex min-h-12 items-center justify-center gap-2 rounded-xl px-4 py-3 text-sm font-semibold transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand disabled:cursor-not-allowed disabled:opacity-60",
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
        "card-surface rounded-2xl border border-line bg-surface",
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
  label: ReactNode;
  value: string;
  detail?: string;
}) {
  return (
    <div className="space-y-1">
      <p className="text-sm text-muted-foreground">{label}</p>
      <p className="text-lg font-semibold tracking-tight tabular-nums">
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

export function Term({
  children,
  explanation,
  prompt,
  learnMoreHref,
  code = false,
}: {
  children: ReactNode;
  explanation: string;
  prompt: string;
  learnMoreHref?: string;
  /** Monospace for identifiers such as fund codes; labels keep the surrounding font. */
  code?: boolean;
}) {
  // A button and span, not <details>: Term sits inside sentences, and <details> inside a
  // <p> is invalid HTML that browsers repair by splitting the paragraph.
  const [open, setOpen] = useState(false);
  const explanationId = useId();
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <span data-term-copy className={code ? "font-mono text-xs" : undefined}>
        {children}
      </span>
      <span className="inline-block text-xs text-muted-foreground">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={explanationId}
          onClick={() => setOpen((value) => !value)}
          className="cursor-pointer underline decoration-dotted underline-offset-2"
        >
          {prompt}
        </button>
        <span id={explanationId} hidden={!open}>
          <span className="ml-2">{explanation}</span>
          {learnMoreHref ? (
            <a
              className="ml-2 whitespace-nowrap text-brand underline underline-offset-2"
              href={learnMoreHref}
            >
              Learn more
            </a>
          ) : null}
        </span>
      </span>
    </span>
  );
}

export function Progress({
  value,
  max = 100,
  label,
  quiet = false,
}: {
  value: number;
  max?: number;
  label: string;
  quiet?: boolean;
}) {
  const boundedValue = Math.min(Math.max(value, 0), max);
  return (
    <ProgressPrimitive.Root
      className={classes(
        "w-full overflow-hidden rounded-full bg-muted",
        quiet ? "h-1.5" : "h-2.5",
      )}
      value={boundedValue}
      max={max}
      aria-label={label}
    >
      <ProgressPrimitive.Indicator
        className={classes(
          "h-full rounded-full bg-brand transition-transform",
          quiet && "opacity-70",
        )}
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
  icon,
  tone = "neutral",
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  icon?: ReactNode;
  tone?: "neutral" | "success";
}) {
  const iconTone =
    tone === "success"
      ? "bg-success-soft text-success-foreground"
      : "bg-muted text-muted-foreground";
  return (
    <div className="flex min-h-52 flex-col items-center justify-center gap-3 rounded-2xl border border-line bg-surface p-6 text-center">
      <span
        className={classes(
          "grid size-11 place-items-center rounded-full",
          iconTone,
        )}
      >
        {icon ?? <AlertCircle aria-hidden="true" className="size-5" />}
      </span>
      <h2 className="text-card-title font-semibold">{title}</h2>
      {description ? (
        <p className="max-w-sm text-base text-muted-foreground">
          {description}
        </p>
      ) : null}
      {action}
    </div>
  );
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
