import { useController } from "react-hook-form";
import type { Control, FieldValues, Path } from "react-hook-form";
import { AlertCircle } from "lucide-react";
import { z } from "zod";
import { cloneElement, isValidElement } from "react";
import type {
  InputHTMLAttributes,
  ReactElement,
  ReactNode,
  SelectHTMLAttributes,
} from "react";
import { formatInr, parseInr } from "../lib/money";
import { fieldErrorMessage } from "./primitives";
import type { ApiFieldError } from "../api/client";

export const inrAmountSchema = z.string().transform((value, context) => {
  try {
    return parseInr(value);
  } catch {
    context.addIssue({
      code: "custom",
      message: "Enter an amount in rupees, such as 5 lakh or 1,25,000.",
    });
    return z.NEVER;
  }
});

export function Field({
  id,
  errorName = id,
  label,
  help,
  error,
  required,
  prefix,
  children,
}: {
  id: string;
  errorName?: string;
  label: string;
  help?: string;
  error?: string | ApiFieldError[];
  required?: boolean;
  prefix?: string;
  children: ReactNode;
}) {
  const message = fieldErrorMessage(error, errorName)?.trim() || undefined;
  const helpText = help?.trim() || undefined;
  const helpId = helpText ? `${id}-help` : undefined;
  const errorId = message ? `${id}-error` : undefined;
  const childDescribedBy = isValidElement<{ "aria-describedby"?: string }>(
    children,
  )
    ? children.props["aria-describedby"]
    : undefined;
  const describedBy =
    [
      ...new Set(
        [childDescribedBy, helpId, errorId]
          .filter(Boolean)
          .flatMap((value) => value?.split(" ") ?? []),
      ),
    ].join(" ") || undefined;
  const control = isValidElement(children)
    ? cloneElement(
        children as ReactElement<{
          "aria-describedby"?: string;
          "aria-invalid"?: boolean;
        }>,
        {
          "aria-describedby": describedBy,
          "aria-invalid": Boolean(message) || undefined,
        },
      )
    : children;
  const prefixedControl = prefix ? (
    <div className="relative">
      <span
        aria-hidden="true"
        className={`pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-muted-foreground ${prefix.length > 1 ? "min-w-10" : ""}`}
      >
        {prefix}
      </span>
      {control}
    </div>
  ) : (
    control
  );
  return (
    <div className="grid gap-1.5">
      <label htmlFor={id} className="text-sm font-medium">
        {label}
        {required ? (
          <>
            <span aria-hidden="true" className="ml-1 text-muted-foreground">
              *
            </span>
            <span className="sr-only"> (required)</span>
          </>
        ) : null}
      </label>
      {prefixedControl}
      {helpText ? (
        <p id={helpId} className="min-h-5 text-sm text-muted-foreground">
          {helpText}
        </p>
      ) : null}
      {message ? (
        <p id={errorId} className="text-sm text-danger-foreground">
          <AlertCircle
            aria-hidden="true"
            className="mr-1.5 inline size-4 align-text-bottom"
          />
          {message}
        </p>
      ) : null}
    </div>
  );
}

export interface MoneyInputProps<T extends FieldValues> {
  control: Control<T>;
  name: Path<T>;
  label: string;
  help?: string;
  problemErrors?: ApiFieldError[];
  validationError?: string;
  required?: boolean;
  placeholder?: string;
  onValueChange?: (value: string) => void;
  id?: string;
  errorName?: string;
}

export function MoneyInput<T extends FieldValues>({
  control,
  name,
  label,
  help,
  problemErrors,
  validationError,
  required,
  placeholder = "0.00",
  onValueChange,
  id: idProp,
  errorName,
}: MoneyInputProps<T>) {
  const { field, fieldState } = useController({ control, name });
  const id = idProp ?? String(name).replaceAll(".", "-");
  const errorField = errorName ?? String(name).replaceAll(".", "-");
  const textValue = String(field.value ?? "");
  let parsed: string | undefined;
  let parseError: string | undefined;
  if (textValue.trim()) {
    try {
      parsed = parseInr(textValue);
    } catch {
      parseError = "Enter a rupee amount such as 5 lakh or 50k.";
    }
  }
  const error =
    fieldState.error?.message ?? validationError ?? parseError ?? problemErrors;
  const describedBy =
    [
      help?.trim() ? `${id}-help` : undefined,
      fieldState.error || validationError || parseError
        ? `${id}-error`
        : undefined,
      problemErrors?.length ? `${id}-error` : undefined,
      parsed ? `${id}-parsed` : undefined,
    ]
      .filter(Boolean)
      .join(" ") || undefined;
  const parsedDecimals = parsed?.endsWith(".00") ? 0 : 2;
  return (
    <div className="grid gap-1.5">
      <Field
        id={id}
        errorName={errorField}
        label={label}
        help={help}
        error={error}
        required={required}
      >
        <div className="relative">
          <span
            aria-hidden="true"
            className="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-muted-foreground"
          >
            ₹
          </span>
          <input
            {...field}
            id={id}
            ref={field.ref}
            type="text"
            inputMode="decimal"
            autoComplete="off"
            required={required}
            placeholder={placeholder}
            value={textValue}
            aria-invalid={
              Boolean(
                fieldState.error ||
                validationError ||
                parseError ||
                problemErrors?.length,
              ) || undefined
            }
            aria-describedby={describedBy}
            className="min-h-12 w-full rounded-xl border border-line bg-surface pl-9 pr-3 text-base tabular-nums shadow-sm outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30"
            onChange={(event) => {
              field.onChange(event);
              onValueChange?.(event.currentTarget.value);
            }}
            onBlur={(event) => {
              const text = event.currentTarget.value.trim();
              if (text) {
                try {
                  field.onChange(parseInr(text));
                } catch {
                  // The connected form schema supplies the validation message.
                }
              }
              field.onBlur();
            }}
          />
        </div>
      </Field>
      {parsed ? (
        <p
          id={`${id}-parsed`}
          className="text-sm tabular-nums text-muted-foreground"
          aria-live="polite"
        >
          = {formatInr(parsed, "full", parsedDecimals)}
        </p>
      ) : null}
    </div>
  );
}

export function MonthYearInput({
  id,
  label,
  help,
  error,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  id: string;
  label: string;
  help?: string;
  error?: string;
}) {
  return (
    <Field id={id} label={label} help={help} error={error}>
      <input
        {...props}
        id={id}
        type="month"
        className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30"
      />
    </Field>
  );
}

export function Select({
  id,
  label,
  help,
  error,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement> & {
  id: string;
  label: string;
  help?: string;
  error?: string;
  children: ReactNode;
}) {
  return (
    <Field id={id} label={label} help={help} error={error}>
      <select
        {...props}
        id={id}
        className="min-h-12 w-full rounded-xl border border-line bg-surface px-3 text-base outline-none focus-visible:border-brand focus-visible:ring-2 focus-visible:ring-brand/30"
      >
        {children}
      </select>
    </Field>
  );
}

export function Checkbox({
  id,
  label,
  help,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & {
  id: string;
  label: string;
  help?: string;
}) {
  return (
    <label
      htmlFor={id}
      className="flex min-h-12 cursor-pointer items-center gap-3 rounded-xl px-1 py-2"
    >
      <input
        {...props}
        id={id}
        type="checkbox"
        className="size-5 accent-brand focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand"
      />
      <span className="grid gap-0.5">
        <span className="text-sm font-medium">{label}</span>
        {help ? (
          <span className="text-sm text-muted-foreground">{help}</span>
        ) : null}
      </span>
    </label>
  );
}

export function TileToggle({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: Array<{ value: string; label: string; description?: string }>;
  value: string;
  onChange(value: string): void;
}) {
  return (
    <div role="group" aria-label={label} className="grid grid-cols-2 gap-3">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={`min-h-20 rounded-2xl border p-3 text-left transition focus-visible:outline-2 focus-visible:outline-brand ${value === option.value ? "border-brand bg-brand/10" : "border-line bg-surface hover:bg-muted"}`}
        >
          <span className="block text-sm font-semibold">{option.label}</span>
          {option.description ? (
            <span className="mt-1 block text-xs text-muted-foreground">
              {option.description}
            </span>
          ) : null}
        </button>
      ))}
    </div>
  );
}
