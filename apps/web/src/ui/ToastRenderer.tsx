import * as ToastPrimitive from "@radix-ui/react-toast";
import { X } from "lucide-react";
import type { ToastItem } from "./toast-context";

export function ToastRenderer({
  items,
  onDismiss,
}: {
  items: ToastItem[];
  onDismiss(id: number): void;
}) {
  return (
    <ToastPrimitive.Provider swipeDirection="down">
      {items.map((item) => (
        <ToastPrimitive.Root
          key={item.id}
          duration={item.duration}
          className="rounded-xl border border-line bg-surface p-4 text-foreground shadow-lg"
          onOpenChange={(open) => {
            if (!open) onDismiss(item.id);
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
                className="mt-3 min-h-12 rounded-lg bg-brand px-4 text-sm font-semibold text-brand-foreground"
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
  );
}
