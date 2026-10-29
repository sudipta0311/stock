import {
  lazy,
  Suspense,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";
import type { ReactNode } from "react";

export interface ToastItem {
  id: number;
  title: string;
  description?: string;
  action?: { label: string; altText: string; onClick(): void };
}

interface ToastContextValue {
  toast(item: Omit<ToastItem, "id">): void;
}

const ToastContext = createContext<ToastContextValue | null>(null);
const ToastRenderer = lazy(() =>
  import("./ToastRenderer").then((module) => ({
    default: module.ToastRenderer,
  })),
);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const toast = useCallback((item: Omit<ToastItem, "id">) => {
    setItems((current) => [
      ...current,
      { ...item, id: Date.now() + Math.random() },
    ]);
  }, []);
  const context = useMemo(() => ({ toast }), [toast]);
  const dismiss = useCallback((id: number) => {
    setItems((current) => current.filter((item) => item.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={context}>
      {children}
      {items.length > 0 ? (
        <Suspense fallback={null}>
          <ToastRenderer items={items} onDismiss={dismiss} />
        </Suspense>
      ) : null}
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside ToastProvider.");
  return context;
}
