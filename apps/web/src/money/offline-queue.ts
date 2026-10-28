import { createAsset, patchAsset } from "./api";
import type { AssetResponse } from "./api";
import type { AssetDraft } from "./forms.generated";

const DATABASE = "wealth-money-map-offline";
const STORE = "asset-writes";

function notifyPendingChanged(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event("wealth:pending-changes"));
  }
}

export interface QueuedAssetWrite {
  key: string;
  draftKey: string;
  assetId: string | null;
  body: Record<string, unknown>;
  updatedAt: number;
}

function openDatabase(): Promise<IDBDatabase> {
  if (typeof indexedDB === "undefined") {
    return Promise.reject(new Error("IndexedDB is not available."));
  }
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE)) {
        database.createObjectStore(STORE, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Queue database could not be opened."));
  });
}

async function withStore<T>(
  mode: IDBTransactionMode,
  operation: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  const database = await openDatabase();
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE, mode);
    const request = operation(transaction.objectStore(STORE));
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Offline queue operation failed."));
    transaction.oncomplete = () => database.close();
    transaction.onerror = () => {
      database.close();
      reject(
        transaction.error ?? new Error("Offline queue transaction failed."),
      );
    };
  });
}

export async function queueAssetWrite(
  draftKey: string,
  assetId: string | null,
  body: Record<string, unknown>,
): Promise<void> {
  const key = assetId ? `asset:${assetId}` : `draft:${draftKey}`;
  await withStore("readwrite", (store) =>
    store.put({ key, draftKey, assetId, body, updatedAt: Date.now() }),
  );
  notifyPendingChanged();
}

export async function listQueuedAssetWrites(): Promise<QueuedAssetWrite[]> {
  return withStore("readonly", (store) => store.getAll());
}

async function removeQueuedAssetWrite(key: string): Promise<void> {
  await withStore("readwrite", (store) => store.delete(key));
  notifyPendingChanged();
}

export async function clearOfflineAssetWrites(): Promise<void> {
  try {
    await withStore("readwrite", (store) => store.clear());
    notifyPendingChanged();
  } catch {
    // Offline storage may not have been initialized on this browser.
  }
}

export async function replayOfflineWrites(): Promise<void> {
  if (typeof navigator !== "undefined" && !navigator.onLine) return;
  let writes: QueuedAssetWrite[];
  try {
    writes = (await listQueuedAssetWrites()).sort(
      (left, right) => left.updatedAt - right.updatedAt,
    );
  } catch {
    return;
  }

  for (const write of writes) {
    try {
      const saved = write.assetId
        ? await patchAsset(write.assetId, write.body)
        : await createAsset(write.body as AssetDraft);
      await removeQueuedAssetWrite(write.key);
      if (typeof window !== "undefined") {
        window.dispatchEvent(
          new CustomEvent<{ draftKey: string; asset: AssetResponse }>(
            "wealth:offline-asset-saved",
            {
              detail: { draftKey: write.draftKey, asset: saved },
            },
          ),
        );
      }
    } catch {
      if (typeof navigator !== "undefined" && !navigator.onLine) break;
    }
  }
}
