const DATABASE_NAME = "wealth-import-share";
const STORE_NAME = "shared-files";
const SHARED_FILE_KEY = "pending-statement";

interface StoredSharedFile {
  bytes: ArrayBuffer;
  name: string;
  type: string;
  lastModified: number;
}

let databasePromise: Promise<IDBDatabase> | undefined;

function openDatabase(): Promise<IDBDatabase> {
  if (databasePromise) return databasePromise;
  databasePromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () =>
      reject(request.error ?? new Error("Could not open shared file storage."));
  });
  return databasePromise;
}

export function isShareTargetPost(method: string, url: string): boolean {
  try {
    return (
      method.toUpperCase() === "POST" &&
      new URL(url).pathname === "/import/share"
    );
  } catch {
    return false;
  }
}

export function shareTargetRedirect(
  origin: string,
  hasFile: boolean,
): Response {
  const location = hasFile
    ? "/import?shared=1"
    : "/import?shareError=missing-file";
  return new Response(null, {
    status: 303,
    headers: { Location: new URL(location, origin).href },
  });
}

export async function handleShareTargetPost(
  method: string,
  url: string,
  file: File | null,
  store: (sharedFile: File) => Promise<void>,
): Promise<Response | null> {
  if (!isShareTargetPost(method, url)) return null;
  const origin = new URL(url).origin;
  if (!file) return shareTargetRedirect(origin, false);
  await store(file);
  return shareTargetRedirect(origin, true);
}

export async function storeSharedFile(file: File): Promise<void> {
  const database = await openDatabase();
  const record: StoredSharedFile = {
    bytes: await file.arrayBuffer(),
    name: file.name,
    type: file.type,
    lastModified: file.lastModified,
  };
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    transaction.objectStore(STORE_NAME).put(record, SHARED_FILE_KEY);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(transaction.error ?? new Error("Could not save the shared file."));
    transaction.onabort = () =>
      reject(transaction.error ?? new Error("Could not save the shared file."));
  });
}

export async function takeSharedFile(): Promise<File | null> {
  const database = await openDatabase();
  return new Promise<File | null>((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, "readwrite");
    const store = transaction.objectStore(STORE_NAME);
    const request = store.get(SHARED_FILE_KEY) as IDBRequest<
      StoredSharedFile | undefined
    >;
    let sharedFile: File | null = null;
    request.onsuccess = () => {
      const record = request.result;
      store.delete(SHARED_FILE_KEY);
      if (
        record &&
        (record.bytes instanceof ArrayBuffer ||
          Object.prototype.toString.call(record.bytes) ===
            "[object ArrayBuffer]")
      ) {
        sharedFile = new File([record.bytes], record.name, {
          type: record.type,
          lastModified: record.lastModified,
        });
      }
    };
    transaction.oncomplete = () => resolve(sharedFile);
    transaction.onerror = () =>
      reject(transaction.error ?? new Error("Could not read the shared file."));
    transaction.onabort = () =>
      reject(transaction.error ?? new Error("Could not read the shared file."));
  });
}

export async function clearSharedFileStorage(): Promise<void> {
  databasePromise?.then((database) => database.close()).catch(() => undefined);
  databasePromise = undefined;
  await new Promise<void>((resolve, reject) => {
    const request = indexedDB.deleteDatabase(DATABASE_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () =>
      reject(
        request.error ?? new Error("Could not clear shared file storage."),
      );
    request.onblocked = () =>
      reject(new Error("Shared file storage is still in use."));
  });
}
