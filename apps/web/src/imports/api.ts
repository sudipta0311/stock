import type { components } from "../api/schema";
import { apiFetch } from "../api/client";

type Schema<Name extends keyof components["schemas"]> =
  components["schemas"][Name];

export type ImportJob = Schema<"ImportJobResponse">;
export type ImportDetail = Schema<"ImportDetailResponse">;
export type ImportItem = Schema<"ImportItemResponse">;
export type ImportUpload = Schema<"ImportUploadResponse">;
export type ImportConfirmation = Schema<"ImportConfirmationResponse">;

export type ImportFormat = "nsdl_cas" | "cams_kfin_cas" | "broker_statement";
export type ImportItemUpdate = {
  status?: "accepted" | "rejected";
  override?: {
    kind: "equity" | "etf" | "mf" | "bond" | "sgb";
    sleeve:
      | "equity_india"
      | "equity_global"
      | "debt"
      | "gold"
      | "cash"
      | "real_estate"
      | "protection";
  };
};

export function listImports(): Promise<ImportJob[]> {
  return apiFetch("/v1/imports");
}

export function uploadImport(
  file: File,
  password: string,
  docType: ImportFormat,
  signal?: AbortSignal,
): Promise<ImportUpload> {
  const body = new FormData();
  body.append("file", file, file.name);
  body.append("doc_type", docType);
  if (password) body.append("password", password);
  return apiFetch("/v1/imports", { method: "POST", body, signal });
}

export function getImport(jobId: string): Promise<ImportDetail> {
  return apiFetch(`/v1/imports/${encodeURIComponent(jobId)}`);
}

export function patchImportItem(
  jobId: string,
  itemId: string,
  payload: ImportItemUpdate,
): Promise<ImportItem> {
  return apiFetch(
    `/v1/imports/${encodeURIComponent(jobId)}/items/${encodeURIComponent(itemId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export function confirmImport(jobId: string): Promise<ImportConfirmation> {
  return apiFetch(`/v1/imports/${encodeURIComponent(jobId)}/confirm`, {
    method: "POST",
  });
}

export function discardImport(jobId: string): Promise<void> {
  return apiFetch(`/v1/imports/${encodeURIComponent(jobId)}`, {
    method: "DELETE",
  });
}
