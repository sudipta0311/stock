import type { components, paths } from "../api/schema";

type ImportSource = components["schemas"]["ImportUploadResponse"]["source"];

export type ImportFeatureSupport = {
  broker: Extract<ImportSource, "broker"> extends never ? false : true;
  mapping: "/v1/imports/{job_id}/mapping" extends keyof paths ? true : false;
};

export const importFeatureSupport: ImportFeatureSupport = {
  broker: false,
  mapping: false,
};
