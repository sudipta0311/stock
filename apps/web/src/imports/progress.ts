export type ImportProgressPhase = "idle" | "uploading" | "reading" | "ready";

export interface ImportProgressState {
  phase: ImportProgressPhase;
  waking: boolean;
}

export type ImportProgressAction =
  | { type: "start" }
  | { type: "reading" }
  | { type: "ready" }
  | { type: "cancel" }
  | { type: "waking"; value: boolean };

export const initialImportProgress: ImportProgressState = {
  phase: "idle",
  waking: false,
};

export function importProgressReducer(
  state: ImportProgressState,
  action: ImportProgressAction,
): ImportProgressState {
  switch (action.type) {
    case "start":
      return { phase: "uploading", waking: false };
    case "reading":
      return state.phase === "uploading"
        ? { ...state, phase: "reading" }
        : state;
    case "ready":
      return { phase: "ready", waking: false };
    case "cancel":
      return initialImportProgress;
    case "waking":
      return { ...state, waking: action.value };
  }
}

export function progressLabel(phase: ImportProgressPhase): string {
  switch (phase) {
    case "uploading":
      return "Uploading your statement";
    case "reading":
      return "Reading your statement";
    case "ready":
      return "Ready to review";
    default:
      return "Choose a statement to begin";
  }
}
