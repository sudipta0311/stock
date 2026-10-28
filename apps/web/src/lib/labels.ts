import englishCopy from "../copy/en.json";

const labels = englishCopy.labels;

export const ACTION_LABELS = labels.action_kinds;
export const EXIT_LABELS = labels.exit_actions;
export const INSTRUMENT_LABELS = labels.instrument_categories;
export const TRIGGER_KEY_LABELS = labels.trigger_keys;
export const TRIGGER_LABELS = labels.trigger_states;
export const CONFIDENCE_LABELS = labels.confidence;
export const SLEEVE_LABELS = labels.sleeves;

export const NEUTRAL_CODE_LABEL = labels.unknown_code;

const allLabels: Record<string, string> = {
  ...ACTION_LABELS,
  ...EXIT_LABELS,
  ...INSTRUMENT_LABELS,
  ...TRIGGER_KEY_LABELS,
  ...TRIGGER_LABELS,
  ...CONFIDENCE_LABELS,
  ...SLEEVE_LABELS,
};

export function labelForCode(code: string): string {
  const label = allLabels[code];
  if (label) return label;
  console.warn(`Unknown display code: ${code}`);
  return NEUTRAL_CODE_LABEL;
}

export function actionLabel(code: string): string {
  return (
    ACTION_LABELS[code as keyof typeof ACTION_LABELS] ?? labelForCode(code)
  );
}

export function exitLabel(code: string): string {
  return EXIT_LABELS[code as keyof typeof EXIT_LABELS] ?? labelForCode(code);
}

export function instrumentLabel(code: string): string {
  return (
    INSTRUMENT_LABELS[code as keyof typeof INSTRUMENT_LABELS] ??
    labelForCode(code)
  );
}

export function triggerKeyLabel(code: string): string {
  return (
    TRIGGER_KEY_LABELS[code as keyof typeof TRIGGER_KEY_LABELS] ??
    labelForCode(code)
  );
}
