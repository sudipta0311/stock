import { afterEach, describe, expect, it, vi } from "vitest";
import {
  ACTION_LABELS,
  CONFIDENCE_LABELS,
  EXIT_LABELS,
  INSTRUMENT_LABELS,
  labelForCode,
  NEUTRAL_CODE_LABEL,
  SLEEVE_LABELS,
  TRIGGER_KEY_LABELS,
  TRIGGER_LABELS,
  triggerKeyLabel,
} from "../../src/lib/labels";
import { reasonCopy, routeHowTo } from "../../src/features/data";
import englishCopy from "../../src/copy/en.json";

afterEach(() => vi.restoreAllMocks());

describe("plain-language labels", () => {
  it("labels every action, instrument, exit, and trigger code in generated copy", () => {
    const groups = [
      englishCopy.labels.action_kinds,
      englishCopy.labels.instrument_categories,
      englishCopy.labels.exit_actions,
      englishCopy.labels.trigger_keys,
    ];
    for (const group of groups) {
      for (const [code, label] of Object.entries(group)) {
        expect(label.trim().length, code).toBeGreaterThan(0);
        expect(labelForCode(code), code).not.toBe(NEUTRAL_CODE_LABEL);
      }
    }
    expect(ACTION_LABELS).toEqual(englishCopy.labels.action_kinds);
    expect(EXIT_LABELS).toEqual(englishCopy.labels.exit_actions);
    expect(INSTRUMENT_LABELS).toEqual(englishCopy.labels.instrument_categories);
    expect(TRIGGER_KEY_LABELS).toEqual(englishCopy.labels.trigger_keys);
    expect(TRIGGER_LABELS).toEqual(englishCopy.labels.trigger_states);
    expect(CONFIDENCE_LABELS).toEqual(englishCopy.labels.confidence);
    expect(SLEEVE_LABELS).toEqual(englishCopy.labels.sleeves);
    expect(triggerKeyLabel("drift")).toBe(
      englishCopy.labels.trigger_keys.drift,
    );
  });

  it("uses a neutral fallback and logs unknown codes", () => {
    const warning = vi
      .spyOn(console, "warn")
      .mockImplementation(() => undefined);
    expect(labelForCode("UNKNOWN_TEST_CODE")).toBe(NEUTRAL_CODE_LABEL);
    expect(warning).toHaveBeenCalledWith(
      "Unknown display code: UNKNOWN_TEST_CODE",
    );
    expect(reasonCopy("unknown_test_reason")).toContain(
      "This action follows your plan",
    );
    expect(warning).toHaveBeenCalledWith(
      "Unknown reason code: unknown_test_reason",
    );
  });
});

describe("route instructions", () => {
  it.each([
    ["A", /mutual fund app.*S&P 500 index fund.*monthly investment/],
    ["B", /Indian broker app.*fund that tracks shares abroad.*buy order/],
    ["C", /overseas investment account.*broad index fund.*invest/],
  ])("explains how to use route %s", (route, expected) => {
    expect(routeHowTo(route, "₹10,000")).toMatch(expected);
  });
});
