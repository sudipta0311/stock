import { afterEach, describe, expect, it, vi } from "vitest";
import {
  importProgressReducer,
  initialImportProgress,
  progressLabel,
} from "../../src/imports/progress";

afterEach(() => vi.useRealTimers());

describe("import progress", () => {
  it("moves from upload to reading to ready and shows cold-start retries", async () => {
    vi.useFakeTimers();
    let state = importProgressReducer(initialImportProgress, { type: "start" });
    expect(progressLabel(state.phase)).toBe("Uploading your statement");

    const readingTimer = window.setTimeout(() => {
      state = importProgressReducer(state, { type: "reading" });
    }, 1200);
    await vi.advanceTimersByTimeAsync(1200);
    window.clearTimeout(readingTimer);
    expect(progressLabel(state.phase)).toBe("Reading your statement");

    state = importProgressReducer(state, { type: "waking", value: true });
    expect(state.waking).toBe(true);
    state = importProgressReducer(state, { type: "waking", value: false });
    state = importProgressReducer(state, { type: "ready" });
    expect(progressLabel(state.phase)).toBe("Ready to review");
    expect(state.waking).toBe(false);
  });

  it("resets cleanly when an upload is cancelled", () => {
    const uploading = importProgressReducer(initialImportProgress, {
      type: "start",
    });
    expect(importProgressReducer(uploading, { type: "cancel" })).toEqual(
      initialImportProgress,
    );
  });
});
