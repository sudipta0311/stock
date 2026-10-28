import { afterEach, describe, expect, it, vi } from "vitest";
import {
  clearSharedFileStorage,
  handleShareTargetPost,
  isShareTargetPost,
  shareTargetRedirect,
  storeSharedFile,
  takeSharedFile,
} from "../../src/imports/share-target";

afterEach(async () => {
  vi.restoreAllMocks();
  await clearSharedFileStorage();
});

describe("Android share target handoff", () => {
  it("recognizes a same-path POST and keeps the redirect pure", () => {
    expect(isShareTargetPost("POST", "https://wealth.test/import/share")).toBe(
      true,
    );
    expect(isShareTargetPost("GET", "https://wealth.test/import/share")).toBe(
      false,
    );
    expect(isShareTargetPost("POST", "https://wealth.test/import/other")).toBe(
      false,
    );

    const response = shareTargetRedirect("https://wealth.test", true);
    expect(response.status).toBe(303);
    expect(response.headers.get("Location")).toBe(
      "https://wealth.test/import?shared=1",
    );
  });

  it("stores the shared file before redirecting to the import screen", async () => {
    const file = new File(["%PDF synthetic statement"], "monthly.pdf", {
      type: "application/pdf",
    });
    const stored: File[] = [];
    const response = await handleShareTargetPost(
      "POST",
      "https://wealth.test/import/share",
      file,
      async (sharedFile) => {
        stored.push(sharedFile);
      },
    );
    expect(stored).toHaveLength(1);
    expect(response?.status).toBe(303);
    expect(response?.headers.get("Location")).toBe(
      "https://wealth.test/import?shared=1",
    );
  });

  it("deletes the IndexedDB copy as soon as the import page picks it up", async () => {
    const file = new File(["%PDF synthetic statement"], "monthly.pdf", {
      type: "application/pdf",
    });
    await storeSharedFile(file);

    const received = await takeSharedFile();
    expect(received?.name).toBe("monthly.pdf");
    expect(received?.type).toBe("application/pdf");
    expect(await takeSharedFile()).toBeNull();
  });

  it("does not save when the share request has no file", async () => {
    const store = vi.fn(async () => undefined);
    const response = await handleShareTargetPost(
      "POST",
      "https://wealth.test/import/share",
      null,
      store,
    );
    expect(store).not.toHaveBeenCalled();
    expect(response?.headers.get("Location")).toBe(
      "https://wealth.test/import?shareError=missing-file",
    );
  });
});
