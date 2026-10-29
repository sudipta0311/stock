import { fireEvent, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithQueries } from "../features/render";
import { GmailStatementsSettingsPage } from "../../src/settings/GmailStatementsSettingsPage";

const mocks = vi.hoisted(() => ({
  createKey: vi.fn(),
  forgetPasswords: vi.fn(),
  listKeys: vi.fn(),
  revokeKey: vi.fn(),
  role: "owner" as "owner" | "partner" | "viewer",
}));

vi.mock("../../src/settings/api", () => ({
  createStatementInboxKey: mocks.createKey,
  forgetStatementPasswords: mocks.forgetPasswords,
  listStatementInboxKeys: mocks.listKeys,
  revokeStatementInboxKey: mocks.revokeKey,
}));

vi.mock("../../src/household/HouseholdContext", () => ({
  useHousehold: () => ({ role: mocks.role }),
}));

const now = "2026-10-29T10:00:00Z";

afterEach(() => vi.unstubAllGlobals());

function renderPage() {
  return renderWithQueries(
    <MemoryRouter>
      <GmailStatementsSettingsPage />
    </MemoryRouter>,
  );
}

describe("Gmail statement settings", () => {
  beforeEach(() => {
    mocks.role = "owner";
    mocks.listKeys.mockResolvedValue([]);
    mocks.createKey.mockResolvedValue({
      id: "synthetic-key-id",
      label: "My Gmail",
      created_at: now,
      last_used_at: null,
      key: "synthetic-one-time-key",
    });
    mocks.revokeKey.mockResolvedValue(undefined);
    mocks.forgetPasswords.mockResolvedValue(undefined);
  });

  it("creates a key, loads the script only when asked, then revokes it", async () => {
    const copiedScript = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: copiedScript },
    });
    const fetchScript = vi.fn().mockResolvedValue({
      ok: true,
      text: async () =>
        'const API_URL = "https://example.invalid";\nconst UPLOAD_KEY = "placeholder";',
    });
    vi.stubGlobal("fetch", fetchScript);
    const keyList: Array<{
      id: string;
      label: string;
      created_at: string;
      last_used_at: string | null;
    }> = [];
    mocks.listKeys.mockImplementation(async () => keyList);
    mocks.createKey.mockImplementation(async (label: string) => {
      keyList.push({
        id: "synthetic-key-id",
        label,
        created_at: now,
        last_used_at: null,
      });
      return {
        id: "synthetic-key-id",
        label,
        created_at: now,
        last_used_at: null,
        key: "synthetic-one-time-key",
      };
    });
    mocks.revokeKey.mockImplementation(async () => {
      keyList.length = 0;
    });

    renderPage();
    expect(fetchScript).not.toHaveBeenCalled();
    fireEvent.click(
      await screen.findByRole("button", { name: "Create an upload key" }),
    );
    expect(
      await screen.findByText("Your new key is ready"),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Copy setup script" }));
    await waitFor(() => expect(copiedScript).toHaveBeenCalledTimes(1));
    expect(fetchScript).toHaveBeenCalledTimes(1);
    expect(copiedScript.mock.calls.at(0)?.[0]).toContain(
      'const UPLOAD_KEY = "synthetic-one-time-key";',
    );
    expect(await screen.findByText("My Gmail")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Revoke My Gmail" }));
    await waitFor(() => expect(mocks.revokeKey).toHaveBeenCalled());
    expect(mocks.revokeKey.mock.calls.at(0)?.[0]).toBe("synthetic-key-id");
    await waitFor(() =>
      expect(screen.queryByText("My Gmail")).not.toBeInTheDocument(),
    );
  });

  it("lets an owner forget saved passwords", async () => {
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: "Forget saved passwords" }),
    );

    await waitFor(() => expect(mocks.forgetPasswords).toHaveBeenCalledTimes(1));
    expect(
      await screen.findByText("Saved statement passwords were removed."),
    ).toBeInTheDocument();
  });

  it("does not offer key changes to a viewer", async () => {
    mocks.role = "viewer";

    renderPage();

    await screen.findByText("Finish the setup");
    expect(
      screen.queryByRole("button", { name: "Create an upload key" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Forget saved passwords" }),
    ).not.toBeInTheDocument();
  });
});
