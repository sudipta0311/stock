import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";
import englishCopy from "../../src/copy/en.json";
import { InstallExperience } from "../../src/app/InstallExperience";
import { markValueMoment } from "../../src/pwa/install";

function setBrowser({
  userAgent,
  platform,
}: {
  userAgent: string;
  platform: string;
}): void {
  Object.defineProperty(navigator, "userAgent", {
    configurable: true,
    value: userAgent,
  });
  Object.defineProperty(navigator, "platform", {
    configurable: true,
    value: platform,
  });
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn(() => ({
      matches: false,
      media: "",
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

function renderInstall() {
  return render(
    <MemoryRouter>
      <InstallExperience />
    </MemoryRouter>,
  );
}

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

describe("install guidance", () => {
  it("waits for a value moment before showing the Android install prompt", async () => {
    const user = userEvent.setup();
    setBrowser({ userAgent: "Android", platform: "Linux armv8l" });
    const prompt = vi.fn().mockResolvedValue(undefined);
    const choice = Promise.resolve({ outcome: "accepted" as const });
    const event = new Event("beforeinstallprompt", { cancelable: true });
    Object.defineProperty(event, "prompt", { value: prompt });
    Object.defineProperty(event, "userChoice", { value: choice });
    renderInstall();
    window.dispatchEvent(event);
    expect(
      screen.queryByRole("button", { name: englishCopy.install.button }),
    ).not.toBeInTheDocument();

    markValueMoment();
    expect(
      await screen.findByText(englishCopy.install.card_sentence),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: englishCopy.install.button }),
    );
    expect(prompt).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(
        screen.queryByText(englishCopy.install.card_sentence),
      ).not.toBeInTheDocument(),
    );
  });

  it("shows iOS install steps once, then waits 30 days before showing them again", async () => {
    const user = userEvent.setup();
    setBrowser({ userAgent: "iPhone", platform: "iPhone" });
    markValueMoment();
    const first = renderInstall();
    expect(
      await screen.findByRole("dialog", {
        name: englishCopy.install.ios_title,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(englishCopy.install.ios_sentence),
    ).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: englishCopy.install.ios_close }),
    );
    first.unmount();

    renderInstall();
    expect(
      screen.queryByRole("dialog", { name: englishCopy.install.ios_title }),
    ).not.toBeInTheDocument();
  });
});
