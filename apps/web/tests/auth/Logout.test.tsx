import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";
import { AuthProvider, useAuth } from "../../src/auth/AuthContext";
import { getAuthToken, setAuthToken } from "../../src/auth/token";
import { queryClient } from "../../src/api/query-client";

const mocks = vi.hoisted(() => ({
  goToLogin: vi.fn(),
  clearAllDeviceData: vi.fn().mockResolvedValue(undefined),
  disablePush: vi.fn().mockResolvedValue(undefined),
  removeUser: vi.fn().mockResolvedValue(undefined),
  signoutRedirect: vi.fn(),
}));

vi.mock("../../src/auth/navigation", () => ({
  goToLogin: mocks.goToLogin,
}));
vi.mock("../../src/pwa/offline-cache", () => ({
  clearAllDeviceData: mocks.clearAllDeviceData,
}));
vi.mock("../../src/pwa/push", () => ({ disablePush: mocks.disablePush }));
vi.mock("../../src/auth/oidc", () => ({
  getOidcUserManager: async () => ({
    removeUser: mocks.removeUser,
    signoutRedirect: mocks.signoutRedirect,
  }),
}));

function SignOutButton({ disablePush }: { disablePush: boolean }) {
  const { logout } = useAuth();
  return (
    <button
      type="button"
      onClick={() => {
        setAuthToken("synthetic-token");
        window.sessionStorage.setItem("other-session-data", "temporary");
        queryClient.setQueryData(["private"], { value: "synthetic" });
        void logout({ disablePush });
      }}
    >
      Sign out
    </button>
  );
}

beforeEach(() => {
  window.sessionStorage.clear();
  mocks.goToLogin.mockClear();
  mocks.clearAllDeviceData.mockClear();
  mocks.disablePush.mockClear();
  mocks.removeUser.mockClear();
  mocks.signoutRedirect.mockClear();
  queryClient.clear();
  vi.stubEnv("VITE_AUTH_MODE", "dev");
});

async function signOut(disablePush: boolean): Promise<void> {
  const user = userEvent.setup();
  render(
    <AuthProvider>
      <SignOutButton disablePush={disablePush} />
    </AuthProvider>,
  );
  await user.click(screen.getByRole("button", { name: "Sign out" }));
}

it("clears the session while keeping this phone's push registration by default", async () => {
  await signOut(false);

  await waitFor(() => expect(mocks.goToLogin).toHaveBeenCalledOnce());
  expect(getAuthToken()).toBeNull();
  expect(window.sessionStorage.getItem("other-session-data")).toBeNull();
  expect(window.sessionStorage.getItem("wealth.auth.notice")).toBe(
    "signed-out",
  );
  expect(queryClient.getQueryData(["private"])).toBeUndefined();
  expect(mocks.disablePush).not.toHaveBeenCalled();
  expect(mocks.clearAllDeviceData).toHaveBeenCalledWith({
    preservePushSubscription: true,
    preserveTourHistory: true,
  });
});

it("removes this phone's push registration when requested", async () => {
  await signOut(true);

  await waitFor(() => expect(mocks.goToLogin).toHaveBeenCalledOnce());
  expect(mocks.disablePush).toHaveBeenCalledOnce();
  expect(mocks.clearAllDeviceData).toHaveBeenCalledWith({
    preservePushSubscription: false,
    preserveTourHistory: true,
  });
});

it("clears the app's OIDC session without logging out of Google", async () => {
  vi.stubEnv("VITE_AUTH_MODE", "oidc");
  await signOut(false);
  await waitFor(() => expect(mocks.goToLogin).toHaveBeenCalledOnce());
  expect(mocks.removeUser).toHaveBeenCalledOnce();
  expect(mocks.signoutRedirect).not.toHaveBeenCalled();
});
