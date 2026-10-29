import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import { AuthProvider } from "../../src/auth/AuthContext";
import { LoginPage } from "../../src/app/AuthPages";
import { RouteGuard } from "../../src/app/RouteGuard";
import { clearAuthToken, setAuthToken } from "../../src/auth/token";

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

beforeEach(() => {
  window.sessionStorage.clear();
  clearAuthToken();
  vi.stubEnv("VITE_AUTH_MODE", "dev");
});

it("returns to the same screen after an expired session is renewed", async () => {
  setAuthToken("synthetic-expired-token");
  const fetchMock = vi
    .fn<typeof fetch>()
    .mockResolvedValueOnce(
      response({ code: "expired", title: "Sign in again", errors: [] }, 401),
    )
    .mockResolvedValueOnce(
      response({
        id: "synthetic-user",
        display_name: "Morgan Sample",
        email: "morgan@example.test",
        household_owner_id: "synthetic-household",
        profile: {},
      }),
    );
  vi.stubGlobal("fetch", fetchMock);

  const user = userEvent.setup();
  render(
    <MemoryRouter initialEntries={["/money"]}>
      <AuthProvider>
        <Routes>
          <Route
            path="/money"
            element={
              <RouteGuard>
                <p>Money screen</p>
              </RouteGuard>
            }
          />
          <Route path="/login" element={<LoginPage />} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>,
  );

  expect(await screen.findByText("Please sign in again.")).toBeInTheDocument();
  expect(window.sessionStorage.getItem("wealth.auth.return-path")).toBe(
    "/money",
  );
  await user.type(
    screen.getByLabelText("Development token"),
    "synthetic-new-token",
  );
  await user.click(screen.getByRole("button", { name: "Continue" }));
  expect(await screen.findByText("Money screen")).toBeInTheDocument();
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2));
});

it("shows the signed-out notice once on the login screen", async () => {
  window.sessionStorage.setItem("wealth.auth.notice", "signed-out");
  const first = render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <LoginPage />
      </AuthProvider>
    </MemoryRouter>,
  );
  expect(await screen.findByText("You've signed out.")).toBeInTheDocument();
  first.unmount();

  render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <LoginPage />
      </AuthProvider>
    </MemoryRouter>,
  );
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: "Continue" }),
    ).toBeInTheDocument(),
  );
  expect(screen.queryByText("You've signed out.")).not.toBeInTheDocument();
});

it("names Google and invited family members in OIDC mode", async () => {
  vi.stubEnv("VITE_AUTH_MODE", "oidc");
  render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <LoginPage />
      </AuthProvider>
    </MemoryRouter>,
  );
  expect(
    await screen.findByRole("button", { name: "Continue with Google" }),
  ).toBeInTheDocument();
  expect(
    screen.getByText("Only invited family members can sign in."),
  ).toBeInTheDocument();
});
