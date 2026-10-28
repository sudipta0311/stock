import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { beforeEach, expect, it, vi } from "vitest";
import { RouteGuard } from "../../src/app/RouteGuard";

const { auth } = vi.hoisted(() => ({
  auth: { state: "anonymous", hasProfile: null as boolean | null },
}));
vi.mock("../../src/auth/AuthContext", () => ({ useAuth: () => auth }));

function renderGuard() {
  return render(
    <MemoryRouter initialEntries={["/money"]}>
      <Routes>
        <Route path="/login" element={<p>Login screen</p>} />
        <Route path="/onboarding" element={<p>Onboarding screen</p>} />
        <Route
          path="/money"
          element={
            <RouteGuard>
              <p>Money screen</p>
            </RouteGuard>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  auth.state = "anonymous";
  auth.hasProfile = null;
});

it("sends signed-out visitors to login", () => {
  renderGuard();
  expect(screen.getByText("Login screen")).toBeInTheDocument();
});

it("sends a first login without a profile to onboarding", () => {
  auth.state = "authenticated";
  auth.hasProfile = false;
  renderGuard();
  expect(screen.getByText("Onboarding screen")).toBeInTheDocument();
});
