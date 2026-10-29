import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { OnboardingPage } from "../../src/onboarding/OnboardingPage";
import { server } from "../msw";
import { axe } from "vitest-axe";

vi.mock("../../src/auth/AuthContext", () => ({
  useAuth: () => ({ refreshProfile: async () => undefined }),
}));

describe("Money Map onboarding", () => {
  it.each(["welcome", "household", "goals", "risk", "consent", "done"])(
    "shows one primary action on the %s step",
    async (step) => {
      server.use(
        http.get(/\/v1\/me$/, () =>
          HttpResponse.json({
            id: "user-test",
            email: "user@example.test",
            display_name: null,
            created_at: "2026-10-28T00:00:00Z",
            profile: null,
            members: [],
            me_member_id: null,
          }),
        ),
        http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
        http.get(/\/v1\/members$/, () => HttpResponse.json([])),
        http.get(/\/v1\/meta\/risk-questions$/, () =>
          HttpResponse.json(
            Array.from({ length: 6 }, (_, index) => ({
              id: `question-${index + 1}`,
              text: `Risk question ${index + 1}`,
              options: [
                { id: "steady", text: `Steady option ${index + 1}` },
                { id: "growth", text: `Growth option ${index + 1}` },
              ],
            })),
          ),
        ),
      );
      render(
        <MemoryRouter initialEntries={[`/onboarding/${step}`]}>
          <Routes>
            <Route path="/onboarding/*" element={<OnboardingPage />} />
          </Routes>
        </MemoryRouter>,
      );

      await screen.findByRole("heading");
      expect(document.querySelectorAll('[data-primary="true"]')).toHaveLength(
        1,
      );
    },
  );

  it("asks for the household basics and keeps extra questions closed", async () => {
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          email: "user@example.test",
          display_name: null,
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/household"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(
      await screen.findByRole("heading", { name: "Who are we planning for?" }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Name")).toHaveValue("You");
    expect(screen.getByLabelText("Tax residence")).toBeInTheDocument();
    expect(document.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    expect(
      screen.getByText("Add more details (optional)").closest("details"),
    ).not.toHaveAttribute("open");
    expect(await axe(document.body)).toEqual(
      expect.objectContaining({ violations: [] }),
    );
  });

  it("reaches the first plan in five taps when adding a spouse", async () => {
    const user = userEvent.setup();
    localStorage.clear();
    const memberBodies: Record<string, unknown>[] = [];
    let goalRequests = 0;
    let riskRequests = 0;
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          email: "user@example.test",
          display_name: null,
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.post(/\/v1\/members$/, async ({ request }) => {
        const body = (await request.json()) as Record<string, unknown>;
        memberBodies.push(body);
        return HttpResponse.json(
          { id: `member-${memberBodies.length}`, ...body },
          { status: 201 },
        );
      }),
      http.put(/\/v1\/me\/profile$/, async ({ request }) =>
        HttpResponse.json({
          ...((await request.json()) as Record<string, unknown>),
          tax_slab_pct: null,
          language: "en",
          financial_context: null,
          risk_answers: null,
          risk_profile: null,
        }),
      ),
      http.post(/\/v1\/goals$/, () => {
        goalRequests += 1;
        return HttpResponse.json({ id: "goal-test" }, { status: 201 });
      }),
      http.put(/\/v1\/me\/risk-answers$/, () => {
        riskRequests += 1;
        return HttpResponse.json({ final: "Balanced" });
      }),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/welcome"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/plan" element={<p>Plan screen</p>} />
        </Routes>
      </MemoryRouter>,
    );

    let taps = 0;
    await user.click(
      await screen.findByRole("button", { name: /Set up my Money Map/ }),
    );
    taps += 1;
    await screen.findByRole("heading", { name: "Who are we planning for?" });
    expect(document.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    await user.click(screen.getByRole("button", { name: /Me and my spouse/ }));
    taps += 1;
    const names = screen.getAllByLabelText("Name");
    expect(names[0]).toHaveValue("You");
    expect(names[1]).toHaveValue("Spouse");
    const residences = screen.getAllByLabelText("Tax residence");
    await user.selectOptions(residences[1]!, "NRI");
    taps += 1;
    await user.click(screen.getByRole("button", { name: "Continue" }));
    taps += 1;
    expect(
      await screen.findByRole("heading", { name: "Your plan is ready" }),
    ).toBeInTheDocument();
    expect(memberBodies).toHaveLength(2);
    expect(memberBodies[0]).toMatchObject({
      display_name: "You",
      relationship: "self",
      residency: { status: "RI", country: "IN" },
    });
    expect(memberBodies[1]).toMatchObject({
      display_name: "Spouse",
      relationship: "spouse",
      residency: { status: "NRI", country: "DK" },
    });
    expect(document.querySelectorAll('[data-primary="true"]')).toHaveLength(1);
    expect(goalRequests).toBe(0);
    expect(riskRequests).toBe(0);
    await user.click(screen.getByRole("button", { name: /View your plan/ }));
    taps += 1;
    expect(await screen.findByText("Plan screen")).toBeInTheDocument();
    expect(taps).toBeLessThanOrEqual(5);
  });
});
