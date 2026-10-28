import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it, vi } from "vitest";
import { OnboardingPage } from "../../src/onboarding/OnboardingPage";
import { ONBOARDING_STORAGE_KEY } from "../../src/money/config";
import { server } from "../msw";
import { axe } from "vitest-axe";

vi.mock("../../src/auth/AuthContext", () => ({
  useAuth: () => ({ refreshProfile: async () => undefined }),
}));

const riskQuestions = Array.from({ length: 6 }, (_, index) => ({
  id: `question-${index + 1}`,
  text: `Risk question ${index + 1}`,
  options: [
    { id: "steady", text: `Steady option ${index + 1}` },
    { id: "growth", text: `Growth option ${index + 1}` },
  ],
}));

describe("Money Map onboarding", () => {
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
    expect(screen.getByLabelText("Name")).toBeRequired();
    expect(screen.getByLabelText("Tax residence")).toBeInTheDocument();
    expect(
      screen.getByText("Add more details (optional)").closest("details"),
    ).not.toHaveAttribute("open");
    expect(await axe(document.body)).toEqual(
      expect.objectContaining({ violations: [] }),
    );
  });

  it("saves two adult residences, then completes the C20 flow", async () => {
    const user = userEvent.setup();
    localStorage.clear();
    let goalBody: Record<string, unknown> | null = null;
    let answersBody: Record<string, unknown> | null = null;
    const memberBodies: Record<string, unknown>[] = [];
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
      http.post(/\/v1\/members$/, async ({ request }) =>
        HttpResponse.json(
          { id: "member-test-1", ...((await request.json()) as object) },
          { status: 201 },
        ),
      ),
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
      http.post(/\/v1\/goals$/, async ({ request }) => {
        goalBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          { id: "goal-test", ...goalBody },
          { status: 201 },
        );
      }),
      http.get(/\/v1\/meta\/risk-questions$/, () =>
        HttpResponse.json(riskQuestions),
      ),
      http.put(/\/v1\/me\/risk-answers$/, async ({ request }) => {
        answersBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json({
          tolerance: "70",
          capacity: "60",
          final: "Growth",
          limited_by: "capacity",
        });
      }),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/welcome"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/money" element={<p>Money Map landing</p>} />
        </Routes>
      </MemoryRouter>,
    );

    await user.click(
      await screen.findByRole("button", { name: /Set up my Money Map/ }),
    );
    await screen.findByRole("heading", { name: "Who are we planning for?" });
    await user.click(screen.getByRole("button", { name: /Me and my spouse/ }));
    const names = screen.getAllByLabelText("Name");
    await user.type(names[0]!, "Demo Adult");
    await user.type(names[1]!, "Demo Partner");
    const residences = screen.getAllByLabelText("Tax residence");
    await user.selectOptions(residences[0]!, "NRI");
    await user.selectOptions(residences[1]!, "RI");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await screen.findByRole("heading", { name: "What are you saving for?" });
    expect(memberBodies).toHaveLength(2);
    expect(memberBodies[0]).toMatchObject({
      display_name: "Demo Adult",
      relationship: "self",
      residency: { status: "NRI", country: "DK" },
    });
    expect(memberBodies[1]).toMatchObject({
      display_name: "Demo Partner",
      relationship: "spouse",
      residency: { status: "RI", country: "IN" },
    });
    await user.click(screen.getByRole("button", { name: /Wealth/ }));
    await user.type(screen.getByLabelText(/Target amount/), "5L");
    await user.click(
      screen.getByRole("button", { name: "Add Wealth building" }),
    );
    await waitFor(() =>
      expect(goalBody).toMatchObject({
        goal_type: "wealth",
        target_amount: "500000.00",
        currency: "INR",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await screen.findByRole("heading", { name: "How do you feel about risk?" });

    for (const question of riskQuestions) {
      const group = screen.getByRole("group", { name: question.text });
      await user.click(
        within(group).getByRole("button", { name: question.options[1]?.text }),
      );
    }
    await user.click(screen.getByRole("button", { name: "Save answers" }));
    expect(await screen.findByText("Your plan: Growth")).toBeInTheDocument();
    expect(answersBody).toMatchObject({
      answers: { "question-1": "growth", "question-6": "growth" },
    });
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await screen.findByRole("heading", {
      name: "How should we keep you updated?",
    });
    expect(
      screen.getByRole("checkbox", { name: /Weekly email/ }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", { name: /Reminders/ }),
    ).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", { name: /Push notifications/ }),
    ).not.toBeChecked();
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await screen.findByRole("heading", {
      name: "You're ready to build your Money Map",
    });
    await user.click(screen.getByRole("button", { name: /Go to Money Map/ }));
    expect(await screen.findByText("Money Map landing")).toBeInTheDocument();
    expect(
      JSON.parse(localStorage.getItem(ONBOARDING_STORAGE_KEY) ?? "{}").consents,
    ).toEqual({ weeklyEmail: false, reminders: false, push: false });
  });
});
