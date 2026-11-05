import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import {
  MemoryRouter,
  Route,
  Routes,
  useLocation,
  useNavigate,
} from "react-router";
import { describe, expect, it, vi } from "vitest";
import { OnboardingPage } from "../../src/onboarding/OnboardingPage";
import { server } from "../msw";
import { axe } from "vitest-axe";

function CurrentPath() {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <>
      <p>
        {location.pathname}
        {location.search}
      </p>
      <button type="button" onClick={() => navigate(-1)}>
        Back
      </button>
    </>
  );
}

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

  it("reaches the ready screen in five taps when adding a spouse", async () => {
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
          <Route path="/money" element={<CurrentPath />} />
          <Route path="/" element={<CurrentPath />} />
        </Routes>
      </MemoryRouter>,
    );

    let taps = 0;
    await user.click(
      await screen.findByRole("button", { name: "Get started" }),
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
      await screen.findByRole("heading", { name: "You're ready to start" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("progressbar", { name: "Onboarding progress" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/\bof 6\b/)).not.toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: "Add holdings" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Add holdings" }));
    taps += 1;
    expect(await screen.findByText("/money?add=1")).toBeInTheDocument();
    expect(taps).toBeLessThanOrEqual(5);
    await user.click(screen.getByRole("button", { name: "Back" }));
    await user.click(screen.getByRole("button", { name: "Explore the app" }));
    expect(await screen.findByText("/")).toBeInTheDocument();
  });

  it("saves a typed goal before continuing and confirms the save", async () => {
    const user = userEvent.setup();
    localStorage.clear();
    let goalBody: Record<string, unknown> | null = null;
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          display_name: "Synthetic user",
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
      http.get(/\/v1\/meta\/risk-questions$/, () => HttpResponse.json([])),
      http.post(/\/v1\/goals$/, async ({ request }) => {
        goalBody = (await request.json()) as Record<string, unknown>;
        return HttpResponse.json(
          { id: "goal-synthetic", ...goalBody },
          { status: 201 },
        );
      }),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/goals"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/" element={<CurrentPath />} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "What are you saving for?" });
    await user.type(screen.getByLabelText(/Target amount/), "500000");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByText("Goal saved")).toBeVisible();
    expect(
      screen.getByRole("link", { name: "Add another goal" }),
    ).toBeVisible();
    expect(goalBody).toMatchObject({
      target_amount: "500000.00",
      target_year: new Date().getFullYear() + 5,
      currency: "INR",
      priority: 1,
    });
  });

  it("keeps the goal amount when saving fails", async () => {
    const user = userEvent.setup();
    localStorage.clear();
    let goalPosts = 0;
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          display_name: "Synthetic user",
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
      http.post(/\/v1\/goals$/, () => {
        goalPosts += 1;
        return HttpResponse.json(
          { detail: "Synthetic failure" },
          { status: 422 },
        );
      }),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/goals"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/onboarding/risk" element={<p>Risk step</p>} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "What are you saving for?" });
    const amount = screen.getByLabelText(/Target amount/);
    await user.type(amount, "250000");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await waitFor(() => expect(goalPosts).toBe(1));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter a target amount and year, then try again.",
    );
    expect(amount).toHaveValue("250000.00");
    expect(screen.queryByText("Risk step")).not.toBeInTheDocument();
  });

  it("continues setup with an empty goal and keeps one primary action", async () => {
    const user = userEvent.setup();
    localStorage.clear();
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          display_name: "Synthetic user",
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
    );
    render(
      <MemoryRouter initialEntries={["/onboarding/goals"]}>
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/onboarding/risk" element={<p>Risk step</p>} />
        </Routes>
      </MemoryRouter>,
    );
    await screen.findByRole("heading", { name: "What are you saving for?" });
    expect(
      screen.getByText("Pick a goal, then add a rough target and year."),
    ).toBeVisible();
    expect(screen.getByRole("button", { name: "Continue" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByText("Risk step")).toBeVisible();
  });

  it("uses a simple home return when a goal form is opened from Home", async () => {
    localStorage.clear();
    server.use(
      http.get(/\/v1\/me$/, () =>
        HttpResponse.json({
          id: "user-test",
          display_name: "Synthetic user",
          created_at: "2026-10-28T00:00:00Z",
          profile: null,
          members: [],
          me_member_id: null,
        }),
      ),
      http.get(/\/v1\/members$/, () => HttpResponse.json([])),
      http.get(/\/v1\/goals$/, () => HttpResponse.json([])),
    );
    render(
      <MemoryRouter
        initialEntries={[
          { pathname: "/onboarding/goals", state: { returnTo: "/" } },
        ]}
      >
        <Routes>
          <Route path="/onboarding/*" element={<OnboardingPage />} />
          <Route path="/" element={<CurrentPath />} />
        </Routes>
      </MemoryRouter>,
    );
    expect(
      await screen.findByRole("heading", { name: "Add a goal" }),
    ).toBeVisible();
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(
      screen.getByText("Pick a goal, then add a rough target and year."),
    ).toBeVisible();
    const user = userEvent.setup();
    expect(screen.getByRole("button", { name: "Save goal" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Skip for now" })).toBeNull();
    expect(
      screen.queryByText("You can skip goals and add them later."),
    ).toBeNull();
    await user.click(screen.getByRole("button", { name: "Save goal" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Enter a target amount.",
    );
    await user.click(screen.getByRole("link", { name: "Back to Home" }));
    expect(await screen.findByText("/")).toBeInTheDocument();
  });
});
