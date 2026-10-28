import { EmptyState } from "../ui";

const pageCopy: Record<string, { title: string; description: string }> = {
  Home: {
    title: "Your week, at a glance",
    description:
      "Your net worth, plan health, and next steps will appear here.",
  },
  Money: {
    title: "Your Money Map starts here",
    description:
      "Add assets or import a statement to see your household wealth in one place.",
  },
  Plan: {
    title: "Your plan will take shape here",
    description:
      "See your targets, goals, and protection checks in one clear view.",
  },
  Ideas: {
    title: "Ideas for your next step",
    description:
      "Research-backed ideas and their buying routes will appear here.",
  },
  Inbox: {
    title: "Nothing needs your attention yet",
    description:
      "Weekly briefs, reminders, and important updates will land here.",
  },
};

export function StubPage({
  section,
  detail,
}: {
  section: string;
  detail?: string;
}) {
  const copy = pageCopy[section] ?? {
    title: section,
    description: detail ?? "This area is ready for the next product update.",
  };
  return (
    <EmptyState title={copy.title} description={detail ?? copy.description} />
  );
}

export function OnboardingPage() {
  return (
    <StubPage
      section="Start with your household"
      detail="A few simple questions will help build your first Money Map."
    />
  );
}
