import { useQuery } from "@tanstack/react-query";
import { apiPathAvailable } from "../api/feature-support";
import { ApiError } from "../api/client";
import { Badge, Card, EmptyState, Skeleton } from "../ui";
import type { ReactNode } from "react";

export function useFeatureQuery<T>(
  queryKey: readonly unknown[],
  path: string,
  queryFn: () => Promise<T>,
) {
  const supported = apiPathAvailable(path);
  const query = useQuery({ queryKey, queryFn, enabled: supported });
  return { ...query, supported };
}

export function isMissingFeature(error: unknown): boolean {
  return (
    error instanceof ApiError && (error.status === 404 || error.status === 405)
  );
}

export function ComingSoon({
  title,
  description = "This information will appear here when the service is available.",
}: {
  title: string;
  description?: string;
}) {
  return (
    <EmptyState
      title={title}
      description={description}
      action={<Badge>Coming soon</Badge>}
    />
  );
}

export function ScreenLoading({
  label = "Loading this screen",
}: {
  label?: string;
}) {
  return (
    <div className="grid gap-4" aria-label={label}>
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-44 w-full" />
      <Skeleton className="h-32 w-full" />
    </div>
  );
}

export function ScreenError({
  title = "We couldn’t load this yet",
}: {
  title?: string;
}) {
  return (
    <Card>
      <p className="text-sm font-semibold">{title}</p>
      <p className="mt-1 text-sm text-muted-foreground">
        Try again in a little while. Your saved information is safe.
      </p>
    </Card>
  );
}

export function OptionalSection({
  supported,
  loading,
  error,
  children,
  title,
}: {
  supported: boolean;
  loading: boolean;
  error: unknown;
  children: ReactNode;
  title: string;
}) {
  if (!supported || isMissingFeature(error)) {
    return <ComingSoon title={`${title} is coming soon`} />;
  }
  if (loading) return <Skeleton className="h-36 w-full" />;
  if (error) return <ScreenError title={`${title} couldn’t load`} />;
  return <>{children}</>;
}
