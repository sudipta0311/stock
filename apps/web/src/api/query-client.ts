import { QueryClient, useMutation } from "@tanstack/react-query";
import type { MutationFunction, QueryKey } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
    mutations: { retry: 0 },
  },
});

export function useApiMutation<TData, TVariables>(
  mutationFn: MutationFunction<TData, TVariables>,
  invalidates: readonly QueryKey[],
) {
  return useMutation({
    mutationFn,
    onSuccess: async () => {
      await Promise.all(
        invalidates.map((queryKey) =>
          queryClient.invalidateQueries({ queryKey }),
        ),
      );
    },
  });
}
