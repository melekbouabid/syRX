import { useCallback, useEffect, useState, type DependencyList } from "react";
import { errorMessage } from "./api";

interface QueryState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/**
 * Load data when the component mounts and whenever `deps` change. A request that is no
 * longer wanted is aborted, so a slow old answer can never overwrite a newer one.
 * The previous data stays visible while the next page loads.
 */
export function useQuery<T>(load: (signal: AbortSignal) => Promise<T>, deps: DependencyList) {
  const [state, setState] = useState<QueryState<T>>({ data: null, error: null, loading: true });
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setState((previous) => ({ ...previous, loading: true }));
    load(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ data, error: null, loading: false });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) {
          setState({ data: null, error: errorMessage(error), loading: false });
        }
      },
    );
    return () => controller.abort();
    // `load` is a fresh closure on every render; `deps` says when it really changed.
  }, [...deps, version]);

  const reload = useCallback(() => setVersion((current) => current + 1), []);
  return { ...state, reload };
}

/** The value, but only after it has stopped changing for `delay` milliseconds. */
export function useDebounced<T>(value: T, delay = 250): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}
