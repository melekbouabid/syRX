import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api, setSessionEndedHandler, type User } from "./api";

interface AuthState {
  /** The signed-in user, or null. Meaningful once `ready` is true. */
  user: User | null;
  /** False until the first "who am I?" request has answered. */
  ready: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (account: { email: string; full_name: string; password: string }) => Promise<void>;
  /** Ends the session on the server, then reloads on the sign-in page. Rejects if it failed. */
  signOut: () => Promise<void>;
  /** Replace the cached user after the profile was edited. */
  setUser: (user: User) => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .me()
      .then((current) => !cancelled && setUser(current))
      .catch(() => undefined) // not signed in, or the API is down: show the sign-in page
      .finally(() => !cancelled && setReady(true));
    setSessionEndedHandler(() => setUser(null));
    return () => {
      cancelled = true;
      setSessionEndedHandler(null);
    };
  }, []);

  const value = useMemo<AuthState>(
    () => ({
      user,
      ready,
      signIn: async (email, password) => setUser(await api.signIn(email, password)),
      signUp: async (account) => setUser(await api.signUp(account)),
      signOut: async () => {
        await api.signOut();
        // A full page load, not a route change: nothing the previous account loaded stays
        // in memory, and the next sign-in starts on that account's own start page.
        window.location.assign("/login");
      },
      setUser,
    }),
    [user, ready],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthState {
  const state = useContext(AuthContext);
  if (!state) throw new Error("useAuth must be used inside <AuthProvider>");
  return state;
}

/** Where each role lands after signing in. The server enforces the same split. */
export function homePath(user: User): string {
  return user.role === "admin" ? "/admin" : "/home";
}
