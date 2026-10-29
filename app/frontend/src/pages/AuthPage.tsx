import { useEffect, useState, type FormEvent } from "react";
import { Link, Navigate, useLocation } from "react-router-dom";
import { api, errorMessage, type CatalogSummary, type User } from "../api";
import { homePath, useAuth } from "../auth";
import { APP_NAME } from "../brand";
import {
  Alert,
  Logo,
  LogoMark,
  PasswordStrength,
  Spinner,
  SubmitButton,
  TextField,
} from "../components";
import { useQuery } from "../hooks";
import { CountUp } from "../motion";

const MIN_PASSWORD_LENGTH = 8;

const FACTS: { key: keyof CatalogSummary; label: string }[] = [
  { key: "synergy_measurements", label: "measurements" },
  { key: "drugs", label: "drugs" },
  { key: "cell_lines", label: "cell lines" },
];

/** Go back to the page that asked for a sign-in, but only inside the user's own area. */
function destination(user: User, from: unknown): string {
  const home = homePath(user);
  if (typeof from !== "string") return home;
  const adminArea = from.startsWith("/admin");
  return adminArea === (user.role === "admin") ? from : home;
}

export default function AuthPage({ mode }: { mode: "signin" | "signup" }) {
  const { user, ready, signIn, signUp } = useAuth();
  const location = useLocation();
  const from = (location.state as { from?: unknown } | null)?.from;
  const summary = useQuery((signal) => api.catalogSummary(signal), []);

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signingUp = mode === "signup";

  useEffect(() => {
    setError(null);
    setPassword("");
    setConfirm("");
  }, [mode]);

  if (!ready) return <Spinner label="Checking your session" />;
  // Also what moves the user on after a successful sign-in or sign-up.
  if (user) return <Navigate to={destination(user, from)} replace />;

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (signingUp && password !== confirm) {
      setError("The two passwords do not match.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (signingUp) {
        await signUp({ email, full_name: fullName, password });
      } else {
        await signIn(email, password);
      }
    } catch (failure) {
      setError(errorMessage(failure));
      setBusy(false);
    }
  };

  const facts = FACTS.flatMap((fact) => {
    const value = summary.data?.[fact.key];
    return value == null ? [] : [{ ...fact, value }];
  });

  return (
    <div className="auth">
      <aside className="auth-aside">
        <div className="orbs" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <p className="eyebrow">Drug-combination synergy in cancer cell lines</p>
        <div className="auth-aside-body">
          <div className="emblem-stage" aria-hidden="true">
            <span className="emblem-ring" />
            <span className="emblem-ring" />
            <LogoMark size={150} className="emblem-float" />
          </div>
          <h1>Find drug combinations that work better together.</h1>
          <p>
            Explore measured synergy between drug pairs across cancer cell lines, built from
            DrugComb, TDC, O'Neil and DepMap.
          </p>
          {facts.length > 0 && (
            <ul className="auth-facts">
              {facts.map((fact) => (
                <li key={fact.key}>
                  <strong>
                    <CountUp value={fact.value} duration={1400} />
                  </strong>
                  {fact.label}
                </li>
              ))}
            </ul>
          )}
        </div>
        <p className="auth-footnote">Research use only. Not medical advice.</p>
      </aside>

      <main className="auth-main">
        <div className="auth-column">
          <Link to="/" className="auth-logo" title={`About ${APP_NAME}`}>
            <Logo height={132} />
          </Link>

          <div className="auth-card">
            <h2>{signingUp ? "Create your account" : "Welcome back"}</h2>
            <p className="muted">
              {signingUp ? "It takes less than a minute." : "Sign in to continue."}
            </p>

            <nav className="tabs" data-mode={mode} aria-label="Sign in or sign up">
              <Link
                to="/login"
                state={location.state}
                aria-current={signingUp ? undefined : "page"}
                className={signingUp ? "tab" : "tab tab-active"}
              >
                Sign in
              </Link>
              <Link
                to="/signup"
                state={location.state}
                aria-current={signingUp ? "page" : undefined}
                className={signingUp ? "tab tab-active" : "tab"}
              >
                Sign up
              </Link>
            </nav>

            <form onSubmit={submit} className="form">
              {error && <Alert>{error}</Alert>}
              {signingUp && (
                <div className="field-enter">
                  <TextField
                    label="Full name"
                    value={fullName}
                    onChange={setFullName}
                    autoComplete="name"
                    maxLength={120}
                    required
                    autoFocus
                  />
                </div>
              )}
              <TextField
                label="Email"
                type="email"
                value={email}
                onChange={setEmail}
                autoComplete={signingUp ? "email" : "username"}
                placeholder="you@example.com"
                required
                autoFocus={!signingUp}
              />
              <TextField
                label="Password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={setPassword}
                autoComplete={signingUp ? "new-password" : "current-password"}
                minLength={signingUp ? MIN_PASSWORD_LENGTH : undefined}
                maxLength={128}
                hint={signingUp ? `At least ${MIN_PASSWORD_LENGTH} characters.` : undefined}
                required
              />
              {signingUp && (
                <>
                  <PasswordStrength password={password} minLength={MIN_PASSWORD_LENGTH} />
                  <div className="field-enter">
                    <TextField
                      label="Confirm password"
                      type={showPassword ? "text" : "password"}
                      value={confirm}
                      onChange={setConfirm}
                      autoComplete="new-password"
                      maxLength={128}
                      required
                    />
                  </div>
                </>
              )}
              <label className="check">
                <input
                  type="checkbox"
                  checked={showPassword}
                  onChange={(event) => setShowPassword(event.target.checked)}
                />
                Show password
              </label>
              <SubmitButton busy={busy} className="btn btn-primary btn-lg btn-block">
                {signingUp ? "Create account" : "Sign in"}
              </SubmitButton>
            </form>

            <p className="auth-switch muted">
              {signingUp ? "Already have an account? " : "New here? "}
              <Link to={signingUp ? "/login" : "/signup"} state={location.state}>
                {signingUp ? "Sign in" : "Create an account"}
              </Link>
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
