import { useState, type FormEvent } from "react";
import { api, errorMessage } from "../api";
import { useAuth } from "../auth";
import {
  Alert,
  Avatar,
  Badge,
  formatDate,
  PasswordStrength,
  SubmitButton,
  TextField,
} from "../components";

const MIN_PASSWORD_LENGTH = 8;

type Notice = { kind: "success" | "error"; text: string } | null;

function NameForm() {
  const { user, setUser } = useAuth();
  const [fullName, setFullName] = useState(user?.full_name ?? "");
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setNotice(null);
    try {
      setUser(await api.updateProfile(fullName));
      setNotice({ kind: "success", text: "Your name was updated." });
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card form" onSubmit={submit}>
      <h2>Your details</h2>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      <TextField
        label="Full name"
        value={fullName}
        onChange={setFullName}
        autoComplete="name"
        maxLength={120}
        required
      />
      <TextField
        label="Email"
        value={user?.email ?? ""}
        onChange={() => undefined}
        hint="Ask an administrator to change your email."
        disabled
      />
      <div className="form-actions">
        <SubmitButton busy={busy} disabled={fullName.trim() === user?.full_name}>
          Save
        </SubmitButton>
      </div>
    </form>
  );
}

function PasswordForm() {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (next !== confirm) {
      setNotice({ kind: "error", text: "The two new passwords do not match." });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      await api.changePassword(current, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      setNotice({
        kind: "success",
        text: "Password changed. Your other devices have been signed out.",
      });
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card form" onSubmit={submit}>
      <h2>Change password</h2>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      <TextField
        label="Current password"
        type="password"
        value={current}
        onChange={setCurrent}
        autoComplete="current-password"
        maxLength={128}
        required
      />
      <TextField
        label="New password"
        type="password"
        value={next}
        onChange={setNext}
        autoComplete="new-password"
        minLength={MIN_PASSWORD_LENGTH}
        maxLength={128}
        hint={`At least ${MIN_PASSWORD_LENGTH} characters.`}
        required
      />
      <PasswordStrength password={next} minLength={MIN_PASSWORD_LENGTH} />
      <TextField
        label="Confirm new password"
        type="password"
        value={confirm}
        onChange={setConfirm}
        autoComplete="new-password"
        maxLength={128}
        required
      />
      <div className="form-actions">
        <SubmitButton busy={busy}>Change password</SubmitButton>
      </div>
    </form>
  );
}

export default function Profile() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <>
      <header className="profile-header">
        <Avatar name={user.full_name} size="lg" />
        <div>
          <h1>{user.full_name}</h1>
          <p className="muted">{user.email}</p>
          <p className="profile-meta">
            <Badge tone={user.role}>{user.role}</Badge>
            <span className="muted">Member since {formatDate(user.created_at)}</span>
            <span className="muted">Last sign-in {formatDate(user.last_login_at, true)}</span>
          </p>
        </div>
      </header>
      <div className="two-columns">
        <NameForm />
        <PasswordForm />
      </div>
    </>
  );
}
