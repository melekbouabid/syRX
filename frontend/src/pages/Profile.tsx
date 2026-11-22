import { useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { api, errorMessage, type User } from "../services/api";
import { useAuth } from "../services/auth";
import {
  Alert,
  Avatar,
  Badge,
  formatDate,
  PasswordStrength,
  SubmitButton,
  TextField,
} from "../components";
import { Icon } from "../components/icons";
import { SignaturePad } from "../components/SignaturePad";
import { useT } from "../i18n";

const MIN_PASSWORD_LENGTH = 8;

type Notice = { kind: "success" | "error"; text: string } | null;

function NameForm() {
  const { user, setUser } = useAuth();
  const t = useT();
  const [fullName, setFullName] = useState(user?.full_name ?? "");
  const [phone, setPhone] = useState(user?.phone ?? "");
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setNotice(null);
    try {
      setUser(await api.updateProfile(fullName, phone.trim()));
      setNotice({ kind: "success", text: t.profile.detailsUpdated });
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card form" onSubmit={submit}>
      <h2>{t.profile.yourDetails}</h2>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      <TextField
        label={t.common.fullName}
        value={fullName}
        onChange={setFullName}
        autoComplete="name"
        maxLength={120}
        required
      />
      <TextField
        label={t.common.email}
        value={user?.email ?? ""}
        onChange={() => undefined}
        hint={t.profile.emailHint}
        disabled
      />
      <TextField
        label={t.profile.phone}
        type="tel"
        value={phone}
        onChange={setPhone}
        autoComplete="tel"
        maxLength={32}
        pattern="[0-9+()./\- ]*"
        hint={t.profile.phoneHint}
      />
      <div className="form-actions">
        <SubmitButton
          busy={busy}
          disabled={fullName.trim() === user?.full_name && phone.trim() === (user?.phone ?? "")}
        >
          {t.common.save}
        </SubmitButton>
      </div>
    </form>
  );
}

function PasswordForm() {
  const t = useT();
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (next !== confirm) {
      setNotice({ kind: "error", text: t.profile.mismatch });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      await api.changePassword(current, next);
      setCurrent("");
      setNext("");
      setConfirm("");
      setNotice({ kind: "success", text: t.profile.changed });
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="card form" onSubmit={submit}>
      <h2>{t.profile.changePassword}</h2>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      <TextField
        label={t.profile.current}
        type="password"
        value={current}
        onChange={setCurrent}
        autoComplete="current-password"
        maxLength={128}
        required
      />
      <TextField
        label={t.profile.newPassword}
        type="password"
        value={next}
        onChange={setNext}
        autoComplete="new-password"
        minLength={MIN_PASSWORD_LENGTH}
        maxLength={128}
        hint={t.common.atLeast(MIN_PASSWORD_LENGTH)}
        required
      />
      <PasswordStrength password={next} minLength={MIN_PASSWORD_LENGTH} />
      <TextField
        label={t.profile.confirmNew}
        type="password"
        value={confirm}
        onChange={setConfirm}
        autoComplete="new-password"
        maxLength={128}
        required
      />
      <div className="form-actions">
        <SubmitButton busy={busy}>{t.profile.changePassword}</SubmitButton>
      </div>
    </form>
  );
}

/** The profile photo: chosen from a file, cropped to a square by the server. */
function PhotoControl() {
  const { user, setUser } = useAuth();
  const t = useT();
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  if (!user) return null;

  const run = async (action: () => Promise<User>) => {
    setBusy(true);
    setProblem(null);
    try {
      setUser(await action());
    } catch (failure) {
      setProblem(errorMessage(failure));
    } finally {
      setBusy(false);
    }
  };

  const choose = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void run(() => api.setImage("photo", file, file.name));
  };

  return (
    <div className="photo-control">
      <Avatar name={user.full_name} size="lg" src={user.photo_url} />
      <div className="photo-actions">
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={choose} />
        <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => input.current?.click()}>
          <Icon name="camera" size={16} />
          {user.photo_url ? t.profile.changePhoto : t.profile.addPhoto}
        </button>
        {user.photo_url && (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={busy}
            onClick={() => void run(() => api.deleteImage("photo"))}
          >
            {t.common.remove}
          </button>
        )}
      </div>
      {problem && <Alert>{problem}</Alert>}
    </div>
  );
}

/** The signature printed on the user's own reports. Never shown to administrators. */
function SignatureCard() {
  const { user, setUser } = useAuth();
  const t = useT();
  const input = useRef<HTMLInputElement>(null);
  const [editing, setEditing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice>(null);
  if (!user) return null;

  const run = async (action: () => Promise<User>, done: string) => {
    setBusy(true);
    setNotice(null);
    try {
      setUser(await action());
      setEditing(false);
      setNotice({ kind: "success", text: done });
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    } finally {
      setBusy(false);
    }
  };

  const upload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) void run(() => api.setImage("signature", file, file.name), t.profile.signatureSaved);
  };

  return (
    <section className="card form">
      <h2>
        <Icon name="pen" size={18} /> {t.profile.signature}
      </h2>
      <p className="muted">{t.profile.signatureText}</p>
      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      {user.signature_url && !editing && (
        <div className="signature-preview">
          <img src={user.signature_url} alt={t.profile.yourSignature} />
        </div>
      )}
      {(editing || !user.signature_url) && (
        <SignaturePad
          busy={busy}
          onDone={(png) =>
            void run(() => api.setImage("signature", png, "signature.png"), t.profile.signatureSaved)
          }
        />
      )}
      <div className="form-actions">
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={upload} />
        <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => input.current?.click()}>
          <Icon name="upload" size={16} />
          {t.profile.uploadScan}
        </button>
        {user.signature_url && !editing && (
          <button type="button" className="btn btn-ghost btn-sm" disabled={busy} onClick={() => setEditing(true)}>
            {t.profile.drawNew}
          </button>
        )}
        {user.signature_url && (
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            disabled={busy}
            onClick={() => void run(() => api.deleteImage("signature"), t.profile.signatureRemoved)}
          >
            {t.common.remove}
          </button>
        )}
      </div>
    </section>
  );
}

export default function Profile() {
  const { user } = useAuth();
  const t = useT();
  if (!user) return null;

  return (
    <>
      <header className="profile-header">
        <PhotoControl />
        <div>
          <h1>{user.full_name}</h1>
          <p className="muted">{user.email}</p>
          <p className="profile-meta">
            <Badge tone={user.role}>{t.common.roles[user.role] ?? user.role}</Badge>
            <span className="muted">{t.profile.memberSince(formatDate(user.created_at))}</span>
            <span className="muted">{t.profile.lastSignIn(formatDate(user.last_login_at, true))}</span>
            {user.phone && <span className="muted">{user.phone}</span>}
          </p>
        </div>
      </header>
      <div className="two-columns">
        <NameForm />
        <PasswordForm />
      </div>
      <div className="two-columns mt-5">
        <SignatureCard />
      </div>
    </>
  );
}
