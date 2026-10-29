import { useEffect, useState, type FormEvent } from "react";
import { api, errorMessage, type Role, type User } from "../../api";
import { useAuth } from "../../auth";
import {
  Alert,
  Avatar,
  Badge,
  formatDate,
  Modal,
  Pagination,
  PasswordStrength,
  staggerIndex,
  SubmitButton,
  TableSkeleton,
  TextField,
} from "../../components";
import { useDebounced, useQuery } from "../../hooks";
import { Icon } from "../../icons";

const PAGE_SIZE = 10;
const MIN_PASSWORD_LENGTH = 8;

type Status = "active" | "inactive";
type Notice = { kind: "success" | "error"; text: string } | null;
type Dialog =
  | { kind: "create" }
  | { kind: "edit"; user: User }
  | { kind: "password"; user: User }
  | { kind: "delete"; user: User };

/** 16 characters from the browser's secure random source, without look-alikes (0/O, 1/l). */
function generatePassword(length = 16): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
  const numbers = crypto.getRandomValues(new Uint32Array(length));
  return Array.from(numbers, (n) => alphabet[n % alphabet.length]).join("");
}

interface PasswordInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
}

/** Shown in clear on request: the administrator has to pass the password on. */
function PasswordInput({ label, value, onChange }: PasswordInputProps) {
  const [visible, setVisible] = useState(false);
  return (
    <>
      <TextField
        label={label}
        type={visible ? "text" : "password"}
        value={value}
        onChange={onChange}
        autoComplete="new-password"
        minLength={MIN_PASSWORD_LENGTH}
        maxLength={128}
        hint={`At least ${MIN_PASSWORD_LENGTH} characters. Only a hash is stored: it cannot be shown again.`}
        required
      />
      <PasswordStrength password={value} minLength={MIN_PASSWORD_LENGTH} />
      <div className="inline-actions">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={() => {
            onChange(generatePassword());
            setVisible(true);
          }}
        >
          Generate a strong one
        </button>
        <label className="check">
          <input
            type="checkbox"
            checked={visible}
            onChange={(event) => setVisible(event.target.checked)}
          />
          Show
        </label>
      </div>
    </>
  );
}

interface DialogActionsProps {
  onClose: () => void;
  busy: boolean;
  submitLabel: string;
  danger?: boolean;
}

function DialogActions({ onClose, busy, submitLabel, danger = false }: DialogActionsProps) {
  return (
    <div className="form-actions">
      <button type="button" className="btn btn-secondary" onClick={onClose}>
        Cancel
      </button>
      <SubmitButton busy={busy} className={danger ? "btn btn-danger" : "btn btn-primary"}>
        {submitLabel}
      </SubmitButton>
    </div>
  );
}

/** Run one request for a dialog, tracking its busy and error state. */
function useSubmit(action: () => Promise<void>) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (failure) {
      setError(errorMessage(failure));
      setBusy(false);
    }
  };
  return { error, busy, submit };
}

interface UserFormProps {
  /** The account being edited, or null to create one. */
  target: User | null;
  isSelf: boolean;
  onClose: () => void;
  onSaved: (user: User) => void;
}

function UserFormDialog({ target, isSelf, onClose, onSaved }: UserFormProps) {
  const [fullName, setFullName] = useState(target?.full_name ?? "");
  const [email, setEmail] = useState(target?.email ?? "");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<Role>(target?.role ?? "user");
  const [active, setActive] = useState(target?.is_active ?? true);

  const { error, busy, submit } = useSubmit(async () => {
    const details = { full_name: fullName, email };
    if (!target) {
      onSaved(await api.admin.createUser({ ...details, password, role, is_active: active }));
    } else if (isSelf) {
      onSaved(await api.admin.updateUser(target.id, details));
    } else {
      onSaved(await api.admin.updateUser(target.id, { ...details, role, is_active: active }));
    }
  });

  return (
    <Modal title={target ? `Edit ${target.full_name}` : "New user"} onClose={onClose}>
      <form className="form" onSubmit={submit}>
        {error && <Alert>{error}</Alert>}
        <TextField
          label="Full name"
          value={fullName}
          onChange={setFullName}
          maxLength={120}
          autoComplete="off"
          required
        />
        <TextField
          label="Email"
          type="email"
          value={email}
          onChange={setEmail}
          autoComplete="off"
          required
        />
        {!target && <PasswordInput label="Password" value={password} onChange={setPassword} />}
        <div className="field">
          <label htmlFor="user-role">Role</label>
          <select
            id="user-role"
            value={role}
            disabled={isSelf}
            onChange={(event) => setRole(event.target.value as Role)}
          >
            <option value="user">User: the application only</option>
            <option value="admin">Administrator: full control of accounts</option>
          </select>
        </div>
        <label className="check">
          <input
            type="checkbox"
            checked={active}
            disabled={isSelf}
            onChange={(event) => setActive(event.target.checked)}
          />
          Active (can sign in)
        </label>
        {isSelf && (
          <p className="hint">
            You cannot change your own role or deactivate yourself. Ask another administrator.
          </p>
        )}
        <DialogActions
          onClose={onClose}
          busy={busy}
          submitLabel={target ? "Save changes" : "Create user"}
        />
      </form>
    </Modal>
  );
}

interface TargetDialogProps {
  target: User;
  isSelf: boolean;
  onClose: () => void;
  onDone: () => void;
}

function PasswordDialog({ target, isSelf, onClose, onDone }: TargetDialogProps) {
  const [password, setPassword] = useState("");
  const { error, busy, submit } = useSubmit(async () => {
    await api.admin.resetPassword(target.id, password);
    onDone();
  });

  return (
    <Modal title={`Reset password of ${target.full_name}`} onClose={onClose}>
      <form className="form" onSubmit={submit}>
        {error && <Alert>{error}</Alert>}
        <p className="muted">
          {isSelf
            ? "Your other devices will be signed out. This one stays signed in."
            : `${target.email} will be signed out everywhere and must use the new password.`}
        </p>
        <PasswordInput label="New password" value={password} onChange={setPassword} />
        <DialogActions onClose={onClose} busy={busy} submitLabel="Set password" />
      </form>
    </Modal>
  );
}

function DeleteDialog({ target, onClose, onDone }: Omit<TargetDialogProps, "isSelf">) {
  const { error, busy, submit } = useSubmit(async () => {
    await api.admin.deleteUser(target.id);
    onDone();
  });

  return (
    <Modal title="Delete this account?" onClose={onClose}>
      <form className="form" onSubmit={submit}>
        {error && <Alert>{error}</Alert>}
        <p>
          <strong>{target.full_name}</strong> ({target.email}) will be removed for good and
          signed out. To block someone temporarily, deactivate the account instead.
        </p>
        <DialogActions onClose={onClose} busy={busy} submitLabel="Delete account" danger />
      </form>
    </Modal>
  );
}

export default function Users() {
  const { user: me, setUser: setMe } = useAuth();
  const [search, setSearch] = useState("");
  const [role, setRole] = useState<Role | "">("");
  const [status, setStatus] = useState<Status | "">("");
  const [page, setPage] = useState(1);
  const [dialog, setDialog] = useState<Dialog | null>(null);
  const [notice, setNotice] = useState<Notice>(null);

  const q = useDebounced(search.trim());
  const users = useQuery(
    (signal) => api.admin.users({ q, role, status, page, page_size: PAGE_SIZE }, signal),
    [q, role, status, page],
  );

  // Deleting the last row of the last page would otherwise leave an empty table.
  const pageIsEmpty = users.data?.page === page && users.data.items.length === 0;
  useEffect(() => {
    if (pageIsEmpty && page > 1) setPage(page - 1);
  }, [pageIsEmpty, page]);

  useEffect(() => {
    if (notice?.kind !== "success") return;
    const timer = window.setTimeout(() => setNotice(null), 6000);
    return () => window.clearTimeout(timer);
  }, [notice]);

  const finish = (text: string) => {
    setDialog(null);
    setNotice({ kind: "success", text });
    users.reload();
  };

  const toggleActive = async (target: User) => {
    try {
      await api.admin.updateUser(target.id, { is_active: !target.is_active });
      finish(
        target.is_active
          ? `${target.full_name} was deactivated and signed out.`
          : `${target.full_name} can sign in again.`,
      );
    } catch (failure) {
      setNotice({ kind: "error", text: errorMessage(failure) });
    }
  };

  const close = () => setDialog(null);
  const isSelf = (target: User) => target.id === me?.id;

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="users" size={24} />
          </span>
          <div>
            <h1>Users</h1>
            <p className="muted">
              Create accounts, change roles, reset passwords, block or delete.
            </p>
          </div>
        </div>
        <button type="button" className="btn btn-primary" onClick={() => setDialog({ kind: "create" })}>
          <Icon name="plus" />
          New user
        </button>
      </header>

      <div className="toolbar">
        <label className="search-box">
          <Icon name="search" />
          <input
            type="search"
            aria-label="Search by name or email"
            placeholder="Search by name or email"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />
        </label>
        <select
          aria-label="Filter by role"
          value={role}
          onChange={(event) => {
            setRole(event.target.value as Role | "");
            setPage(1);
          }}
        >
          <option value="">All roles</option>
          <option value="admin">Administrators</option>
          <option value="user">Users</option>
        </select>
        <select
          aria-label="Filter by status"
          value={status}
          onChange={(event) => {
            setStatus(event.target.value as Status | "");
            setPage(1);
          }}
        >
          <option value="">Any status</option>
          <option value="active">Active</option>
          <option value="inactive">Deactivated</option>
        </select>
      </div>

      {notice && <Alert kind={notice.kind}>{notice.text}</Alert>}
      {users.error && <Alert>{users.error}</Alert>}
      {!users.data && users.loading && <TableSkeleton rows={6} />}

      {users.data && (
        <div className="card table-card" aria-busy={users.loading}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Last sign-in</th>
                  <th className="num">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.data.items.map((user, index) => (
                  <tr key={user.id} className="row-enter" style={staggerIndex(index)}>
                    <td>
                      <div className="cell-user">
                        <Avatar name={user.full_name} size="sm" />
                        <div>
                          <span className="cell-title">
                            {user.full_name} {isSelf(user) && <Badge tone="neutral">You</Badge>}
                          </span>
                          <span className="cell-sub">{user.email}</span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <Badge tone={user.role}>{user.role}</Badge>
                    </td>
                    <td>
                      <Badge tone={user.is_active ? "active" : "inactive"}>
                        {user.is_active ? "Active" : "Deactivated"}
                      </Badge>
                    </td>
                    <td>{formatDate(user.created_at)}</td>
                    <td>{formatDate(user.last_login_at, true)}</td>
                    <td className="num">
                      <div className="row-actions">
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          onClick={() => setDialog({ kind: "edit", user })}
                        >
                          Edit
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          onClick={() => setDialog({ kind: "password", user })}
                        >
                          Password
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          disabled={isSelf(user)}
                          title={isSelf(user) ? "You cannot deactivate yourself" : undefined}
                          onClick={() => toggleActive(user)}
                        >
                          {user.is_active ? "Deactivate" : "Activate"}
                        </button>
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm btn-ghost-danger"
                          disabled={isSelf(user)}
                          title={isSelf(user) ? "You cannot delete yourself" : undefined}
                          onClick={() => setDialog({ kind: "delete", user })}
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {users.data.items.length === 0 && (
                  <tr>
                    <td colSpan={6} className="empty">
                      No account matches these filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <Pagination page={page} pageSize={PAGE_SIZE} total={users.data.total} onPage={setPage} />
        </div>
      )}

      {dialog?.kind === "create" && (
        <UserFormDialog
          target={null}
          isSelf={false}
          onClose={close}
          onSaved={(saved) => finish(`${saved.full_name} was created.`)}
        />
      )}
      {dialog?.kind === "edit" && (
        <UserFormDialog
          target={dialog.user}
          isSelf={isSelf(dialog.user)}
          onClose={close}
          onSaved={(saved) => {
            if (saved.id === me?.id) setMe(saved);
            finish(`${saved.full_name} was updated.`);
          }}
        />
      )}
      {dialog?.kind === "password" && (
        <PasswordDialog
          target={dialog.user}
          isSelf={isSelf(dialog.user)}
          onClose={close}
          onDone={() => finish(`The password of ${dialog.user.full_name} was changed.`)}
        />
      )}
      {dialog?.kind === "delete" && (
        <DeleteDialog
          target={dialog.user}
          onClose={close}
          onDone={() => finish(`${dialog.user.full_name} was deleted.`)}
        />
      )}
    </>
  );
}
