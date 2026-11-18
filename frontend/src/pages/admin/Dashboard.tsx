import type { CSSProperties } from "react";
import { Link } from "react-router-dom";
import { api, type AdminStats } from "../../services/api";
import { useAuth } from "../../services/auth";
import { APP_NAME } from "../../brand";
import {
  Alert,
  Avatar,
  Badge,
  formatDate,
  greeting,
  StatTile,
  staggerIndex,
  TableSkeleton,
} from "../../components";
import { useQuery } from "../../hooks";
import { Icon, type IconName } from "../../components/icons";
import { actionLabel, GROUP_LABEL, GROUP_TONE } from "../../lib/activity";

const TILES: { key: keyof AdminStats; label: string; note: string; icon: IconName }[] = [
  { key: "users_total", label: "Accounts", note: "users and administrators", icon: "users" },
  { key: "users_active", label: "Active", note: "can sign in", icon: "check" },
  { key: "users_inactive", label: "Deactivated", note: "blocked from signing in", icon: "ban" },
  { key: "admins", label: "Administrators", note: "active, with full access", icon: "shield" },
  {
    key: "signups_last_7_days",
    label: "New this week",
    note: "created in the last 7 days",
    icon: "spark",
  },
  {
    key: "active_sessions",
    label: "Open sessions",
    note: "browsers signed in right now",
    icon: "clock",
  },
];

const WEEK_TILES: { key: keyof AdminStats; label: string; note: string; icon: IconName }[] = [
  { key: "scans_last_7_days", label: "Scans", note: "photos scanned in 7 days", icon: "scan" },
  { key: "questions_last_7_days", label: "Questions", note: "asked to the assistant in 7 days", icon: "book" },
  { key: "reports_total", label: "Reports", note: "saved by users, in total", icon: "file" },
  {
    key: "failed_sign_ins_last_7_days",
    label: "Refused sign-ins",
    note: "wrong password or blocked, 7 days",
    icon: "alert",
  },
];

/** The last few events of the activity log, with a link to all of it. */
function LatestActivity() {
  const log = useQuery((signal) => api.admin.activity({ page_size: 6, days: 0 }, signal), []);
  return (
    <section className="card latest-activity" aria-label="Latest activity">
      <header className="pill-card-head">
        <h2>Latest activity</h2>
        <Link to="/admin/activity" className="link-btn">
          See all
        </Link>
      </header>
      {log.error && <Alert>{log.error}</Alert>}
      {!log.data && log.loading && <TableSkeleton rows={4} />}
      {log.data && (
        <ul className="event-list">
          {log.data.items.map((event) => (
            <li key={event.id}>
              <Badge tone={GROUP_TONE[event.group]}>{GROUP_LABEL[event.group]}</Badge>
              <span>
                <span>
                  <strong>{event.user_name}</strong> · {actionLabel(event.action).toLowerCase()}
                  {event.target_email ? ` (${event.target_email})` : ""}
                </span>
                <span className="cell-sub">{formatDate(event.created_at, true)}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

/** Share of accounts that can sign in, as a bar that fills when the numbers arrive. */
function AccessMeter({ stats }: { stats: AdminStats }) {
  const share = stats.users_total ? Math.round((stats.users_active / stats.users_total) * 100) : 0;
  return (
    <div className="card meter-card">
      <div className="meter-head">
        <h2>Account access</h2>
        <strong>{share}% active</strong>
      </div>
      <div
        className="meter"
        role="img"
        aria-label={`${stats.users_active} of ${stats.users_total} accounts are active`}
      >
        <span style={{ "--w": `${share}%` } as CSSProperties} />
      </div>
      <p className="meter-legend muted">
        <span>
          <i className="dot dot-active" /> {stats.users_active} active
        </span>
        <span>
          <i className="dot dot-inactive" /> {stats.users_inactive} deactivated
        </span>
      </p>
    </div>
  );
}

export default function Dashboard() {
  const { user } = useAuth();
  const stats = useQuery((signal) => api.admin.stats(signal), []);
  const recent = useQuery((signal) => api.admin.users({ page_size: 5 }, signal), []);
  const firstName = user?.full_name.split(/\s+/)[0] ?? "";

  return (
    <>
      <section className="banner banner-admin">
        <div className="orbs" aria-hidden="true">
          <span />
          <span />
        </div>
        <div className="banner-copy">
          <p className="eyebrow">Administration</p>
          <h1>Dashboard</h1>
          <p>
            {greeting()}, {firstName}. Here is who has access to {APP_NAME}, at a glance.
          </p>
          <div className="banner-actions">
            <Link className="btn btn-light" to="/admin/users">
              <Icon name="users" />
              Manage users
            </Link>
            <Link className="btn btn-on-dark" to="/admin/usage">
              <Icon name="chart" />
              Usage
            </Link>
            <Link className="btn btn-on-dark" to="/home">
              <Icon name="eye" />
              View the app
            </Link>
          </div>
        </div>
      </section>

      {stats.error && <Alert>{stats.error}</Alert>}
      <section aria-label="Account numbers" className="stat-grid stat-grid-3">
        {TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={tile.label}
            note={tile.note}
            value={stats.data?.[tile.key]}
            loading={!stats.data && stats.loading}
          />
        ))}
      </section>

      <h2 className="section-title">Use of the app</h2>
      <section aria-label="Use of the app" className="stat-grid stat-grid-4">
        {WEEK_TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={tile.label}
            note={tile.note}
            value={stats.data?.[tile.key]}
            loading={!stats.data && stats.loading}
          />
        ))}
      </section>

      <div className="grid items-start gap-5 lg:grid-cols-2">
        {stats.data ? <AccessMeter stats={stats.data} /> : <div />}
        <LatestActivity />
      </div>

      <h2 className="section-title">Newest accounts</h2>
      {recent.error && <Alert>{recent.error}</Alert>}
      {!recent.data && recent.loading && <TableSkeleton rows={5} />}
      {recent.data && (
        <div className="card table-card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Last sign-in</th>
                </tr>
              </thead>
              <tbody>
                {recent.data.items.map((account, index) => (
                  <tr key={account.id} className="row-enter" style={staggerIndex(index)}>
                    <td>
                      <div className="cell-user">
                        <Avatar name={account.full_name} size="sm" src={account.photo_url} />
                        <div>
                          <span className="cell-title">{account.full_name}</span>
                          <span className="cell-sub">{account.email}</span>
                        </div>
                      </div>
                    </td>
                    <td>
                      <Badge tone={account.role}>{account.role}</Badge>
                    </td>
                    <td>
                      <Badge tone={account.is_active ? "active" : "inactive"}>
                        {account.is_active ? "Active" : "Deactivated"}
                      </Badge>
                    </td>
                    <td>{formatDate(account.created_at)}</td>
                    <td>{formatDate(account.last_login_at, true)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
