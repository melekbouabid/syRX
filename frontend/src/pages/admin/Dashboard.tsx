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
import { actionLabel, GROUP_TONE, groupLabel } from "../../lib/activity";
import { useT } from "../../i18n";

// Labels and notes in src/i18n (admin.dashboard.tiles), by key.
const TILES: { key: keyof AdminStats; icon: IconName }[] = [
  { key: "users_total", icon: "users" },
  { key: "users_active", icon: "check" },
  { key: "users_inactive", icon: "ban" },
  { key: "admins", icon: "shield" },
  { key: "signups_last_7_days", icon: "spark" },
  { key: "active_sessions", icon: "clock" },
];

const WEEK_TILES: { key: keyof AdminStats; icon: IconName }[] = [
  { key: "scans_last_7_days", icon: "scan" },
  { key: "questions_last_7_days", icon: "book" },
  { key: "reports_total", icon: "file" },
  { key: "failed_sign_ins_last_7_days", icon: "alert" },
];

/** The last few events of the activity log, with a link to all of it. */
function LatestActivity() {
  const t = useT();
  const log = useQuery((signal) => api.admin.activity({ page_size: 6, days: 0 }, signal), []);
  return (
    <section className="card latest-activity" aria-label={t.admin.dashboard.latest}>
      <header className="pill-card-head">
        <h2>{t.admin.dashboard.latest}</h2>
        <Link to="/admin/activity" className="link-btn">
          {t.admin.dashboard.seeAll}
        </Link>
      </header>
      {log.error && <Alert>{log.error}</Alert>}
      {!log.data && log.loading && <TableSkeleton rows={4} />}
      {log.data && (
        <ul className="event-list">
          {log.data.items.map((event) => (
            <li key={event.id}>
              <Badge tone={GROUP_TONE[event.group]}>{groupLabel(event.group)}</Badge>
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
  const d = useT().admin.dashboard;
  const share = stats.users_total ? Math.round((stats.users_active / stats.users_total) * 100) : 0;
  return (
    <div className="card meter-card">
      <div className="meter-head">
        <h2>{d.access}</h2>
        <strong>{d.activeShare(share)}</strong>
      </div>
      <div className="meter" role="img" aria-label={d.activeOf(stats.users_active, stats.users_total)}>
        <span style={{ "--w": `${share}%` } as CSSProperties} />
      </div>
      <p className="meter-legend muted">
        <span>
          <i className="dot dot-active" /> {d.nActive(stats.users_active)}
        </span>
        <span>
          <i className="dot dot-inactive" /> {d.nDeactivated(stats.users_inactive)}
        </span>
      </p>
    </div>
  );
}

export default function Dashboard() {
  const { user } = useAuth();
  const t = useT();
  const d = t.admin.dashboard;
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
          <p className="eyebrow">{d.eyebrow}</p>
          <h1>{d.title}</h1>
          <p>{d.lead(greeting(), firstName, APP_NAME)}</p>
          <div className="banner-actions">
            <Link className="btn btn-light" to="/admin/users">
              <Icon name="users" />
              {d.manage}
            </Link>
            <Link className="btn btn-on-dark" to="/admin/usage">
              <Icon name="chart" />
              {d.usage}
            </Link>
            <Link className="btn btn-on-dark" to="/home">
              <Icon name="eye" />
              {d.viewApp}
            </Link>
          </div>
        </div>
      </section>

      {stats.error && <Alert>{stats.error}</Alert>}
      <section aria-label={d.accountNumbers} className="stat-grid stat-grid-3">
        {TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={d.tiles[tile.key].label}
            note={d.tiles[tile.key].note}
            value={stats.data?.[tile.key]}
            loading={!stats.data && stats.loading}
          />
        ))}
      </section>

      <h2 className="section-title">{d.use}</h2>
      <section aria-label={d.use} className="stat-grid stat-grid-4">
        {WEEK_TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={d.tiles[tile.key].label}
            note={d.tiles[tile.key].note}
            value={stats.data?.[tile.key]}
            loading={!stats.data && stats.loading}
          />
        ))}
      </section>

      <div className="grid items-start gap-5 lg:grid-cols-2">
        {stats.data ? <AccessMeter stats={stats.data} /> : <div />}
        <LatestActivity />
      </div>

      <h2 className="section-title">{d.newest}</h2>
      {recent.error && <Alert>{recent.error}</Alert>}
      {!recent.data && recent.loading && <TableSkeleton rows={5} />}
      {recent.data && (
        <div className="card table-card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{d.columns.user}</th>
                  <th>{d.columns.role}</th>
                  <th>{d.columns.status}</th>
                  <th>{d.columns.created}</th>
                  <th>{d.columns.lastSignIn}</th>
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
                      <Badge tone={account.role}>{t.common.roles[account.role] ?? account.role}</Badge>
                    </td>
                    <td>
                      <Badge tone={account.is_active ? "active" : "inactive"}>
                        {account.is_active ? t.common.active : t.common.deactivated}
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
