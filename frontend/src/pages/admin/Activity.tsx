import { useSearchParams } from "react-router-dom";
import { useState } from "react";
import { api, exportUrl, type ActivityFilters, type ActivityGroup } from "../../services/api";
import {
  Alert,
  Avatar,
  Badge,
  formatDate,
  formatNumber,
  Pagination,
  staggerIndex,
  TableSkeleton,
} from "../../components";
import { useDebounced, useQuery } from "../../hooks";
import { Icon } from "../../components/icons";
import {
  ACTION_OPTIONS,
  actionLabel,
  browserName,
  eventDetail,
  GROUP_LABEL,
  GROUP_TONE,
  GROUPS,
} from "../../lib/activity";

const PAGE_SIZE = 25;
const PERIODS = [
  { days: 1, label: "Last 24 hours" },
  { days: 7, label: "Last 7 days" },
  { days: 30, label: "Last 30 days" },
  { days: 90, label: "Last 90 days" },
  { days: 0, label: "All time" },
];

function isGroup(value: string | null): value is ActivityGroup {
  return GROUPS.some((g) => g.group === value);
}

/** The account the log is filtered on (?user=ID), with a way back to everyone. */
function UserFilter({ userId, onClear }: { userId: number; onClear: () => void }) {
  const account = useQuery((signal) => api.admin.user(userId, signal), [userId]);
  return (
    <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
      {account.data ? (
        <>
          <Avatar name={account.data.full_name} src={account.data.photo_url} />
          <div className="min-w-0 flex-1">
            <span className="cell-title">{account.data.full_name}</span>
            <span className="cell-sub">
              {account.data.email} · last sign-in {formatDate(account.data.last_login_at, true)}
            </span>
          </div>
        </>
      ) : (
        <span className="muted flex-1">
          {account.error ? `Account #${userId} (deleted)` : "Loading the account…"}
        </span>
      )}
      <button type="button" className="btn btn-ghost btn-sm" onClick={onClear}>
        Show everyone
      </button>
    </div>
  );
}

export default function Activity() {
  const [params, setParams] = useSearchParams();
  const userParam = Number(params.get("user"));
  const userId = Number.isInteger(userParam) && userParam > 0 ? userParam : undefined;
  const groupParam = params.get("group");
  const group: ActivityGroup | "" = isGroup(groupParam) ? groupParam : "";
  const [search, setSearch] = useState("");
  const [action, setAction] = useState("");
  const [days, setDays] = useState(30);
  const [page, setPage] = useState(1);
  const q = useDebounced(search.trim());

  const filters: ActivityFilters = { user_id: userId, group, action, q, days };
  const log = useQuery(
    (signal) => api.admin.activity({ ...filters, page, page_size: PAGE_SIZE }, signal),
    [userId, group, action, q, days, page],
  );

  const setParam = (key: string, value: string | null) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    setParams(next, { replace: true });
    setPage(1);
  };
  const total = log.data ? Object.values(log.data.counts).reduce((a, b) => a + b, 0) : null;

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="activity" size={24} />
          </span>
          <div>
            <h1>Activity</h1>
            <p className="muted">
              Who signed in, scanned, asked or changed an account, and when. Counts only: never a
              photo, a medicine or the text of a question.
            </p>
          </div>
        </div>
        <a className="btn btn-ghost" href={exportUrl("activity", filters)} download>
          <Icon name="download" />
          Export CSV
        </a>
      </header>

      {userId !== undefined && <UserFilter userId={userId} onClear={() => setParam("user", null)} />}

      <div className="mb-4 flex flex-wrap gap-2" role="tablist" aria-label="Kind of event">
        <button
          type="button"
          role="tab"
          aria-selected={group === ""}
          className={`chip-btn ${group === "" ? "chip-btn-on" : ""}`}
          onClick={() => setParam("group", null)}
        >
          All {total !== null && <span className="chip-count">{formatNumber(total)}</span>}
        </button>
        {GROUPS.map((item) => (
          <button
            key={item.group}
            type="button"
            role="tab"
            aria-selected={group === item.group}
            className={`chip-btn ${group === item.group ? "chip-btn-on" : ""}`}
            onClick={() => setParam("group", item.group)}
          >
            {item.label}
            {log.data && <span className="chip-count">{formatNumber(log.data.counts[item.group])}</span>}
          </button>
        ))}
      </div>

      <div className="toolbar">
        <label className="search-box">
          <Icon name="search" />
          <input
            type="search"
            aria-label="Search by name, email or IP address"
            placeholder="Name, email or IP address"
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />
        </label>
        <select
          aria-label="Action"
          value={action}
          onChange={(event) => {
            setAction(event.target.value);
            setPage(1);
          }}
        >
          <option value="">Every action</option>
          {ACTION_OPTIONS.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          aria-label="Period"
          value={days}
          onChange={(event) => {
            setDays(Number(event.target.value));
            setPage(1);
          }}
        >
          {PERIODS.map((period) => (
            <option key={period.days} value={period.days}>
              {period.label}
            </option>
          ))}
        </select>
      </div>

      {log.error && <Alert>{log.error}</Alert>}
      {!log.data && log.loading && <TableSkeleton rows={8} />}
      {log.data && log.data.items.length === 0 && (
        <div className="card empty-state">
          <Icon name="activity" size={28} />
          <p>Nothing matches these filters.</p>
        </div>
      )}
      {log.data && log.data.items.length > 0 && (
        <div className="card table-card" aria-busy={log.loading}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Account</th>
                  <th>What</th>
                  <th>From</th>
                </tr>
              </thead>
              <tbody>
                {log.data.items.map((event, index) => (
                  <tr key={event.id} className="row-enter" style={staggerIndex(index)}>
                    <td className="whitespace-nowrap">{formatDate(event.created_at, true)}</td>
                    <td>
                      <button
                        type="button"
                        className="link-btn cell-title"
                        title="Show only this account"
                        onClick={() => setParam("user", String(event.user_id))}
                      >
                        {event.user_name}
                      </button>
                      <span className="cell-sub">{event.user_email}</span>
                    </td>
                    <td>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={GROUP_TONE[event.group]}>{GROUP_LABEL[event.group]}</Badge>
                        <span>{actionLabel(event.action)}</span>
                      </div>
                      <span className="cell-sub">
                        {event.target_email && (
                          <>
                            account <strong>{event.target_email}</strong>
                            {eventDetail(event) ? " · " : ""}
                          </>
                        )}
                        {eventDetail(event)}
                      </span>
                    </td>
                    <td>
                      <span className="cell-title">{event.ip ?? "—"}</span>
                      <span className="cell-sub" title={event.user_agent ?? ""}>
                        {browserName(event.user_agent)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pagination page={page} pageSize={PAGE_SIZE} total={log.data.total} onPage={setPage} />
        </div>
      )}
    </>
  );
}
