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
  actionLabel,
  actionOptions,
  browserName,
  eventDetail,
  GROUP_TONE,
  groupLabel,
  GROUPS,
} from "../../lib/activity";
import { useT } from "../../i18n";

const PAGE_SIZE = 25;
const PERIODS = [1, 7, 30, 90, 0]; // days; 0 = all time (labels in src/i18n)

function isGroup(value: string | null): value is ActivityGroup {
  return GROUPS.some((g) => g.group === value);
}

/** The account the log is filtered on (?user=ID), with a way back to everyone. */
function UserFilter({ userId, onClear }: { userId: number; onClear: () => void }) {
  const a = useT().admin.activity;
  const account = useQuery((signal) => api.admin.user(userId, signal), [userId]);
  return (
    <div className="card mb-4 flex flex-wrap items-center gap-3 p-3">
      {account.data ? (
        <>
          <Avatar name={account.data.full_name} src={account.data.photo_url} />
          <div className="min-w-0 flex-1">
            <span className="cell-title">{account.data.full_name}</span>
            <span className="cell-sub">
              {account.data.email} · {a.lastSignIn(formatDate(account.data.last_login_at, true))}
            </span>
          </div>
        </>
      ) : (
        <span className="muted flex-1">
          {account.error ? a.deletedAccount(userId) : a.loadingAccount}
        </span>
      )}
      <button type="button" className="btn btn-ghost btn-sm" onClick={onClear}>
        {a.everyone}
      </button>
    </div>
  );
}

export default function Activity() {
  const t = useT();
  const a = t.admin.activity;
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
            <h1>{a.title}</h1>
            <p className="muted">{a.lead}</p>
          </div>
        </div>
        <a className="btn btn-ghost" href={exportUrl("activity", filters)} download>
          <Icon name="download" />
          {t.admin.users.export}
        </a>
      </header>

      {userId !== undefined && <UserFilter userId={userId} onClear={() => setParam("user", null)} />}

      <div className="mb-4 flex flex-wrap gap-2" role="tablist" aria-label={a.kind}>
        <button
          type="button"
          role="tab"
          aria-selected={group === ""}
          className={`chip-btn ${group === "" ? "chip-btn-on" : ""}`}
          onClick={() => setParam("group", null)}
        >
          {a.all} {total !== null && <span className="chip-count">{formatNumber(total)}</span>}
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
            {groupLabel(item.group)}
            {log.data && <span className="chip-count">{formatNumber(log.data.counts[item.group])}</span>}
          </button>
        ))}
      </div>

      <div className="toolbar">
        <label className="search-box">
          <Icon name="search" />
          <input
            type="search"
            aria-label={a.searchLabel}
            placeholder={a.searchPlaceholder}
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />
        </label>
        <select
          aria-label={a.action}
          value={action}
          onChange={(event) => {
            setAction(event.target.value);
            setPage(1);
          }}
        >
          <option value="">{a.everyAction}</option>
          {actionOptions().map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <select
          aria-label={a.period}
          value={days}
          onChange={(event) => {
            setDays(Number(event.target.value));
            setPage(1);
          }}
        >
          {PERIODS.map((period) => (
            <option key={period} value={period}>
              {a.periods[period]}
            </option>
          ))}
        </select>
      </div>

      {log.error && <Alert>{log.error}</Alert>}
      {!log.data && log.loading && <TableSkeleton rows={8} />}
      {log.data && log.data.items.length === 0 && (
        <div className="card empty-state">
          <Icon name="activity" size={28} />
          <p>{a.nothing}</p>
        </div>
      )}
      {log.data && log.data.items.length > 0 && (
        <div className="card table-card" aria-busy={log.loading}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{a.columns.when}</th>
                  <th>{a.columns.account}</th>
                  <th>{a.columns.what}</th>
                  <th>{a.columns.from}</th>
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
                        title={a.onlyThis}
                        onClick={() => setParam("user", String(event.user_id))}
                      >
                        {event.user_name}
                      </button>
                      <span className="cell-sub">{event.user_email}</span>
                    </td>
                    <td>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={GROUP_TONE[event.group]}>{groupLabel(event.group)}</Badge>
                        <span>{actionLabel(event.action)}</span>
                      </div>
                      <span className="cell-sub">
                        {event.target_email && (
                          <>
                            {a.account} <strong>{event.target_email}</strong>
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
