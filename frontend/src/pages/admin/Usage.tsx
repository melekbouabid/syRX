import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type UsageDay } from "../../services/api";
import { Alert, formatNumber, Skeleton, StatTile } from "../../components";
import { useQuery } from "../../hooks";
import { Icon, type IconName } from "../../components/icons";
import { DailyLine, Legend, shortDay, StackedColumns, type Series } from "../../components/charts";
import { useT, type Messages } from "../../i18n";

const PERIODS = [7, 14, 30, 90];

// Fixed order: a series keeps its colour whatever the period (styles.css, .viz).
const SERIES_KEYS: { key: string; color: string }[] = [
  { key: "scans", color: "var(--series-1)" },
  { key: "interactions", color: "var(--series-2)" },
  { key: "questions", color: "var(--series-3)" },
  { key: "toxicity", color: "var(--series-4)" },
  { key: "reports", color: "var(--series-5)" },
];

function seriesOf(t: Messages): Series[] {
  return SERIES_KEYS.map((s) => ({ ...s, label: t.admin.usage.series[s.key] }));
}

const TILES: { key: string; icon: IconName }[] = [
  { key: "active_users", icon: "users" },
  { key: "sign_ins", icon: "check" },
  { key: "scans", icon: "scan" },
  { key: "interactions", icon: "pill" },
  { key: "questions", icon: "book" },
  { key: "reports", icon: "file" },
];

function UsageTable({ days }: { days: UsageDay[] }) {
  const t = useT();
  const u = t.admin.usage;
  const SERIES = seriesOf(t);
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>{u.day}</th>
            <th className="num">{u.series.active_users}</th>
            <th className="num">{u.series.sign_ins}</th>
            {SERIES.map((s) => (
              <th key={s.key} className="num">
                {s.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[...days].reverse().map((day) => (
            <tr key={day.day}>
              <td>{shortDay(day.day)}</td>
              <td className="num">{formatNumber(day.active_users)}</td>
              <td className="num">{formatNumber(day.sign_ins)}</td>
              {SERIES.map((s) => (
                <td key={s.key} className="num">
                  {formatNumber(Number(day[s.key as keyof UsageDay]))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Usage() {
  const t = useT();
  const u = t.admin.usage;
  const SERIES = seriesOf(t);
  const [days, setDays] = useState(30);
  const [view, setView] = useState<"chart" | "table">("chart");
  const usage = useQuery((signal) => api.admin.usage(days, signal), [days]);
  const rows = usage.data?.days ?? [];
  const empty = rows.every((d) => SERIES.every((s) => !d[s.key as keyof UsageDay]));

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="chart" size={24} />
          </span>
          <div>
            <h1>{u.title}</h1>
            <p className="muted">{u.lead}</p>
          </div>
        </div>
      </header>

      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label={u.period}>
          {PERIODS.map((period) => (
            <button
              key={period}
              type="button"
              role="tab"
              aria-selected={days === period}
              onClick={() => setDays(period)}
            >
              {u.days(period)}
            </button>
          ))}
        </div>
        <div className="segmented" role="tablist" aria-label={u.view}>
          <button type="button" role="tab" aria-selected={view === "chart"} onClick={() => setView("chart")}>
            {u.charts}
          </button>
          <button type="button" role="tab" aria-selected={view === "table"} onClick={() => setView("table")}>
            {u.table}
          </button>
        </div>
      </div>

      {usage.error && <Alert>{usage.error}</Alert>}

      <section aria-label={u.totals} className="stat-grid stat-grid-3">
        {TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={u.series[tile.key]}
            note={u.inLast(days)}
            value={usage.data?.totals[tile.key]}
            loading={!usage.data && usage.loading}
          />
        ))}
      </section>

      {!usage.data && usage.loading && (
        <div className="card p-4">
          <Skeleton height="240px" />
        </div>
      )}

      {usage.data && view === "table" && (
        <div className="card table-card">
          <UsageTable days={rows} />
        </div>
      )}

      {usage.data && view === "chart" && (
        <div className="viz grid grid-cols-1 gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
          <section className="card viz-card">
            <h2>{u.perDay}</h2>
            <Legend series={SERIES} />
            {empty ? (
              <p className="muted viz-empty">{u.empty}</p>
            ) : (
              <StackedColumns rows={rows} series={SERIES} label={u.perDayLabel(days)} />
            )}
          </section>
          <section className="card viz-card">
            <h2>{u.activePerDay}</h2>
            <p className="muted viz-sub">{u.activeSub}</p>
            <DailyLine
              rows={rows}
              field="active_users"
              name={u.activeName}
              color="var(--series-1)"
              label={u.activeLabel(days)}
            />
          </section>
        </div>
      )}

      {usage.data && (
        <>
          <h2 className="section-title">{u.top}</h2>
          {usage.data.top_users.length === 0 ? (
            <p className="muted">{u.noActivity}</p>
          ) : (
            <div className="card table-card">
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>{u.account}</th>
                      <th className="num">{u.actions}</th>
                      <th className="num" />
                    </tr>
                  </thead>
                  <tbody>
                    {usage.data.top_users.map((top) => (
                      <tr key={top.user_id}>
                        <td>
                          <span className="cell-title">{top.name}</span>
                          <span className="cell-sub">{top.email}</span>
                        </td>
                        <td className="num">{formatNumber(top.actions)}</td>
                        <td className="num">
                          <Link className="btn btn-ghost btn-sm" to={`/admin/activity?user=${top.user_id}`}>
                            {u.activity}
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
