import { useState } from "react";
import { Link } from "react-router-dom";
import { api, type UsageDay } from "../../services/api";
import { Alert, formatNumber, Skeleton, StatTile } from "../../components";
import { useQuery } from "../../hooks";
import { Icon, type IconName } from "../../components/icons";
import { DailyLine, Legend, shortDay, StackedColumns, type Series } from "../../components/charts";

const PERIODS = [7, 14, 30, 90];

// Fixed order: a series keeps its colour whatever the period (styles.css, .viz).
const SERIES: Series[] = [
  { key: "scans", label: "Scans", color: "var(--series-1)" },
  { key: "interactions", label: "Interaction checks", color: "var(--series-2)" },
  { key: "questions", label: "Questions", color: "var(--series-3)" },
  { key: "toxicity", label: "Toxicity", color: "var(--series-4)" },
  { key: "reports", label: "Reports saved", color: "var(--series-5)" },
];

const TILES: { key: string; label: string; icon: IconName }[] = [
  { key: "active_users", label: "Active accounts", icon: "users" },
  { key: "sign_ins", label: "Sign-ins", icon: "check" },
  { key: "scans", label: "Scans", icon: "scan" },
  { key: "interactions", label: "Interaction checks", icon: "pill" },
  { key: "questions", label: "Questions", icon: "book" },
  { key: "reports", label: "Reports saved", icon: "file" },
];

function UsageTable({ days }: { days: UsageDay[] }) {
  return (
    <div className="table-wrap">
      <table className="table">
        <thead>
          <tr>
            <th>Day</th>
            <th className="num">Active accounts</th>
            <th className="num">Sign-ins</th>
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
            <h1>Usage</h1>
            <p className="muted">
              What Tujya is used for, day by day (UTC), counted from the activity log.
            </p>
          </div>
        </div>
      </header>

      <div className="toolbar">
        <div className="segmented" role="tablist" aria-label="Period">
          {PERIODS.map((period) => (
            <button
              key={period}
              type="button"
              role="tab"
              aria-selected={days === period}
              onClick={() => setDays(period)}
            >
              {period} days
            </button>
          ))}
        </div>
        <div className="segmented" role="tablist" aria-label="View">
          <button type="button" role="tab" aria-selected={view === "chart"} onClick={() => setView("chart")}>
            Charts
          </button>
          <button type="button" role="tab" aria-selected={view === "table"} onClick={() => setView("table")}>
            Table
          </button>
        </div>
      </div>

      {usage.error && <Alert>{usage.error}</Alert>}

      <section aria-label="Totals" className="stat-grid stat-grid-3">
        {TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={tile.label}
            note={`in the last ${days} days`}
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
            <h2>Actions per day</h2>
            <Legend series={SERIES} />
            {empty ? (
              <p className="muted viz-empty">No scan, check, question or report in this period yet.</p>
            ) : (
              <StackedColumns rows={rows} series={SERIES} label={`Actions per day, last ${days} days`} />
            )}
          </section>
          <section className="card viz-card">
            <h2>Active accounts per day</h2>
            <p className="muted viz-sub">Accounts that signed in or used the app that day.</p>
            <DailyLine
              rows={rows}
              field="active_users"
              name="active accounts"
              color="var(--series-1)"
              label={`Active accounts per day, last ${days} days`}
            />
          </section>
        </div>
      )}

      {usage.data && (
        <>
          <h2 className="section-title">Most active accounts</h2>
          {usage.data.top_users.length === 0 ? (
            <p className="muted">No activity in this period.</p>
          ) : (
            <div className="card table-card">
              <div className="table-wrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Account</th>
                      <th className="num">Actions</th>
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
                            Activity
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
