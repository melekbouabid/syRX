import { api, type DriftFeature, type DriftReport } from "../services/api";
import { useQuery } from "../hooks";
import { Alert, Badge, Spinner, formatNumber } from "../components";
import { Icon } from "../components/icons";
import { driftStatusLabel, driftTone, formatStat } from "../lib/drift";
import { useT } from "../i18n";

// Data drift of the scanned photos against the reference photos (GET /monitoring/drift, step 2.8).

function Summary({ report }: { report: DriftReport }) {
  const m = useT().monitoring;
  const facts: [string, string][] = [
    [m.status, driftStatusLabel(report.status)],
    [m.window, `${formatNumber(report.window_size)} / ${formatNumber(report.window_max)}`],
    [m.seen, formatNumber(report.photos_seen)],
    [m.reference, formatNumber(Number(report.reference.photos ?? 0))],
    [m.tests, m.testsValue(report.thresholds.ks_alpha, report.thresholds.psi_drift)],
  ];
  return (
    <section className="card mb-4 p-4" aria-label={m.summary}>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt className="cell-sub">{label}</dt>
            <dd className="cell-title m-0">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="cell-sub mt-3">{m.privacy}</p>
    </section>
  );
}

function FeatureRow({ feature }: { feature: DriftFeature }) {
  const t = useT();
  return (
    <tr>
      <td>
        <div className="cell-title">{feature.feature}</div>
        <div className="cell-sub">{feature.description}</div>
      </td>
      <td className="num">{formatStat(feature.reference_mean)}</td>
      <td className="num">{formatStat(feature.current_mean)}</td>
      <td className="num">{formatNumber(feature.current_n)}</td>
      <td className="num">{formatStat(feature.ks_statistic)}</td>
      <td className="num">{feature.ks_pvalue === null ? t.common.na : feature.ks_pvalue.toExponential(1)}</td>
      <td className="num">{formatStat(feature.psi)}</td>
      <td>
        <Badge tone={driftTone(feature.status)}>{driftStatusLabel(feature.status)}</Badge>
      </td>
    </tr>
  );
}

export default function MonitoringPage() {
  const drift = useQuery((signal) => api.drift(signal), []);
  const report = drift.data;
  const m = useT().monitoring;
  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="eye" size={24} />
          </span>
          <div>
            <h1>{m.title}</h1>
            <p className="muted">{m.lead}</p>
          </div>
        </div>
        <button type="button" className="btn btn-secondary" onClick={drift.reload}>
          {m.refresh}
        </button>
      </header>

      {drift.error && <Alert>{drift.error}</Alert>}
      {!report && drift.loading && <Spinner label={m.loading} />}
      {report && report.status === "drift" && <Alert>{m.drift(report.drifted.join(", "))}</Alert>}
      {report && report.status === "unavailable" && <Alert kind="info">{m.noReference}</Alert>}
      {report && report.status === "insufficient" && (
        <Alert kind="info">{m.notEnough(report.thresholds.min_samples)}</Alert>
      )}
      {report && <Summary report={report} />}
      {report && report.features.length > 0 && (
        <div className="card table-card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{m.columns.statistic}</th>
                  <th className="num">{m.columns.referenceMean}</th>
                  <th className="num">{m.columns.windowMean}</th>
                  <th className="num">{m.columns.photos}</th>
                  <th className="num">{m.columns.ks}</th>
                  <th className="num">{m.columns.pvalue}</th>
                  <th className="num">{m.columns.psi}</th>
                  <th>{m.columns.status}</th>
                </tr>
              </thead>
              <tbody>
                {report.features.map((feature) => (
                  <FeatureRow key={feature.feature} feature={feature} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  );
}
