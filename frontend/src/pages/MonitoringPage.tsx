import { api, type DriftFeature, type DriftReport } from "../services/api";
import { useQuery } from "../hooks";
import { Alert, Badge, Spinner, formatNumber } from "../components";
import { Icon } from "../components/icons";
import { driftStatusLabel, driftTone, formatStat } from "../lib/drift";

// Data drift of the scanned photos against the reference photos (GET /monitoring/drift, step 2.8).

function Summary({ report }: { report: DriftReport }) {
  const facts: [string, string][] = [
    ["Status", driftStatusLabel(report.status)],
    ["Photos in the window", `${formatNumber(report.window_size)} / ${formatNumber(report.window_max)}`],
    ["Photos since start-up", formatNumber(report.photos_seen)],
    ["Reference photos", formatNumber(Number(report.reference.photos ?? 0))],
    ["Tests", `KS p < ${report.thresholds.ks_alpha} and PSI ≥ ${report.thresholds.psi_drift}`],
  ];
  return (
    <section className="card mb-4 p-4" aria-label="Drift summary">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt className="cell-sub">{label}</dt>
            <dd className="cell-title m-0">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="cell-sub mt-3">
        Only eight numbers per scanned photo are kept, in memory: no image and nothing about you.
        The reference is 600 photos of the C3PI v2 validation split: the same kind as the
        training photos, never trained on.
      </p>
    </section>
  );
}

function FeatureRow({ feature }: { feature: DriftFeature }) {
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
      <td className="num">{feature.ks_pvalue === null ? "n/a" : feature.ks_pvalue.toExponential(1)}</td>
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
  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="eye" size={24} />
          </span>
          <div>
            <h1>Monitoring</h1>
            <p className="muted">
              Do the photos scanned recently still look like the photos the models were trained
              on? Each statistic is compared with reference photos of the same kind (Kolmogorov-Smirnov
              test and population stability index).
            </p>
          </div>
        </div>
        <button type="button" className="btn btn-secondary" onClick={drift.reload}>
          Refresh
        </button>
      </header>

      {drift.error && <Alert>{drift.error}</Alert>}
      {!report && drift.loading && <Spinner label="Loading the drift report" />}
      {report && report.status === "drift" && (
        <Alert>
          Data drift: the recent photos differ from the reference photos ({report.drifted.join(", ")}).
          The confidence shown for a pill is less reliable; retake the photo in good light, sharp,
          on a plain background.
        </Alert>
      )}
      {report && report.status === "unavailable" && (
        <Alert kind="info">No reference statistics on the server (dvc repro drift_reference).</Alert>
      )}
      {report && report.status === "insufficient" && (
        <Alert kind="info">
          Not enough photos yet: at least {report.thresholds.min_samples} scans are needed before
          the tests say anything.
        </Alert>
      )}
      {report && <Summary report={report} />}
      {report && report.features.length > 0 && (
        <div className="card table-card">
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Statistic</th>
                  <th className="num">Reference mean</th>
                  <th className="num">Window mean</th>
                  <th className="num">Photos</th>
                  <th className="num">KS statistic</th>
                  <th className="num">KS p-value</th>
                  <th className="num">PSI</th>
                  <th>Status</th>
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
