import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, errorMessage, type Report, type ReportSummary, type ShareState } from "../services/api";
import { Alert, Badge, formatDate, Modal, Spinner, staggerIndex, TableSkeleton } from "../components";
import { Icon } from "../components/icons";
import { SEVERITY_TONE } from "../components/interactions";
import { ReportDocument } from "../components/ReportDocument";
import { useQuery } from "../hooks";
import { useT } from "../i18n";

const SHARE_DAYS = [1, 7, 30];

/** The link a reader opens: the token stays after #, so it is never sent to a server log. */
export function shareUrl(token: string): string {
  return `${window.location.origin}/shared#${token}`;
}

function ShareBadge({ share }: { share: ShareState | null }) {
  const t = useT();
  if (!share) return null;
  if (!share.active) return <Badge tone="neutral">{t.reports.linkEnded}</Badge>;
  return <Badge tone="active">{t.reports.sharedUntil(formatDate(share.expires_at), share.views)}</Badge>;
}

function SeverityBadge({ report }: { report: ReportSummary | Report }) {
  const t = useT();
  return report.highest_severity ? (
    <Badge tone={SEVERITY_TONE[report.highest_severity]}>{t.severity[report.highest_severity]}</Badge>
  ) : (
    <Badge tone="neutral">{t.reports.noneKnown}</Badge>
  );
}

function DeleteDialog({ id, onClose, onDeleted }: { id: number; onClose: () => void; onDeleted: () => void }) {
  const t = useT();
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const remove = async () => {
    setBusy(true);
    try {
      await api.my.deleteReport(id);
      onDeleted();
    } catch (failure) {
      setProblem(errorMessage(failure));
      setBusy(false);
    }
  };
  return (
    <Modal title={t.reports.deleteTitle} onClose={onClose}>
      <p>{t.reports.deleteText}</p>
      {problem && <Alert>{problem}</Alert>}
      <div className="form-actions">
        <button type="button" className="btn btn-ghost" onClick={onClose}>
          {t.common.cancel}
        </button>
        <button type="button" className="btn btn-danger" disabled={busy} onClick={remove}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          {t.common.delete}
        </button>
      </div>
    </Modal>
  );
}

/** The user's saved reports, newest first. */
export default function ReportsPage() {
  const t = useT();
  const r = t.reports;
  const reports = useQuery((signal) => api.my.reports(signal), []);
  const [deleting, setDeleting] = useState<number | null>(null);

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="file" size={24} />
          </span>
          <div>
            <h1>{r.title}</h1>
            <p className="muted">{r.lead}</p>
          </div>
        </div>
        <Link className="btn btn-primary" to="/scan">
          <Icon name="scan" />
          {r.newCheck}
        </Link>
      </header>

      {reports.error && <Alert>{reports.error}</Alert>}
      {!reports.data && reports.loading && <TableSkeleton rows={4} />}
      {reports.data && reports.data.items.length === 0 && (
        <div className="card empty-state">
          <Icon name="file" size={28} />
          <p>
            {r.emptyBefore}
            <Link to="/scan">{t.shell.nav.scan}</Link>
            {r.emptyMiddle}
            <strong>{r.emptySave}</strong>.
          </p>
        </div>
      )}
      {reports.data && reports.data.items.length > 0 && (
        <ul className="report-list">
          {reports.data.items.map((report, index) => (
            <li key={report.id} className="card report-item row-enter" style={staggerIndex(index)}>
              <div className="min-w-0 flex-1">
                <Link to={`/reports/${report.id}`} className="report-item-title">
                  {report.title}
                </Link>
                <span className="cell-sub">
                  {formatDate(report.created_at, true)} · {report.ingredients.join(", ") || r.noIngredient}
                  {report.chat_answers ? r.answers(report.chat_answers) : ""}
                </span>
                <span className="badge-row mt-2">
                  <SeverityBadge report={report} />
                  <ShareBadge share={report.share} />
                </span>
              </div>
              <div className="row-actions">
                <Link className="btn btn-ghost btn-sm" to={`/reports/${report.id}`}>
                  {r.open}
                </Link>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDeleting(report.id)}>
                  <Icon name="trash" size={16} />
                  <span className="sr-only">{t.common.delete}</span>
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {reports.data && (
        <p className="muted mt-3">
          {r.count(reports.data.items.length, reports.data.limit)}
        </p>
      )}
      {deleting !== null && (
        <DeleteDialog
          id={deleting}
          onClose={() => setDeleting(null)}
          onDeleted={() => {
            setDeleting(null);
            reports.reload();
          }}
        />
      )}
    </>
  );
}

function ShareDialog({ report, onClose, onChange }: { report: Report; onClose: () => void; onChange: () => void }) {
  const t = useT();
  const r = t.reports;
  const [days, setDays] = useState(7);
  const [link, setLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const create = async () => {
    setBusy(true);
    setProblem(null);
    try {
      const { token } = await api.my.share(report.id, days);
      setLink(shareUrl(token));
      onChange();
    } catch (failure) {
      setProblem(errorMessage(failure));
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      await api.my.unshare(report.id);
      setLink(null);
      onChange();
      onClose();
    } catch (failure) {
      setProblem(errorMessage(failure));
      setBusy(false);
    }
  };

  const copy = async () => {
    if (!link) return;
    await navigator.clipboard.writeText(link).catch(() => undefined);
    setCopied(true);
  };

  return (
    <Modal title={r.shareTitle} onClose={onClose}>
      <p className="muted">{r.shareText}</p>
      {problem && <Alert>{problem}</Alert>}
      {link ? (
        <>
          <div className="share-link">
            <input readOnly value={link} aria-label={r.shareLink} onFocus={(e) => e.target.select()} />
            <button type="button" className="btn btn-primary btn-sm" onClick={copy}>
              <Icon name="link" size={16} />
              {copied ? r.copied : r.copy}
            </button>
          </div>
          <p className="hint">{r.onlyNow}</p>
        </>
      ) : (
        <div className="field">
          <label htmlFor="share-days">{r.validFor}</label>
          <select id="share-days" value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {SHARE_DAYS.map((d) => (
              <option key={d} value={d}>
                {r.days(d)}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="form-actions">
        {report.share?.active && (
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={stop}>
            {r.stop}
          </button>
        )}
        {!link && (
          <button type="button" className="btn btn-primary" disabled={busy} onClick={create}>
            {busy && <span className="btn-spinner" aria-hidden="true" />}
            {r.create}
          </button>
        )}
        {link && (
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            {r.done}
          </button>
        )}
      </div>
    </Modal>
  );
}

/** One saved report: read, print, share or delete it. */
export function ReportPage() {
  const t = useT();
  const r = t.reports;
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const report = useQuery((signal) => api.my.report(id, signal), [id]);
  const [sharing, setSharing] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (report.error) {
    return (
      <>
        <Alert>{report.error}</Alert>
        <Link to="/reports">{r.back}</Link>
      </>
    );
  }
  if (!report.data) return <Spinner label={r.loading} />;

  return (
    <>
      <p className="no-print mb-3">
        <Link to="/reports" className="link-btn">
          {r.backShort}
        </Link>
      </p>
      <ReportDocument
        report={report.data}
        actions={
          <>
            <ShareBadge share={report.data.share} />
            <span className="flex-1" />
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => window.print()}>
              <Icon name="printer" size={16} />
              {r.print}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setSharing(true)}>
              <Icon name="share" size={16} />
              {r.share}
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDeleting(true)}>
              <Icon name="trash" size={16} />
              {t.common.delete}
            </button>
          </>
        }
      />
      {sharing && <ShareDialog report={report.data} onClose={() => setSharing(false)} onChange={report.reload} />}
      {deleting && (
        <DeleteDialog id={id} onClose={() => setDeleting(false)} onDeleted={() => navigate("/reports")} />
      )}
    </>
  );
}
