import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, errorMessage, type Report, type ReportSummary, type ShareState } from "../services/api";
import { Alert, Badge, formatDate, Modal, Spinner, staggerIndex, TableSkeleton } from "../components";
import { Icon } from "../components/icons";
import { SEVERITY_TONE } from "../components/interactions";
import { ReportDocument } from "../components/ReportDocument";
import { useQuery } from "../hooks";

const SHARE_DAYS = [1, 7, 30];

/** The link a reader opens: the token stays after #, so it is never sent to a server log. */
export function shareUrl(token: string): string {
  return `${window.location.origin}/shared#${token}`;
}

function ShareBadge({ share }: { share: ShareState | null }) {
  if (!share) return null;
  if (!share.active) return <Badge tone="neutral">link ended</Badge>;
  return (
    <Badge tone="active">
      shared until {formatDate(share.expires_at)} · {share.views} view{share.views === 1 ? "" : "s"}
    </Badge>
  );
}

function SeverityBadge({ report }: { report: ReportSummary | Report }) {
  return report.highest_severity ? (
    <Badge tone={SEVERITY_TONE[report.highest_severity]}>{report.highest_severity}</Badge>
  ) : (
    <Badge tone="neutral">none known</Badge>
  );
}

function DeleteDialog({ id, onClose, onDeleted }: { id: number; onClose: () => void; onDeleted: () => void }) {
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
    <Modal title="Delete this report?" onClose={onClose}>
      <p>The report and its share link are deleted for good.</p>
      {problem && <Alert>{problem}</Alert>}
      <div className="form-actions">
        <button type="button" className="btn btn-ghost" onClick={onClose}>
          Cancel
        </button>
        <button type="button" className="btn btn-danger" disabled={busy} onClick={remove}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          Delete
        </button>
      </div>
    </Modal>
  );
}

/** The user's saved reports, newest first. */
export default function ReportsPage() {
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
            <h1>My reports</h1>
            <p className="muted">
              Interaction checks you saved from the Scanner, kept as they were on that day. Print
              them or share a read-only link with a pharmacist or a doctor.
            </p>
          </div>
        </div>
        <Link className="btn btn-primary" to="/scan">
          <Icon name="scan" />
          New check
        </Link>
      </header>

      {reports.error && <Alert>{reports.error}</Alert>}
      {!reports.data && reports.loading && <TableSkeleton rows={4} />}
      {reports.data && reports.data.items.length === 0 && (
        <div className="card empty-state">
          <Icon name="file" size={28} />
          <p>
            No report yet. In the <Link to="/scan">Scanner</Link>, check your medicines, then choose
            <strong> Save as report</strong>.
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
                  {formatDate(report.created_at, true)} · {report.ingredients.join(", ") || "no ingredient"}
                  {report.chat_answers ? ` · ${report.chat_answers} answer(s)` : ""}
                </span>
                <span className="badge-row mt-2">
                  <SeverityBadge report={report} />
                  <ShareBadge share={report.share} />
                </span>
              </div>
              <div className="row-actions">
                <Link className="btn btn-ghost btn-sm" to={`/reports/${report.id}`}>
                  Open
                </Link>
                <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDeleting(report.id)}>
                  <Icon name="trash" size={16} />
                  <span className="sr-only">Delete</span>
                </button>
              </div>
            </li>
          ))}
        </ul>
      )}
      {reports.data && (
        <p className="muted mt-3">
          {reports.data.items.length} of {reports.data.limit} reports.
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
    <Modal title="Share a read-only link" onClose={onClose}>
      <p className="muted">
        Anyone with the link can read this report, without an account, until it expires. A new
        link replaces the previous one.
      </p>
      {problem && <Alert>{problem}</Alert>}
      {link ? (
        <>
          <div className="share-link">
            <input readOnly value={link} aria-label="Share link" onFocus={(e) => e.target.select()} />
            <button type="button" className="btn btn-primary btn-sm" onClick={copy}>
              <Icon name="link" size={16} />
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <p className="hint">The link is shown only now: copy it before closing.</p>
        </>
      ) : (
        <div className="field">
          <label htmlFor="share-days">Valid for</label>
          <select id="share-days" value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {SHARE_DAYS.map((d) => (
              <option key={d} value={d}>
                {d} day{d === 1 ? "" : "s"}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="form-actions">
        {report.share?.active && (
          <button type="button" className="btn btn-ghost" disabled={busy} onClick={stop}>
            Stop sharing
          </button>
        )}
        {!link && (
          <button type="button" className="btn btn-primary" disabled={busy} onClick={create}>
            {busy && <span className="btn-spinner" aria-hidden="true" />}
            Create the link
          </button>
        )}
        {link && (
          <button type="button" className="btn btn-ghost" onClick={onClose}>
            Done
          </button>
        )}
      </div>
    </Modal>
  );
}

/** One saved report: read, print, share or delete it. */
export function ReportPage() {
  const id = Number(useParams().id);
  const navigate = useNavigate();
  const report = useQuery((signal) => api.my.report(id, signal), [id]);
  const [sharing, setSharing] = useState(false);
  const [deleting, setDeleting] = useState(false);

  if (report.error) {
    return (
      <>
        <Alert>{report.error}</Alert>
        <Link to="/reports">Back to my reports</Link>
      </>
    );
  }
  if (!report.data) return <Spinner label="Loading the report" />;

  return (
    <>
      <p className="no-print mb-3">
        <Link to="/reports" className="link-btn">
          ← My reports
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
              Print or save as PDF
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setSharing(true)}>
              <Icon name="share" size={16} />
              Share
            </button>
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setDeleting(true)}>
              <Icon name="trash" size={16} />
              Delete
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
