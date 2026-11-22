import { Link, useLocation } from "react-router-dom";
import { api } from "../services/api";
import { Alert, Logo, Spinner } from "../components";
import { Icon } from "../components/icons";
import { ReportDocument } from "../components/ReportDocument";
import { SiteFooter } from "../components/layouts";
import { Preferences } from "../components/prefs";
import { useQuery } from "../hooks";
import { messages, useT } from "../i18n";

/**
 * A report opened from its share link, /shared#<token>, without signing in. The token is
 * read from the URL fragment (never sent with the page request) and posted to the API.
 */
export default function SharedReportPage() {
  const token = useLocation().hash.slice(1);
  const t = useT();
  const report = useQuery(
    (signal) => (token ? api.shared(token, signal) : Promise.reject(new Error(messages().shared.incomplete))),
    [token],
  );

  return (
    <div className="shell">
      <header className="topbar no-print">
        <div className="topbar-inner">
          <Link className="brand" to="/">
            <Logo />
          </Link>
          <span className="muted">{t.shared.readOnly}</span>
          <div className="topbar-user">
            <Preferences />
            {report.data && (
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => window.print()}>
                <Icon name="printer" size={16} />
                {t.shared.print}
              </button>
            )}
          </div>
        </div>
      </header>
      <main className="page">
        {report.error && (
          <div className="card empty-state">
            <Icon name="link" size={28} />
            <Alert>{report.error}</Alert>
            <p className="muted">{t.shared.askNew}</p>
          </div>
        )}
        {!report.data && !report.error && <Spinner label={t.shared.opening} />}
        {report.data && <ReportDocument report={report.data} />}
      </main>
      <SiteFooter />
    </div>
  );
}
