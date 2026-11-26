import type { Role } from "../../services/api";
import { useAuth } from "../../services/auth";
import { Icon, type IconName } from "../../components/icons";
import { useT, type Messages } from "../../i18n";
import { ADMINER_LOGIN, toolLinks, type ToolId } from "../../lib/tools";

const ICONS: Record<ToolId, IconName> = {
  mlflow: "flask",
  qdrant: "search",
  adminer: "grid",
  swagger: "book",
  spark: "spark",
};

interface ViewProps {
  role: Role | undefined;
  hostname: string;
  t: Messages;
}

/** One card per tool, each opening in a new tab. Nothing at all for an account that is not admin. */
export function ToolsSectionView({ role, hostname, t }: ViewProps) {
  if (role !== "admin") return null;
  const s = t.admin.tools;
  return (
    <section aria-labelledby="admin-tools-title" className="admin-tools">
      <h2 id="admin-tools-title" className="section-title">
        {s.title}
      </h2>
      <p className="muted admin-tools-lead">{s.lead}</p>
      <ul className="tool-grid">
        {toolLinks(hostname).map((tool) => (
          <li key={tool.id} className="card tool-card">
            <div className="tool-head">
              <span className="tool-icon" aria-hidden="true">
                <Icon name={ICONS[tool.id]} />
              </span>
              <h3>{s.items[tool.id].name}</h3>
            </div>
            <p className="muted">{s.items[tool.id].role}</p>
            {tool.id === "spark" && <p className="tool-note">{s.sparkProfile}</p>}
            <a className="btn btn-primary tool-open" href={tool.url} target="_blank" rel="noopener noreferrer">
              {s.open}
              <span className="sr-only"> {s.items[tool.id].name}</span>
              <Icon name="link" size={16} />
            </a>
            {tool.id === "adminer" && (
              <dl className="tool-login" aria-label={s.login.title}>
                <dt>{s.login.system}</dt>
                <dd>
                  <code>{ADMINER_LOGIN.system}</code>
                </dd>
                <dt>{s.login.server}</dt>
                <dd>
                  <code>{ADMINER_LOGIN.server}</code>
                </dd>
                <dt>{s.login.username}</dt>
                <dd>
                  <code>{ADMINER_LOGIN.username}</code>
                </dd>
                <dt>{s.login.database}</dt>
                <dd>
                  <code>{ADMINER_LOGIN.databases[0]}</code> {s.login.or} <code>{ADMINER_LOGIN.databases[1]}</code>
                </dd>
              </dl>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

/** The tools section of the admin dashboard, for the machine the app is opened from. */
export default function ToolsSection() {
  const { user } = useAuth();
  const t = useT();
  return <ToolsSectionView role={user?.role} hostname={window.location.hostname} t={t} />;
}
