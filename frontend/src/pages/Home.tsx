import { Link } from "react-router-dom";
import { useAuth } from "../services/auth";
import { APP_NAME } from "../brand";
import { greeting, LogoMark, staggerIndex } from "../components";
import { Icon, type IconName } from "../components/icons";
import { useT } from "../i18n";

// Their texts are in src/i18n (home.shortcuts).
const SHORTCUTS: { to: string; key: "scan" | "toxicity" | "profile"; icon: IconName }[] = [
  { to: "/scan", key: "scan", icon: "scan" },
  { to: "/toxicity", key: "toxicity", icon: "flask" },
  { to: "/profile", key: "profile", icon: "user" },
];

export default function Home() {
  const { user } = useAuth();
  const t = useT();
  const firstName = user?.full_name.split(/\s+/)[0] ?? "";

  return (
    <>
      <section className="banner">
        <div className="orbs" aria-hidden="true">
          <span />
          <span />
        </div>
        <div className="banner-copy">
          <p className="eyebrow">{t.home.eyebrow}</p>
          <h1>
            {greeting()}, {firstName}
          </h1>
          <p>{t.home.lead(APP_NAME)}</p>
        </div>
        <LogoMark size={170} className="banner-emblem emblem-float" />
      </section>

      <h2 className="section-title">{t.home.whereToGo}</h2>
      <section className="link-grid">
        {SHORTCUTS.map((shortcut, index) => (
          <Link
            className="card link-card lift"
            to={shortcut.to}
            key={shortcut.to}
            style={staggerIndex(index)}
          >
            <span className="link-card-icon">
              <Icon name={shortcut.icon} size={22} />
            </span>
            <h3>{t.home.shortcuts[shortcut.key].title}</h3>
            <p>{t.home.shortcuts[shortcut.key].text}</p>
            <span className="link-card-arrow" aria-hidden="true">
              <Icon name="arrow" />
            </span>
          </Link>
        ))}
      </section>
    </>
  );
}
