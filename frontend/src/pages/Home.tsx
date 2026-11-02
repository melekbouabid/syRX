import { Link } from "react-router-dom";
import { useAuth } from "../services/auth";
import { APP_NAME } from "../brand";
import { greeting, LogoMark, staggerIndex } from "../components";
import { Icon, type IconName } from "../components/icons";

const SHORTCUTS: { to: string; title: string; text: string; icon: IconName }[] = [
  {
    to: "/scan",
    icon: "scan",
    title: "Scan pills",
    text: "Identify pills from a photo or the webcam, check their interactions and ask a question.",
  },
  {
    to: "/toxicity",
    icon: "flask",
    title: "Toxicity (bonus)",
    text: "See what an in vitro model predicts for an active ingredient, with its limits.",
  },
  {
    to: "/profile",
    icon: "user",
    title: "Your profile",
    text: "Update your name or change your password.",
  },
];

export default function Home() {
  const { user } = useAuth();
  const firstName = user?.full_name.split(/\s+/)[0] ?? "";

  return (
    <>
      <section className="banner">
        <div className="orbs" aria-hidden="true">
          <span />
          <span />
        </div>
        <div className="banner-copy">
          <p className="eyebrow">Home</p>
          <h1>
            {greeting()}, {firstName}
          </h1>
          <p>
            {APP_NAME} identifies pills from a photo, checks whether their active ingredients
            interact and answers your questions with cited sources. Information only: consult a
            pharmacist or doctor.
          </p>
        </div>
        <LogoMark size={170} className="banner-emblem emblem-float" />
      </section>

      <h2 className="section-title">Where to go</h2>
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
            <h3>{shortcut.title}</h3>
            <p>{shortcut.text}</p>
            <span className="link-card-arrow" aria-hidden="true">
              <Icon name="arrow" />
            </span>
          </Link>
        ))}
      </section>
    </>
  );
}
