import { useState } from "react";
import { Link, Navigate, NavLink, Outlet, useLocation } from "react-router-dom";
import { errorMessage, type Role } from "../services/api";
import { homePath, useAuth } from "../services/auth";
import { Alert, Avatar, Logo, Spinner } from "./index";
import { APP_NAME } from "../brand";
import { useT } from "../i18n";
import { Icon, type IconName } from "./icons";
import { Preferences } from "./prefs";

/** Gate for a group of routes: signed in, and (when `role` is given) with that role. */
export function RequireAuth({ role }: { role?: Role }) {
  const { user, ready } = useAuth();
  const location = useLocation();
  const t = useT();

  if (!ready) return <Spinner label={t.common.checkingSession} />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  if (role && user.role !== role) return <Navigate to={homePath(user)} replace />;
  return <Outlet />;
}

interface NavItem {
  to: string;
  label: string;
  icon: IconName;
  /** Match the path exactly, so "/admin" is not highlighted on "/admin/users". */
  end?: boolean;
}

function AppShell({ items, admin = false }: { items: NavItem[]; admin?: boolean }) {
  const { user, signOut } = useAuth();
  const location = useLocation();
  const t = useT();
  const [signOutError, setSignOutError] = useState<string | null>(null);
  if (!user) return null;

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (failure) {
      // Never let someone walk away from a shared computer believing they signed out.
      setSignOutError(t.shell.stillSignedIn(errorMessage(failure)));
    }
  };

  return (
    <div className={admin ? "shell shell-admin" : "shell"}>
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" to={admin ? "/admin" : "/home"}>
            <Logo />
            {admin && <span className="brand-tag">{t.shell.admin}</span>}
          </Link>
          <nav className="nav" aria-label={t.shell.main}>
            {items.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className="nav-link">
                <Icon name={item.icon} />
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="topbar-user">
            <Preferences />
            <Link
              to={admin ? "/admin/account" : "/profile"}
              className="user-link"
              title={`${user.email} · ${admin ? t.shell.myAccount : t.shell.profile}`}
            >
              <Avatar name={user.full_name} src={user.photo_url} />
              <span className="user-id">
                <span className="user-name">{user.full_name}</span>
                <span className="user-role">
                  {user.role === "admin" ? t.shell.administrator : t.shell.member}
                </span>
              </span>
            </Link>
            <button type="button" className="btn btn-ghost btn-sm" onClick={handleSignOut}>
              <Icon name="logout" size={16} />
              {t.common.signOut}
            </button>
          </div>
        </div>
      </header>
      <main className="page">
        {signOutError && <Alert>{signOutError}</Alert>}
        {/* Keyed by address, so every page plays its entrance when you navigate to it. */}
        <div className="page-enter" key={location.pathname}>
          <Outlet />
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}

/** Links to the information pages, at the bottom of every page (and of the public ones). */
export function SiteFooter() {
  const t = useT();
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <span className="muted">
          © {new Date().getFullYear()} {APP_NAME} · {t.shell.footer}
        </span>
        <nav aria-label={t.shell.information}>
          <Link to="/about">{t.shell.about}</Link>
          <Link to="/privacy">{t.shell.privacy}</Link>
          <Link to="/help">{t.shell.help}</Link>
        </nav>
      </div>
    </footer>
  );
}

/** The public information pages: the logo, the page, the footer; no sign-in needed. */
export function InfoLayout() {
  const { user } = useAuth();
  const location = useLocation();
  const t = useT();
  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" to="/">
            <Logo />
          </Link>
          <nav className="nav" aria-label={t.shell.information}>
            <NavLink to="/about" className="nav-link">
              {t.shell.about}
            </NavLink>
            <NavLink to="/privacy" className="nav-link">
              {t.shell.privacy}
            </NavLink>
            <NavLink to="/help" className="nav-link">
              {t.shell.help}
            </NavLink>
          </nav>
          <div className="topbar-user">
            <Preferences />
            {user ? (
              <Link className="btn btn-primary btn-sm" to={homePath(user)}>
                {t.shell.backToApp}
              </Link>
            ) : (
              <Link className="btn btn-primary btn-sm" to="/login">
                {t.common.signIn}
              </Link>
            )}
          </div>
        </div>
      </header>
      <main className="page">
        <div className="page-enter" key={location.pathname}>
          <Outlet />
        </div>
      </main>
      <SiteFooter />
    </div>
  );
}

export function UserLayout() {
  const { user } = useAuth();
  const nav = useT().shell.nav;
  const items: NavItem[] = [
    { to: "/home", label: nav.home, icon: "home" },
    { to: "/scan", label: nav.scan, icon: "scan" },
    { to: "/reports", label: nav.reports, icon: "file" },
    { to: "/toxicity", label: nav.toxicity, icon: "flask" },
    { to: "/documents", label: nav.documents, icon: "book" },
    { to: "/monitoring", label: nav.monitoring, icon: "eye" },
  ];
  if (user?.role === "admin") items.push({ to: "/admin", label: nav.admin, icon: "shield" });
  return <AppShell items={items} />;
}

export function AdminLayout() {
  const nav = useT().shell.nav;
  return (
    <AppShell
      admin
      items={[
        { to: "/admin", label: nav.dashboard, icon: "grid", end: true },
        { to: "/admin/users", label: nav.users, icon: "users" },
        { to: "/admin/activity", label: nav.activity, icon: "activity" },
        { to: "/admin/usage", label: nav.usage, icon: "chart" },
        { to: "/home", label: nav.viewApp, icon: "eye" },
      ]}
    />
  );
}
