import { useState } from "react";
import { Link, Navigate, NavLink, Outlet, useLocation } from "react-router-dom";
import { errorMessage, type Role } from "../services/api";
import { homePath, useAuth } from "../services/auth";
import { Alert, Avatar, Logo, Spinner } from "./index";
import { APP_NAME } from "../brand";
import { Icon, type IconName } from "./icons";

/** Gate for a group of routes: signed in, and (when `role` is given) with that role. */
export function RequireAuth({ role }: { role?: Role }) {
  const { user, ready } = useAuth();
  const location = useLocation();

  if (!ready) return <Spinner label="Checking your session" />;
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
  const [signOutError, setSignOutError] = useState<string | null>(null);
  if (!user) return null;

  const handleSignOut = async () => {
    try {
      await signOut();
    } catch (failure) {
      // Never let someone walk away from a shared computer believing they signed out.
      setSignOutError(`You are still signed in. ${errorMessage(failure)}`);
    }
  };

  return (
    <div className={admin ? "shell shell-admin" : "shell"}>
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" to={admin ? "/admin" : "/home"}>
            <Logo />
            {admin && <span className="brand-tag">Admin</span>}
          </Link>
          <nav className="nav" aria-label="Main">
            {items.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className="nav-link">
                <Icon name={item.icon} />
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className="topbar-user">
            <Link
              to={admin ? "/admin/account" : "/profile"}
              className="user-link"
              title={`${user.email} · ${admin ? "My account" : "Profile"}`}
            >
              <Avatar name={user.full_name} src={user.photo_url} />
              <span className="user-id">
                <span className="user-name">{user.full_name}</span>
                <span className="user-role">{user.role === "admin" ? "Administrator" : "Member"}</span>
              </span>
            </Link>
            <button type="button" className="btn btn-ghost btn-sm" onClick={handleSignOut}>
              <Icon name="logout" size={16} />
              Sign out
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
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <span className="muted">
          © {new Date().getFullYear()} {APP_NAME} · Information uniquement, consultez un pharmacien
          ou un médecin.
        </span>
        <nav aria-label="Information">
          <Link to="/about">About</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/help">Help</Link>
        </nav>
      </div>
    </footer>
  );
}

/** The public information pages: the logo, the page, the footer; no sign-in needed. */
export function InfoLayout() {
  const { user } = useAuth();
  const location = useLocation();
  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <Link className="brand" to="/">
            <Logo />
          </Link>
          <nav className="nav" aria-label="Information">
            <NavLink to="/about" className="nav-link">
              About
            </NavLink>
            <NavLink to="/privacy" className="nav-link">
              Privacy
            </NavLink>
            <NavLink to="/help" className="nav-link">
              Help
            </NavLink>
          </nav>
          <div className="topbar-user">
            {user ? (
              <Link className="btn btn-primary btn-sm" to={homePath(user)}>
                Back to the app
              </Link>
            ) : (
              <Link className="btn btn-primary btn-sm" to="/login">
                Sign in
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
  const items: NavItem[] = [
    { to: "/home", label: "Home", icon: "home" },
    { to: "/scan", label: "Scanner", icon: "scan" },
    { to: "/reports", label: "Reports", icon: "file" },
    { to: "/toxicity", label: "Toxicity", icon: "flask" },
    { to: "/documents", label: "Documents", icon: "book" },
    { to: "/monitoring", label: "Monitoring", icon: "eye" },
  ];
  if (user?.role === "admin") items.push({ to: "/admin", label: "Admin", icon: "shield" });
  return <AppShell items={items} />;
}

export function AdminLayout() {
  return (
    <AppShell
      admin
      items={[
        { to: "/admin", label: "Dashboard", icon: "grid", end: true },
        { to: "/admin/users", label: "Users", icon: "users" },
        { to: "/admin/activity", label: "Activity", icon: "activity" },
        { to: "/admin/usage", label: "Usage", icon: "chart" },
        { to: "/home", label: "View app", icon: "eye" },
      ]}
    />
  );
}
