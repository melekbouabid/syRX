import { Link, Navigate, Route, Routes } from "react-router-dom";
import { homePath, useAuth } from "./auth";
import { Logo, Spinner } from "./components";
import { AdminLayout, RequireAuth, UserLayout } from "./layouts";
import Dashboard from "./pages/admin/Dashboard";
import Users from "./pages/admin/Users";
import AuthPage from "./pages/AuthPage";
import { CellLinesPage, DrugsPage } from "./pages/CatalogPage";
import Home from "./pages/Home";
import Intro from "./pages/Intro";
import Profile from "./pages/Profile";
import ScanPage from "./pages/ScanPage";
import ToxicityPage from "./pages/ToxicityPage";

/** "/" shows the introduction to visitors and sends signed-in accounts to their own area. */
function Landing() {
  const { user, ready } = useAuth();
  if (!ready) return <Spinner label="Checking your session" />;
  return user ? <Navigate to={homePath(user)} replace /> : <Intro />;
}

function NotFound() {
  return (
    <div className="not-found">
      <Logo height={72} />
      <h1>Page not found</h1>
      <p className="muted">There is nothing at this address.</p>
      <Link className="btn btn-primary" to="/">
        Go to the start page
      </Link>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<AuthPage mode="signin" />} />
      <Route path="/signup" element={<AuthPage mode="signup" />} />

      {/* Any signed-in account */}
      <Route element={<RequireAuth />}>
        <Route element={<UserLayout />}>
          <Route path="/home" element={<Home />} />
          <Route path="/scan" element={<ScanPage />} />
          <Route path="/toxicity" element={<ToxicityPage />} />
          <Route path="/drugs" element={<DrugsPage />} />
          <Route path="/cell-lines" element={<CellLinesPage />} />
          <Route path="/profile" element={<Profile />} />
        </Route>
      </Route>

      {/* Administrators only. The API checks the role again on every request. */}
      <Route element={<RequireAuth role="admin" />}>
        <Route element={<AdminLayout />}>
          <Route path="/admin" element={<Dashboard />} />
          <Route path="/admin/users" element={<Users />} />
          <Route path="/admin/account" element={<Profile />} />
        </Route>
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
