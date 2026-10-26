import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, type CatalogSummary } from "../api";
import { useAuth } from "../auth";
import { APP_NAME } from "../brand";
import { Alert, greeting, LogoMark, StatTile, staggerIndex } from "../components";
import { useQuery } from "../hooks";
import { Icon, type IconName } from "../icons";

const TILES: { key: keyof CatalogSummary; label: string; note: string; icon: IconName }[] = [
  { key: "drugs", label: "Drugs", note: "with identifiers and synonyms", icon: "pill" },
  { key: "cell_lines", label: "Cell lines", note: "mapped to DepMap", icon: "cells" },
  {
    key: "synergy_measurements",
    label: "Synergy measurements",
    note: "Bliss, Loewe, HSA, ZIP",
    icon: "flask",
  },
  { key: "abstracts", label: "PubMed abstracts", note: "on drug combinations", icon: "book" },
];

const SHORTCUTS: { to: string; title: string; text: string; icon: IconName }[] = [
  {
    to: "/drugs",
    icon: "pill",
    title: "Browse drugs",
    text: "Search by name, synonym or ChEMBL ID and open the PubChem record.",
  },
  {
    to: "/cell-lines",
    icon: "cells",
    title: "Browse cell lines",
    text: "See tissue, disease and how many measurements each line has.",
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
  const navigate = useNavigate();
  const summary = useQuery((signal) => api.catalogSummary(signal), []);
  const [search, setSearch] = useState("");
  const firstName = user?.full_name.split(/\s+/)[0] ?? "";

  const openDrugs = (event: FormEvent) => {
    event.preventDefault();
    const q = search.trim();
    navigate(q ? `/drugs?q=${encodeURIComponent(q)}` : "/drugs");
  };

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
            {APP_NAME} brings together published drug-combination screens on cancer cell lines,
            so you can see which pairs were measured, where, and how strongly they interact.
          </p>
          <form className="banner-search" role="search" onSubmit={openDrugs}>
            <Icon name="search" size={20} />
            <input
              type="search"
              aria-label="Search a drug"
              placeholder="Search a drug, for example erlotinib"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <button type="submit" className="btn btn-light">
              Search
            </button>
          </form>
        </div>
        <LogoMark size={170} className="banner-emblem emblem-float" />
      </section>

      {summary.error && <Alert>{summary.error}</Alert>}

      <section aria-label="Dataset size" className="stat-grid">
        {TILES.map((tile) => (
          <StatTile
            key={tile.key}
            icon={tile.icon}
            label={tile.label}
            note={summary.data && summary.data[tile.key] == null ? "not built yet" : tile.note}
            value={summary.data?.[tile.key]}
            loading={summary.loading && !summary.data}
          />
        ))}
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
