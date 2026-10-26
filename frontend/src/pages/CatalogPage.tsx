import { useEffect, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type CellLine, type Drug, type Page } from "../api";
import {
  Alert,
  Badge,
  formatNumber,
  Pagination,
  staggerIndex,
  TableSkeleton,
} from "../components";
import { useDebounced, useQuery } from "../hooks";
import { Icon, type IconName } from "../icons";

const PAGE_SIZE = 25;

interface Column<T> {
  header: string;
  cell: (row: T) => ReactNode;
  numeric?: boolean;
}

interface CatalogProps<T> {
  title: string;
  subtitle: string;
  icon: IconName;
  searchPlaceholder: string;
  load: (query: { q: string; page: number; page_size: number }, signal: AbortSignal) => Promise<Page<T>>;
  rowKey: (row: T) => string;
  columns: Column<T>[];
}

/** A searchable, paginated table over one of the processed tables. */
function Catalog<T>({ title, subtitle, icon, searchPlaceholder, load, rowKey, columns }: CatalogProps<T>) {
  // The search lives in the address (?q=...), so a search can be linked to and survives reload.
  const [params, setParams] = useSearchParams();
  const [search, setSearch] = useState(params.get("q") ?? "");
  const [page, setPage] = useState(1);
  const q = useDebounced(search.trim());
  const result = useQuery((signal) => load({ q, page, page_size: PAGE_SIZE }, signal), [q, page]);

  useEffect(() => {
    if ((params.get("q") ?? "") !== q) setParams(q ? { q } : {}, { replace: true });
  }, [q, params, setParams]);

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name={icon} size={24} />
          </span>
          <div>
            <h1>{title}</h1>
            <p className="muted">{subtitle}</p>
          </div>
        </div>
        {result.data && (
          <span className="result-count" aria-live="polite">
            <strong>{formatNumber(result.data.total)}</strong> {q ? "matching" : "in total"}
          </span>
        )}
      </header>

      <div className="toolbar">
        <label className="search-box">
          <Icon name="search" />
          <input
            type="search"
            aria-label={searchPlaceholder}
            placeholder={searchPlaceholder}
            value={search}
            onChange={(event) => {
              setSearch(event.target.value);
              setPage(1);
            }}
          />
        </label>
      </div>

      {result.error && <Alert>{result.error}</Alert>}
      {!result.data && result.loading && <TableSkeleton rows={10} />}

      {result.data && (
        <div className="card table-card" aria-busy={result.loading}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  {columns.map((column) => (
                    <th key={column.header} className={column.numeric ? "num" : undefined}>
                      {column.header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {/* Rows are keyed by record, so the ones a new page brings in play their entrance. */}
                {result.data.items.map((row, index) => (
                  <tr key={rowKey(row)} className="row-enter" style={staggerIndex(index)}>
                    {columns.map((column) => (
                      <td key={column.header} className={column.numeric ? "num" : undefined}>
                        {column.cell(row)}
                      </td>
                    ))}
                  </tr>
                ))}
                {result.data.items.length === 0 && (
                  <tr>
                    <td colSpan={columns.length} className="empty">
                      {q ? `Nothing matches “${q}”.` : "This table is empty."}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={result.data.total}
            onPage={setPage}
          />
        </div>
      )}
    </>
  );
}

const dash = <span className="muted">—</span>;

function Sources({ names }: { names: string[] }) {
  return (
    <span className="badge-row">
      {names.map((name) => (
        <Badge tone="neutral" key={name}>
          {name}
        </Badge>
      ))}
    </span>
  );
}

function External({ href, children }: { href: string; children: ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

const DRUG_COLUMNS: Column<Drug>[] = [
  {
    header: "Drug",
    cell: (drug) => (
      <>
        <span className="cell-title">{drug.name}</span>
        {drug.synonyms.length > 0 && (
          <span className="cell-sub">{drug.synonyms.slice(0, 3).join(" · ")}</span>
        )}
      </>
    ),
  },
  {
    header: "PubChem",
    cell: (drug) =>
      drug.pubchem_cid == null ? (
        dash
      ) : (
        <External href={`https://pubchem.ncbi.nlm.nih.gov/compound/${drug.pubchem_cid}`}>
          {drug.pubchem_cid}
        </External>
      ),
  },
  {
    header: "ChEMBL",
    cell: (drug) =>
      drug.chembl_id ? (
        <External
          href={`https://www.ebi.ac.uk/chembl/explore/compound/${encodeURIComponent(drug.chembl_id)}`}
        >
          {drug.chembl_id}
        </External>
      ) : (
        dash
      ),
  },
  { header: "Sources", cell: (drug) => <Sources names={drug.sources} /> },
  { header: "Measurements", numeric: true, cell: (drug) => formatNumber(drug.n_measurements) },
];

const CELL_LINE_COLUMNS: Column<CellLine>[] = [
  { header: "Cell line", cell: (line) => <span className="cell-title">{line.name}</span> },
  { header: "Tissue", cell: (line) => line.tissue ?? dash },
  { header: "Disease", cell: (line) => line.disease ?? dash },
  {
    header: "DepMap ID",
    cell: (line) =>
      line.depmap_id ? (
        <External href={`https://depmap.org/portal/cell_line/${encodeURIComponent(line.depmap_id)}`}>
          {line.depmap_id}
        </External>
      ) : (
        dash
      ),
  },
  { header: "Sources", cell: (line) => <Sources names={line.sources} /> },
  { header: "Measurements", numeric: true, cell: (line) => formatNumber(line.n_measurements) },
];

export function DrugsPage() {
  return (
    <Catalog
      title="Drugs"
      subtitle="Every compound found in the collected screens, most measured first."
      icon="pill"
      searchPlaceholder="Search by name, synonym or ChEMBL ID"
      load={api.drugs}
      rowKey={(drug) => drug.drug_id}
      columns={DRUG_COLUMNS}
    />
  );
}

export function CellLinesPage() {
  return (
    <Catalog
      title="Cell lines"
      subtitle="Cancer cell lines the combinations were tested on, most measured first."
      icon="cells"
      searchPlaceholder="Search by name, tissue or disease"
      load={api.cellLines}
      rowKey={(line) => line.cell_line_id}
      columns={CELL_LINE_COLUMNS}
    />
  );
}
