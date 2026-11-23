import { useState } from "react";
import { api, type DocumentsPage as Page, type IndexedDocument, type IndexInfo } from "../services/api";
import { useDebounced, useQuery } from "../hooks";
import {
  Alert,
  Badge,
  Modal,
  Pagination,
  Spinner,
  TableSkeleton,
  formatDate,
  formatNumber,
  staggerIndex,
} from "../components";
import { Icon } from "../components/icons";
import { useT } from "../i18n";
import {
  docTypeLabel,
  facetOptions,
  formatDocDate,
  sectionLabel,
  shortHash,
} from "../lib/documents";

// The documents the assistant searches, as indexed in Qdrant (GET /api/documents).
const PAGE_SIZE = 25;

function IndexCard({ index }: { index: IndexInfo }) {
  const t = useT();
  const w = t.documents.index;
  const facts: [string, string][] = [
    [w.store, index.store === "qdrant" ? `Qdrant · ${index.collection}` : `${index.store} · ${index.collection}`],
    [w.documents, formatNumber(index.documents)],
    [w.chunks, formatNumber(index.points)],
    [w.embeddings, w.dimensions(index.embedding_model, index.dim)],
    [w.indexed, index.indexed_at ? formatDate(index.indexed_at, true) : t.common.na],
  ];
  return (
    <section className="card mb-4 p-4" aria-label={w.label}>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label}>
            <dt className="cell-sub">{label}</dt>
            <dd className="cell-title m-0">{value}</dd>
          </div>
        ))}
      </dl>
      <p className="cell-sub mt-3">
        {w.filtered(index.payload_indexes.join(", ") || t.common.none)}{" "}
        {index.dashboard_url && (
          <a href={index.dashboard_url} target="_blank" rel="noreferrer">
            {w.dashboard}
          </a>
        )}
      </p>
    </section>
  );
}

function DocumentDialog({ docId, onClose }: { docId: string; onClose: () => void }) {
  const detail = useQuery((signal) => api.document(docId, signal), [docId]);
  const doc = detail.data;
  const t = useT();
  const w = t.documents.dialog;
  return (
    <Modal title={doc?.title ?? w.document} onClose={onClose}>
      {detail.error && <Alert>{detail.error}</Alert>}
      {!doc && detail.loading && <Spinner label={w.loading} />}
      {doc && (
        <div className="grid gap-4">
          <dl className="grid grid-cols-1 gap-x-6 gap-y-1 sm:grid-cols-2">
            {(
              [
                [w.identifier, doc.doc_id],
                [w.type, docTypeLabel(doc.doc_type) + (doc.article_type ? ` · ${doc.article_type}` : "")],
                [w.source, doc.source_name],
                [w.date, formatDocDate(doc.date)],
                [w.version, doc.version],
                [w.language, doc.language],
                [w.licence, doc.license],
                [w.about, doc.ingredients.join(", ") || t.common.na],
                [w.mentions, doc.mentioned_ingredients.join(", ") || t.common.none],
              ] as [string, string][]
            ).map(([label, value]) => (
              <div key={label}>
                <dt className="cell-sub">{label}</dt>
                <dd className="m-0 break-words">{value}</dd>
              </div>
            ))}
          </dl>
          <p>
            <a href={doc.url} target="_blank" rel="noreferrer">
              {w.original}
            </a>
          </p>
          <h3>{w.chunks(doc.chunks.length)}</h3>
          <ol className="grid gap-3">
            {doc.chunks.map((chunk) => (
              <li key={chunk.chunk_id} className="card p-3">
                <p className="cell-sub">
                  #{chunk.chunk_index} · {sectionLabel(chunk.section)} · id {chunk.chunk_id} ·
                  SHA-256 <span title={chunk.content_hash}>{shortHash(chunk.content_hash)}</span>
                </p>
                <p className="mt-1 whitespace-pre-line">{chunk.text}</p>
              </li>
            ))}
          </ol>
          <p className="disclaimer">
            <Icon name="shield" size={16} />
            {w.doses} {t.common.disclaimer}
          </p>
        </div>
      )}
    </Modal>
  );
}

function DocumentRow({ doc, index, onOpen }: { doc: IndexedDocument; index: number; onOpen: () => void }) {
  const t = useT();
  return (
    <tr className="row-enter" style={staggerIndex(index)}>
      <td>
        <button
          type="button"
          className="cursor-pointer border-0 bg-transparent p-0 text-left"
          onClick={onOpen}
        >
          <span className="cell-title">{doc.title}</span>
          <span className="cell-sub">{doc.doc_id}</span>
        </button>
      </td>
      <td>
        <Badge tone={doc.doc_type === "label" ? "info" : "neutral"}>{docTypeLabel(doc.doc_type)}</Badge>
        {doc.is_retracted && <Badge tone="danger">{t.documents.retracted}</Badge>}
      </td>
      <td>{doc.source_name}</td>
      <td>{doc.ingredients.join(", ")}</td>
      <td>{formatDocDate(doc.date)}</td>
      <td className="num">{doc.chunk_count}</td>
    </tr>
  );
}

export default function DocumentsPage() {
  const t = useT();
  const d = t.documents;
  const [search, setSearch] = useState("");
  const [ingredientInput, setIngredientInput] = useState("");
  const [docType, setDocType] = useState("");
  const [source, setSource] = useState("");
  const [page, setPage] = useState(1);
  const [open, setOpen] = useState<string | null>(null);

  const q = useDebounced(search.trim());
  const ingredient = useDebounced(ingredientInput.trim());
  const documents = useQuery(
    (signal) =>
      api.documents(
        {
          q: q || undefined,
          ingredient: ingredient || undefined,
          doc_type: docType || undefined,
          source: source || undefined,
          limit: PAGE_SIZE,
          offset: (page - 1) * PAGE_SIZE,
        },
        signal,
      ),
    [q, ingredient, docType, source, page],
  );
  // The options keep the counts of the last unfiltered answer for that field.
  const [facets, setFacets] = useState<Page["facets"] | null>(null);
  if (documents.data && !docType && !source && !facets) setFacets(documents.data.facets);

  const reset = (apply: () => void) => {
    apply();
    setPage(1);
  };

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="book" size={24} />
          </span>
          <div>
            <h1>{d.title}</h1>
            <p className="muted">{d.lead}</p>
          </div>
        </div>
      </header>

      {documents.data && <IndexCard index={documents.data.index} />}

      <div className="toolbar">
        <label className="search-box">
          <Icon name="search" />
          <input
            type="search"
            aria-label={d.searchTitles}
            placeholder={d.titleWords}
            value={search}
            onChange={(event) => reset(() => setSearch(event.target.value))}
          />
        </label>
        <label className="search-box">
          <Icon name="pill" />
          <input
            type="search"
            aria-label={d.ingredient}
            placeholder={d.ingredientPlaceholder}
            value={ingredientInput}
            onChange={(event) => reset(() => setIngredientInput(event.target.value))}
          />
        </label>
        <select
          aria-label={d.byType}
          value={docType}
          onChange={(event) => reset(() => setDocType(event.target.value))}
        >
          <option value="">{d.allTypes}</option>
          {facetOptions(facets?.doc_type).map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <select
          aria-label={d.bySource}
          value={source}
          onChange={(event) => reset(() => setSource(event.target.value))}
        >
          <option value="">{d.allSources}</option>
          {Object.entries(facets?.source ?? {}).map(([value, count]) => (
            <option key={value} value={value}>
              {value} ({formatNumber(count)})
            </option>
          ))}
        </select>
      </div>

      {documents.error && <Alert>{documents.error}</Alert>}
      {!documents.data && documents.loading && <TableSkeleton rows={8} />}

      {documents.data && (
        <div className="card table-card" aria-busy={documents.loading}>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>{d.columns.document}</th>
                  <th>{d.columns.type}</th>
                  <th>{d.columns.source}</th>
                  <th>{d.columns.about}</th>
                  <th>{d.columns.date}</th>
                  <th className="num">{d.columns.chunks}</th>
                </tr>
              </thead>
              <tbody>
                {documents.data.items.map((doc, index) => (
                  <DocumentRow key={doc.doc_id} doc={doc} index={index} onOpen={() => setOpen(doc.doc_id)} />
                ))}
              </tbody>
            </table>
          </div>
          {documents.data.total === 0 && <p className="muted p-4">{d.none}</p>}
          <Pagination page={page} pageSize={PAGE_SIZE} total={documents.data.total} onPage={setPage} />
        </div>
      )}

      {open && <DocumentDialog docId={open} onClose={() => setOpen(null)} />}
    </>
  );
}
