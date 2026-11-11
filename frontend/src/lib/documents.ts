// Display helpers of the Documents page (GET /api/documents, docs/api.md).

export const DOC_TYPE_LABELS: Record<string, string> = {
  label: "Drug label",
  abstract: "PubMed abstract",
  interaction: "DDInter pair",
};

export function docTypeLabel(docType: string): string {
  return DOC_TYPE_LABELS[docType] ?? docType;
}

/** "20260910" (openFDA effective time) -> "2026-09-10"; other dates unchanged. */
export function formatDocDate(date: string): string {
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(date);
  return match ? `${match[1]}-${match[2]}-${match[3]}` : date || "n/a";
}

/** "drug_interactions" -> "Drug interactions". */
export function sectionLabel(section: string): string {
  const words = section.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** The first 12 hex digits of a SHA-256, enough to tell chunks apart on screen. */
export function shortHash(hash: string): string {
  return hash ? `${hash.slice(0, 12)}…` : "n/a";
}

/** Facet options for a filter: "label (82)", most frequent first. */
export function facetOptions(counts: Record<string, number> | undefined): [string, string][] {
  return Object.entries(counts ?? {}).map(([value, count]) => [
    value,
    `${docTypeLabel(value)} (${count.toLocaleString("en-US")})`,
  ]);
}
