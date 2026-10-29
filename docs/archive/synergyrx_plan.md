# SynergyRx data pipeline: plan

Purpose: collect public drug-combination, cell-line, chemistry and literature data for
SynergyRx AI in a polite, reproducible way. It is a set of source connectors, not a crawler.

## Architecture

```
CLI (typer) ──> pipeline.py ──> connectors/*  ──> PoliteClient ──> network
                    │                │                 ├─ guardrails (hosts, https, honest User-Agent)
                    │                │                 ├─ robots.txt (RFC 9309) for HTML and bulk
                    │                │                 ├─ per-host token bucket + semaphore
                    │                │                 ├─ retry (backoff + jitter, Retry-After)
                    │                │                 ├─ circuit breaker per source
                    │                │                 └─ disk cache (ETag / Last-Modified)
                    │                └─ discover → fetch → parse → validate
                    ├─ cleaning/*    (SMILES, drug identity, cell lines, pair keys, scores)
                    ├─ literature/*  (queries, abstracts, sentences, mentions, annotation sample)
                    ├─ quality/*     (coverage, conflicts, drift, licenses, link check)
                    └─ export/*      (frontend JSON)
```

Every connector implements the same four steps:

| Step | Network | Writes | Purpose |
|------|---------|--------|---------|
| `discover` | no | no | list what would be fetched (this is what `--dry-run` prints) |
| `fetch` | yes | `data/raw` + `manifest.json` | download or call the API, idempotently |
| `parse` | no | no | raw files to tidy tables with provenance columns |
| `validate` | no | `data/interim` | Pydantic check per row; rejects are kept and counted |

## Storage layers

- `data/raw/<source>/<version>/` untouched files as retrieved, hashed in `manifest.json`
- `data/interim/<source>/` one Parquet table per source, validated, with provenance
- `data/parquet/` the six processed tables the team uses
- `data/csv/`, `data/json/`, `data/reports/` the other generated files, grouped by format:
  CSV reports and the annotation sample, frontend JSON, and the quality report (md / html)
- `data/manual/<source>/` files a person must place (DrugBank, robots-blocked downloads)
- `data/cache/http/` HTTP cache, `data/state/` connector state and schema snapshots

## Phases

1. Core collection: HTTP stack, connector interface, synergy / cell-line / chemistry / PubMed
   connectors, manifest, CLI, mocked tests.
2. Cleaning and quality: pair keys, RDKit standardisation, identity resolution, cell-line
   mapping, conflict / coverage / drift reports, licence registry, frontend export.
3. Literature: Europe PMC, PubTator, baseline, DDInter, DrugBank (manual), query builder,
   sentence store, mention detection, annotation sample.
4. Operations: live smoke tests, Docker, Makefile, pre-commit, README, data dictionary,
   scheduler examples, preflight, `summary`.

After each phase: tests, linter, five lines in `docs/STATUS.md`. Choices made along the way
are recorded in `docs/DECISIONS.md`; what each source really returned is in `docs/SOURCES.md`.
