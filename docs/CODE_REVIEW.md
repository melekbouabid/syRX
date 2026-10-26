# Code review notes

What a reviewer checks before a change to this pipeline is merged. The first section is the
source-access policy; it is the part most easily eroded one "small exception" at a time.

## Source access: API first, bulk second, HTML last

1. **Is there an official API?** Use it. Follow its documented limits and identification
   rules (`tool` / `email` / `api_key` for NCBI).
2. **No API, but an official bulk file?** Download that, with a pinned version and a
   checksum where the publisher gives one.
3. **Neither?** Stop and ask whether the source is needed. Parsing HTML is the last resort,
   needs a written reason in `docs/DECISIONS.md`, and must go through
   `client.fetch(..., kind="html")` so robots.txt is checked first.

No connector parses HTML today. `tests/unit/test_policy.py` fails if one starts to.

## Checklist for a new or changed connector

- [ ] It subclasses `Connector` and implements `discover`, `fetch`, `parse`, `validate`.
- [ ] `discover` touches neither the network nor the disk (`--dry-run` relies on it).
- [ ] Every request goes through `PoliteClient`. No `requests`, `urllib` or raw `httpx`.
- [ ] Every host it contacts is listed in `SourceInfo.hosts`.
- [ ] `SourceInfo` has the licence, the terms URL, a citation and a redistribution note.
- [ ] A host limit exists in `config.yaml`, with what the provider documents recorded in
      `quality/linkcheck.py` (`RATE_LIMITS`).
- [ ] Licence-gated or non-commercial source: disabled by default and gated explicitly.
- [ ] Every output row carries the five provenance columns and passes a Pydantic schema.
- [ ] Rows that fail validation are written to `*.rejects.parquet`, not dropped.
- [ ] A second run with unchanged inputs makes no request and rewrites nothing.
- [ ] Unit tests mock HTTP with respx; nothing in `tests/unit` reaches the network.

## Never

- Bypass a CAPTCHA, login, paywall or bot check, or retry after a 401 / 403.
- Set `Cookie`, `Authorization`, a browser User-Agent or forwarding headers.
- Add proxies, IP rotation or any other way to look like several clients.
- Call an undocumented or authenticated endpoint.
- Put a secret in YAML, a fixture, a log line or a stored URL.
- Merge or convert synergy scores of different types.
- Commit anything under `data/`.
