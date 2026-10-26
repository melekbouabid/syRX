## What changed and why

## Source access (see docs/CODE_REVIEW.md)

- [ ] No new HTML parsing. If there is: the API and bulk options were ruled out and the
      reason is in `docs/DECISIONS.md`.
- [ ] New hosts are declared in the connector's `SourceInfo.hosts` and have a rate limit.
- [ ] Licence, terms URL and citation are filled in for any new source.
- [ ] Nothing bypasses a login, CAPTCHA, paywall or robots.txt rule.

## Data and tests

- [ ] Provenance columns and schema validation are in place for new tables.
- [ ] `make check` passes (unit tests use mocked HTTP only).
- [ ] No secret and no collected data is part of this change.
