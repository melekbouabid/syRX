"""Tests of the code under src/: `python -m pytest` from the repository root.

Reuses the web app fixtures (backend/tests/support) and the data pipeline fixtures
(tests/support/sources_fixtures.py); pgvector tests need TUJYA_TEST_PG_URL.
"""

pytest_plugins = ["backend.tests.support.fixtures", "tests.support.sources_fixtures"]
