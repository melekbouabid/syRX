"""Tests of the code under src/, run from the repository root: `python -m pytest`.

They reuse two fixture sets, importable from the project venv (.venv): the web app's
(`app`, `user`, `pharma`, ..., backend/tujya_testing/fixtures.py) for the RAG, agent, chat
and MCP tests, and the data pipeline's (`settings`, `mock_http`, `make_client`, ...,
src/data/tujya_data/testing.py) for the corpus tests, which run on it.
The pgvector tests also need TUJYA_TEST_PG_URL (PostgreSQL + pgvector); skipped otherwise."""

pytest_plugins = ["tujya_testing.fixtures", "src.data.tujya_data.testing"]
