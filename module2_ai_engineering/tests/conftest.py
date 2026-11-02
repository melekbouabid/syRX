"""Module 2 tests reuse two fixture sets: the web app's (`app`, `user`, `pharma`, ...,
backend/tujya_testing/fixtures.py) for the RAG, agent, chat and MCP tests, and the
pipeline's (`settings`, `mock_http`, `make_client`, ...) for the corpus tests, which run on
the SynergyRx pipeline (data_pipeline/src/synergyrx_data/testing.py)."""

pytest_plugins = ["tujya_testing.fixtures", "synergyrx_data.testing"]
