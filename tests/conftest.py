"""Tests of the code under src/, run from the repository root: `python -m pytest`.

They reuse the web app's fixtures (`app`, `user`, `pharma`, ...,
backend/tujya_testing/fixtures.py), importable from the project venv (.venv)."""

pytest_plugins = ["tujya_testing.fixtures"]
