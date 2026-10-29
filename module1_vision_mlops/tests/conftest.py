"""The scan API tests reuse the web app's fixtures (`app`, `user`, ...):
app/backend/tujya_testing/fixtures.py, importable from the backend venv."""

pytest_plugins = ["tujya_testing.fixtures"]
