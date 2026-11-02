"""Test support shared by the web app's tests and the module test suites.

`tujya_testing.fixtures` is a pytest plugin (the `app`, `user`, `admin`, `pharma`, ...
fixtures); every test suite that calls the API loads it from its conftest.py with
`pytest_plugins = ["tujya_testing.fixtures"]`. Importable from the project venv (.venv)
(pyproject.toml); never part of the Docker image.
"""
