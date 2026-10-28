"""Swagger UI and the OpenAPI schema: /docs and /openapi.json, plus the older /api/ addresses."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_swagger_ui_is_served_at_docs(anonymous: TestClient) -> None:
    response = anonymous.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text
    assert "/openapi.json" in response.text


def test_openapi_schema_lists_the_api_routes(anonymous: TestClient) -> None:
    schema = anonymous.get("/openapi.json").json()
    assert schema["info"]["title"] == "Tujya API"
    assert "/api/health" in schema["paths"]
    assert "/api/auth/login" in schema["paths"]
    # The compatibility addresses are not API routes and stay out of the schema.
    assert "/api/docs" not in schema["paths"]
    assert "/api/openapi.json" not in schema["paths"]


def test_old_docs_address_redirects_to_docs(anonymous: TestClient) -> None:
    response = anonymous.get("/api/docs", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
    assert anonymous.get("/api/docs").status_code == 200  # following it lands on Swagger


def test_old_schema_address_serves_the_same_schema(anonymous: TestClient) -> None:
    assert anonymous.get("/api/openapi.json").json() == anonymous.get("/openapi.json").json()
