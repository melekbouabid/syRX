"""Activity log, daily usage and CSV exports of the admin pages (app/activity.py)."""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.activity import GROUP_OF, GROUPS, prune
from app.db import utc_now
from app.models import Action, ActivityEvent
from tujya_testing.fixtures import ADMIN, USER, sign_in


def actions(admin: TestClient, **params: object) -> list[str]:
    response = admin.get("/api/admin/activity", params=params)
    assert response.status_code == 200, response.text
    return [event["action"] for event in response.json()["items"]]


def test_every_action_has_one_group() -> None:
    assert set(GROUP_OF) == {a.value for a in Action}
    assert sum(len(acts) for acts in GROUPS.values()) == len(Action)


def test_activity_is_for_administrators_only(user: TestClient, anonymous: TestClient) -> None:
    for path in ("/api/admin/activity", "/api/admin/usage", "/api/admin/export/users.csv"):
        assert user.get(path).status_code == 403
        assert anonymous.get(path).status_code == 401


def test_sign_in_events(
    admin: TestClient, browser: Callable[[], TestClient], user: TestClient
) -> None:
    wrong = browser().post("/api/auth/login", json={**USER, "password": "not-the-password"})
    assert wrong.status_code == 401
    # An unknown email leaves no event: there is no account to attach it to.
    browser().post("/api/auth/login", json={"email": "nobody@mail.com", "password": "x"})
    user.post("/api/auth/logout")

    found = actions(admin, q=USER["email"])
    assert found[:3] == ["sign_out", "sign_in_failed", "sign_in"]  # newest first
    page = admin.get("/api/admin/activity", params={"group": "security"}).json()
    assert [e["action"] for e in page["items"]] == ["sign_in_failed"]
    assert page["counts"]["security"] == 1 and page["counts"]["access"] >= 3
    event = page["items"][0]
    assert event["user_email"] == USER["email"] and event["group"] == "security"
    assert event["ip"] == "testclient" and event["user_agent"]


def test_blocked_account_is_logged(admin: TestClient, browser: Callable[[], TestClient]) -> None:
    target = admin.get("/api/admin/users", params={"q": USER["email"]}).json()["items"][0]
    admin.patch(f"/api/admin/users/{target['id']}", json={"is_active": False})
    assert browser().post("/api/auth/login", json=USER).status_code == 403

    events = admin.get("/api/admin/activity", params={"user_id": target["id"]}).json()["items"]
    assert [e["action"] for e in events[:2]] == ["sign_in_blocked", "user_update"]
    update = events[1]
    assert update["user_email"] == ADMIN["email"]  # done by the administrator, to the user
    assert update["target_email"] == USER["email"]
    assert update["details"]["fields"] == ["is_active"]


def test_deleted_account_keeps_the_admin_event(admin: TestClient) -> None:
    created = admin.post(
        "/api/admin/users",
        json={"email": "temp@mail.com", "full_name": "Temp", "password": "temporary-1"},
    ).json()
    sign_in(TestClient(admin.app), {"email": "temp@mail.com", "password": "temporary-1"})
    assert admin.delete(f"/api/admin/users/{created['id']}").status_code == 204

    events = admin.get("/api/admin/activity", params={"user_id": created["id"]}).json()["items"]
    # The account's own sign-in is gone with it; what the administrator did stays.
    assert [e["action"] for e in events] == ["user_delete", "user_create"]
    assert events[0]["details"]["name"] == "Temp"


def test_no_health_data_in_the_log(admin: TestClient, user: TestClient, pharma: str) -> None:
    response = user.post("/api/interactions", json={"pills": [7], "ingredients": ["fluconazole"]})
    assert response.status_code == 200
    event = admin.get("/api/admin/activity", params={"group": "usage"}).json()["items"][0]
    assert event["action"] == "interactions"
    assert event["details"] == {"items": 2, "pairs_with_interaction": 1}
    text = str(event).lower()
    assert "warfarin" not in text and "fluconazole" not in text


def test_usage_by_day(admin: TestClient, user: TestClient, pharma: str) -> None:
    user.post("/api/interactions", json={"ingredients": ["warfarin", "fluconazole"]})
    body = admin.get("/api/admin/usage", params={"days": 7}).json()
    assert len(body["days"]) == 7
    today = body["days"][-1]
    assert today["day"] == utc_now().date().isoformat()
    assert today["interactions"] == 1 and today["sign_ins"] == 2
    assert today["active_users"] == 2
    assert body["totals"]["interactions"] == 1
    assert body["top_users"][0]["email"] in (USER["email"], ADMIN["email"])
    assert admin.get("/api/admin/usage", params={"days": 3}).status_code == 422


def test_stats_count_the_week(admin: TestClient, browser: Callable[[], TestClient]) -> None:
    browser().post("/api/auth/login", json={**USER, "password": "wrong-password"})
    stats = admin.get("/api/admin/stats").json()
    assert stats["failed_sign_ins_last_7_days"] == 1


def test_csv_exports(admin: TestClient) -> None:
    admin.post(
        "/api/admin/users",
        json={"email": "csv@mail.com", "full_name": "=HYPERLINK(1)", "password": "password-1"},
    )
    response = admin.get("/api/admin/export/users.csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text.lstrip("﻿"))))
    assert rows[0][:3] == ["id", "email", "full_name"]
    assert "password" not in ",".join(rows[0]).replace("_", "")
    names = [row[2] for row in rows[1:]]
    assert "'=HYPERLINK(1)" in names  # never run as a formula by a spreadsheet

    activity = admin.get("/api/admin/export/activity.csv", params={"group": "admin"})
    lines = list(csv.reader(io.StringIO(activity.text.lstrip("﻿"))))
    assert lines[0][0] == "created_at"
    assert {line[2] for line in lines[1:]} >= {"user_create", "data_export"}
    assert actions(admin, action="data_export")  # the exports themselves are on record


def test_prune_old_events(app: FastAPI, admin: TestClient) -> None:
    with app.state.session_factory() as db:
        old = db.scalars(select(ActivityEvent)).first()
        assert old is not None
        old.created_at = utc_now() - timedelta(days=400)
        db.commit()
        assert prune(db, 0) == 0
        assert prune(db, 180) == 1
