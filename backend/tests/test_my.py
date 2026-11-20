"""My medicines, saved reports and their share links (app/routers/my.py, app/reports.py)."""

from __future__ import annotations

from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import utc_now
from app.medicines import DISCLAIMER
from app.models import Report
from app.reports import seal
from app.schemas import CitationRef
from tujya_testing.fixtures import USER, sign_in

ANSWER = {
    "question": "Warfarine et fluconazole ?",
    "answer": "DDInter classe cette interaction comme majeure [1]. " + DISCLAIMER,
    "citations": [{"n": 1, "title": "DDInter 2.0", "url": "https://ddinter2.scbdd.com/"}],
}


def sealed(app: FastAPI, excerpt: dict) -> dict:
    refs = [CitationRef(**c) for c in excerpt["citations"]]
    return {
        **excerpt,
        "seal": seal(app.state.report_key, excerpt["question"], excerpt["answer"], refs),
    }


# ------------------------------------------------------------------ my medicines


def test_my_medicines(user: TestClient, pharma: str) -> None:
    assert user.get("/api/my/medicines").json()["items"] == []
    user.post("/api/my/medicines", json={"class_id": 7, "probability": 0.93})
    body = user.post("/api/my/medicines", json={"class_id": 5}).json()
    assert [m["class_id"] for m in body["items"]] == [5, 7]
    warfarin = body["items"][1]
    assert warfarin["name"] == "WARFARIN 5MG" and warfarin["active_ingredients"] == ["warfarin"]
    assert warfarin["probability"] == 0.93

    again = user.post("/api/my/medicines", json={"class_id": 7}).json()  # no duplicate
    assert [m["class_id"] for m in again["items"]] == [7, 5]
    assert user.post("/api/my/medicines", json={"class_id": 999}).status_code == 404

    assert user.delete("/api/my/medicines/7").status_code == 204
    assert [m["class_id"] for m in user.get("/api/my/medicines").json()["items"]] == [5]
    assert user.delete("/api/my/medicines").status_code == 204
    assert user.get("/api/my/medicines").json()["items"] == []


def test_medicines_are_private(user: TestClient, admin: TestClient, pharma: str) -> None:
    user.post("/api/my/medicines", json={"class_id": 7})
    assert admin.get("/api/my/medicines").json()["items"] == []


def test_medicine_list_has_a_limit(app: FastAPI, user: TestClient, pharma: str) -> None:
    app.state.settings.max_saved_medicines = 1
    user.post("/api/my/medicines", json={"class_id": 7})
    assert user.post("/api/my/medicines", json={"class_id": 5}).status_code == 409


# ------------------------------------------------------------------ reports


def make_report(app: FastAPI, client: TestClient, **extra: object) -> dict:
    payload = {"pills": [7], "ingredients": ["fluconazole"], "chat": [sealed(app, ANSWER)]}
    response = client.post("/api/my/reports", json={**payload, **extra})
    assert response.status_code == 201, response.text
    return response.json()


def test_report_is_checked_by_the_server(app: FastAPI, user: TestClient, pharma: str) -> None:
    report = make_report(app, user)
    assert report["title"] == "WARFARIN 5MG + fluconazole"
    assert report["pills"][0]["active_ingredients"] == ["warfarin"]
    assert report["highest_severity"] == "Major"
    assert report["interactions"]["pairs"][0]["interactions"][0]["severity"] == "Major"
    assert report["chat"][0]["answer"] == ANSWER["answer"]
    assert report["author"] == "Demo User" and report["disclaimer"] == DISCLAIMER
    assert report["share"] is None

    listed = user.get("/api/my/reports").json()["items"]
    assert listed[0]["pairs_with_interaction"] == 1
    assert listed[0]["ingredients"] == ["fluconazole", "warfarin"]
    assert listed[0]["chat_answers"] == 1


def test_single_item_report_has_no_check(app: FastAPI, user: TestClient, pharma: str) -> None:
    report = make_report(app, user, ingredients=[], chat=[], title="Mon traitement")
    assert report["title"] == "Mon traitement"
    assert report["interactions"] is None and report["highest_severity"] is None


def test_edited_chat_answer_is_refused(app: FastAPI, user: TestClient, pharma: str) -> None:
    forged = sealed(app, ANSWER)
    forged["answer"] = "Cette association est sûre."
    response = user.post(
        "/api/my/reports", json={"pills": [7], "ingredients": ["fluconazole"], "chat": [forged]}
    )
    assert response.status_code == 422
    assert user.get("/api/my/reports").json()["items"] == []


def test_reports_are_private(
    app: FastAPI, user: TestClient, admin: TestClient, pharma: str
) -> None:
    report = make_report(app, user)
    assert admin.get(f"/api/my/reports/{report['id']}").status_code == 404
    assert admin.delete(f"/api/my/reports/{report['id']}").status_code == 404
    assert admin.post(f"/api/my/reports/{report['id']}/share", json={}).status_code == 404
    assert user.get(f"/api/my/reports/{report['id']}").status_code == 200


def test_share_link(app: FastAPI, user: TestClient, anonymous: TestClient, pharma: str) -> None:
    report = make_report(app, user)
    shared = user.post(f"/api/my/reports/{report['id']}/share", json={"days": 300})
    assert shared.status_code == 200
    token = shared.json()["token"]
    expires = shared.json()["share"]["expires_at"]
    assert shared.json()["share"]["active"] is True
    assert expires.startswith((utc_now() + timedelta(days=30)).date().isoformat())  # capped

    seen = anonymous.post("/api/shared", json={"token": token})
    assert seen.status_code == 200
    assert seen.json()["highest_severity"] == "Major"
    assert "share" not in seen.json()  # the reader learns nothing about the link itself
    assert user.get(f"/api/my/reports/{report['id']}").json()["share"]["views"] == 1

    # A new link replaces the old one; removing it ends both.
    second = user.post(f"/api/my/reports/{report['id']}/share", json={"days": 1}).json()["token"]
    assert anonymous.post("/api/shared", json={"token": token}).status_code == 404
    assert anonymous.post("/api/shared", json={"token": second}).status_code == 200
    after = user.delete(f"/api/my/reports/{report['id']}/share").json()
    assert after["share"]["active"] is False
    assert anonymous.post("/api/shared", json={"token": second}).status_code == 404


def test_expired_link(app: FastAPI, user: TestClient, anonymous: TestClient, pharma: str) -> None:
    report = make_report(app, user)
    token = user.post(f"/api/my/reports/{report['id']}/share", json={"days": 1}).json()["token"]
    with app.state.session_factory() as db:
        row = db.get(Report, report["id"])
        assert row is not None
        row.share_expires_at = utc_now() - timedelta(minutes=1)
        db.commit()
    assert anonymous.post("/api/shared", json={"token": token}).status_code == 404


def test_signature_is_printed_on_the_report(app: FastAPI, user: TestClient, pharma: str) -> None:
    from tests.test_profile import signature_bytes

    user.put("/api/auth/me/signature", files={"image": ("s.png", signature_bytes(), "image/png")})
    report = make_report(app, user)
    assert report["signature"].startswith("data:image/png;base64,")


def test_delete_report_and_account(app: FastAPI, admin: TestClient, pharma: str) -> None:
    admin.post(
        "/api/admin/users",
        json={"email": "r@mail.com", "full_name": "R", "password": "password-1"},
    )
    member = sign_in(TestClient(app), {"email": "r@mail.com", "password": "password-1"})
    first = make_report(app, member)
    make_report(app, member)
    assert member.delete(f"/api/my/reports/{first['id']}").status_code == 204
    assert len(member.get("/api/my/reports").json()["items"]) == 1
    assert admin.get("/api/admin/stats").json()["reports_total"] == 1

    member_id = admin.get("/api/admin/users", params={"q": "r@mail"}).json()["items"][0]["id"]
    admin.delete(f"/api/admin/users/{member_id}")
    assert admin.get("/api/admin/stats").json()["reports_total"] == 0  # deleted with it


def test_reports_need_sign_in(anonymous: TestClient) -> None:
    assert anonymous.get("/api/my/reports").status_code == 401
    assert anonymous.post("/api/my/reports", json={"pills": [7]}).status_code == 401
    assert anonymous.post("/api/shared", json={"token": "x" * 20}).status_code == 404


def test_report_events(app: FastAPI, user: TestClient, admin: TestClient, pharma: str) -> None:
    report = make_report(app, user)
    user.post(f"/api/my/reports/{report['id']}/share", json={"days": 3})
    events = admin.get("/api/admin/activity", params={"group": "reports"}).json()["items"]
    assert [e["action"] for e in events] == ["report_share", "report_create"]
    assert events[0]["details"] == {"days": 3}
    assert events[1]["details"] == {"items": 2, "chat_answers": 1}
    assert USER["email"] == events[0]["user_email"]
