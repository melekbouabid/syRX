from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from tests.conftest import ADMIN, USER, sign_in

NEW_USER = {"email": "Bob@Mail.com", "full_name": "Bob Stone", "password": "bob-password"}


def user_id(admin: TestClient, email: str) -> int:
    items = admin.get("/api/admin/users", params={"q": email}).json()["items"]
    assert len(items) == 1, items
    return int(items[0]["id"])


def test_regular_user_cannot_use_any_admin_route(user: TestClient, admin: TestClient) -> None:
    target = user_id(admin, ADMIN["email"])
    attempts = [
        user.get("/api/admin/stats"),
        user.get("/api/admin/users"),
        user.post("/api/admin/users", json=NEW_USER),
        user.get(f"/api/admin/users/{target}"),
        user.patch(f"/api/admin/users/{target}", json={"full_name": "Hacked"}),
        user.post(f"/api/admin/users/{target}/password", json={"new_password": "hacked-pass"}),
        user.delete(f"/api/admin/users/{target}"),
    ]
    assert [response.status_code for response in attempts] == [403] * len(attempts)
    sign_in(TestClient(admin.app), ADMIN)  # the administrator's password is untouched


def test_stats(admin: TestClient, user: TestClient) -> None:
    assert admin.get("/api/admin/stats").json() == {
        "users_total": 2,
        "users_active": 2,
        "users_inactive": 0,
        "admins": 1,
        "signups_last_7_days": 2,
        "active_sessions": 2,
    }


def test_list_search_filter_and_paginate(admin: TestClient) -> None:
    for index in range(5):
        payload = {**NEW_USER, "email": f"member{index}@mail.com", "full_name": f"Member {index}"}
        assert admin.post("/api/admin/users", json=payload).status_code == 201

    everyone = admin.get("/api/admin/users").json()
    assert everyone["total"] == 7
    assert "password_hash" not in everyone["items"][0]

    page = admin.get("/api/admin/users", params={"page": 2, "page_size": 3}).json()
    assert (page["total"], page["page"], len(page["items"])) == (7, 2, 3)

    assert admin.get("/api/admin/users", params={"q": "MEMBER 3"}).json()["total"] == 1
    assert admin.get("/api/admin/users", params={"q": "member"}).json()["total"] == 5
    # % and _ are searched for literally, not as wildcards.
    assert admin.get("/api/admin/users", params={"q": "%"}).json()["total"] == 0
    assert admin.get("/api/admin/users", params={"role": "admin"}).json()["total"] == 1
    assert admin.get("/api/admin/users", params={"status": "inactive"}).json()["total"] == 0
    assert admin.get("/api/admin/users", params={"page_size": 1000}).status_code == 422


def test_create_user_and_admin(admin: TestClient, browser: Callable[[], TestClient]) -> None:
    created = admin.post("/api/admin/users", json={**NEW_USER, "role": "admin"})
    assert created.status_code == 201, created.text
    assert created.json()["email"] == "bob@mail.com"
    assert created.json()["role"] == "admin"
    assert admin.post("/api/admin/users", json=NEW_USER).status_code == 409

    bob = sign_in(browser(), {"email": "bob@mail.com", "password": NEW_USER["password"]})
    assert bob.get("/api/admin/stats").status_code == 200


def test_edit_user(admin: TestClient) -> None:
    target = user_id(admin, USER["email"])
    response = admin.patch(
        f"/api/admin/users/{target}",
        json={"full_name": "New Name", "email": "Renamed@Mail.com", "role": "admin"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["full_name"], body["email"], body["role"]) == (
        "New Name",
        "renamed@mail.com",
        "admin",
    )
    taken = admin.patch(f"/api/admin/users/{target}", json={"email": ADMIN["email"]})
    assert taken.status_code == 409
    assert admin.patch("/api/admin/users/9999", json={"full_name": "x"}).status_code == 404


def test_demotion_applies_to_an_open_session(
    admin: TestClient, browser: Callable[[], TestClient]
) -> None:
    admin.post("/api/admin/users", json={**NEW_USER, "role": "admin"})
    bob = sign_in(browser(), {"email": "bob@mail.com", "password": NEW_USER["password"]})
    assert bob.get("/api/admin/stats").status_code == 200

    target = user_id(admin, "bob@mail.com")
    assert admin.patch(f"/api/admin/users/{target}", json={"role": "user"}).status_code == 200
    assert bob.get("/api/admin/stats").status_code == 403
    assert bob.get("/api/auth/me").json()["role"] == "user"


def test_deactivate_and_reactivate(
    admin: TestClient, user: TestClient, browser: Callable[[], TestClient]
) -> None:
    target = user_id(admin, USER["email"])
    off = admin.patch(f"/api/admin/users/{target}", json={"is_active": False})
    assert off.json()["is_active"] is False
    assert user.get("/api/auth/me").status_code == 401  # signed out at once

    blocked = browser().post("/api/auth/login", json=USER)
    assert blocked.status_code == 403
    assert "deactivated" in blocked.json()["detail"]
    # Someone without the password learns nothing about the account's state.
    assert browser().post("/api/auth/login", json={**USER, "password": "x"}).status_code == 401

    admin.patch(f"/api/admin/users/{target}", json={"is_active": True})
    sign_in(browser(), USER)


def test_reset_password_signs_the_user_out(
    admin: TestClient, user: TestClient, browser: Callable[[], TestClient]
) -> None:
    target = user_id(admin, USER["email"])
    weak = admin.post(f"/api/admin/users/{target}/password", json={"new_password": "123"})
    assert weak.status_code == 422

    ok = admin.post(f"/api/admin/users/{target}/password", json={"new_password": "fresh-pass-1"})
    assert ok.status_code == 204
    assert user.get("/api/auth/me").status_code == 401
    assert browser().post("/api/auth/login", json=USER).status_code == 401
    sign_in(browser(), {"email": USER["email"], "password": "fresh-pass-1"})


def test_admin_resetting_own_password_stays_signed_in(
    admin: TestClient, browser: Callable[[], TestClient]
) -> None:
    other_browser = sign_in(browser(), ADMIN)
    me = admin.get("/api/auth/me").json()["id"]
    ok = admin.post(f"/api/admin/users/{me}/password", json={"new_password": "new-admin-pass"})
    assert ok.status_code == 204
    assert admin.get("/api/auth/me").status_code == 200
    assert other_browser.get("/api/auth/me").status_code == 401


def test_delete_user(
    admin: TestClient, user: TestClient, browser: Callable[[], TestClient]
) -> None:
    target = user_id(admin, USER["email"])
    assert admin.delete(f"/api/admin/users/{target}").status_code == 204
    assert admin.get(f"/api/admin/users/{target}").status_code == 404
    assert user.get("/api/auth/me").status_code == 401
    assert browser().post("/api/auth/login", json=USER).status_code == 401
    assert admin.delete(f"/api/admin/users/{target}").status_code == 404
    assert admin.get("/api/admin/stats").json()["active_sessions"] == 1


def test_a_deleted_users_id_is_never_reused(admin: TestClient) -> None:
    first = admin.post("/api/admin/users", json=NEW_USER).json()["id"]
    assert admin.delete(f"/api/admin/users/{first}").status_code == 204
    second = admin.post("/api/admin/users", json=NEW_USER).json()["id"]
    assert second > first


def test_admin_cannot_lock_themselves_out(admin: TestClient) -> None:
    me = admin.get("/api/auth/me").json()["id"]
    assert admin.patch(f"/api/admin/users/{me}", json={"role": "user"}).status_code == 409
    assert admin.patch(f"/api/admin/users/{me}", json={"is_active": False}).status_code == 409
    assert admin.delete(f"/api/admin/users/{me}").status_code == 409
    # Harmless edits of one's own account still work.
    same = admin.patch(
        f"/api/admin/users/{me}", json={"full_name": "Boss", "role": "admin", "is_active": True}
    )
    assert same.status_code == 200
    assert admin.get("/api/admin/stats").json()["admins"] == 1


def test_accounts_are_not_recreated_on_restart(admin: TestClient) -> None:
    """Deleting the demo user must stick: seeding only happens on an empty database."""
    from app.main import create_app

    admin.delete(f"/api/admin/users/{user_id(admin, USER['email'])}")
    with TestClient(create_app(admin.app.state.settings)) as restarted:
        assert restarted.post("/api/auth/login", json=USER).status_code == 401
        assert restarted.post("/api/auth/login", json=ADMIN).status_code == 200
