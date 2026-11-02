"""GET /api/pills and POST /api/interactions on the tiny medicine tables of pharma_data.py."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

NO_KNOWN = "aucune interaction connue dans cette base de données"


def check(client: TestClient, **payload: Any) -> dict[str, Any]:
    response = client.post("/api/interactions", json=payload)
    assert response.status_code == 200, response.text
    # Safety wording: never "safe", always the disclaimer and a source.
    text = response.text.lower()
    assert "safe" not in text and "sûr" not in text
    body: dict[str, Any] = response.json()
    assert body["disclaimer"] == "Information uniquement, consultez un pharmacien ou un médecin."
    assert any(s["name"] == "ddinter" for s in body["sources"])
    return body


def test_routes_require_sign_in(anonymous: TestClient, pharma: str) -> None:
    assert anonymous.get("/api/pills").status_code == 401
    assert anonymous.post("/api/interactions", json={"pills": [5, 7]}).status_code == 401


def test_pill_search(user: TestClient, pharma: str) -> None:
    def names(**params: Any) -> list[str]:
        response = user.get("/api/pills", params=params)
        assert response.status_code == 200, response.text
        return [p["name"] for p in response.json()["items"]]

    assert names() == ["COUMADIN 2MG", "FORMULA B PLUS TAB", "HYZAAR 50MG", "WARFARIN 5MG"]
    assert names(q="warfarin") == ["COUMADIN 2MG", "WARFARIN 5MG"]  # generic name, ingredient
    assert names(q="LOSARTAN") == ["HYZAAR 50MG"]
    assert names(q="mrk") == ["HYZAAR 50MG"]  # imprint
    assert names(q="warfarin taro") == ["WARFARIN 5MG"]  # every word must match
    assert names(q="11") == ["COUMADIN 2MG"]  # class id
    assert names(q="nothing-like-this") == []
    assert names(limit=1) == ["COUMADIN 2MG"]

    body = user.get("/api/pills", params={"q": "hyzaar"}).json()
    (hyzaar,) = body["items"]
    assert hyzaar["class_id"] == 5
    assert hyzaar["generic_name"] == "LOSARTAN AND HYDROCHLOROTHIAZIDE"
    assert hyzaar["active_ingredients"] == ["hydrochlorothiazide", "losartan"]
    assert {s["name"] for s in body["sources"]} == {"c3pi", "rxnav"}


def test_two_pills_with_an_interaction(user: TestClient, pharma: str) -> None:
    body = check(user, pills=[5, 7])
    (pair,) = body["pairs"]
    assert (pair["item_a"]["label"], pair["item_b"]["label"]) == ("HYZAAR 50MG", "WARFARIN 5MG")
    (found,) = pair["interactions"]  # not hydrochlorothiazide + losartan: same product
    assert (found["ingredient_a"], found["ingredient_b"]) == ("losartan", "warfarin")
    assert found["severity"] == "Moderate"
    assert found["source"]["url"] == "https://ddinter2.scbdd.com/"
    assert "Tian Y" in found["source"]["citation"]
    assert pair["message"] is None
    assert body["highest_severity"] == "Moderate"
    assert body["message"].startswith("1 paire(s) sur 1")
    assert {s["name"] for s in body["sources"]} == {"c3pi", "ddinter", "rxnav"}


def test_pairs_are_ordered_by_severity(user: TestClient, pharma: str) -> None:
    body = check(user, pills=[5, 7], ingredients=["Fluconazole"])
    severities = [p["highest_severity"] for p in body["pairs"]]
    assert severities == ["Major", "Moderate", None]  # warfarin+fluconazole first
    assert body["pairs"][0]["item_b"] == {
        "kind": "ingredient",
        "label": "fluconazole",
        "class_id": None,
        "found": True,
        "active_ingredients": ["fluconazole"],
    }
    assert body["highest_severity"] == "Major"
    assert NO_KNOWN in body["pairs"][2]["message"].lower()  # losartan, hctz vs fluconazole


def test_no_interaction_is_never_called_safe(user: TestClient, pharma: str) -> None:
    body = check(user, pills=[7, 11])  # two warfarin products
    (pair,) = body["pairs"]
    assert pair["interactions"] == [] and pair["highest_severity"] is None
    assert pair["shared_ingredients"] == ["warfarin"]
    assert NO_KNOWN in pair["message"].lower()
    assert NO_KNOWN in body["message"].lower()
    assert body["highest_severity"] is None


def test_typed_ingredients_only(user: TestClient, pharma: str) -> None:
    body = check(user, ingredients=["ibuprofen", "paracetamol"])
    assert body["pairs"][0]["interactions"][0]["severity"] == "Minor"
    assert {s["name"] for s in body["sources"]} == {"ddinter"}


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"pills": [7, 9]}, "aucun principe actif connu"),  # the multivitamin
        ({"pills": [7, 99]}, "absent de cette base"),  # unknown class
        ({"pills": [7], "ingredients": ["notadrug"]}, "absent de cette base"),
    ],
)
def test_items_that_cannot_be_checked_say_so(
    user: TestClient, pharma: str, payload: dict[str, Any], expected: str
) -> None:
    body = check(user, **payload)
    (pair,) = body["pairs"]
    assert pair["interactions"] == []
    assert expected in pair["message"].lower()
    assert "n'ont pas pu être vérifiées" in body["message"]


@pytest.mark.parametrize(
    "payload",
    [
        {"pills": [5]},
        {"ingredients": ["warfarin"]},
        {},
        {"pills": list(range(6)), "ingredients": ["a" * 5] * 5},  # 11 items
        {"pills": [-1, 5]},
        {"ingredients": ["warfarin", "x" * 101]},
    ],
)
def test_interaction_payload_limits(user: TestClient, pharma: str, payload: Any) -> None:
    assert user.post("/api/interactions", json=payload).status_code == 422


def test_tables_not_loaded_answer_503(user: TestClient) -> None:
    assert user.get("/api/pills").status_code == 503
    response = user.post("/api/interactions", json={"pills": [5, 7]})
    assert response.status_code == 503
    assert "pharma_loader" in response.json()["detail"]
