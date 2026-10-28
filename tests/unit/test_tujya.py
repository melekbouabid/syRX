"""Tujya data steps: RxNav ingredient mapping (mocked HTTP) and DDInter filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import respx

from synergyrx_data.tujya import ingredients as ing
from synergyrx_data.tujya.__main__ import update_class_file
from synergyrx_data.tujya.interactions import ddinter_names, filter_interactions, read_aliases
from tests.conftest import ClientFactory

R = ing.RXNAV


def _history(*ingredients: tuple[str, str], name: str = "product") -> dict[str, object]:
    return {
        "rxcuiStatusHistory": {
            "attributes": {"name": name},
            "definitionalFeatures": {
                "ingredientAndStrength": [
                    {
                        "baseRxcui": cui,
                        "baseName": base,
                        "activeIngredientName": f"{base} hydrochloride",
                        "numeratorValue": "10",  # strengths exist upstream but are never kept
                    }
                    for cui, base in ingredients
                ]
            },
        }
    }


async def test_ndc_to_ingredients_uses_history_for_obsolete_ndcs(
    mock_http: respx.MockRouter, make_client: ClientFactory
) -> None:
    mock_http.get(f"{R}/ndcstatus.json", params={"ndc": "11111111101"}).respond(
        json={"ndcStatus": {"status": "ACTIVE", "rxcui": "100", "conceptName": "A 10 MG [Brand]"}}
    )
    mock_http.get(f"{R}/ndcstatus.json", params={"ndc": "11111111102"}).respond(
        json={
            "ndcStatus": {
                "status": "OBSOLETE",
                "rxcui": "",
                "ndcHistory": [{"activeRxcui": "", "originalRxcui": "100"}],
            }
        }
    )
    mock_http.get(f"{R}/rxcui/100/historystatus.json").respond(
        json=_history(("2", "losartan"), ("1", "Hydrochlorothiazide"))
    )
    client = make_client(hosts={ing.HOST})
    rows = await ing.map_class(
        client,
        {"class_id": "7", "class_key": "111111111_1", "ndc11_list": "11111111101;11111111102"},
    )
    assert [(r.ndc11, r.ndc_status, r.ingredient) for r in rows] == [
        ("11111111101", "ACTIVE", "hydrochlorothiazide"),
        ("11111111101", "ACTIVE", "losartan"),
        ("11111111102", "OBSOLETE", "hydrochlorothiazide"),
        ("11111111102", "OBSOLETE", "losartan"),
    ]
    assert {r.method for r in rows} == {"ndc"}
    summary = ing.summarise(rows)[7]
    assert summary == {
        "rxcui": "100",
        "active_ingredients": "hydrochlorothiazide; losartan",
        "ingredient_rxcuis": "1; 2",
        "ingredient_method": "ndc",
    }


async def test_unknown_ndc_falls_back_to_the_generic_name(
    mock_http: respx.MockRouter, make_client: ClientFactory
) -> None:
    mock_http.get(f"{R}/ndcstatus.json").respond(json={"ndcStatus": {"status": "UNKNOWN"}})
    term = mock_http.get(f"{R}/approximateTerm.json").respond(
        json={"approximateGroup": {"candidate": [{"rxcui": "55"}]}}
    )
    mock_http.get(f"{R}/rxcui/55/historystatus.json").respond(json=_history(("11289", "warfarin")))
    client = make_client(hosts={ing.HOST})
    rows = await ing.map_class(
        client,
        {
            "class_id": "1",
            "class_key": "k",
            "ndc11_list": "00000000000",
            "generic_name": "Warfarin Sodium Tablets, USP",
        },
    )
    assert term.calls.last.request.url.params["term"] == "Warfarin Sodium"
    assert [(r.ingredient, r.method, r.ndc11) for r in rows] == [("warfarin", "name", "")]


def test_rows_round_trip_and_class_file_update(tmp_path: Path) -> None:
    rows = [ing.IngredientRow(0, "k0", "1", "ACTIVE", "9", "p", "11289", "warfarin", "ndc")]
    ing.write_rows(rows, tmp_path / "rows.csv")
    assert ing.read_rows(tmp_path / "rows.csv") == rows

    classes = tmp_path / "pill_classes.csv"
    classes.write_text("class_id,name\n0,A\n1,B\n", encoding="utf-8")
    update_class_file(classes, ing.summarise(rows))
    update_class_file(classes, ing.summarise(rows))  # idempotent: columns are not repeated
    table = pd.read_csv(classes, dtype="string", keep_default_na=False)
    assert list(table.columns) == [
        "class_id",
        "name",
        "rxcui",
        "active_ingredients",
        "ingredient_rxcuis",
        "ingredient_method",
    ]
    assert table["active_ingredients"].tolist() == ["warfarin", ""]


def _ddinter() -> pd.DataFrame:
    rows = [
        ("Warfarin", "Fluconazole", "Major", "DDInter1", "DDInter2"),
        ("Fluconazole", "Warfarin", "Moderate", "DDInter2", "DDInter1"),  # reversed, other level
        ("Warfarin", "Aspirin", "Major", "DDInter1", "DDInter3"),
        ("Valproic acid", "Warfarin", None, "DDInter4", "DDInter1"),
        ("Aspirin", "Ibuprofen", "Moderate", "DDInter3", "DDInter5"),
    ]
    return pd.DataFrame(
        rows,
        columns=["drug_a_name", "drug_b_name", "level", "drug_a_source_id", "drug_b_source_id"],
    )


def test_interactions_between_ingredients_are_ordered_and_deduplicated(tmp_path: Path) -> None:
    aliases_file = tmp_path / "aliases.csv"
    aliases_file.write_text(
        "rxnorm_ingredient,ddinter_name,note\nvalproate,Valproic acid,x\n", encoding="utf-8"
    )
    aliases = read_aliases(aliases_file)
    assert aliases == {"valproic acid": "valproate"}
    assert "valproate" in ddinter_names(_ddinter(), aliases)

    table = filter_interactions(
        _ddinter(), ["warfarin", "fluconazole", "valproate"], aliases=aliases
    )
    assert table["pair"].tolist() == ["fluconazole + warfarin", "valproate + warfarin"]
    first, second = table.to_dict("records")
    assert first["severity"] == "Major" and first["conflict"]  # most severe level kept
    assert (first["ddinter_id_a"], first["ddinter_id_b"]) == ("DDInter2", "DDInter1")
    assert second["severity"] == "Unknown" and not second["conflict"]
    assert first["source"] == "DDInter 2.0" and first["license"] == "CC BY-NC-SA 4.0"
    assert "safe" not in " ".join(table["description"]).lower()


def test_interactions_with_any_drug() -> None:
    table = filter_interactions(_ddinter(), ["aspirin"], both=False)
    assert table["pair"].tolist() == ["aspirin + ibuprofen", "aspirin + warfarin"]
    assert filter_interactions(_ddinter(), ["paracetamol"]).empty
