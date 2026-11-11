"""Scores of the brand benchmark (src/benchmark/bench_brands.py, step 2.7) and its questions."""

from __future__ import annotations

from app.medicines import DISCLAIMER
from src.benchmark import bench_brands as bb

ITEM = {
    "id": "brand-panadol-warfarin-fr",
    "brands": ["Panadol"],
    "ingredients": ["acetaminophen", "warfarin"],
}


def state(*, brands: bool, cite_brand: bool) -> dict:
    sources = [{"n": 1, "doc_type": "interaction"}]
    if brands:
        sources.insert(0, {"n": 0, "doc_type": "brand"})
    pair = {"ingredient_a": "acetaminophen", "ingredient_b": "warfarin", "severity": "Moderate"}
    return {
        "brands": [{"brand": "Panadol"}] if brands else [],
        "ingredients": ["warfarin", "acetaminophen"] if brands else ["warfarin"],
        "sources": sources,
        "tool_calls": [{"name": "check_interaction", "ok": True}] if brands else [],
        "tool_results": (
            [{"name": "check_interaction", "result": {"pairs": [{"interactions": [pair]}]}}]
            if brands
            else []
        ),
        "answer": f"Le Panadol contient du paracétamol {'[0]' if cite_brand else ''} [1]. "
        + DISCLAIMER,
        "safety": {"action": "ok"},
    }


def test_score_with_and_without_the_brand_table() -> None:
    good = bb.score_state(ITEM, state(brands=True, cite_brand=True), {"acetaminophen", "warfarin"})
    assert all(good[m] for m in bb.METRICS)
    before = bb.score_state(
        ITEM, state(brands=False, cite_brand=False), {"acetaminophen", "warfarin"}
    )
    assert not before["brand_found"] and not before["ingredients_found"]
    assert not before["interaction_tool"] and not before["brand_cited"] and before["cited"]
    assert bb.summarise([good, before])["brand_found"] == 0.5


def test_brand_questions_of_the_benchmark() -> None:
    items = bb.brand_questions()
    assert len(items) >= 8 and any("Panadol" in i["brands"] for i in items)
    for item in items:
        assert len(bb.pair_of(item)) == 2 and bb.pair_of(item) <= set(item["ingredients"])
