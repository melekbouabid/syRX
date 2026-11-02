"""Pharma tables and their loader, on a temporary SQLite database and tiny CSV inputs."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import make_engine
from app.pharma import Ingredient, Interaction, Pill, PillIngredient, Source
from app.pharma_loader import Inputs, load

CLASSES = """class_id,class_key,ndc9,part,ndc11_list,name,generic_name,labeled_by,imprint,color,shape,n_consumer_available,n_consumer_eligible,n_consumer_selected,n_reference_selected,consumer_mb,rxcui,active_ingredients,ingredient_rxcuis,ingredient_method
0,111111111_1,111111111,1,11111111101;11111111102,HYZAAR 50MG,LOSARTAN AND HYDROCHLOROTHIAZIDE,MERCK,MRK;717,YELLOW,OVAL,40,35,30,2,30.1,979464,hydrochlorothiazide; losartan,5487; 52175,ndc
1,222222222_1,222222222,1,22222222201,WARFARIN 5MG,WARFARIN,TARO,TV;5,PEACH,ROUND,30,25,25,1,20.0,855332,warfarin,11289,ndc
2,333333333_1,333333333,1,33333333301,FORMULA B PLUS TAB,VITAMINS AND MINERALS,MAJOR,CPC;1027,ORANGE,OVAL,36,25,25,0,30.8,,,,
"""  # noqa: E501

INGREDIENTS = """class_id,class_key,ndc11,ndc_status,product_rxcui,product_name,ingredient_rxcui,ingredient,method
0,111111111_1,11111111101,ACTIVE,979464,p,5487,hydrochlorothiazide,ndc
0,111111111_1,11111111101,ACTIVE,979464,p,52175,losartan,ndc
0,111111111_1,11111111102,OBSOLETE,979464,p,5487,hydrochlorothiazide,ndc
0,111111111_1,11111111102,OBSOLETE,979464,p,52175,losartan,ndc
1,222222222_1,22222222201,ACTIVE,855332,p,11289,warfarin,ndc
"""  # noqa: E501

INTERACTIONS = """ingredient_a,ingredient_b,pair,severity,description,conflict,ddinter_id_a,ddinter_id_b,source,source_url,license
fluconazole,warfarin,fluconazole + warfarin,Major,DDInter 2.0 lists an interaction between fluconazole and warfarin with severity Major.,False,DDInter2,DDInter1,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
hydrochlorothiazide,losartan,hydrochlorothiazide + losartan,Unknown,x,False,DDInter7,DDInter8,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
losartan,warfarin,losartan + warfarin,Moderate,y,False,DDInter8,DDInter1,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
"""  # noqa: E501


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    files = {"classes": CLASSES, "ingredients": INGREDIENTS, "interactions": INTERACTIONS}
    for name, text in files.items():
        (tmp_path / f"{name}.csv").write_text(text, encoding="utf-8")
    return Inputs(
        classes=tmp_path / "classes.csv",
        ingredients=tmp_path / "ingredients.csv",
        interactions=tmp_path / "interactions.csv",
    )


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[Engine]:
    engine = make_engine(f"sqlite:///{(tmp_path / 'pharma.db').as_posix()}")
    yield engine
    engine.dispose()


def _count(db: Session, table: type) -> int:
    return db.scalar(select(func.count()).select_from(table)) or 0


def test_load_fills_every_table(engine: Engine, inputs: Inputs) -> None:
    report = load(engine, inputs)
    assert (report.pills, report.ingredients, report.pill_ingredients, report.interactions) == (
        3,
        4,  # hydrochlorothiazide, losartan, warfarin + fluconazole from DDInter
        3,  # duplicated NDC rows count once; the vitamin class has none
        3,
    )
    with Session(engine) as db:
        hyzaar = db.scalars(select(Pill).where(Pill.class_id == 0)).one()
        assert sorted(link.ingredient.name for link in hyzaar.ingredients) == [
            "hydrochlorothiazide",
            "losartan",
        ]
        assert hyzaar.n_images == 32 and hyzaar.source.name == "c3pi"
        vitamins = db.scalars(select(Pill).where(Pill.class_id == 2)).one()
        assert vitamins.ingredients == []
        warfarin = db.scalars(select(Ingredient).where(Ingredient.name == "warfarin")).one()
        assert (warfarin.rxcui, warfarin.ddinter_id) == ("11289", "DDInter1")
        for row in db.scalars(select(Interaction)):
            assert row.ingredient_a_id < row.ingredient_b_id
            assert row.source.name == "ddinter"
        assert {s.name for s in db.scalars(select(Source))} == {"c3pi", "rxnav", "ddinter"}


def test_load_twice_gives_the_same_database(engine: Engine, inputs: Inputs) -> None:
    first = load(engine, inputs)
    assert load(engine, inputs) == first
    with Session(engine) as db:
        assert _count(db, Source) == 3
        assert _count(db, Pill) == 3
        assert _count(db, PillIngredient) == 3
        assert _count(db, Interaction) == 3


def test_a_pair_is_stored_in_one_order_only(engine: Engine, inputs: Inputs) -> None:
    load(engine, inputs)
    with Session(engine) as db:
        row = db.scalars(select(Interaction)).first()
        assert row is not None
        db.add(
            Interaction(
                ingredient_a_id=row.ingredient_b_id,
                ingredient_b_id=row.ingredient_a_id,
                severity="Minor",
                source_id=row.source_id,
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
