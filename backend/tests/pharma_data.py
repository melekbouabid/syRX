"""A tiny version of the medicine tables for the API tests (scan, pills, interactions).

Classes 5, 7 and 9 match the fake classifier's colours in fake_models.py: red -> 5 (a
losartan + hydrochlorothiazide combination), green -> 7 (warfarin), blue -> 9 (a
multivitamin RxNav has no ingredient for). Class 11 is a second warfarin product.
"""

from __future__ import annotations

from pathlib import Path

from app.db import make_engine
from app.pharma_loader import Inputs, load

CLASSES = """class_id,class_key,ndc9,part,ndc11_list,name,generic_name,labeled_by,imprint,color,shape,n_consumer_selected,n_reference_selected
5,111111111_1,111111111,1,11111111101,HYZAAR 50MG,LOSARTAN AND HYDROCHLOROTHIAZIDE,MERCK,MRK;717,YELLOW,OVAL,30,2
7,222222222_1,222222222,1,22222222201,WARFARIN 5MG,WARFARIN,TARO,TV;5,PEACH,ROUND,25,1
9,333333333_1,333333333,1,33333333301,FORMULA B PLUS TAB,VITAMINS AND MINERALS,MAJOR,CPC;1027,ORANGE,OVAL,25,0
11,444444444_1,444444444,1,44444444401,COUMADIN 2MG,WARFARIN SODIUM,BMS,COUMADIN;2,LAVENDER,ROUND,24,1
"""  # noqa: E501

INGREDIENTS = """class_id,class_key,ndc11,ndc_status,product_rxcui,product_name,ingredient_rxcui,ingredient,method
5,111111111_1,11111111101,ACTIVE,979464,p,5487,hydrochlorothiazide,ndc
5,111111111_1,11111111101,ACTIVE,979464,p,52175,losartan,ndc
7,222222222_1,22222222201,ACTIVE,855332,p,11289,warfarin,ndc
11,444444444_1,44444444401,ACTIVE,855333,p,11289,warfarin,ndc
"""  # noqa: E501

INTERACTIONS = """ingredient_a,ingredient_b,pair,severity,description,conflict,ddinter_id_a,ddinter_id_b,source,source_url,license
fluconazole,warfarin,fluconazole + warfarin,Major,DDInter 2.0 lists an interaction between fluconazole and warfarin with severity Major.,False,DDInter2,DDInter1,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
hydrochlorothiazide,losartan,hydrochlorothiazide + losartan,Unknown,DDInter 2.0 lists an interaction between hydrochlorothiazide and losartan with severity Unknown.,False,DDInter7,DDInter8,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
losartan,warfarin,losartan + warfarin,Moderate,DDInter 2.0 lists an interaction between losartan and warfarin with severity Moderate.,False,DDInter8,DDInter1,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
ibuprofen,paracetamol,ibuprofen + paracetamol,Minor,DDInter 2.0 lists an interaction between ibuprofen and paracetamol with severity Minor.,False,DDInter3,DDInter4,DDInter 2.0,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0
"""  # noqa: E501


def load_pharma(url: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = {"classes": CLASSES, "ingredients": INGREDIENTS, "interactions": INTERACTIONS}
    for name, text in files.items():
        (directory / f"{name}.csv").write_text(text, encoding="utf-8")
    engine = make_engine(url)
    try:
        load(
            engine,
            Inputs(
                classes=directory / "classes.csv",
                ingredients=directory / "ingredients.csv",
                interactions=directory / "interactions.csv",
            ),
        )
    finally:
        engine.dispose()
