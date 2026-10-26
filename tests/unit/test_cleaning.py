"""Pair keys, SMILES standardisation, drug identity, cell lines and scores."""

from __future__ import annotations

import pandas as pd
import pytest

from synergyrx_data.cleaning.cell_lines import (
    build_cell_line_table,
    cell_line_key,
    collect_cell_line_names,
)
from synergyrx_data.cleaning.drugs import (
    build_drug_table,
    collect_mentions,
    name_key,
    normalize_name,
)
from synergyrx_data.cleaning.pairs import add_pair_key, build_synergy_pairs, canonical_pair
from synergyrx_data.cleaning.scores import (
    SCORE_COLUMNS,
    aggregate_replicates,
    find_conflicts,
    flag_outliers,
    source_agreement,
)
from synergyrx_data.cleaning.smiles import standardize_smiles
from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.provenance import PROVENANCE_COLUMNS

FU = "O=c1[nH]cc(F)c(=O)[nH]1"
VELIPARIB = "CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def synergy(source: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = synergy_frame(pd.DataFrame(rows))
    return frame.assign(
        source=source,
        source_url=f"https://example.org/{source}",
        retrieved_at="2026-01-01T00:00:00Z",
        source_version="1",
        license="CC-BY-4.0",
    )


# -------------------------------------------------------------- pair keys


def test_pair_key_ignores_order() -> None:
    assert canonical_pair("B", "A") == canonical_pair("A", "B") == ("A", "B", "A|B")
    frame = pd.DataFrame({"drug_a_id": ["B", "A", "C"], "drug_b_id": ["A", "B", "A"]})
    keyed = add_pair_key(frame)
    assert list(keyed["pair_key"]) == ["A|B", "A|B", "A|C"]
    assert list(keyed["drug_a_id"]) == ["A", "A", "A"]
    assert list(keyed["drug_b_id"]) == ["B", "B", "C"]


def build(frames: dict[str, pd.DataFrame], pubchem: pd.DataFrame | None = None):  # type: ignore[no-untyped-def]
    drugs = build_drug_table(collect_mentions(frames), pubchem=pubchem)
    cells = build_cell_line_table(collect_cell_line_names(frames))
    pairs, stats = build_synergy_pairs(
        frames, drugs.name_map, cells.name_map, drugs.drugs, cells.cell_lines
    )
    return drugs, cells, pairs, stats


def test_a_b_and_b_a_end_up_as_one_row() -> None:
    """The acceptance criterion: the same pair stored both ways is one row."""
    frames = {
        "screen": synergy(
            "screen",
            [
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "ABT-888",
                    "cell_line_name": "A2058",
                    "synergy_loewe": 4.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": VELIPARIB,
                },
                {
                    "drug_a_name": "ABT-888",
                    "drug_b_name": "5-FU",
                    "cell_line_name": "A2058",
                    "synergy_loewe": 8.0,
                    "drug_a_smiles": VELIPARIB,
                    "drug_b_smiles": FU,
                },
            ],
        )
    }
    _, _, pairs, stats = build(frames)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row.synergy_loewe == 6.0  # replicates within a study are averaged...
    assert row.n_replicates == 2 and row.synergy_loewe_std == 2.0  # ...and their spread is kept
    assert row.drug_a_id < row.drug_b_id
    assert row.pair_key == f"{row.drug_a_id}|{row.drug_b_id}"
    assert stats["replicates_averaged"] == 1


def test_same_drug_under_two_names_gives_one_pair_and_self_pairs_are_dropped() -> None:
    frames = {
        "a": synergy(
            "a",
            [
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "ABT-888",
                    "cell_line_name": "A2058",
                    "synergy_bliss": 1.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": VELIPARIB,
                },
                {
                    "drug_a_name": "Veliparib",
                    "drug_b_name": "Fluorouracil",
                    "cell_line_name": "A-2058",
                    "synergy_bliss": 3.0,
                    "drug_a_smiles": VELIPARIB,
                    "drug_b_smiles": FU,
                },
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "Fluorouracil",
                    "cell_line_name": "A2058",
                    "synergy_bliss": 0.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": FU,
                },
            ],
        )
    }
    drugs, cells, pairs, stats = build(frames)
    assert len(drugs.drugs) == 2 and len(cells.cell_lines) == 1
    assert len(pairs) == 1 and pairs["synergy_bliss"].iloc[0] == 2.0
    assert stats["self_pairs_dropped"] == 1


def test_scores_from_different_sources_are_never_merged() -> None:
    row = {
        "drug_a_name": "5-FU",
        "drug_b_name": "ABT-888",
        "cell_line_name": "A2058",
        "drug_a_smiles": FU,
        "drug_b_smiles": VELIPARIB,
    }
    frames = {
        "one": synergy("one", [{**row, "synergy_loewe": 12.0, "synergy_bliss": 5.0}]),
        "two": synergy("two", [{**row, "synergy_loewe": -9.0, "study_id": "S1"}]),
    }
    _, _, pairs, _ = build(frames)
    assert len(pairs) == 2 and set(pairs["source"]) == {"one", "two"}
    assert set(pairs["n_sources"]) == {2}
    one = pairs[pairs["source"] == "one"].iloc[0]
    two = pairs[pairs["source"] == "two"].iloc[0]
    assert (one.synergy_loewe, one.synergy_bliss) == (12.0, 5.0)
    assert two.synergy_loewe == -9.0 and pd.isna(two.synergy_bliss)  # not borrowed from "one"
    assert all(c in pairs.columns for c in [*SCORE_COLUMNS, *PROVENANCE_COLUMNS])

    conflicts = find_conflicts(pairs, tolerance=10.0)
    assert len(conflicts) == 1
    conflict = conflicts.iloc[0]
    assert conflict.synergy_loewe_range == 21.0 and conflict.sign_disagreement
    assert conflict.is_conflict and conflict.sources == "one; two"
    assert conflict.synergy_bliss_n == 1 and pd.isna(conflict.synergy_bliss_range)

    agreement = source_agreement(
        pd.concat([pairs, pairs.assign(cell_line_id="other", synergy_loewe=[1.0, 2.0])])
    )
    loewe = agreement[agreement["score"] == "synergy_loewe"].iloc[0]
    assert loewe.shared_triplets == 2 and loewe.sign_disagreement_rate == 0.5


def test_small_disagreement_is_a_duplicate_not_a_conflict() -> None:
    row = {
        "drug_a_name": "5-FU",
        "drug_b_name": "ABT-888",
        "cell_line_name": "A2058",
        "drug_a_smiles": FU,
        "drug_b_smiles": VELIPARIB,
    }
    frames = {
        "one": synergy("one", [{**row, "synergy_loewe": 12.0}]),
        "two": synergy("two", [{**row, "synergy_loewe": 14.5}]),
    }
    conflicts = find_conflicts(build(frames)[2], tolerance=10.0)
    assert len(conflicts) == 1 and not conflicts["is_conflict"].iloc[0]


def test_replicate_aggregation_and_outlier_flag() -> None:
    frame = pd.DataFrame(
        {
            "k": ["a", "a", "b"],
            "study_id": pd.array([None, None, "S"], dtype="string"),
            **{c: [1.0, 3.0, 500.0] for c in SCORE_COLUMNS},
        }
    )
    out = aggregate_replicates(frame, ["k", "study_id"])
    assert len(out) == 2  # a missing study id still forms a group
    a = out[out["k"] == "a"].iloc[0]
    assert a.synergy_zip == 2.0 and a.n_replicates == 2 and a.synergy_zip_std == 1.0
    assert list(flag_outliers(out, 200.0)) == [False, True]
    assert out.loc[out["k"] == "b", "synergy_zip"].iloc[0] == 500.0  # flagged, not clipped


# ------------------------------------------------------------------ SMILES


def test_standardisation_removes_salts_neutralises_and_canonicalises() -> None:
    plain = standardize_smiles(ASPIRIN)
    variants = [
        "OC(=O)c1ccccc1OC(C)=O",  # same molecule, different atom order
        "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",  # sodium salt
        "CC(=O)Oc1ccccc1C(=O)O.Cl",  # hydrochloride written as a second fragment
    ]
    for variant in variants:
        result = standardize_smiles(variant)
        assert result.valid
        assert result.smiles == plain.smiles
        assert result.inchikey == plain.inchikey == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert "salt or solvent removed" in (standardize_smiles(variants[1]).note or "")


def test_metal_complexes_keep_their_metal() -> None:
    carboplatin = standardize_smiles("N.N.O=C(O)C1(C(=O)O)CCC1.[Pt]")
    assert carboplatin.valid and "Pt" in (carboplatin.smiles or "")
    assert "metal complex" in (carboplatin.note or "")


@pytest.mark.parametrize(
    ("smiles", "issue"),
    [
        ("not a smiles", "unparseable"),
        ("C1CC", "unparseable"),  # unclosed ring
        ("c1ccccc1(C)(C)(C)C", "unparseable"),  # impossible valence
        ("", "missing"),
        (None, "missing"),
    ],
)
def test_invalid_smiles_are_flagged_with_a_reason(smiles: str | None, issue: str) -> None:
    result = standardize_smiles(smiles)
    assert not result.valid and result.smiles is None and result.inchikey is None
    assert issue in (result.issue or "")


def test_invalid_smiles_are_counted_not_silently_removed() -> None:
    """The acceptance criterion: bad structures stay in the table, flagged."""
    frames = {
        "s": synergy(
            "s",
            [
                {
                    "drug_a_name": "Good",
                    "drug_b_name": "Broken",
                    "cell_line_name": "A2058",
                    "synergy_zip": 1.0,
                    "drug_a_smiles": ASPIRIN,
                    "drug_b_smiles": "C1CC((",
                },
                {
                    "drug_a_name": "Good",
                    "drug_b_name": "Nameless",
                    "cell_line_name": "A2058",
                    "synergy_zip": 2.0,
                    "drug_a_smiles": ASPIRIN,
                },
            ],
        )
    }
    drugs, _, pairs, _ = build(frames)
    table = drugs.drugs.set_index("name")
    assert len(table) == 3  # nothing dropped
    assert int((~table["smiles_valid"]).sum()) == 2  # and the bad ones can be counted
    assert table.loc["Broken", "smiles_valid"] == False  # noqa: E712
    assert "unparseable" in table.loc["Broken", "review_reason"]
    assert table.loc["Broken", "smiles_raw"] == "C1CC(("  # the original is kept for review
    assert table.loc["Broken", "drug_id"] == "name:broken"
    assert table.loc["Good", "drug_id"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert set(drugs.review_queue["name"]) == {"Broken", "Nameless"}
    assert len(pairs) == 2  # their measurements are kept too


# ------------------------------------------------------------ drug identity


def test_name_keys() -> None:
    assert normalize_name(f"  5{chr(0x2010)}FU  ") == "5-FU"  # a Unicode hyphen becomes plain
    assert name_key("ABT-888") == name_key("abt 888") == name_key("ABT888") == "abt888"
    assert name_key("5-FU") == "5fu"
    # Enantiomers and the racemate are different compounds: their names must not collide.
    keys = {name_key(n) for n in ("(+)-JQ1", "(-)-JQ1", "(+/-)-JQ1", "JQ1", "(+)-jq1")}
    assert keys == {"plusjq1", "minusjq1", "racjq1", "jq1"}


def pubchem_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "n_cids": 1,
        "status": "found",
        "synonyms": [],
        "title": None,
        "inchikey": None,
        "smiles": None,
        "pubchem_cid": None,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_identity_resolution_confidence_and_review_queue() -> None:
    frames = {
        "with_smiles": synergy(
            "with_smiles",
            [
                {
                    "drug_a_name": "Aspirin",
                    "drug_b_name": "Mismatch",
                    "cell_line_name": "X",
                    "synergy_zip": 1.0,
                    "drug_a_smiles": ASPIRIN,
                    "drug_b_smiles": FU,
                }
            ],
        ),
        "names_only": synergy(
            "names_only",
            [
                {
                    "drug_a_name": "ASPIRIN",
                    "drug_b_name": "Veliparib",
                    "cell_line_name": "X",
                    "synergy_zip": 1.0,
                },
                {
                    "drug_a_name": "Vague",
                    "drug_b_name": "Unknownium",
                    "cell_line_name": "X",
                    "synergy_zip": 1.0,
                },
            ],
        ),
    }
    pubchem = pubchem_table(
        [
            {
                "query_name": "Aspirin",
                "pubchem_cid": 2244,
                "smiles": ASPIRIN,
                "title": "Aspirin",
                "synonyms": ["Acetylsalicylic acid", "ASA"],
            },
            {
                "query_name": "Veliparib",
                "pubchem_cid": 11960529,
                "smiles": VELIPARIB,
                "title": "Veliparib",
                "synonyms": ["ABT-888"],
            },
            {"query_name": "Mismatch", "pubchem_cid": 5, "smiles": VELIPARIB},
            {
                "query_name": "Vague",
                "pubchem_cid": 7,
                "smiles": "CCO",
                "status": "ambiguous",
                "n_cids": 4,
            },
            {"query_name": "Unknownium", "status": "not_found", "n_cids": 0},
        ]
    )
    chembl = pd.DataFrame(
        {
            "query": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"],
            "query_type": ["inchikey"],
            "chembl_id": ["CHEMBL25"],
        }
    )
    result = build_drug_table(collect_mentions(frames), pubchem=pubchem, chembl=chembl)
    drugs = result.drugs.set_index("name")

    aspirin = drugs.loc["Aspirin"]
    assert aspirin.resolution_method == "source_smiles+pubchem"
    assert aspirin.resolution_confidence == 1.0 and not aspirin.needs_review
    assert aspirin.pubchem_cid == 2244 and aspirin.chembl_id == "CHEMBL25"
    assert {"Acetylsalicylic acid", "ASA"} <= set(aspirin.synonyms)
    assert "ASPIRIN" not in aspirin.synonyms  # same word as the name: not listed twice
    assert set(aspirin.sources) == {"with_smiles", "names_only"}  # two spellings, one drug

    assert drugs.loc["Veliparib", "resolution_method"] == "pubchem_name"
    assert drugs.loc["Veliparib", "resolution_confidence"] == 0.9

    mismatch = drugs.loc["Mismatch"]
    assert mismatch.resolution_method == "source_smiles_conflict" and mismatch.needs_review
    assert "disagree" in mismatch.review_reason and pd.isna(mismatch.pubchem_cid)

    vague = drugs.loc["Vague"]
    assert vague.resolution_method == "pubchem_name_ambiguous" and vague.needs_review
    assert "4 PubChem compounds" in vague.review_reason

    unknown = drugs.loc["Unknownium"]
    assert unknown.resolution_method == "unresolved" and unknown.resolution_confidence == 0.0
    assert not unknown.smiles_valid and "not found in PubChem" in unknown.review_reason

    assert set(result.review_queue["name"]) == {"Mismatch", "Vague", "Unknownium"}
    assert (
        dict(zip(result.name_map["raw_name"], result.name_map["drug_id"], strict=True))["ASPIRIN"]
        == aspirin.drug_id
    )
    assert all(c in result.drugs.columns for c in PROVENANCE_COLUMNS)


def test_manual_override_wins() -> None:
    frames = {
        "s": synergy(
            "s",
            [
                {
                    "drug_a_name": "Codename-1",
                    "drug_b_name": "Aspirin",
                    "cell_line_name": "X",
                    "synergy_zip": 1.0,
                }
            ],
        )
    }
    overrides = pd.DataFrame(
        {"name": ["codename 1"], "smiles": [FU], "pubchem_cid": ["3385"], "note": ["from paper"]}
    ).astype("string")
    drugs = build_drug_table(collect_mentions(frames), overrides=overrides).drugs.set_index("name")
    row = drugs.loc["Codename-1"]
    assert row.resolution_method == "manual_override" and row.pubchem_cid == 3385
    assert row.smiles_valid and not row.needs_review


# -------------------------------------------------------------- cell lines


def test_cell_line_keys_strip_punctuation_and_supplier_suffixes() -> None:
    assert cell_line_key("MDA-MB-231") == cell_line_key("MDAMB231") == "MDAMB231"
    assert cell_line_key("MDA-MB-231/ATCC") == "MDAMB231"
    assert cell_line_key("HL-60(TB)") == "HL60"
    assert cell_line_key("NCI/ADR-RES") == "NCIADRRES"


def test_cell_lines_are_mapped_to_depmap_and_unmapped_ones_are_queued() -> None:
    frames = {
        "a": synergy(
            "a",
            [
                {
                    "drug_a_name": "A",
                    "drug_b_name": "B",
                    "cell_line_name": "OVCAR-3",
                    "tissue": "ovarian",
                    "synergy_zip": 1.0,
                },
                {
                    "drug_a_name": "A",
                    "drug_b_name": "B",
                    "cell_line_name": "MDA-MB-231/ATCC",
                    "synergy_zip": 1.0,
                },
                {
                    "drug_a_name": "A",
                    "drug_b_name": "B",
                    "cell_line_name": "Mystery-9",
                    "tissue": "unknown tissue",
                    "synergy_zip": 1.0,
                },
            ],
        ),
        "b": synergy(
            "b",
            [
                {
                    "drug_a_name": "A",
                    "drug_b_name": "B",
                    "cell_line_name": "MDAMB231",
                    "synergy_zip": 2.0,
                }
            ],
        ),
    }
    models = pd.DataFrame(
        {
            "depmap_id": ["ACH-000001", "ACH-000768"],
            "cell_line_name": ["NIH:OVCAR-3", "MDA-MB-231"],
            "stripped_name": ["NIHOVCAR3", "MDAMB231"],
            "ccle_name": ["NIHOVCAR3_OVARY", "MDAMB231_BREAST"],
            "lineage": ["Ovary/Fallopian Tube", "Breast"],
            "primary_disease": ["Ovarian Epithelial Tumor", "Invasive Breast Carcinoma"],
        }
    )
    result = build_cell_line_table(
        collect_cell_line_names(frames), models, aliases={"OVCAR3": "NIHOVCAR3"}
    )
    table = result.cell_lines.set_index("cell_line_id")
    assert set(table.index) == {"ACH-000001", "ACH-000768", "name:MYSTERY9"}
    breast = table.loc["ACH-000768"]
    assert breast["name"] == "MDA-MB-231" and breast.tissue == "Breast" and breast.mapped
    assert set(breast.source_names) == {"MDA-MB-231/ATCC", "MDAMB231"}  # two spellings merged
    assert set(breast.sources) == {"a", "b"}
    assert table.loc["ACH-000001", "match_method"] == "alias"
    mystery = table.loc["name:MYSTERY9"]
    assert not mystery.mapped and mystery.needs_review and mystery.tissue == "unknown tissue"
    assert list(result.review_queue["cell_line_id"]) == ["name:MYSTERY9"]


def test_without_depmap_everything_is_unmapped_but_kept() -> None:
    frames = {
        "a": synergy(
            "a",
            [
                {
                    "drug_a_name": "A",
                    "drug_b_name": "B",
                    "cell_line_name": "HT-29",
                    "synergy_zip": 1.0,
                }
            ],
        )
    }
    table = build_cell_line_table(collect_cell_line_names(frames)).cell_lines
    assert list(table["cell_line_id"]) == ["name:HT29"] and not table["mapped"].any()


def test_derived_lines_do_not_make_the_parent_ambiguous() -> None:
    """Found on real DepMap data: `A375_SKIN_CJ1` must not compete with `A375`."""
    models = pd.DataFrame(
        {
            "depmap_id": ["ACH-000219", "ACH-002001", "ACH-000900", "ACH-000901"],
            "cell_line_name": ["A-375", "A375 SKIN CJ1", "Twin one", "Twin two"],
            "stripped_name": ["A375", "A375SKINCJ1", "TWIN", "TWIN"],
            "ccle_name": ["A375_SKIN", "A375_SKIN_CJ1", "TWIN_LUNG", "TWIN_SKIN"],
            "lineage": ["Skin", "Skin", "Lung", "Skin"],
            "primary_disease": ["Melanoma"] * 4,
        }
    )
    frames = {
        "a": synergy(
            "a",
            [
                {"drug_a_name": "A", "drug_b_name": "B", "cell_line_name": "A375",
                 "synergy_zip": 1.0},
                {"drug_a_name": "A", "drug_b_name": "B", "cell_line_name": "TWIN",
                 "synergy_zip": 1.0},
            ],
        )
    }  # fmt: skip
    table = build_cell_line_table(collect_cell_line_names(frames), models).cell_lines
    by_name = table.set_index("stripped_name")
    assert by_name.loc["A375", "cell_line_id"] == "ACH-000219" and by_name.loc["A375", "mapped"]
    # Two models with the same stripped name is a real ambiguity: not guessed, queued.
    twin = by_name.loc["TWIN"]
    assert not twin.mapped and twin.needs_review and "2 DepMap models" in twin.review_reason
