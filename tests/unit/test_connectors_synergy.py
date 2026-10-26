"""Synergy and cell-line connectors, end to end through `pipeline.fetch`, with mocked HTTP."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import pickle
import sys
import types
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from synergyrx_data import pipeline
from synergyrx_data.config import Settings, SourceConfig
from synergyrx_data.connectors._safe_pickle import load_dataframe_pickle
from synergyrx_data.manifest import Manifest
from synergyrx_data.provenance import PROVENANCE_COLUMNS, has_provenance
from synergyrx_data.state import MANUAL_STEP, NEEDS_ATTENTION, StateStore

DRUGCOMB_URL = "https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content"
DRUGCOMB_CSV = (
    b'"block_id","drug_row","drug_col","cell_line_name","conc_r_unit","conc_c_unit","css",'
    b'"synergy_zip","synergy_bliss","synergy_loewe","synergy_hsa","ic50_row","ic50_col",'
    b'"ri_row","ri_col","css_row","css_col","S"\n'
    b'"1","5-FU","ABT-888","A2058","uM","uM","30.8","3.86","6.25","-2.95","5.53","5","3","1","1","2","3","19"\n'
    b'"2","ABT-888","5-FU","A2058","uM","uM","27.4","8.24","12.33","3.12","11.61","5","3","1","1","2","3","16"\n'
    b'"3","5-FU","NULL","A2058","uM","uM","NULL","NULL","NULL","NULL","NULL","5","NULL","1","NULL","2","NULL","NULL"\n'
    b'"4","BORTEZOMIB","DASATINIB","HT29","uM","uM","\\N","\\N","\\N","\\N","\\N","5","3","1","1","2","3","\\N"\n'
    b'"5","BORTEZOMIB","DASATINIB","HT29","uM","uM","47.2","-3.99","-0.42","-6.51","-2.89","5","3","1","1","2","3","-2"\n'
)


def serve(payload: bytes, content_type: str = "text/plain"):  # type: ignore[no-untyped-def]
    """A handler that honours Range requests, like the real file servers do."""

    def handler(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": content_type}
        if "range" in request.headers:
            spec = request.headers["range"].removeprefix("bytes=")
            start = int(spec.split("-")[0])
            end = int(spec.split("-")[1]) if spec.split("-")[1] else len(payload) - 1
            end = min(end, len(payload) - 1)
            headers["content-range"] = f"bytes {start}-{end}/{len(payload)}"
            return httpx.Response(206, content=payload[start : end + 1], headers=headers)
        return httpx.Response(200, content=payload, headers=headers)

    return handler


def pin(settings: Settings, source: str, **options: object) -> None:
    settings.sources[source] = SourceConfig(enabled=True, options=dict(options))


async def run(settings: Settings, source: str) -> pipeline.SourceRun:
    return (await pipeline.fetch(settings, [source]))[0]


def network_calls(router: respx.MockRouter) -> int:
    return len(router.calls)


# ---------------------------------------------------------------- DrugComb


@pytest.fixture
def drugcomb(settings: Settings, mock_http: respx.MockRouter, allow_robots: None) -> respx.Route:
    pin(
        settings,
        "drugcomb",
        size_bytes=len(DRUGCOMB_CSV),
        md5=hashlib.md5(DRUGCOMB_CSV).hexdigest(),
    )
    return mock_http.get(DRUGCOMB_URL).mock(side_effect=serve(DRUGCOMB_CSV))


async def test_drugcomb_full_fetch_parse_validate(
    settings: Settings, drugcomb: respx.Route
) -> None:
    result = await run(settings, "drugcomb")
    assert result.outcome == "ok"
    assert result.tables == {"synergy": 3}  # 5 rows - 1 single-agent - 1 without any score
    assert result.rejects == {"synergy": 1}
    assert result.stats["monotherapy_rows_skipped"] == 1

    table = pd.read_parquet(settings.interim_dir / "drugcomb" / "synergy.parquet")
    assert has_provenance(table)
    assert set(table["source"]) == {"drugcomb"}
    assert set(table["source_version"]) == {"1.4"}
    assert set(table["license"]) == {"CC-BY-4.0"}
    assert table["source_url"].iloc[0] == DRUGCOMB_URL
    # The four scores stay in four columns.
    first = table.iloc[0]
    assert (first.synergy_zip, first.synergy_bliss, first.synergy_loewe, first.synergy_hsa) == (
        3.86,
        6.25,
        -2.95,
        5.53,
    )
    assert table["study_id"].isna().all()  # the bulk table has no study column

    rejects = pd.read_parquet(settings.interim_dir / "drugcomb" / "synergy.rejects.parquet")
    assert list(rejects["source_record_id"]) == ["4"]  # kept for inspection, not dropped

    manifest = Manifest(settings)
    raw = manifest.entries["data/raw/drugcomb/1.4/summary_table_v1.4.csv"]
    assert raw.sha256 == hashlib.sha256(DRUGCOMB_CSV).hexdigest()
    assert (raw.size_bytes, raw.version, raw.license) == (len(DRUGCOMB_CSV), "1.4", "CC-BY-4.0")
    assert raw.url == DRUGCOMB_URL and raw.terms_url and raw.retrieved_at.endswith("Z")
    interim = manifest.entries["data/interim/drugcomb/synergy.parquet"]
    assert interim.rows == 3 and interim.derived_from == [raw.path]


async def test_second_run_downloads_and_duplicates_nothing(
    settings: Settings, drugcomb: respx.Route, mock_http: respx.MockRouter
) -> None:
    await run(settings, "drugcomb")
    interim = settings.interim_dir / "drugcomb" / "synergy.parquet"
    before = (network_calls(mock_http), interim.stat().st_mtime_ns, interim.read_bytes())
    manifest_before = Manifest(settings).entries

    again = await run(settings, "drugcomb")

    assert again.outcome == "unchanged"
    assert again.tables == {"synergy": 3}
    assert network_calls(mock_http) == before[0]  # not one request
    assert interim.stat().st_mtime_ns == before[1]  # not rewritten
    assert interim.read_bytes() == before[2]
    assert Manifest(settings).entries == manifest_before  # same hashes, same timestamps
    assert len(pd.read_parquet(interim)) == 3  # no duplicated rows


async def test_modified_raw_file_is_detected_and_fetched_again(
    settings: Settings, drugcomb: respx.Route
) -> None:
    await run(settings, "drugcomb")
    raw = settings.raw_dir / "drugcomb" / "1.4" / "summary_table_v1.4.csv"
    raw.write_bytes(raw.read_bytes() + b"tampered")
    result = await run(settings, "drugcomb")
    assert result.outcome == "ok" and drugcomb.call_count == 2
    assert raw.read_bytes() == DRUGCOMB_CSV


async def test_sample_mode_fetches_only_the_head_and_is_labelled(
    settings: Settings, drugcomb: respx.Route
) -> None:
    settings.sample = 2
    result = await run(settings, "drugcomb")
    assert result.tables == {"synergy": 2}
    request = drugcomb.calls.last.request
    assert request.headers["range"].startswith("bytes=0-")
    sample = settings.raw_dir / "drugcomb" / "1.4" / "summary_table_v1.4.sample2.csv"
    assert sample.read_bytes().count(b"\n") == 3  # header + 2 rows
    manifest = Manifest(settings)
    assert manifest.get(sample).is_sample  # type: ignore[union-attr]
    assert manifest.entries["data/interim/drugcomb/synergy.parquet"].is_sample
    assert StateStore(settings.state_dir / "connectors.json").get("drugcomb")["is_sample"]


async def test_wrong_checksum_puts_the_source_in_needs_attention(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    pin(settings, "drugcomb", size_bytes=len(DRUGCOMB_CSV), md5="0" * 32)
    mock_http.get(DRUGCOMB_URL).mock(side_effect=serve(DRUGCOMB_CSV))
    result = await run(settings, "drugcomb")
    assert result.outcome == NEEDS_ATTENTION and "MD5" in result.message
    assert not (settings.interim_dir / "drugcomb").exists()
    state = StateStore(settings.state_dir / "connectors.json").get("drugcomb")
    assert state["state"] == NEEDS_ATTENTION


async def test_dead_link_is_reported_and_other_sources_still_run(
    settings: Settings, drugcomb: respx.Route, mock_http: respx.MockRouter
) -> None:
    mock_http.get(url__startswith="https://bioinf.jku.at/software/DeepSynergy/").respond(404)
    runs = await pipeline.fetch(settings, ["drugcomb", "oneil"])
    outcomes = {r.source: r.outcome for r in runs}
    assert outcomes == {"drugcomb": "ok", "oneil": NEEDS_ATTENTION}
    assert "404" in next(r.message for r in runs if r.source == "oneil")


# ------------------------------------------------------------------ O'Neil

LABELS = (
    b",drug_a_name,drug_b_name,cell_line,synergy,fold\n"
    b"5-FU_ABT-888_A2058,5-FU,ABT-888,A2058,7.69,2\n"
    b"5-FU_ABT-888_A2780,5-FU,ABT-888,A2780,7.77,2\n"
    b"MYSTERY_ABT-888_A2780,MYSTERY,ABT-888,A2780,1.5,0\n"
)
SMILES = b"5-FU,O=c1[nH]cc(F)c(=O)[nH]1\nABT-888,CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1\n"


async def test_oneil_joins_smiles_and_reports_loewe_only(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    base = "https://bioinf.jku.at/software/DeepSynergy/"
    mock_http.get(base + "labels.csv").mock(side_effect=serve(LABELS, "text/csv"))
    mock_http.get(base + "smiles.csv").mock(side_effect=serve(SMILES, "text/csv"))
    result = await run(settings, "oneil")
    assert result.tables == {"synergy": 2}
    assert result.rejects == {"synergy": 1}  # MYSTERY has no SMILES: rejected, and counted
    table = pd.read_parquet(settings.interim_dir / "oneil" / "synergy.parquet")
    assert list(table["synergy_loewe"]) == [7.69, 7.77]
    assert table[["synergy_bliss", "synergy_hsa", "synergy_zip"]].isna().all().all()
    assert table["drug_a_smiles"].iloc[0] == "O=c1[nH]cc(F)c(=O)[nH]1"
    assert set(table["study_id"]) == {"ONEIL"}
    assert "Unspecified" in table["license"].iloc[0]


# --------------------------------------------------------------------- TDC


def tdc_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Drug1_ID": ["5-FU", "ABT-888", "5-FU"],
            "Drug2_ID": ["ABT-888", "MK-1775", "MK-1775"],
            "Cell_Line_ID": ["A2058", "A2058", "HT29"],
            "CSS": [30.0, 20.0, 10.0],
            "Synergy_ZIP": [1.0, 2.0, 3.0],
            "Synergy_Bliss": [1.5, 2.5, 3.5],
            "Synergy_Loewe": [-1.0, -2.0, -3.0],
            "Synergy_HSA": [0.5, 0.6, 0.7],
            "Drug1": ["O=c1[nH]cc(F)c(=O)[nH]1", "CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1", "CCO"],
            "Drug2": ["CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1", "CCN", "CCN"],
            "CellLine": [[0.1, 0.2]] * 3,
        }
    )


@pytest.fixture
def tdc(settings: Settings, mock_http: respx.MockRouter) -> respx.Route:
    # Plain object columns, like the file TDC published in 2021 (no Arrow-backed strings).
    frame = tdc_frame().drop(columns="CellLine")
    text = [c for c in frame.columns if not c.startswith(("CSS", "Synergy"))]
    frame = frame.astype(dict.fromkeys(text, object))
    frame.columns = pd.Index(list(frame.columns), dtype=object)
    payload = pickle.dumps(frame, protocol=4)
    pin(
        settings,
        "tdc",
        files={"drugcomb": {"file_id": 4215720, "size_bytes": len(payload)}},
        loader="builtin",
    )
    return mock_http.get("https://dataverse.harvard.edu/api/access/datafile/4215720").mock(
        side_effect=serve(payload, "application/octet-stream")
    )


async def test_tdc_builtin_loader_keeps_all_scores_and_smiles(
    settings: Settings, tdc: respx.Route
) -> None:
    result = await run(settings, "tdc")
    assert result.tables == {"synergy": 3} and result.stats["loader"] == "builtin"
    table = pd.read_parquet(settings.interim_dir / "tdc" / "synergy.parquet")
    assert list(table.columns[-5:]) == PROVENANCE_COLUMNS
    assert has_provenance(table)
    assert list(table["synergy_bliss"]) == [1.5, 2.5, 3.5]
    assert list(table["css"]) == [30.0, 20.0, 10.0]
    assert table["drug_b_smiles"].iloc[1] == "CCN"
    assert set(table["source_version"]) == {"dataverse-4215720"}


async def test_tdc_sample_is_reproducible(settings: Settings, tdc: respx.Route) -> None:
    settings.sample = 2
    await run(settings, "tdc")
    first = pd.read_parquet(settings.interim_dir / "tdc" / "synergy.parquet")
    assert len(first) == 2
    settings.refresh = True
    await run(settings, "tdc")
    again = pd.read_parquet(settings.interim_dir / "tdc" / "synergy.parquet")
    # A forced re-download gets a new retrieval time; the sampled rows must be the same.
    assert again.drop(columns="retrieved_at").equals(first.drop(columns="retrieved_at"))


async def test_tdc_uses_pytdc_drugsyn_when_installed(
    settings: Settings, tdc: respx.Route, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PyTDC cannot be installed on this Python, so `tdc.multi_pred.DrugSyn` is stubbed."""
    seen: dict[str, str] = {}

    class DrugSyn:
        def __init__(self, name: str, path: str) -> None:
            seen.update(name=name, path=path)

        def get_data(self) -> pd.DataFrame:
            return tdc_frame()

    module = types.ModuleType("tdc.multi_pred")
    module.DrugSyn = DrugSyn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tdc", types.ModuleType("tdc"))
    monkeypatch.setitem(sys.modules, "tdc.multi_pred", module)
    settings.sources["tdc"].options["loader"] = "auto"
    result = await run(settings, "tdc")
    assert result.stats["loader"] == "pytdc"
    assert seen["name"] == "drugcomb"
    # PyTDC is pointed at the folder our own client downloaded into: it never fetches.
    assert Path(seen["path"]) == settings.raw_dir / "tdc" / "dataverse-4215720"
    assert (Path(seen["path"]) / "drugcomb.pkl").is_file()
    assert tdc.call_count == 1


def test_restricted_loader_refuses_pickles_that_would_run_code(tmp_path: Path) -> None:
    class Exploit:
        def __reduce__(self):  # type: ignore[no-untyped-def]
            return (print, ("arbitrary code would run here",))

    evil = tmp_path / "evil.pkl"
    evil.write_bytes(pickle.dumps(Exploit()))
    with pytest.raises(pickle.UnpicklingError, match=r"refusing to load builtins\.print"):
        load_dataframe_pickle(evil)
    not_a_frame = tmp_path / "list.pkl"
    not_a_frame.write_bytes(pickle.dumps([1, 2, 3]))
    with pytest.raises(pickle.UnpicklingError, match="DataFrame"):
        load_dataframe_pickle(not_a_frame)


# ------------------------------------------------------------- NCI-ALMANAC

NCI_ROBOTS = "User-agent: *\nCrawl-delay: 15\nDisallow: /\nAllow: /$\nAllow: /spaces/*/pages/\n"
ALMANAC_HEADER = (
    "COMBODRUGSEQ,SCREENER,STUDY,TESTDATE,PLATE,PANELNBR,CELLNBR,PREFIX1,NSC1,SAMPLE1,"
    "CONCINDEX1,CONC1,CONCUNIT1,PREFIX2,NSC2,SAMPLE2,CONCINDEX2,CONC2,CONCUNIT2,PERCENTGROWTH,"
    "PERCENTGROWTHNOTZ,TESTVALUE,CONTROLVALUE,TZVALUE,EXPECTEDGROWTH,SCORE,VALID,PANEL,CELLNAME"
)


def almanac_zip() -> bytes:
    def row(nsc2: str, c1: int, c2: int, score: str, plate: str = "P1") -> str:
        return (
            f"1,1A,S100,01/01/2012,{plate},1,1,S,740,1,{c1},1e-6,M,S,{nsc2},1,{c2},1e-6,M,50,,"
            f"1,1,1,60,{score},Y,Breast Cancer,MCF7"
        )

    rows = [
        row("750", 1, 1, "10"),
        row("750", 1, 1, "20", plate="P2"),  # plate replicate of the same dose pair: mean 15
        row("750", 1, 2, "5"),
        row("750", 2, 1, "-2"),
        row("", 1, 0, ""),  # single-agent record: not a combination
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ComboDrugGrowth_Nov2017.csv", "\n".join([ALMANAC_HEADER, *rows]) + "\n")
    return buffer.getvalue()


async def test_almanac_respects_robots_and_asks_for_a_manual_download(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    mock_http.get("https://wiki.nci.nih.gov/robots.txt").respond(200, text=NCI_ROBOTS)
    download = mock_http.get(url__startswith="https://wiki.nci.nih.gov/download/").respond(
        200, content=almanac_zip()
    )
    result = await run(settings, "nci_almanac")
    assert result.outcome == MANUAL_STEP
    assert "data/manual/nci_almanac/ComboDrugGrowth_Nov2017.zip" in result.message
    assert not download.called  # the disallowed path was never requested


async def test_almanac_parses_a_manually_placed_file_without_any_request(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    mock_http.get("https://wiki.nci.nih.gov/robots.txt").respond(200, text=NCI_ROBOTS)
    manual = settings.manual_dir / "nci_almanac"
    manual.mkdir(parents=True)
    (manual / "ComboDrugGrowth_Nov2017.zip").write_bytes(almanac_zip())
    (manual / "ComboCompoundNames_small.txt").write_text(
        "740\tMethotrexate\n750\tBusulfan\n750\tMyleran\n", encoding="utf-8"
    )
    result = await run(settings, "nci_almanac")
    assert result.outcome == "ok" and result.tables == {"synergy": 1}
    row = pd.read_parquet(settings.interim_dir / "nci_almanac" / "synergy.parquet").iloc[0]
    assert (row.drug_a_name, row.drug_b_name) == ("Methotrexate", "Busulfan")
    assert (row.drug_a_source_id, row.drug_b_source_id) == ("NSC-740", "NSC-750")
    assert row.combo_score == pytest.approx(15 + 5 - 2)  # sum over dose pairs
    assert pd.isna(row.synergy_bliss) and pd.isna(row.synergy_loewe)  # ComboScore is its own column
    assert (row.cell_line_name, row.tissue, row.study_id) == ("MCF7", "Breast Cancer", "ALMANAC:1A")
    assert all(call.request.url.path == "/robots.txt" for call in mock_http.calls) or not len(
        mock_http.calls
    )
    assert (
        Manifest(settings).entries["data/manual/nci_almanac/ComboDrugGrowth_Nov2017.zip"].layer
        == "manual"
    )


# ------------------------------------------------------------------ DepMap

MODEL_CSV = (
    b"ModelID,PatientID,CellLineName,StrippedCellLineName,OncotreeLineage,"
    b"OncotreePrimaryDisease,CCLEName,SangerModelID\n"
    b"ACH-000001,PT-1,NIH:OVCAR-3,NIHOVCAR3,Ovary/Fallopian Tube,Ovarian Epithelial Tumor,"
    b"NIHOVCAR3_OVARY,SIDM00105\n"
    b"ACH-000788,PT-2,A2058,A2058,Skin,Melanoma,A2058_SKIN,SIDM00797\n"
    b"BAD-ID,PT-3,Broken,BROKEN,Skin,Melanoma,BROKEN_SKIN,\n"
)
EXPRESSION_CSV = (
    b",TSPAN6 (7105),TNMD (64102),DPM1 (8813)\nACH-000001,4.33,0.0,6.08\nACH-000788,3.1,0.5,5.9\n"
)


@pytest.fixture
def depmap(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> dict[str, respx.Route]:
    def entry(file_id: int, name: str, payload: bytes) -> dict[str, object]:
        return {
            "id": file_id,
            "name": name,
            "size": len(payload),
            "computed_md5": hashlib.md5(payload).hexdigest(),
            "download_url": f"https://ndownloader.figshare.com/files/{file_id}",
        }

    listing = {
        "title": "DepMap 24Q4 Public",
        "files": [
            entry(1, "Model.csv", MODEL_CSV),
            entry(2, "OmicsExpressionProteinCodingGenesTPMLogp1.csv", EXPRESSION_CSV),
        ],
    }
    return {
        "listing": mock_http.get("https://api.figshare.com/v2/articles/27993248").respond(
            json=listing
        ),
        "models": mock_http.get("https://ndownloader.figshare.com/files/1").mock(
            side_effect=serve(MODEL_CSV, "text/csv")
        ),
        "expression": mock_http.get("https://ndownloader.figshare.com/files/2").mock(
            side_effect=serve(EXPRESSION_CSV, "text/csv")
        ),
    }


async def test_depmap_models_and_expression_from_figshare(
    settings: Settings, depmap: dict[str, respx.Route]
) -> None:
    result = await run(settings, "depmap")
    assert result.outcome == "ok"
    assert result.tables == {"models": 2, "expression": 2}
    assert result.rejects == {"models": 1}  # BAD-ID is not a DepMap identifier
    models = pd.read_parquet(settings.interim_dir / "depmap" / "models.parquet")
    assert list(models["depmap_id"]) == ["ACH-000001", "ACH-000788"]
    assert models["stripped_name"].iloc[0] == "NIHOVCAR3"
    assert models["lineage"].iloc[1] == "Skin"
    expression = pd.read_parquet(settings.interim_dir / "depmap" / "expression.parquet")
    assert list(expression.columns[:4]) == [
        "depmap_id",
        "TSPAN6 (7105)",
        "TNMD (64102)",
        "DPM1 (8813)",
    ]
    assert expression["TSPAN6 (7105)"].dtype == "float32"
    assert has_provenance(expression)
    assert set(expression["source_version"]) == {"24Q4"}

    again = await run(settings, "depmap")
    assert again.outcome == "unchanged"
    assert depmap["models"].call_count == 1 and depmap["expression"].call_count == 1
    assert depmap["listing"].call_count == 1  # the file list came from the disk cache


async def test_depmap_new_expression_layout_is_understood(
    settings: Settings, mock_http: respx.MockRouter, depmap: dict[str, respx.Route]
) -> None:
    from synergyrx_data.connectors.base import Artifact
    from synergyrx_data.connectors.depmap import DepMapConnector

    path = settings.raw_dir / "new.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        ",SequencingID,ModelID,IsDefaultEntryForModel,TSPAN6 (7105)\n"
        "0,CDS-1,ACH-000001,Yes,4.3\n1,CDS-2,ACH-000001,No,9.9\n",
        encoding="utf-8",
    )
    artifact = Artifact("depmap", path, "u", "25Q2", "x", 1, "2026-01-01T00:00:00Z", "CC-BY-4.0")
    frame = DepMapConnector._read_expression(artifact)
    assert frame.to_dict("records") == [
        {"depmap_id": "ACH-000001", "TSPAN6 (7105)": pytest.approx(4.3)}
    ]


async def test_gdsc_reads_the_cell_model_passports_list(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    csv = (
        "model_id,model_name,synonyms,model_type,tissue,cancer_type,COSMIC_ID,BROAD_ID\n"
        'SIDM00105,OVCAR-3,"NIH:OVCAR-3;OVCAR3",Cell Line,Ovary,Ovarian Carcinoma,905933,'
        "ACH-000001\n"
        "SIDM01274,HCM-SANG-0314-C15,NA,Organoid,Esophagus,Esophageal Carcinoma,NA,NA\n"
    )
    route = mock_http.get("https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz").mock(
        side_effect=serve(gzip.compress(csv.encode()), "application/gzip")
    )
    result = await run(settings, "gdsc")  # disabled by default, but runs when asked for
    assert result.tables == {"models": 2}
    models = pd.read_parquet(settings.interim_dir / "gdsc" / "models.parquet")
    assert models["depmap_id"].iloc[0] == "ACH-000001" and pd.isna(models["depmap_id"].iloc[1])
    assert (await run(settings, "gdsc")).outcome == "unchanged" and route.call_count == 1


def test_only_sources_enabled_by_default_run_unless_named(settings: Settings) -> None:
    default = pipeline.select_sources(settings, None)
    assert {
        "drugcomb",
        "tdc",
        "oneil",
        "nci_almanac",
        "depmap",
        "pubchem",
        "chembl",
        "pubmed",
        "europepmc",
    } == set(default)
    assert {"drugbank", "ddinter", "gdsc", "pubmed_baseline", "pubtator"}.isdisjoint(default)
    assert default.index("drugcomb") < default.index("pubchem") < default.index("chembl")
    assert default.index("chembl") < default.index("pubmed") < default.index("europepmc")
    with pytest.raises(KeyError, match="unknown source"):
        pipeline.select_sources(settings, ["not-a-source"])


def test_manifest_json_is_stable_and_complete(settings: Settings, tmp_path: Path) -> None:
    file = settings.raw_dir / "x" / "a.csv"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"a,b\n1,2\n")
    manifest = Manifest(settings)
    entry = manifest.record_file(
        file, source="x", layer="raw", url="https://example.org/a.csv", version="1", license="CC0"
    )
    manifest.save()
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    stored = payload["files"]["data/raw/x/a.csv"]
    assert {"sha256", "size_bytes", "url", "version", "retrieved_at", "license"} <= set(stored)
    assert stored["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    # Recording the same bytes again keeps the original retrieval time.
    assert Manifest(settings).record_file(file, source="x", layer="raw").retrieved_at == (
        entry.retrieved_at
    )
    assert Manifest(settings).is_current(file, url="https://example.org/a.csv", version="1")
    assert not Manifest(settings).is_current(file, url="https://example.org/a.csv", version="2")


async def test_a_new_source_version_is_checked_for_drift(
    settings: Settings, drugcomb: respx.Route, mock_http: respx.MockRouter
) -> None:
    first = await run(settings, "drugcomb")
    assert first.drift == []  # a first snapshot is not drift
    # Version 1.5 arrives with most of its rows gone.
    smaller = b"\n".join(DRUGCOMB_CSV.split(b"\n")[:2]) + b"\n"
    settings.sources["drugcomb"] = SourceConfig(
        enabled=True,
        version="1.5",
        options={"size_bytes": len(smaller), "md5": hashlib.md5(smaller).hexdigest()},
    )
    mock_http.get(DRUGCOMB_URL).mock(side_effect=serve(smaller))
    second = await run(settings, "drugcomb")
    assert second.tables == {"synergy": 1}
    assert [(d["table"], d["kind"]) for d in second.drift] == [
        ("interim/drugcomb/synergy", "rows_changed")
    ]
    state = StateStore(settings.state_dir / "connectors.json").get("drugcomb")
    assert state["drift"][0]["detail"].startswith("3 -> 1 rows")


def test_two_processes_do_not_erase_each_others_manifest_and_state(settings: Settings) -> None:
    """A long download and a `clean` can overlap; each holds its own in-memory copy."""
    folder = settings.raw_dir / "x"
    folder.mkdir(parents=True)
    for name in ("a.csv", "b.csv", "c.csv"):
        (folder / name).write_bytes(name.encode())

    slow, quick = Manifest(settings), Manifest(settings)  # both loaded before either saves
    slow.record_file(folder / "a.csv", source="slow", layer="raw")
    quick.record_file(folder / "b.csv", source="quick", layer="raw")
    quick.save()
    quick.record_file(folder / "c.csv", source="quick", layer="raw")
    quick.save()
    slow.save()  # finishes last, with a stale view of the file
    on_disk = Manifest(settings).entries
    assert set(on_disk) == {"data/raw/x/a.csv", "data/raw/x/b.csv", "data/raw/x/c.csv"}
    assert set(slow.entries) == set(on_disk)  # and it now sees the others' entries too

    path = settings.state_dir / "connectors.json"
    first, second = StateStore(path), StateStore(path)
    first.update("depmap", state="ok")
    second.update("pubmed", state="ok")
    first.update("depmap", rows=5)
    final = StateStore(path)
    assert final.get("pubmed")["state"] == "ok"
    assert final.get("depmap")["state"] == "ok" and final.get("depmap")["rows"] == 5
    assert not list(settings.state_dir.glob("*.lock"))  # locks are released


def test_a_stale_lock_from_a_dead_process_is_cleared(settings: Settings) -> None:
    import os
    import time

    from synergyrx_data.manifest import file_lock

    target = settings.state_dir / "connectors.json"
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True)
    lock.write_text("")
    old = time.time() - 600
    os.utime(lock, (old, old))
    with file_lock(target, timeout=2):
        assert lock.exists()
    assert not lock.exists()
    lock.write_text("")  # a live lock held by someone else: give up after the timeout
    with pytest.raises(TimeoutError), file_lock(target, timeout=0.2):
        pass


async def test_a_corrupt_file_in_one_source_does_not_stop_the_others(
    settings: Settings, drugcomb: respx.Route, mock_http: respx.MockRouter
) -> None:
    manual = settings.manual_dir / "nci_almanac"
    manual.mkdir(parents=True)
    (manual / "ComboDrugGrowth_Nov2017.zip").write_bytes(b"this is not a zip archive")
    (manual / "ComboCompoundNames_small.txt").write_text("740\tMethotrexate\n", encoding="utf-8")
    runs = await pipeline.fetch(settings, ["nci_almanac", "drugcomb"])
    by_source = {r.source: r for r in runs}
    assert by_source["nci_almanac"].outcome == NEEDS_ATTENTION
    assert "BadZipFile" in by_source["nci_almanac"].message
    assert by_source["drugcomb"].outcome == "ok"  # ran after the failure, unaffected
    state = StateStore(settings.state_dir / "connectors.json").get("nci_almanac")
    assert state["state"] == NEEDS_ATTENTION
