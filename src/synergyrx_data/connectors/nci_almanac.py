"""NCI-ALMANAC combination screen.

The official files are attachments of the NCI wiki, whose robots.txt disallows everything
under `/download/` for automated clients. The connector therefore asks for robots.txt
first: if the path is still disallowed it stops and asks a person to download the file in a
browser and drop it under `data/manual/nci_almanac/`. If NCI ever allows it, the same code
downloads it directly.
"""

from __future__ import annotations

import zipfile
from dataclasses import replace
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.connectors.base import (
    Artifact,
    Connector,
    Link,
    ManualStepRequired,
    PlannedAction,
    SourceInfo,
)
from synergyrx_data.connectors.schemas import AlmanacRecord, Record
from synergyrx_data.http.errors import RobotsDisallowedError

PAGE = "https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC"
_KEEP = [
    "NSC1",
    "NSC2",
    "CELLNAME",
    "PANEL",
    "SCREENER",
    "STUDY",
    "CONCINDEX1",
    "CONCINDEX2",
    "SCORE",
]
_GROUP = ["NSC1", "NSC2", "CELLNAME", "PANEL", "SCREENER", "STUDY"]


class NciAlmanacConnector(Connector):
    info = SourceInfo(
        name="nci_almanac",
        title="NCI-ALMANAC combination screen",
        category="synergy",
        access="manual",
        homepage="https://dtp.cancer.gov/ncialmanac",
        license="US Government work (public domain); cite NCI DTP and Holbeck et al. 2017",
        terms_url=PAGE,
        citation=(
            "Holbeck SL, et al. The National Cancer Institute ALMANAC: a comprehensive "
            "screening resource for the detection of anticancer drug pairs with enhanced "
            "therapeutic activity. Cancer Res. 2017;77(13):3564-76. "
            "doi:10.1158/0008-5472.CAN-17-0489"
        ),
        hosts=("wiki.nci.nih.gov", "dtp.cancer.gov"),
        links=(
            Link("https://dtp.cancer.gov/ncialmanac", "NCI-ALMANAC", "html", "checked"),
            Link(PAGE, "NCI-ALMANAC download page (NCI wiki)", "html", ""),
        ),
        notes=(
            "Reports its own ComboScore, not Bliss / Loewe / HSA / ZIP. robots.txt of "
            "wiki.nci.nih.gov disallows /download/, so the file is placed manually."
        ),
    )
    default_version = "2017-11"
    default_options: ClassVar[dict[str, Any]] = {
        "base_url": "https://wiki.nci.nih.gov/download/attachments/338237347/",
        "growth": "ComboDrugGrowth_Nov2017.zip",
        "growth_size_bytes": 85_966_226,
        "names": "ComboCompoundNames_small.txt",
        "chunk_rows": 500_000,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"synergy": AlmanacRecord}

    def _url(self, key: str) -> str:
        return f"{self.options['base_url']}{self.options[key]}"

    def _instructions(self, filename: str) -> str:
        return (
            f"robots.txt of wiki.nci.nih.gov disallows automated download of {filename}. "
            f"Open {PAGE} in a browser, download {filename} and save it as "
            f"{self.rel(self.manual_dir / filename)}, then run the command again."
        )

    async def discover(self) -> list[PlannedAction]:
        plans = []
        for key, size in (("growth", self.options["growth_size_bytes"]), ("names", None)):
            filename = self.options[key]
            if (self.manual_dir / filename).is_file():
                plans.append(
                    PlannedAction(
                        self.name,
                        "manual",
                        self.rel(self.manual_dir / filename),
                        None,
                        None,
                        "read the manually placed file",
                    )
                )
            else:
                plan = self.plan_file(self._url(key), filename, size)
                note = "only if robots.txt allows it; otherwise a manual download is requested"
                plans.append(replace(plan, note=note))
        return plans

    async def _one(self, key: str) -> Artifact:
        filename = self.options[key]
        if (manual := self.manual_artifact(filename, self._url(key))) is not None:
            return manual
        return await self.fetch_file(
            self._url(key),
            filename,
            expected_size=self.options["growth_size_bytes"] if key == "growth" else None,
        )

    async def fetch(self) -> list[Artifact]:
        try:
            artifacts = [await self._one("growth")]
        except RobotsDisallowedError as exc:
            raise ManualStepRequired(self._instructions(self.options["growth"])) from exc
        try:
            artifacts.append(await self._one("names"))
        except RobotsDisallowedError:
            # Optional: without it drugs are named by their NSC number.
            self.log.warning("nci_almanac.names_unavailable", extra={"file": self.options["names"]})
        return artifacts

    @staticmethod
    def _read_names(artifact: Artifact) -> dict[int, str]:
        names = pd.read_csv(
            artifact.path, sep="\t", header=None, names=["nsc", "name"], dtype={"name": "string"}
        ).dropna()
        # Several names per NSC: the first listed is the primary one.
        first = names.drop_duplicates("nsc", keep="first")
        return dict(zip(first["nsc"].astype(int), first["name"].str.strip(), strict=True))

    def _read_combinations(self, artifact: Artifact) -> pd.DataFrame:
        """Stream the CSV inside the zip, keeping valid two-drug records only."""
        kept: list[pd.DataFrame] = []
        raw_rows = 0
        with zipfile.ZipFile(artifact.path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"expected one CSV in {artifact.path.name}, found {members}")
            with archive.open(members[0]) as handle:
                reader = pd.read_csv(
                    handle,
                    chunksize=self.options["chunk_rows"],
                    low_memory=False,
                    nrows=self.sample * 50 if self.sample else None,
                )
                for chunk in reader:
                    raw_rows += len(chunk)
                    chunk.columns = [str(c).strip().upper() for c in chunk.columns]
                    combo = (
                        chunk["NSC2"].notna()
                        & (chunk["CONCINDEX1"] > 0)
                        & (chunk["CONCINDEX2"] > 0)
                        & chunk["SCORE"].notna()
                    )
                    if "VALID" in chunk.columns:
                        combo &= chunk["VALID"].astype("string").str.upper().eq("Y")
                    kept.append(chunk.loc[combo, _KEEP])
        self.stats["raw_rows"] = raw_rows
        return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=_KEEP)

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        growth = artifacts[0]
        names = self._read_names(artifacts[1]) if len(artifacts) > 1 else {}
        combos = self._read_combinations(growth)
        self.stats["combination_rows"] = len(combos)
        # ComboScore of an experiment = sum over its dose pairs of (expected - observed
        # growth). Plate replicates of a dose pair are averaged first.
        per_dose = combos.groupby([*_GROUP, "CONCINDEX1", "CONCINDEX2"], as_index=False)[
            "SCORE"
        ].mean()
        scored = per_dose.groupby(_GROUP, as_index=False).agg(
            combo_score=("SCORE", "sum"), n_dose_points=("SCORE", "size")
        )
        nsc1, nsc2 = scored["NSC1"].astype(int), scored["NSC2"].astype(int)
        table = synergy_frame(
            pd.DataFrame(
                {
                    "drug_a_name": [names.get(n, f"NSC-{n}") for n in nsc1],
                    "drug_b_name": [names.get(n, f"NSC-{n}") for n in nsc2],
                    "drug_a_source_id": "NSC-" + nsc1.astype(str),
                    "drug_b_source_id": "NSC-" + nsc2.astype(str),
                    "cell_line_name": scored["CELLNAME"],
                    "tissue": scored["PANEL"],
                    # The screening site is the study; STUDY is one experiment within it.
                    "study_id": "ALMANAC:" + scored["SCREENER"].astype(str),
                    "source_record_id": scored["STUDY"].astype(str),
                    "combo_score": scored["combo_score"],
                }
            )
        )
        return {"synergy": self.provenance(growth).apply(table)}
