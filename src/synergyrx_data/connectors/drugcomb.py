"""DrugComb: bulk summary table from a pinned Zenodo record."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import DrugCombRecord, Record

_RENAME = {
    "block_id": "source_record_id",
    "drug_row": "drug_a_name",
    "drug_col": "drug_b_name",
}
_NULLS = ["NULL", "\\N", "NA", "NaN", ""]


class DrugCombConnector(Connector):
    info = SourceInfo(
        name="drugcomb",
        title="DrugComb v1.4 summary table",
        category="synergy",
        access="bulk",
        homepage="https://drugcomb.org/",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms_url="https://zenodo.org/records/11102665",
        citation=(
            "Zheng S, et al. DrugComb update: a more comprehensive drug sensitivity data "
            "repository and analysis portal. Nucleic Acids Res. 2021;49(W1):W174-W184. "
            "doi:10.1093/nar/gkab438"
        ),
        hosts=("zenodo.org", "drugcomb.org"),
        links=(
            Link("https://drugcomb.org/", "DrugComb portal", "html", "checked"),
            Link(
                "https://zenodo.org/records/11102665",
                "DrugComb v1.4 bulk dataset (Zenodo)",
                "html",
                "checked",
            ),
        ),
        notes=(
            "The bulk table has no study or tissue column: rows cannot be traced back to "
            "O'Neil, ALMANAC, etc. without the DrugComb API."
        ),
    )
    default_version = "1.4"
    # Pinned to the published record: name, size and checksum come from the Zenodo record.
    default_options: ClassVar[dict[str, Any]] = {
        "record_id": 11102665,
        "filename": "summary_table_v1.4.csv",
        "size_bytes": 193_184_734,
        "md5": "c11efbdcae4a860c2374c1505a66599b",
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"synergy": DrugCombRecord}

    @property
    def url(self) -> str:
        return (
            f"https://zenodo.org/api/records/{self.options['record_id']}"
            f"/files/{self.options['filename']}/content"
        )

    async def discover(self) -> list[PlannedAction]:
        return [
            self.plan_file(
                self.url,
                self.options["filename"],
                self.options["size_bytes"],
                sample_rows=self.sample,
            )
        ]

    async def fetch(self) -> list[Artifact]:
        full = self.sample is None
        return [
            await self.fetch_file(
                self.url,
                self.options["filename"],
                expected_size=self.options["size_bytes"] if full else None,
                expected_md5=self.options["md5"] if full else None,
                sample_rows=self.sample,
            )
        ]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        artifact = artifacts[0]
        raw = pd.read_csv(
            artifact.path,
            dtype={
                "block_id": "string",
                "drug_row": "string",
                "drug_col": "string",
                "cell_line_name": "string",
            },
            na_values=_NULLS,
            keep_default_na=False,
            low_memory=False,
        )
        # Single-agent screens have no second drug; they are not combinations.
        mono = raw["drug_col"].isna() | raw["drug_row"].isna()
        self.stats = {"raw_rows": len(raw), "monotherapy_rows_skipped": int(mono.sum())}
        table = synergy_frame(raw.loc[~mono].rename(columns=_RENAME))
        return {"synergy": self.provenance(artifact).apply(table)}
