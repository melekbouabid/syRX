"""GDSC cell-line annotations (optional, disabled by default).

cancerrxgene.org now answers 410 Gone: GDSC moved into the Sanger DepMap, and its cell-line
annotations are the Cell Model Passports model list, which this connector reads.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import Record, SangerModelRecord


class GdscConnector(Connector):
    info = SourceInfo(
        name="gdsc",
        title="GDSC cell-line annotations (Sanger Cell Model Passports)",
        category="cell_line",
        access="bulk",
        homepage="https://www.cancerrxgene.org/",
        license="Not verified automatically: read the Sanger terms before enabling",
        terms_url="https://cellmodelpassports.sanger.ac.uk/downloads",
        citation=(
            "Yang W, et al. Genomics of Drug Sensitivity in Cancer (GDSC). Nucleic Acids Res. "
            "2013;41(D1):D955-61; van der Meer D, et al. Cell Model Passports. "
            "Nucleic Acids Res. 2019;47(D1):D923-9"
        ),
        hosts=("cog.sanger.ac.uk", "www.cancerrxgene.org", "cellmodelpassports.sanger.ac.uk"),
        links=(
            Link("https://www.cancerrxgene.org/", "GDSC / Cancer RxGene", "html", "verify"),
            Link(
                "https://cellmodelpassports.sanger.ac.uk/downloads",
                "Cell Model Passports downloads",
                "html",
                "",
            ),
        ),
        optional=True,
        default_enabled=False,
        notes="cancerrxgene.org is retired (HTTP 410); data now lives in Cell Model Passports.",
    )
    default_version = "latest"
    default_options: ClassVar[dict[str, Any]] = {
        "url": "https://cog.sanger.ac.uk/cmp/download/model_list_latest.csv.gz",
        "filename": "model_list_latest.csv.gz",
        "max_age_days": 30,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"models": SangerModelRecord}

    async def discover(self) -> list[PlannedAction]:
        return [self.plan_file(self.options["url"], self.options["filename"], 150_000)]

    async def fetch(self) -> list[Artifact]:
        return [
            await self.fetch_file(
                self.options["url"],
                self.options["filename"],
                max_age_days=self.options["max_age_days"],
            )
        ]

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        raw = pd.read_csv(
            artifacts[0].path,
            compression="gzip",
            dtype="string",
            na_values=["NA", ""],
            keep_default_na=False,
            nrows=self.sample,
        )
        models = pd.DataFrame(
            {
                "sanger_model_id": raw["model_id"],
                "cell_line_name": raw["model_name"],
                "depmap_id": raw.get("BROAD_ID"),
                "tissue": raw.get("tissue"),
                "cancer_type": raw.get("cancer_type"),
                "model_type": raw.get("model_type"),
                "cosmic_id": raw.get("COSMIC_ID"),
            }
        )
        return {"models": self.provenance(artifacts[0]).apply(models)}
