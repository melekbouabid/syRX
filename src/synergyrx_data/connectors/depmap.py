"""DepMap / CCLE cell-line models and expression, from the official Figshare release.

The DepMap portal answers automated requests with a "verifying you are a person" page and
asks not to be scraped, so the portal API is not used. Each DepMap release is also published
on Figshare+, which has a documented API, checksums and a fixed version.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import pandas as pd

from synergyrx_data.connectors.base import Artifact, Connector, Link, PlannedAction, SourceInfo
from synergyrx_data.connectors.schemas import DepMapModelRecord, Record
from synergyrx_data.http.cache import atomic_write_bytes

_GENE_COLUMN = re.compile(r"^.+ \(\d+\)$")
_MODEL_COLUMNS = {
    "depmap_id": "ModelID",
    "cell_line_name": "CellLineName",
    "stripped_name": "StrippedCellLineName",
    "ccle_name": "CCLEName",
    "lineage": "OncotreeLineage",
    "primary_disease": "OncotreePrimaryDisease",
    "sanger_model_id": "SangerModelID",
}


class DepMapConnector(Connector):
    info = SourceInfo(
        name="depmap",
        title="DepMap / CCLE cell-line models and expression",
        category="cell_line",
        access="api",
        homepage="https://depmap.org/portal/",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        terms_url="https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248",
        citation=(
            "DepMap, Broad (2024). DepMap 24Q4 Public. Figshare+. Dataset. "
            "doi:10.25452/figshare.plus.27993248.v1; Ghandi M, et al. Next-generation "
            "characterization of the Cancer Cell Line Encyclopedia. Nature. 2019;569:503-8"
        ),
        hosts=(
            "api.figshare.com",
            "ndownloader.figshare.com",
            "plus.figshare.com",
            "depmap.org",
            "forum.depmap.org",
            "sites.broadinstitute.org",
        ),
        links=(
            Link(
                "https://sites.broadinstitute.org/ccle/datasets", "CCLE datasets", "html", "checked"
            ),
            Link(
                "https://forum.depmap.org/t/depmap-portal-datasets/3695",
                "DepMap forum, dataset pointers",
                "html",
                "checked",
            ),
            Link("https://depmap.org/portal/", "DepMap portal", "html", "verify"),
            Link(
                "https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248",
                "DepMap 24Q4 Public (Figshare+)",
                "html",
                "",
            ),
        ),
        notes="Pinned to release 24Q4. Change `article_id` and `version` together to upgrade.",
    )
    default_version = "24Q4"
    default_options: ClassVar[dict[str, Any]] = {
        "article_id": 27993248,
        "models_file": "Model.csv",
        "models_size_bytes": 645_696,
        "expression_file": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
        "expression_size_bytes": 506_628_654,
        "include_expression": True,
    }
    schemas: ClassVar[dict[str, type[Record]]] = {"models": DepMapModelRecord}

    @property
    def listing_url(self) -> str:
        return f"https://api.figshare.com/v2/articles/{self.options['article_id']}"

    async def discover(self) -> list[PlannedAction]:
        plans = [
            PlannedAction(self.name, "api", self.listing_url, None, None, "file list and MD5s"),
            PlannedAction(
                self.name,
                "download",
                f"Figshare file {self.options['models_file']}",
                self.rel(self.raw_dir / self.options["models_file"]),
                self.options["models_size_bytes"],
            ),
        ]
        if self.options["include_expression"]:
            plans.append(
                self.plan_file(
                    f"Figshare file {self.options['expression_file']}",
                    self.options["expression_file"],
                    self.options["expression_size_bytes"],
                    sample_rows=self.sample,
                )
            )
        return plans

    async def fetch(self) -> list[Artifact]:
        listing = await self.client.fetch(self.listing_url, source=self.name)
        files = {entry["name"]: entry for entry in listing.json().get("files", [])}
        listing_path = self.raw_dir / f"figshare_article_{self.options['article_id']}.json"
        if not listing_path.is_file() or listing_path.read_bytes() != listing.content:
            atomic_write_bytes(listing_path, listing.content)
        artifacts = [self.record_raw(listing_path, self.listing_url, auxiliary=True)]

        wanted: list[tuple[str, int | None]] = [(self.options["models_file"], None)]
        if self.options["include_expression"]:
            wanted.append((self.options["expression_file"], self.sample))
        for name, sample_rows in wanted:
            if name not in files:
                raise ValueError(
                    f"{name} is not in Figshare article {self.options['article_id']}; "
                    "check `article_id` and the file names for this release"
                )
            entry = files[name]
            artifacts.append(
                await self.fetch_file(
                    entry["download_url"],
                    name,
                    expected_size=None if sample_rows else entry.get("size"),
                    expected_md5=None
                    if sample_rows
                    else entry.get("computed_md5") or entry.get("supplied_md5"),
                    sample_rows=sample_rows,
                )
            )
        return artifacts

    @staticmethod
    def _read_expression(artifact: Artifact) -> pd.DataFrame:
        raw = pd.read_csv(artifact.path, low_memory=False)
        if "ModelID" in raw.columns:  # layout used from 25Q2 on
            if "IsDefaultEntryForModel" in raw.columns:
                raw = raw[raw["IsDefaultEntryForModel"].astype("string").str.lower() == "yes"]
            ids = raw["ModelID"]
        else:  # up to 24Q4 the first, unnamed column is the model id
            ids = raw.iloc[:, 0]
        genes = [c for c in raw.columns if _GENE_COLUMN.match(str(c))]
        expression = raw[genes].astype("float32")
        model_ids = pd.DataFrame(
            {"depmap_id": ids.astype("string").to_numpy()}, index=expression.index
        )
        return pd.concat([model_ids, expression], axis=1).reset_index(drop=True)

    def parse(self, artifacts: list[Artifact]) -> dict[str, pd.DataFrame]:
        _, models_artifact, *rest = artifacts
        raw = pd.read_csv(models_artifact.path, dtype="string")
        models = pd.DataFrame(
            {
                target: raw[source] if source in raw.columns else pd.NA
                for target, source in _MODEL_COLUMNS.items()
            }
        ).astype("string")
        tables = {"models": self.provenance(models_artifact).apply(models)}
        self.stats["models"] = len(models)
        if rest:
            expression = self._read_expression(rest[0])
            self.stats["expression_models"] = len(expression)
            self.stats["expression_genes"] = expression.shape[1] - 1
            tables["expression"] = self.provenance(rest[0]).apply(expression)
        return tables
